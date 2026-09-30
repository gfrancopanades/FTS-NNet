#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generate a synthetic AP-7 segment-interval dataset with the study's schema.

The corridor data used in the paper (loop-detector traffic, crash affectation
log, weather forecasts) belong to the Catalan Traffic Authority and are not
redistributed. This module writes a *synthetic* file with exactly the same 39
columns, types, separators and filename as the real 5-minute input, so every
pipeline stage runs unchanged on it.

Nothing is sampled from the real data. Static geometry is drawn from the
published summary statistics (Table A.2); traffic, weather and crashes come
from simple parametric processes chosen so that the mechanisms the pipeline
relies on are present:

* traffic follows weekday/weekend and seasonal demand profiles, congests near
  capacity, and the congestion spreads to neighbouring kilometre posts;
* the 1-day and 3-day weather forecasts are noisy versions of a hidden hourly
  weather series, held constant within each hour;
* crash onsets are rare events whose hazard rises with congestion, rain and
  curvature. Each onset stamps an affectation queue upstream of the crash
  (``ACCIDENT`` = 1 over the affected posts and intervals) and depresses speed
  there, as the logged affectation does in the real record.

Results obtained on this file say nothing about the AP-7; it exists to make the
code executable end to end. See ``data/synthetic/README.md`` for the schema.

Usage (from the repository root)::

    # the reference windows of the paper, on a reduced 10-post corridor
    python -m src.data.make_synthetic_data

    # the full 100-post corridor over the whole record (about 5 GB)
    python -m src.data.make_synthetic_data --pk-min 120 --pk-max 219 \\
        --windows 2024-04-04:2025-10-01
"""

from __future__ import annotations

import argparse
import gzip
import io
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.paths import DATA_DIR

# Filename every training/simulation module expects for the 5-minute input.
DATA_FILE = ("CrashGNNLSTM_v1_vel-extinrix_int_geo_mob_wthr_5min_"
             "fund-propag-ltd_from_20240404_to_20251001.csv")

# Stage-1 training windows (June 2024, May 2025) and the held-out month
# (June 2025) of the reference experiment. May and June 2025 are contiguous.
DEFAULT_WINDOWS = "2024-06-01:2024-07-01,2025-05-01:2025-07-01"

COLUMNS = [
    "dat", "via", "pk", "sen", "anyo", "mes", "dia", "diaSem", "hor", "5min",
    "1d_fcst_temperature_2m", "1d_fcst_precipitation", "1d_fcst_snowfall",
    "1d_fcst_cloud_cover", "1d_fcst_wind_speed_10m", "1d_fcst_wind_gusts_10m",
    "3d_fcst_temperature_2m", "3d_fcst_precipitation", "3d_fcst_snowfall",
    "3d_fcst_cloud_cover", "3d_fcst_wind_speed_10m", "3d_fcst_wind_gusts_10m",
    "1d_fcst_rain_binary", "3d_fcst_rain_binary",
    "mean_speed", "speed_imputation", "car", "intensity_imputation",
    "intTot", "intP", "ACCIDENT",
    "C_NIVELL_AFECTACIO", "F_TEMPS_AFECTACIO", "F_LONG_AFECTACIO",
    "mob_esp", "ang_curv", "ang_pend_pos", "ang_pend_neg", "segment",
]

# Public holidays in Catalonia, 2024-2025, plus the main holiday exit days.
# They drive the special-mobility flag ``mob_esp``.
SPECIAL_DAYS = pd.to_datetime([
    "2024-01-01", "2024-01-06", "2024-03-29", "2024-04-01", "2024-05-01",
    "2024-05-20", "2024-06-24", "2024-08-15", "2024-09-11", "2024-10-12",
    "2024-11-01", "2024-12-06", "2024-12-08", "2024-12-25", "2024-12-26",
    "2025-01-01", "2025-01-06", "2025-04-18", "2025-04-21", "2025-05-01",
    "2025-06-09", "2025-06-24", "2025-08-15", "2025-09-11", "2025-10-12",
    "2025-11-01", "2025-12-06", "2025-12-08", "2025-12-25", "2025-12-26",
    # holiday exit / return days
    "2024-06-21", "2024-08-01", "2024-08-31", "2025-06-20", "2025-08-01",
    "2025-08-31",
]).normalize()


def _parse_windows(spec: str) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    out = []
    for part in spec.split(","):
        a, b = part.split(":")
        out.append((pd.Timestamp(a), pd.Timestamp(b)))
    return out


def _timeline(windows) -> pd.DatetimeIndex:
    idx = [pd.date_range(a, b, freq="5min", inclusive="left") for a, b in windows]
    return pd.DatetimeIndex(np.unique(np.concatenate([i.values for i in idx])))


# --------------------------------------------------------------------------
# Weather: one hidden hourly series for the corridor, and two forecasts of it
# --------------------------------------------------------------------------
def _weather(hours: pd.DatetimeIndex, rng: np.random.Generator) -> dict:
    n = len(hours)
    doy = hours.dayofyear.values
    hod = hours.hour.values
    temp = (16 + 8 * np.sin(2 * np.pi * (doy - 110) / 365)
            + 5 * np.sin(2 * np.pi * (hod - 9) / 24))
    ar = np.zeros(n)
    eps = rng.normal(0, 0.6, n)
    for i in range(1, n):
        ar[i] = 0.95 * ar[i - 1] + eps[i]
    temp = temp + ar

    # rain episodes: a two-state Markov chain, wetter in spring and autumn
    p_start = 0.006 + 0.006 * (np.cos(2 * np.pi * (doy - 290) / 182.5) > 0)
    raining = np.zeros(n, dtype=bool)
    u = rng.random(n)
    for i in range(1, n):
        raining[i] = u[i] < (0.70 if raining[i - 1] else p_start[i])
    precip = np.where(raining, rng.gamma(0.8, 2.0, n), 0.0)
    cloud = np.where(raining, rng.uniform(85, 100, n), 100 * rng.beta(0.7, 1.2, n))
    wind = rng.gamma(2.0, 4.5, n) + 3 * raining
    gusts = wind * rng.uniform(1.6, 2.4, n) + rng.normal(0, 1.5, n)
    temp = temp - 2.5 * raining

    def forecast(noise: float, miss: float) -> dict:
        hit = rng.random(n) > miss
        p = np.where(hit, precip * rng.lognormal(0, noise, n), 0.0)
        return {
            "temperature_2m": temp + rng.normal(0, 1.2 * noise + 0.3, n),
            "precipitation": np.round(p, 1),
            "snowfall": np.zeros(n),
            "cloud_cover": np.clip(cloud + rng.normal(0, 25 * noise, n), 0, 100),
            "wind_speed_10m": np.clip(wind + rng.normal(0, 3 * noise, n), 0, None),
            "wind_gusts_10m": np.clip(gusts + rng.normal(0, 5 * noise, n), 1, None),
        }

    return {"truth_rain": raining, "truth_precip": precip,
            "1d": forecast(0.35, 0.10), "3d": forecast(0.70, 0.30)}


# --------------------------------------------------------------------------
# Static corridor description
# --------------------------------------------------------------------------
def _corridor(pks: np.ndarray, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for pk in pks:
        curv = float(np.clip(rng.normal(35.6, 13.5), 10.9, 68.3))
        for sen in ("cre", "dec"):
            rows.append(dict(
                pk=int(pk), sen=sen,
                # Table A.2 moments; the two carriageways share most geometry
                ang_curv=round(float(np.clip(curv + rng.normal(0, 3), 10.9, 68.3)), 2),
                ang_pend_pos=round(float(np.clip(abs(rng.normal(9.1, 7.8)), 0, 43.9)), 2),
                ang_pend_neg=round(-float(np.clip(abs(rng.normal(9.2, 7.7)), 0, 48.6)), 2),
                car=int(rng.choice([2, 3, 3, 3, 4])),
                v_ff=float(rng.uniform(106, 121)),
                demand=float(rng.uniform(0.75, 1.20)),
                hv_base=float(rng.uniform(0.14, 0.24)),
            ))
    df = pd.DataFrame(rows)
    df["segment"] = ((df["pk"] - df["pk"].min()) // 5 + 1).astype(int)
    return df


def _demand_profile(ts: pd.DatetimeIndex, sen: str) -> np.ndarray:
    """Relative demand in [0, 1] by time of day, weekday and season."""
    h = ts.hour.values + ts.minute.values / 60.0
    dow = ts.dayofweek.values
    special = ts.normalize().isin(SPECIAL_DAYS)
    weekend = (dow >= 5) | special
    am, pm = (1.0, 0.8) if sen == "cre" else (0.8, 1.0)
    weekday = (0.12 + am * np.exp(-0.5 * ((h - 8.0) / 1.3) ** 2)
               + pm * np.exp(-0.5 * ((h - 18.0) / 1.8) ** 2)
               + 0.45 * np.exp(-0.5 * ((h - 13.0) / 3.5) ** 2))
    weekend_p = 0.10 + 0.75 * np.exp(-0.5 * ((h - 15.5) / 4.0) ** 2)
    prof = np.where(weekend, weekend_p, weekday)
    summer = np.isin(ts.month.values, [7, 8])
    prof = prof * np.where(summer, 1.12, 1.0) * np.where(dow == 4, 1.08, 1.0)
    return prof / 1.45


# --------------------------------------------------------------------------
# Main generator
# --------------------------------------------------------------------------
def generate(windows, pk_min: int, pk_max: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ts = _timeline(windows)
    nt = len(ts)
    pks = np.arange(pk_min, pk_max + 1)
    corr = _corridor(pks, rng)
    ns = len(corr)

    hours = ts.floor("h")
    uh = pd.DatetimeIndex(np.unique(hours.values))
    wx = _weather(uh, rng)
    hpos = uh.get_indexer(hours)
    rain_t = wx["truth_rain"][hpos]

    # ---- traffic ---------------------------------------------------------
    q = np.zeros((ns, nt))
    v = np.zeros((ns, nt))
    cong = np.zeros((ns, nt))
    prof = {s: _demand_profile(ts, s) for s in ("cre", "dec")}
    for i, r in corr.iterrows():
        cap = 150.0 * r["car"]
        mean_q = cap * 0.95 * r["demand"] * prof[r["sen"]] * np.where(rain_t, 0.95, 1.0)
        q[i] = rng.poisson(np.clip(mean_q, 0.5, None))
        ratio = q[i] / cap
        cong[i] = 1 / (1 + np.exp(-10 * (ratio - 0.82))) + 0.12 * rain_t
    # congestion spreads along the carriageway and persists in time
    for sen in ("cre", "dec"):
        m = (corr["sen"] == sen).values
        c = cong[m]
        pad = np.pad(c, ((1, 1), (0, 0)), mode="edge")
        c = 0.25 * pad[:-2] + 0.5 * pad[1:-1] + 0.25 * pad[2:]
        for t in range(1, nt):
            c[:, t] = 0.6 * c[:, t] + 0.4 * c[:, t - 1]
        cong[m] = c

    # ---- crashes: onset hazard, then an upstream affectation queue -------
    acc = np.zeros((ns, nt))
    lvl = np.zeros((ns, nt))
    dur = np.zeros((ns, nt))
    qlen = np.zeros((ns, nt))
    curv_z = ((corr["ang_curv"] - 35.6) / 13.5).values[:, None]
    haz = 2.8e-5 * np.exp(2.2 * cong + 0.9 * rain_t[None, :] + 0.35 * curv_z)
    onsets = np.argwhere(rng.random((ns, nt)) < haz)
    key = {(r.pk, r.sen): i for i, r in corr.iterrows()}
    for s, t0 in onsets:
        pk0, sen = corr.at[s, "pk"], corr.at[s, "sen"]
        length = int(rng.integers(1, 5))              # kilometre posts affected
        d = int(rng.integers(3, 19))                   # 15 to 90 minutes
        level = int(rng.integers(1, 6))
        step = -1 if sen == "cre" else 1               # upstream direction
        for k in range(length):
            j = key.get((pk0 + step * k, sen))
            if j is None:
                break
            t1 = min(nt, t0 + d)
            acc[j, t0:t1] = 1.0
            lvl[j, t0:t1] = level
            dur[j, t0:t1] = round(d * 5 / 60.0, 2)
            qlen[j, t0:t1] = round(length * rng.uniform(0.6, 1.0), 1)
            cong[j, t0:t1] = np.maximum(cong[j, t0:t1], rng.uniform(0.5, 0.9))

    for i, r in corr.iterrows():
        v[i] = (r["v_ff"] * (1 - 0.8 * np.clip(cong[i], 0, 1.2))
                - 6 * rain_t + rng.normal(0, 3, nt))
    v = np.clip(np.round(v), 1, 126)
    hv_share = np.zeros((ns, nt))
    night = np.isin(ts.hour.values, [0, 1, 2, 3, 4, 5, 22, 23])
    weekend = ts.dayofweek.values >= 5
    for i, r in corr.iterrows():
        hv_share[i] = np.clip(r["hv_base"] + 0.12 * night - 0.09 * weekend, 0.03, 0.6)
    q = np.clip(q, 0, 763).astype(int)
    qp = rng.binomial(q, hv_share)

    # imputation flags: short gaps, filled by the upstream pipeline
    def gaps(rate):
        g = np.zeros((ns, nt), dtype=int)
        for s, t in np.argwhere(rng.random((ns, nt)) < rate):
            g[s, t:t + int(rng.integers(1, 7))] = 1
        return g
    spd_imp, int_imp = gaps(0.002), gaps(0.002)

    # ---- assemble in location-major order --------------------------------
    cal = pd.DataFrame({
        "dat": ts.strftime("%Y-%m-%d %H:%M:%S"),
        "anyo": ts.year, "mes": ts.month, "dia": ts.day,
        "diaSem": ts.dayofweek, "hor": ts.hour, "5min": ts.minute,
        "mob_esp": ts.normalize().isin(SPECIAL_DAYS).astype(float),
    })
    for lead in ("1d", "3d"):
        f = wx[lead]
        for var, vals in f.items():
            col = f"{lead}_fcst_{var}"
            x = vals[hpos]
            cal[col] = np.round(x, 1)
        cal[f"{lead}_fcst_rain_binary"] = (cal[f"{lead}_fcst_precipitation"] >= 0.1).astype(int)
    cal["1d_fcst_cloud_cover"] = cal["1d_fcst_cloud_cover"].round().astype(int)

    parts = []
    for i, r in corr.iterrows():
        d = cal.copy()
        d.insert(1, "via", "AP-7")
        d.insert(2, "pk", r["pk"])
        d.insert(3, "sen", r["sen"])
        d["mean_speed"] = v[i].astype(int)
        d["speed_imputation"] = spd_imp[i]
        d["car"] = r["car"]
        d["intensity_imputation"] = int_imp[i]
        d["intTot"] = q[i]
        d["intP"] = qp[i]
        d["ACCIDENT"] = acc[i]
        d["C_NIVELL_AFECTACIO"] = lvl[i]
        d["F_TEMPS_AFECTACIO"] = dur[i]
        d["F_LONG_AFECTACIO"] = qlen[i]
        for c in ("ang_curv", "ang_pend_pos", "ang_pend_neg", "segment"):
            d[c] = r[c]
        parts.append(d[COLUMNS])
    out = pd.concat(parts, ignore_index=True)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--windows", default=DEFAULT_WINDOWS,
                    help="comma-separated START:END date windows (END exclusive)")
    ap.add_argument("--pk-min", type=int, default=150)
    ap.add_argument("--pk-max", type=int, default=159)
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--out", default=None,
                    help=f"output path (default: $AP7_DATA_DIR/{DATA_FILE}); "
                         "a .gz suffix writes a compressed file")
    a = ap.parse_args(argv)
    if not (120 <= a.pk_min <= a.pk_max <= 220):
        ap.error("kilometre posts must lie in the modelled range 120-220")

    df = generate(_parse_windows(a.windows), a.pk_min, a.pk_max, a.seed)
    out = Path(a.out) if a.out else DATA_DIR / DATA_FILE
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix == ".gz":
        # mtime=0 keeps the compressed bytes identical across regenerations
        buf = io.StringIO()
        df.to_csv(buf, sep=";", index=False)
        with open(out, "wb") as fh, gzip.GzipFile(fileobj=fh, mode="wb",
                                                  mtime=0, filename="") as gz:
            gz.write(buf.getvalue().encode("latin-1"))
    else:
        df.to_csv(out, sep=";", index=False)

    n_pos = int(df["ACCIDENT"].sum())
    print(f"wrote {out}")
    print(f"  {len(df):,} segment-intervals | {df['pk'].nunique()} posts x 2 directions"
          f" | {df['dat'].min()} -> {df['dat'].max()}")
    print(f"  crash-stamped intervals: {n_pos:,} ({100 * n_pos / len(df):.3f} %)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
