#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Event-level and pre-onset operating points for the deployed risk surface.

Every number in Section 5.7 counts 5-minute segment-intervals. An agency
positions a patrol against a crash, so this module re-reads the same frozen
predictions in two other units:

  * per incident   -- was any cell of this incident's footprint flagged, and
                      was any of them flagged BEFORE the incident began?
  * per episode    -- how many distinct runs of flagged cells does the surface
                      produce in a month, and what share of them sit on a real
                      incident? This is the false-alarm count a control room
                      actually experiences, which the interval-level FAR hides
                      because one alarm spans many adjacent cells.

It also recomputes the threshold-free score under a pre-onset-only label, which
keeps a positive only where the crash has not yet happened. Intervals at or
after onset are dropped rather than relabelled negative: the model is not wrong
to fire on them, they simply cannot be acted on preventively.

Incidents are reconstructed from the raw crash stamps, because the label in the
modelling frame is a queue footprint rather than a crash: each stamp covers the
contiguous run of kilometre posts the logged affectation spans, and one incident
re-stamps at successive timestamps as its queue grows. Stamps in the same
direction whose kilometre spans overlap (with a 2 km tolerance) and which fall
within MERGE_HOURS of each other are treated as one incident.

Run:  python3 -m src.evaluation.event_level_metrics
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from src.paths import EXPERIMENTS_ROOT_STR as _EXP, DATA_DIR_STR as _DATA

PRED = os.path.join(_EXP, "benchmark_results", "predictions", "C2_C5_Focal.parquet")
CACHE = os.path.join(_EXP, "benchmark_results", "june2025_crash_stamps.parquet")
OUT = os.path.join(_EXP, "benchmark_results", "event_level_metrics.csv")
DATA_FILE = ("CrashGNNLSTM_v1_vel-extinrix_int_geo_mob_wthr_5min_"
             "fund-propag-ltd_from_20240404_to_20251001.csv")

MERGE_HOURS = 2.0        # stamps of one growing queue, same direction
PK_TOL = 2               # kilometre tolerance when matching spans
STEP_MIN = 5

# Pre-declared tiers of Section 5.7, plus the frozen shipped threshold.
TIERS = {"dispatch (P>=0.80)": 0.523, "balanced (max F1)": 0.446,
         "advisory (R>=0.50)": 0.419}


# ---------------------------------------------------------------- crash stamps
def load_stamps() -> pd.DataFrame:
    """June-2025 rows carrying ACCIDENT = 1, one per (sen, pk, dat)."""
    if os.path.exists(CACHE):
        return pd.read_parquet(CACHE)
    src = os.path.join(_DATA, DATA_FILE)
    cols = ["via", "sen", "pk", "anyo", "mes", "dia", "hor", "5min", "ACCIDENT"]
    parts = []
    for ch in pd.read_csv(src, sep=";", decimal=".", encoding="latin-1",
                          usecols=cols, chunksize=1_000_000):
        m = (ch["anyo"] == 2025) & (ch["mes"] == 6) & (ch["ACCIDENT"] == 1)
        if m.any():
            parts.append(ch[m].copy())
    df = pd.concat(parts, ignore_index=True)
    if df["sen"].dtype == object:
        df["sen"] = df["sen"].map({"dec": 0, "cre": 1}).fillna(1).astype(int)
    df["dat"] = pd.to_datetime(
        df["anyo"].astype(str) + "-" + df["mes"].astype(str).str.zfill(2) + "-"
        + df["dia"].astype(str).str.zfill(2) + " "
        + df["hor"].astype(str).str.zfill(2) + ":"
        + df["5min"].astype(str).str.zfill(2) + ":00")
    df = df.drop_duplicates(["sen", "pk", "dat"])[["sen", "pk", "dat"]]
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    df.to_parquet(CACHE)
    return df


def build_incidents(stamps: pd.DataFrame) -> pd.DataFrame:
    """Merge stamps into incidents; one row per incident with its footprint."""
    ev = (stamps.groupby(["sen", "dat"])
                .agg(pkmin=("pk", "min"), pkmax=("pk", "max"))
                .reset_index().sort_values(["sen", "dat"]))
    out, cur = [], None
    for _, r in ev.iterrows():
        same = (cur is not None and r.sen == cur["sen"]
                and (r.dat - cur["tmax"]) <= pd.Timedelta(hours=MERGE_HOURS)
                and not (r.pkmax < cur["pkmin"] - PK_TOL
                         or r.pkmin > cur["pkmax"] + PK_TOL))
        if same:
            cur["tmax"] = r.dat
            cur["pkmin"] = min(cur["pkmin"], r.pkmin)
            cur["pkmax"] = max(cur["pkmax"], r.pkmax)
        else:
            if cur is not None:
                out.append(cur)
            cur = dict(sen=r.sen, tmin=r.dat, tmax=r.dat,
                       pkmin=r.pkmin, pkmax=r.pkmax)
    if cur is not None:
        out.append(cur)
    inc = pd.DataFrame(out)
    inc["incident_id"] = np.arange(len(inc))
    return inc


def assign_incident(pred: pd.DataFrame, inc: pd.DataFrame) -> pd.DataFrame:
    """Attach incident id and minutes-from-onset to every positive cell."""
    pred["incident_id"] = -1
    pred["mins_from_onset"] = np.nan
    pos = pred["y"].values == 1
    sen = pred["sen"].values
    pk = pred["pk"].values
    t = pred["dat"].values.astype("datetime64[m]").astype(np.int64)
    iid = np.full(len(pred), -1, dtype=np.int64)
    off = np.full(len(pred), np.nan)
    for _, r in inc.iterrows():
        t0 = np.datetime64(r.tmin, "m").astype(np.int64)
        t1 = np.datetime64(r.tmax, "m").astype(np.int64)
        m = (pos & (sen == r.sen) & (pk >= r.pkmin - PK_TOL)
             & (pk <= r.pkmax + PK_TOL) & (t >= t0) & (t <= t1))
        take = m & (iid < 0)
        iid[take] = r.incident_id
        off[take] = t[take] - t0
    pred["incident_id"] = iid
    pred["mins_from_onset"] = off
    return pred


# ------------------------------------------------------------------- episodes
def flagged_episodes(pred: pd.DataFrame, thr: float, gap: int = 1) -> pd.DataFrame:
    """Group flagged cells into runs, per segment, tolerating `gap` empty steps."""
    f = pred.loc[pred["prob"] >= thr, ["sen", "pk", "dat", "incident_id"]]
    if f.empty:
        return pd.DataFrame(columns=["sen", "pk", "start", "end", "n", "hit"])
    f = f.sort_values(["sen", "pk", "dat"])
    step = pd.Timedelta(minutes=STEP_MIN * (gap + 1))
    newloc = (f["sen"].ne(f["sen"].shift()) | f["pk"].ne(f["pk"].shift()))
    newrun = newloc | (f["dat"].diff() > step)
    f["episode"] = newrun.cumsum()
    g = f.groupby("episode").agg(sen=("sen", "first"), pk=("pk", "first"),
                                 start=("dat", "min"), end=("dat", "max"),
                                 n=("dat", "size"),
                                 hit=("incident_id", lambda s: (s >= 0).any()))
    return g.reset_index(drop=True)


def main() -> None:
    if not os.path.exists(PRED):
        sys.exit(f"prediction store not found: {PRED}")
    pred = pd.read_parquet(PRED)
    pred["dat"] = pd.to_datetime(pred["dat"])
    stamps = load_stamps()
    inc = build_incidents(stamps)
    pred = assign_incident(pred, inc)

    n_inc = len(inc)
    print(f"[EVENT] {len(stamps):,} crash stamps -> {n_inc} incidents "
          f"| {int(pred['y'].sum()):,} positive intervals "
          f"| {len(pred):,} rows")
    unmatched = int(((pred["y"] == 1) & (pred["incident_id"] < 0)).sum())
    if unmatched:
        print(f"[EVENT] warning: {unmatched} positive intervals matched no incident")

    rows = []
    for name, thr in TIERS.items():
        flag = pred["prob"].values >= thr
        y = pred["y"].values == 1
        tp = int((flag & y).sum()); fp = int((flag & ~y).sum())
        prec = tp / max(tp + fp, 1)
        rec_int = tp / max(int(y.sum()), 1)

        # per incident: any cell flagged, and any cell flagged before onset
        d = pred.loc[y & (pred["incident_id"] >= 0),
                     ["incident_id", "mins_from_onset"]].copy()
        d["flag"] = flag[(y & (pred["incident_id"] >= 0)).values]
        per = d.groupby("incident_id").agg(
            any_flag=("flag", "any"),
            pre_flag=("flag", lambda s: bool(
                (s.values & (d.loc[s.index, "mins_from_onset"].values < 0)).any())))
        caught = int(per["any_flag"].sum())
        caught_pre = int(per["pre_flag"].sum())

        ep = flagged_episodes(pred, thr)
        n_ep = len(ep); ep_hit = int(ep["hit"].sum()) if n_ep else 0

        rows.append(dict(
            tier=name, threshold=thr,
            interval_precision=round(prec, 3), interval_recall=round(rec_int, 3),
            incidents=n_inc, incidents_caught=caught,
            incident_recall=round(caught / max(n_inc, 1), 3),
            incidents_caught_before_onset=caught_pre,
            incident_recall_pre_onset=round(caught_pre / max(n_inc, 1), 3),
            episodes=n_ep, episodes_on_an_incident=ep_hit,
            episode_precision=round(ep_hit / max(n_ep, 1), 3),
            false_episodes_per_incident=round((n_ep - ep_hit) / max(n_inc, 1), 1)))

    # threshold-free score under a pre-onset-only label
    keep = ~((pred["y"] == 1) & (pred["mins_from_onset"] >= 0))
    y_pre = ((pred["y"] == 1) & (pred["mins_from_onset"] < 0)).values[keep.values]
    p_pre = pred["prob"].values[keep.values]
    ap_pre = average_precision_score(y_pre, p_pre)
    ap_all = average_precision_score(pred["y"].values, pred["prob"].values)
    base_pre = y_pre.mean()
    print(f"\n[EVENT] AUPRC, published label      : {ap_all:.3f} "
          f"(base {pred['y'].mean()*100:.3f} %, lift {ap_all/pred['y'].mean():.0f}x)")
    print(f"[EVENT] AUPRC, pre-onset-only label : {ap_pre:.3f} "
          f"(base {base_pre*100:.3f} %, lift {ap_pre/base_pre:.0f}x, "
          f"{int(y_pre.sum()):,} positives)")

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print("\n" + out.to_string(index=False))
    print(f"\n[EVENT] written -> {OUT}")


if __name__ == "__main__":
    main()
