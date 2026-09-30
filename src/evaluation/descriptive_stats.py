#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Table S1 -- descriptive statistics of the study data.

Accident Analysis & Prevention expects the reader to see the distributions
before the models. This produces three panels:

  A. the segment-interval lattice and its coverage
  B. traffic and geometry variables, over the modelling months
  C. crash counts and base rate by month and direction

Streams the 5-minute CSV in chunks; the file is ~5 GB and does not fit
comfortably in memory alongside the rest of the pipeline.
"""
from __future__ import annotations
import os  # noqa: E402  (portable paths)
from src.paths import RESULTS_DIR as _RESULTS_DIR
from src.paths import (  # portable paths -- see src/paths.py
    PROJECT_ROOT_STR as _AP7_ROOT,
    EXPERIMENTS_ROOT_STR as _AP7_EXPERIMENTS,
    DATA_DIR_STR as _AP7_DATA,
    TABLES_DIR as _AP7_TABLES,
    FIGURES_DIR as _AP7_FIGS,
)
import os, sys
import numpy as np
import pandas as pd

ROOT = _AP7_ROOT
CSV = os.path.join(_AP7_DATA,
    "CrashGNNLSTM_v1_vel-extinrix_int_geo_mob_wthr_5min_fund-propag-ltd_from_20240404_to_20251001.csv")
OUT_T = str(_AP7_TABLES)
OUT_R = str(_RESULTS_DIR)

USE = ["dat","via","pk","sen","mean_speed","intTot","intP","ACCIDENT",
       "ang_curv","ang_pend_pos","ang_pend_neg"]

# The models are trained and evaluated on de-duplicated segment-intervals, so
# a table computed on the raw CSV would describe a different population from
# every result. We import the pipeline's own de-duplication rather than
# reimplementing it, so the two cannot diverge.
sys.path.insert(0, ROOT)
from src.training.ablation_study_v21_xgboost_only import dedup_segment_intervals
# months used anywhere in the paper: training pair, test month, seasonal cycle
SPAN = ("2024-06-01", "2025-09-01")


def main() -> int:
    chunks, kept = [], 0
    for ch in pd.read_csv(CSV, sep=";", usecols=USE, chunksize=2_000_000,
                          low_memory=False):
        ch["dat"] = pd.to_datetime(ch["dat"], errors="coerce")
        ch = ch[(ch.dat >= SPAN[0]) & (ch.dat < SPAN[1])]
        if len(ch):
            chunks.append(ch); kept += len(ch)
        print(f"  ... {kept:,} rows kept", flush=True)
    df = pd.concat(chunks, ignore_index=True); del chunks
    n_raw, pos_raw = len(df), int(df.ACCIDENT.sum())
    print(f"[raw ] {n_raw:,} segment-intervals, {pos_raw:,} crash rows")

    df = dedup_segment_intervals(df, "descriptive")
    n_dedup, pos_dedup = len(df), int(df.ACCIDENT.sum())
    print(f"[dedup] {n_dedup:,} rows, {pos_dedup:,} recorded crash intervals")

    df["car"] = df["intTot"] - df["intP"]          # light-vehicle intensity

    # ---------------- Panel A: lattice ------------------------------------
    A = pd.DataFrame([
        ("Kilometre posts (1 km segments)", f"{df.pk.nunique()}"),
        ("Directions", f"{df.sen.nunique()}"),
        ("Directional segments", f"{df.groupby(['pk','sen']).ngroups:,}"),
        ("Temporal resolution", "5 min"),
        ("Record span", f"{df.dat.min():%Y-%m-%d} to {df.dat.max():%Y-%m-%d}"),
        ("Segment-intervals (raw)", f"{n_raw:,}"),
        ("Segment-intervals (de-duplicated, modelled)", f"{n_dedup:,}"),
        ("Recorded crash intervals", f"{pos_dedup:,}"),
        ("Recorded crash rate", f"{100*pos_dedup/n_dedup:.4f} %"),
        ("Imbalance ratio", f"1 : {int(round((1-df.ACCIDENT.mean())/df.ACCIDENT.mean())):,}"),
    ], columns=["Property", "Value"])

    # ---------------- Panel B: variable distributions ---------------------
    lab = {"mean_speed":"Mean speed (km/h)", "intTot":"Total intensity (veh/5 min)",
           "intP":"Heavy-vehicle intensity (veh/5 min)", "car":"Light-vehicle intensity (veh/5 min)",
           "ang_curv":"Trajectory curvature (deg)", "ang_pend_pos":"Positive slope (deg)",
           "ang_pend_neg":"Negative slope (deg)"}
    rows = []
    for c, name in lab.items():
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        rows.append([name, f"{len(s):,}", f"{s.mean():.2f}", f"{s.std():.2f}",
                     f"{s.min():.2f}", f"{s.quantile(.25):.2f}", f"{s.median():.2f}",
                     f"{s.quantile(.75):.2f}", f"{s.max():.2f}"])
    B = pd.DataFrame(rows, columns=["Variable","N","Mean","SD","Min","P25","Median","P75","Max"])

    # ---------------- Panel C: crashes by month ---------------------------
    g = df.assign(month=df.dat.dt.to_period("M").astype(str)).groupby("month")
    C = pd.DataFrame({
        "Segment-intervals": g.size(),
        "Crash intervals": g.ACCIDENT.sum().astype(int),
        "Base rate (%)": (100*g.ACCIDENT.mean()).map("{:.3f}".format),
    }).reset_index()
    C.columns = ["Month","Segment-intervals","Crash intervals","Base rate (%)"]
    for _c in ("Segment-intervals", "Crash intervals"):
        C[_c] = C[_c].map("{:,}".format)

    d = df.groupby("sen").agg(size=("ACCIDENT","size"),
                              pos=("ACCIDENT","sum"), rate=("ACCIDENT","mean"))
    D = pd.DataFrame({
        "Direction": ["South-north (sen=0)","North-south (sen=1)"],
        "Segment-intervals": [f"{int(v):,}" for v in d["size"]],
        "Crash intervals": [f"{int(v):,}" for v in d["pos"]],
        "Base rate (%)": [f"{100*v:.3f}" for v in d["rate"]],
    })

    os.makedirs(OUT_T, exist_ok=True); os.makedirs(OUT_R, exist_ok=True)
    for nm, t in [("A_lattice",A),("B_variables",B),("C_by_month",C),("D_by_direction",D)]:
        t.to_csv(os.path.join(OUT_R, f"descriptive_{nm}.csv"), index=False)
        print(f"\n--- {nm} ---"); print(t.to_string(index=False))

    with open(os.path.join(OUT_T, "tableS1_descriptive.tex"), "w") as fh:
        fh.write("% Table S1 -- descriptive statistics (generated by descriptive_stats.py)\n")
        # Supplementary floats: number them S1..Sn so they do not consume the
        # main-text table counter (the manuscript inputs this inside Section 3.4).
        fh.write("\\begingroup\n")
        fh.write("\\setcounter{table}{0}\n")
        fh.write("\\renewcommand{\\thetable}{S\\arabic{table}}\n")
        for cap, t in [("Study lattice and coverage",A),
                       ("Distribution of traffic and geometry variables",B),
                       ("Crash counts by month",C),
                       ("Crash counts by direction",D)]:
            fh.write("\n\\begin{table}[htbp]\\centering\n\\caption{%s}\n" % cap)
            fh.write(t.to_latex(index=False, escape=True))
            fh.write("\\end{table}\n")
        fh.write("\\endgroup\n")
    print(f"\n[DONE] {OUT_T}/tableS1_descriptive.tex")
    return 0


if __name__ == "__main__":
    sys.exit(main())
