#!/usr/bin/env python3
"""Archive the 13-month seasonal-cycle simulation into the prediction store.

`archive_benchmark_predictions.py` keys off the registry, whose rows all point
at one-month June simulations, so the cycle frame -- a different simulation
window of the SAME proposed model -- has no entry there and is never archived.
`plot_results_figures.py` reads this file for the durability panel and the
seasonal-cycle table; when it is absent that figure is skipped by design,
rather than silently falling back to a one-month frame presented as a full
seasonal cycle.

Lives in the repo (shared storage) rather than a scratch dir because SLURM
compute nodes have their own node-local /tmp and cannot see the submit host's.
"""
from src.paths import (  # portable paths -- see src/paths.py
    PROJECT_ROOT_STR as _AP7_ROOT,
    EXPERIMENTS_ROOT_STR as _AP7_EXPERIMENTS,
    DATA_DIR_STR as _AP7_DATA,
    TABLES_DIR as _AP7_TABLES,
    FIGURES_DIR as _AP7_FIGS,
)
import glob
import os
import sys

import pandas as pd

EXPERIMENTS_ROOT = _AP7_EXPERIMENTS
OUT_DIR = (f"{_AP7_ROOT}/"
           "experiments/benchmark_results/predictions")
# The Table B.2 pair (BL-F7 Stage 1 + BL-C5 Stage 2) scored over Aug-2024 ->
# Aug-2025. Both run ids are overridable for reproduced runs.
_GNN_DIR = os.environ.get("AP7_CYCLE_GNN_DIR", "v86_gnn_no-w1d_5min_2663351")
_XGB = os.environ.get("AP7_CYCLE_XGB", "2672406")
PATTERN = (f"{EXPERIMENTS_ROOT}/{_GNN_DIR}/xgboost_{_XGB}/"
           "simulation_benchmark_*sim20240801-20250901/*.csv")


def main() -> int:
    files = sorted(glob.glob(PATTERN))
    if not files:
        print(f"[FAIL] no 13-month cycle simulation matched:\n  {PATTERN}")
        return 1
    df = pd.read_csv(files[-1], sep=";")
    df["dat"] = pd.to_datetime(df["dat"])
    out = pd.DataFrame({
        "pk": df["pk"],
        "sen": df["sen"],
        "dat": df["dat"],
        "y": df["ACCIDENT_real"].astype(int),
        "prob": df["accident_probability"].astype(float),
        "pred_binary": df["accident_pred_binary"].astype(int),
    })
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "C2_C5_Focal_cycle.parquet")
    out.to_parquet(path, index=False)
    print(f"[OK] {path}")
    print(f"     {len(out):,} rows | {int(out.y.sum()):,} positives | "
          f"{out.dat.min().date()} .. {out.dat.max().date()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
