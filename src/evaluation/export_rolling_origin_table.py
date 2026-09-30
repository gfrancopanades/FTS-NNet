#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Emit Table 5 (rolling-origin validation) from the fold simulations.

Each rolling-origin fold retrains Stage 1 on a window rolled to end before its
test month, so the folds live in their own experiment trees (v100..v103) rather
than in the June benchmark registry. This script reads each fold's simulation
CSV directly, recomputes AUPRC and lift from the stored per-row predictions, and
writes reports/tables/table4_rolling_origin.tex.

Lift = AUPRC / base rate. It is reported because AUPRC is bounded below by the
month's prevalence, which varies by a third across these folds and is enough to
reverse the ranking (July looks second-best by AUPRC and is last by lift).

A fold with no simulation yet is emitted as "fold in training" rather than being
silently dropped, so the table always states the full design.

    python src/evaluation/export_rolling_origin_table.py
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

import pandas as pd
from sklearn.metrics import average_precision_score

ROOT = _AP7_ROOT
EXP = _AP7_EXPERIMENTS
OUT = os.path.join(ROOT, os.path.join(str(_AP7_TABLES), "table4_rolling_origin.tex"))

# fold -> (pretty month, experiment glob, expected sim window)
# The June reference is the main benchmark configuration (v62 stage-1 + the
# proposed focal classifier), not a v10x fold, so it is addressed separately.
FOLDS = [
    ("May 2025",  "v100_*", "2025-05"),
    ("June 2025", None,     "2025-06"),   # reference fold, from the benchmark tree
    ("July 2025", "v101_*", "2025-07"),
    ("August 2025", "v102_*", "2025-08"),
    ("September 2025", "v103_*", "2025-09"),
]

def _june_reference_csv():
    """The June reference comes from the benchmark registry, not a fixed path.

    It must be the SAME one-month frame the other folds use: scoring June inside
    a longer frame changes the engineered features and yields 0.587 instead of
    0.615, which would silently mix frames within this one table.
    """
    import sys
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    from src.evaluation.benchmark_harvest import (REGISTRY, discover_sim_csv,
                                                  EXPERIMENTS_ROOT)
    entry = next(e for e in REGISTRY if e["key"] == "C2_C5_Focal")
    return discover_sim_csv(entry, EXPERIMENTS_ROOT, "two_stage")


def _score(csv, month):
    """AUPRC / base rate / counts for one month of one simulation CSV."""
    d = pd.read_csv(csv, sep=";", low_memory=False,
                    usecols=["dat", "ACCIDENT_real", "accident_probability"])
    d["dat"] = pd.to_datetime(d["dat"])
    d = d[d["dat"].dt.to_period("M").astype(str) == month]
    if d.empty or d["ACCIDENT_real"].fillna(0).sum() == 0:
        return None
    y = d["ACCIDENT_real"].fillna(0).astype(int)
    p = d["accident_probability"]
    ap = float(average_precision_score(y, p))
    base = float(y.mean())
    return dict(n=len(d), pos=int(y.sum()), base=base, auprc=ap, lift=ap / base)


def _newest_fold_sim(pattern):
    """Newest fold simulation CSV, preferring the highest xgboost job id.

    A fold that was rerun (e.g. v101 after the SLURM comma-truncation fix)
    leaves the superseded run on disk; the later xgboost directory is the valid
    one, so sort by job id and take the last.
    """
    cands = glob.glob(os.path.join(EXP, pattern, "xgboost_*",
                                   "simulation_benchmark_*", "*.csv"))
    if not cands:
        return None

    def _xgb_id(p):
        for part in p.split(os.sep):
            if part.startswith("xgboost_"):
                return int(part.split("_")[1])
        return -1

    return sorted(cands, key=_xgb_id)[-1]


def main():
    rows = []
    for label, pattern, month in FOLDS:
        csv = _june_reference_csv() if pattern is None else _newest_fold_sim(pattern)
        res = _score(csv, month) if csv and os.path.exists(csv) else None
        rows.append((label, res, csv))
        if res:
            print(f"[OK]      {label:16s} n={res['n']:>9,} pos={res['pos']:>5} "
                  f"base={res['base']:.3%} AUPRC={res['auprc']:.3f} "
                  f"lift={res['lift']:.0f}x")
        else:
            print(f"[PENDING] {label:16s} no simulation yet")

    body = []
    for label, res, _ in rows:
        name = label + (r" \emph{(reference)}" if label.startswith("June") else "")
        if res is None:
            body.append(f"{name} & \\multicolumn{{5}}{{c}}{{\\emph{{fold in training}}}} \\\\")
        else:
            body.append(
                f"{name} & {res['n']:,} & {res['pos']:,} & "
                f"{res['base'] * 100:.3f}\\% & {res['auprc']:.3f} & "
                f"{res['lift']:.0f}$\\times$ \\\\".replace(",", "{,}"))

    tex = r"""% Rolling-origin (walk-forward) validation. Requires: booktabs.
% Generated by src/evaluation/export_rolling_origin_table.py -- do not hand-edit.
\begin{table}[t]
\centering
\caption{Rolling-origin validation of the proposed system. For each test month the Stage-1 forecaster is retrained from scratch on a training window rolled forward to end before that month, and the classifier and decision threshold are refitted; no data from the test month or later enters training. Lift is AUPRC divided by the month's own base rate, and is the only cross-month-comparable form because AUPRC is bounded below by prevalence. Folds used a reduced hyper-parameter budget (12 Optuna trials against 30 for the June reference).}
\label{tab:rolling}
\begin{tabular}{l r r r r r}
\toprule
Test month & Segment-intervals & Crash-affected intervals & Base rate & AUPRC & Lift \\
\midrule
""" + "\n".join(body) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        fh.write(tex)
    done = sum(1 for _, r, _ in rows if r)
    print(f"\n[DONE] {done}/{len(rows)} folds -> {OUT}")


if __name__ == "__main__":
    main()
