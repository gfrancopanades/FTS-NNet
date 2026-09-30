#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Emit Tables S6 (lead-time sweep) and S7 (event-level operating points).

Both are supplementary and carry their own S-counter wrapper, so they continue
the S1-S5 sequence instead of consuming a main table number. Numbers are read
from the frozen simulation CSVs and from event_level_metrics.csv, never typed.

Run: python3 -m src.evaluation.export_new_result_tables
"""
from __future__ import annotations

import glob
import os

import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.paths import EXPERIMENTS_ROOT_STR as _EXP, TABLES_DIR as _TAB

# Experiment artefacts normally live off the repository volume; point
# AP7_EXPERIMENTS_DIR at them (see src/paths.py).
_ALT = _EXP


def _exp_dir(name: str) -> str:
    for root in (_EXP, _ALT):
        p = os.path.join(root, name)
        if os.path.isdir(p):
            return p
    return os.path.join(_EXP, name)


V86 = _exp_dir("v86_gnn_no-w1d_5min_2663351")
V62 = _exp_dir("v62_gnn_no-w1d_5min_2628913")
EVENTS = os.path.join(_EXP, "benchmark_results", "event_level_metrics.csv")

SWEEP = [("contemporaneous", V86, "2672505"), ("1", V86, "2674433"),
         ("3", V86, "2674435"), ("8", V86, "2674437"), ("24", V86, "2674439"),
         ("48", V86, "2674441"), ("72", V86, "2672507")]

S_HEAD = (r"""\begingroup
\setcounter{table}{%d}
\renewcommand{\thetable}{S\arabic{table}}
""")


def score(base: str, job: str):
    cs = glob.glob(os.path.join(base, f"xgboost_{job}", "simulation_*", "*.csv"))
    if not cs:
        return None
    d = pd.read_csv(sorted(cs, key=os.path.getmtime)[-1], sep=";")
    p = [c for c in d.columns if "probab" in c.lower()][0]
    y = [c for c in d.columns if "real" in c.lower()][0]
    return (average_precision_score(d[y], d[p]), d[y].mean(),
            roc_auc_score(d[y], d[p]))


def table_s6() -> str:
    rows = []
    for lab, base, job in SWEEP:
        r = score(base, job)
        if r is None:
            continue
        ap, b, auroc = r
        name = "contemporaneous ($H=0$)" if lab == "contemporaneous" else f"{lab}"
        star = r"\,$^{\dagger}$" if job == "2672507" else ""
        rows.append(f"{name}{star} & {ap:.3f} & {ap/b:.0f}$\\times$ & {auroc:.3f} \\\\")
    body = "\n".join(rows)
    return S_HEAD % 5 + r"""
\begin{table}[t]
\centering
\caption{Lead-time sweep on the persistence baseline. The classifier, features,
training windows, search budget and evaluation frame are those of the
lagged-traffic arm of Table 2(c); only the lag at which the traffic state is
read changes. Lead is in hours. $^{\dagger}$the 72-hour row is the arm reported
in Table 2(c) and is reused unchanged, so the sweep ends on the published value.}
\label{tab:horizon}
\begin{tabular}{l c c c}
\toprule
Lead & AUPRC & Lift & AUROC \\
\midrule
""" + body + r"""
\bottomrule
\end{tabular}
\end{table}
\endgroup
"""


def table_s7() -> str:
    e = pd.read_csv(EVENTS)
    rows = []
    for _, r in e.iterrows():
        tier = (str(r["tier"]).replace(">=", r"$\geq$").replace("F1", "$F_1$"))
        # interval-level precision and recall are Table 4's; repeating them here
        # invited a third-decimal disagreement, because Table 4 reads its
        # operating points off the PR curve while this table uses the fixed tier
        # thresholds. The columns are dropped rather than reconciled.
        rows.append(
            f"{tier} & {int(r['incidents_caught'])}/{int(r['incidents'])} & "
            f"{int(r['incidents_caught_before_onset'])}/{int(r['incidents'])} & "
            f"{int(r['episodes'])} & {r['episode_precision']:.3f} \\\\")
    body = "\n".join(rows)
    return S_HEAD % 6 + r"""
\begin{table*}[t]
\centering
\caption{The held-out month read per incident and per alarm episode; the
interval view is Table 4. An incident is a merged run of crash stamps
(Section 4.5.2, merging rule there); an episode
is a maximal run of flagged cells on one segment, which is what a control room
sees as one alarm. `Before onset' counts incidents with at least one cell
flagged strictly earlier than the incident's first stamp.}
\label{tab:eventlevel}
\begin{tabular}{l c c c c}
\toprule
 & \multicolumn{2}{c}{per incident} & \multicolumn{2}{c}{per episode} \\
\cmidrule(lr){2-3} \cmidrule(lr){4-5}
Tier & Caught & Before onset & Episodes & Precision \\
\midrule
""" + body + r"""
\bottomrule
\end{tabular}
\end{table*}
\endgroup
"""


def main() -> None:
    os.makedirs(_TAB, exist_ok=True)
    for name, txt in (("tableS6_horizon", table_s6()),
                      ("tableS7_eventlevel", table_s7())):
        p = os.path.join(_TAB, f"{name}.tex")
        with open(p, "w", encoding="utf-8") as fh:
            fh.write(txt)
        print(f"[DONE] {p}")


if __name__ == "__main__":
    main()
