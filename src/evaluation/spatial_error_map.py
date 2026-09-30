#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Figure 11 -- where along the corridor the model works.

A road authority's first question about a corridor-wide risk model is not what
it scores on average but *where* it succeeds and where it fails. This resolves
the held-out month by kilometre post and direction, reporting per-post recall,
precision and crash load at the frozen dispatch threshold, plus the local
ranking quality (AUPRC lift over that post's own base rate).

Reads only the frozen prediction store, so the figure cannot drift from the
numbers in Tables 2 and 5.
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Shared seaborn theme: one aesthetic across every figure in the paper.
# See src/evaluation/plot_style.py for the palette and its CVD validation.
import sys as _sys
_sys.path.insert(0, _AP7_ROOT)
from src.evaluation.plot_style import apply as _apply_style, PALETTE  # noqa: E402
_apply_style()

from sklearn.metrics import average_precision_score

ROOT = _AP7_ROOT
STORE = os.path.join(ROOT, "experiments/benchmark_results/predictions")
FIGS = str(_AP7_FIGS)
RES = str(_RESULTS_DIR)
KEY = os.environ.get("AP7_SPATIAL_KEY", "C2_proposed")
DIRNAME = {0: "South-north", 1: "North-south"}


def main() -> int:
    df = pd.read_parquet(os.path.join(STORE, f"{KEY}.parquet"))
    print(f"[data] {KEY}: {len(df):,} rows, {int(df.y.sum()):,} positives")

    rows = []
    for (pk, sen), g in df.groupby(["pk", "sen"]):
        n, pos = len(g), int(g.y.sum())
        tp = int(((g.pred_binary == 1) & (g.y == 1)).sum())
        fp = int(((g.pred_binary == 1) & (g.y == 0)).sum())
        rec = tp / pos if pos else np.nan
        prec = tp / (tp + fp) if (tp + fp) else np.nan
        # local ranking quality, comparable across posts only as a lift
        if 0 < pos < n:
            ap = average_precision_score(g.y, g.prob)
            lift = ap / (pos / n)
        else:
            ap = lift = np.nan
        rows.append(dict(pk=pk, sen=sen, n=n, crashes=pos, tp=tp, fp=fp,
                         recall=rec, precision=prec, auprc=ap, lift=lift,
                         flagged=int((g.pred_binary == 1).sum())))
    r = pd.DataFrame(rows).sort_values(["sen", "pk"])
    os.makedirs(RES, exist_ok=True)
    r.to_csv(os.path.join(RES, "spatial_error_by_pk.csv"), index=False)

    # ------------------------------------------------------------------ plot
    fig, axes = plt.subplots(3, 1, figsize=(11, 8.4), sharex=True,
                             gridspec_kw=dict(height_ratios=[1.0, 1.0, 0.8]))
    col = {0: "#2a78d6", 1: "#eb6834"}
    for sen in (0, 1):
        d = r[r.sen == sen]
        axes[0].bar(d.pk + (0.4 * sen - 0.2), d.crashes, width=0.4,
                    color=col[sen], label=DIRNAME[sen])
        axes[1].plot(d.pk, d.lift, ".-", color=col[sen], lw=1.1, ms=4,
                     label=DIRNAME[sen])
        axes[2].plot(d.pk, d.recall, ".-", color=col[sen], lw=1.1, ms=4,
                     label=DIRNAME[sen])

    axes[0].set_ylabel("crash intervals")
    axes[0].set_title("Corridor-resolved performance, June 2025 held-out month",
                      loc="left", fontweight="semibold")
    axes[1].set_ylabel("AUPRC lift over\nlocal base rate")
    axes[1].axhline(1.0, color="grey", lw=.8, ls=":")
    axes[2].set_ylabel("recall at dispatch\nthreshold")
    axes[2].set_xlabel("kilometre post (AP-7)")
    axes[2].set_ylim(-0.03, 1.03)
    for a in axes:
        a.grid(True, alpha=.3); a.set_axisbelow(True)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    axes[0].legend(frameon=False, ncol=2, fontsize=9)
    fig.tight_layout()
    os.makedirs(FIGS, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIGS, f"spatial_error_map.{ext}"), dpi=170)
    plt.close(fig)

    # ------------------------------------------------------------- summary
    ok = r[r.crashes > 0]
    print(f"\n[summary] posts with >=1 crash: {len(ok)} of {len(r)}")
    print(f"  median per-post lift : {ok.lift.median():.1f}x")
    print(f"  posts with lift > 1  : {(ok.lift > 1).sum()} of {ok.lift.notna().sum()}")
    print(f"  median per-post recall: {ok.recall.median():.3f}")
    print("\n  worst 5 posts by lift:")
    print(ok.nsmallest(5, "lift")[["pk","sen","crashes","lift","recall"]].to_string(index=False))
    print("\n  best 5 posts by lift:")
    print(ok.nlargest(5, "lift")[["pk","sen","crashes","lift","recall"]].to_string(index=False))
    print(f"\n[DONE] {FIGS}/spatial_error_map.pdf")
    return 0


if __name__ == "__main__":
    sys.exit(main())
