"""Precision-recall curves for every benchmark model, one figure (3 panels).

Loads each model's simulation CSV via the benchmark harvester registry (so it
always matches the current Table-B anchors), clips to the common June-2025
window, computes the PR curve, and draws one panel per contrast:

  (a) C1 forecaster swap  (b) C2 classifier swap  (c) C3 complete systems

Per panel: baselines in the validated 8-slot categorical palette (fixed order,
never cycled), the proposed/reference cascade in ink-black (thickest), the
perfect-forecast ceiling as a dashed gray reference, and the positive-rate
floor as a dotted line. Legend entries carry each model's AUPRC, sorted
descending. PENDING rows (sim not on disk yet) are skipped with a notice.

Output: visualizations/benchmark_pr_curves.{png,pdf}

Run (needs ~40G to stream 18 sim CSVs):
    sbatch --mem=60G --cpus-per-task=4 \
        --wrap="python -u src/evaluation/plot_benchmark_pr_curves.py"
"""
from src.paths import (  # portable paths -- see src/paths.py
    PROJECT_ROOT_STR as _AP7_ROOT,
    EXPERIMENTS_ROOT_STR as _AP7_EXPERIMENTS,
    DATA_DIR_STR as _AP7_DATA,
    TABLES_DIR as _AP7_TABLES,
    FIGURES_DIR as _AP7_FIGS,
)

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Shared seaborn theme: one aesthetic across every figure in the paper.
# See src/evaluation/plot_style.py for the palette and its CVD validation.
import sys as _sys
_sys.path.insert(0, _AP7_ROOT)
from src.evaluation.plot_style import apply as _apply_style, PALETTE  # noqa: E402
_apply_style()

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve

PROD_ROOT = _AP7_ROOT
if PROD_ROOT not in sys.path:
    sys.path.insert(0, PROD_ROOT)

from src.evaluation.benchmark_harvest import (  # noqa: E402
    REGISTRY, discover_sim_csv)

EVAL_START, EVAL_END = "2025-06-01", "2025-07-01"
OUT_DIR = str(_AP7_FIGS)

# Validated categorical palette (light mode, fixed slot order — never cycled).
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
           "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK = "#1a1a19"        # proposed / primary text
MUTED = "#8a8878"      # ceiling + baselines refs
GRID = "#e9e8e0"


def load_pr(entry, mode="two_stage"):
    """Return (recall, precision, auprc) for one registry entry, or None."""
    csv = discover_sim_csv(entry, _root(), mode)
    if not csv:
        return None
    df = pd.read_csv(csv, sep=";", usecols=["dat", "accident_probability",
                                            "ACCIDENT_real"])
    df["dat"] = pd.to_datetime(df["dat"])
    df = df[(df["dat"] >= pd.Timestamp(EVAL_START))
            & (df["dat"] < pd.Timestamp(EVAL_END))]
    y = df["ACCIDENT_real"].astype(int).to_numpy()
    p = df["accident_probability"].astype(float).to_numpy()
    if y.sum() == 0:
        return None
    prec, rec, _ = precision_recall_curve(y, p)
    # AP identical to the benchmark tables (average_precision_score, not trapezoid)
    return rec, prec, float(average_precision_score(y, p)), float(y.mean())


def _root():
    from src.evaluation import benchmark_harvest as bh
    return bh.EXPERIMENTS_ROOT


def _shown(ap, harvest, key):
    """Prefer the harvested AUPRC, reporting any disagreement with the curve."""
    hv = harvest.get(key)
    if hv is None or abs(hv - ap) < 5e-4:
        return ap
    print(f"[legend] {key}: harvest {hv:.4f} vs curve AP {ap:.4f} -> printing harvest")
    return float(hv)


def _harvest_auprc():
    """AUPRC per registry key as printed by the benchmark tables, or {}."""
    try:
        import pandas as _pd
        from src.evaluation.export_latex_tables import CI_CSV
        return _pd.read_csv(CI_CSV).set_index("key")["auprc"].to_dict()
    except Exception as exc:
        print(f"[warn] harvest AUPRC unavailable ({exc}); legend falls back to "
              "the AP computed from the plotted curve")
        return {}


def _decimate(rec, prec, n=1500):
    if len(rec) <= n:
        return rec, prec
    ix = np.unique(np.linspace(0, len(rec) - 1, n).astype(int))
    return rec[ix], prec[ix]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    # Roman numerals, not letters: Table 2 uses (a)-(d) for a different
    # grouping and reusing the letters here invites a false correspondence.
    panels = [("C1", "(i) Stage-1 forecaster swap"),
              ("C2", "(ii) Stage-2 classifier swap"),
              ("C3", "(iii) Complete systems (end-to-end contrast)")]

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 9.2), sharey=True)
    pos_rate = None

    for ax, (contrast, title) in zip(axes, panels):
        try:
            from src.evaluation.export_latex_tables import (superseded_keys,
                                                            ablation_keys,
                                                            excluded_keys,
                                                            _paper_label)
            _skip = superseded_keys() | ablation_keys() | excluded_keys()
        except Exception:
            _skip = set()
            _paper_label = lambda t: t
        # Registry labels are LaTeX-flavoured; matplotlib renders "--" literally.
        _plot_label = lambda t: _paper_label(t).replace("--", "\u2013")
        # The legend quotes the AUPRC the benchmark tables print, not the AP
        # recomputed here: an arm whose simulation was refreshed after the last
        # harvest would otherwise carry one number in Table 2 and another in the
        # legend (BL-C11 read 0.294 against the table's 0.289).
        _harvest = _harvest_auprc()
        # Table 2 files the covariates-only, lagged-traffic and observed-traffic
        # arms under its no-decomposition and reference panels; keeping them in
        # panel (b) here would show a different membership than the table.
        if contrast == "C2":
            _skip |= {e["key"] for e in REGISTRY
                      if any(t in e["key"] for t in ("H2_", "LAG_", "OBS_"))}
        entries = [e for e in REGISTRY
                   if e["contrast"] == contrast and e["key"] not in _skip]
        proposed = next(e for e in entries if e.get("is_proposed"))
        baselines = [e for e in entries if not e.get("is_proposed")]

        curves = []          # (auprc, label, rec, prec, style)
        # -- baselines: fixed palette order (registry order = slot order)
        for slot, e in enumerate(baselines):
            r = load_pr(e)
            if r is None:
                print(f"[skip] {e['key']} PENDING (sim not found)")
                continue
            rec, prec, ap, pr = r
            pos_rate = pos_rate or pr
            ap = _shown(ap, _harvest, e["key"])
            curves.append((ap, f"{_plot_label(e['label'])} (AP={ap:.3f})", rec, prec,
                           dict(color=PALETTE[slot % len(PALETTE)], lw=1.8,
                                zorder=2)))
        # -- proposed (ink, thickest, on top)
        r = load_pr(proposed)
        if r is not None:
            rec, prec, ap, _ = r
            ap = _shown(ap, _harvest, proposed["key"])
            curves.append((ap, f"{_plot_label(proposed['label'])} (AP={ap:.3f})", rec, prec,
                           dict(color=INK, lw=2.8, zorder=4)))
        # -- ceiling reference (proposed pipeline on real traffic)
        r = load_pr(proposed, mode="no_gnn")
        if r is not None:
            rec, prec, ap, _ = r
            curves.append((ap, f"Ceiling: real traffic (AP={ap:.3f})", rec, prec,
                           dict(color=MUTED, lw=1.8, ls="--", zorder=3)))

        for ap, label, rec, prec, style in sorted(curves, key=lambda c: -c[0]):
            rec, prec = _decimate(rec, prec)
            ax.plot(rec, prec, label=label, solid_capstyle="round", **style)

        if pos_rate:
            ax.axhline(pos_rate, color=MUTED, lw=1.2, ls=":", zorder=1)
            ax.annotate(f"positive rate = {pos_rate:.4f}",
                        xy=(0.985, pos_rate), xytext=(0, 6),
                        textcoords="offset points", ha="right",
                        va="bottom", fontsize=8, color=MUTED,
                        zorder=6)

        ax.set_title(title, fontsize=12, color=INK, loc="left")
        ax.set_xlabel("Recall", fontsize=10, color=INK)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.grid(True, color=GRID, lw=0.7, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(MUTED)
        ax.tick_params(colors=INK, labelsize=9)
        # One column, always: a multi-column legend expands past its own
        # panel and its overflow renders under the neighbouring plot.
        ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.11),
                  ncol=1, fontsize=6.6, frameon=False, handlelength=1.5,
                  labelcolor=INK, borderaxespad=0.0, handletextpad=0.4)

    axes[0].set_ylabel("Precision", fontsize=10, color=INK)
    fig.suptitle("Precision–recall curves — crash prediction, 72 h "
                 "frozen regime (June 2025)",
                 fontsize=13, color=INK, x=0.01, ha="left")
    fig.subplots_adjust(left=0.045, right=0.995, top=0.90, bottom=0.42,
                        wspace=0.10)

    for ext in ("png", "pdf"):
        path = os.path.join(OUT_DIR, f"benchmark_pr_curves.{ext}")
        fig.savefig(path, dpi=300, facecolor="white")
        print(f"[DONE] {path}")


if __name__ == "__main__":
    main()
