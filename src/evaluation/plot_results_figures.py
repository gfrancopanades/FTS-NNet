"""Result-analysis figures and tables, all derived from the archived predictions.

Reads experiments/benchmark_results/predictions/*.parquet (written by
archive_benchmark_predictions.py) so every figure is guaranteed consistent with
the benchmark tables — no re-globbing of simulation folders, no re-derivation
of labels.

Produces (visualizations/):
  1. results_worked_day.{png,pdf}      - Section 5: one worked June day, risk
     surface over the corridor with actual crashes overlaid + the two tiers.
  2. results_durability.{png,pdf}      - Section 4.4: frozen-model AUPRC and
     dispatch precision/recall by month (Jun-Sep).
  3. results_tier_tradeoff.{png,pdf}   - Section 5: precision / recall /
     flagged-exposure vs threshold, with both operating tiers marked.

CPU-only, seconds to run once the archive exists:
    python src/evaluation/plot_results_figures.py
"""
from src.paths import (  # portable paths -- see src/paths.py
    PROJECT_ROOT_STR as _AP7_ROOT,
    EXPERIMENTS_ROOT_STR as _AP7_EXPERIMENTS,
    DATA_DIR_STR as _AP7_DATA,
    TABLES_DIR as _AP7_TABLES,
    FIGURES_DIR as _AP7_FIGS,
)

import json
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
PRED_DIR = os.path.join(PROD_ROOT, "experiments/benchmark_results/predictions")
OUT = str(_AP7_FIGS)

# validated categorical palette (fixed slot order, CVD-safe)
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
PLUM = "#8b5cf6"   # lift (secondary axis, figure 2)
INK, MUTED, GRID = "#1a1a19", "#8a8878", "#e9e8e0"
SEG_KM, BIN_H = 1.0, 5.0 / 60.0          # 1 km segments, 5-min bins
PROPOSED = "C2_C5_Focal"                  # MLP+Focal on the GeoLSTM stage-1
# The seasonal-cycle figure needs the 13-month simulation frame, while the
# benchmark tables need every model scored in the SAME one-month frame. Those
# are different simulations of the same model, so they are archived under
# separate keys; durability reads the cycle frame explicitly.
PROPOSED_CYCLE = "C2_C5_Focal_cycle"


def _load(key):
    p = os.path.join(PRED_DIR, f"{key}.parquet")
    return pd.read_parquet(p) if os.path.exists(p) else None


def _style(ax):
    ax.grid(True, color=GRID, lw=0.7); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=9)


def _thresholds(y, p, p_target=0.80, r_target=0.50):
    # r_target matches ADVISORY_RECALL in export_latex_tables and
    # operational_analysis. At 0.70 the point sits past the precision knee
    # (0.023 precision, 5.7% of corridor-time), so the figures would label a
    # tier the text does not describe.
    prec, rec, thr = precision_recall_curve(y, p)
    prec, rec = prec[:-1], rec[:-1]
    ok_p = np.where(prec >= p_target)[0]
    t_disp = thr[ok_p[np.argmax(rec[ok_p])]] if len(ok_p) else thr[np.argmax(prec)]
    ok_r = np.where(rec >= r_target)[0]
    t_adv = thr[ok_r[np.argmax(prec[ok_r])]] if len(ok_r) else thr[np.argmin(abs(rec - r_target))]
    return float(t_disp), float(t_adv), (prec, rec, thr)


# ---------------------------------------------------------------- figure 1
def fig_worked_day(df, t_disp, t_adv, day="2025-06-17"):
    d = df[(df.dat >= day) & (df.dat < pd.Timestamp(day) + pd.Timedelta("1D"))]
    if d.empty or d.y.sum() == 0:                 # pick the day with most crashes
        jun = df[(df.dat >= "2025-06-01") & (df.dat < "2025-07-01")].copy()
        jun["day"] = jun.dat.dt.date
        day = str(jun.groupby("day").y.sum().idxmax())
        d = jun[jun.day == pd.Timestamp(day).date()]
    piv = d.pivot_table(index="pk", columns=d.dat.dt.hour + d.dat.dt.minute / 60,
                        values="prob", aggfunc="max")
    fig, ax = plt.subplots(figsize=(11, 5.2))
    im = ax.pcolormesh(piv.columns, piv.index, piv.to_numpy(),
                       cmap="magma_r", vmin=0, vmax=max(t_disp, piv.to_numpy().max() * 0.9),
                       shading="auto")
    cr = d[d.y == 1]
    ax.scatter(cr.dat.dt.hour + cr.dat.dt.minute / 60, cr.pk, s=44,
               facecolors="none", edgecolors=AQUA, linewidths=1.1,
               alpha=0.9, label="observed crash", zorder=3)
    ax.set_xlabel("Hour of day", color=INK); ax.set_ylabel("Kilometre post", color=INK)
    ax.set_title(f"Issued 72 h ahead: forecast crash-risk surface, {day}",
                 color=INK, loc="left", fontsize=12)
    ax.legend(frameon=False, loc="upper left", fontsize=9, labelcolor=INK)
    cb = fig.colorbar(im, ax=ax, pad=0.02); cb.set_label("crash probability",
                                                    color=INK, labelpad=34)
    cb.ax.axhline(t_disp, color=BLUE, lw=2); cb.ax.axhline(t_adv, color=ORANGE, lw=2)
    cb.ax.text(3.6, t_disp, "dispatch", color=BLUE, va="center", fontsize=8,
               transform=cb.ax.get_yaxis_transform(), clip_on=False)
    cb.ax.text(3.6, t_adv, "advisory", color=ORANGE, va="center", fontsize=8,
               transform=cb.ax.get_yaxis_transform(), clip_on=False)
    _style(ax)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}/results_worked_day.{ext}", dpi=250, facecolor="white")
    plt.close(fig)
    n_fl = int((d.prob >= t_disp).sum())
    print(f"[fig] worked day {day}: {len(d):,} intervals, {int(d.y.sum())} crashes, "
          f"{n_fl} dispatch-flagged -> results_worked_day.png")


# ---------------------------------------------------------------- figure 2
def fig_durability(df, t_disp):
    # Bi-monthly coverage of a full seasonal cycle. Training months (Jun-2024,
    # May-2025) are excluded. Months before the training end are backcasts and
    # are marked as such in the figure; Jun/Aug-2025 are forward validation.
    months = [("Aug\n2024", "2024-08-01", "2024-09-01"), ("Oct\n2024", "2024-10-01", "2024-11-01"),
              ("Dec\n2024", "2024-12-01", "2025-01-01"), ("Feb\n2025", "2025-02-01", "2025-03-01"),
              ("Apr\n2025", "2025-04-01", "2025-05-01"), ("Jun\n2025", "2025-06-01", "2025-07-01"),
              ("Aug\n2025", "2025-08-01", "2025-09-01")]
    rows = []
    for name, s, e in months:
        d = df[(df.dat >= s) & (df.dat < e)]
        if len(d) == 0 or d.y.sum() == 0:
            continue
        y, p = d.y.to_numpy(), d.prob.to_numpy()
        b = (p >= t_disp)
        tp, fp = int((b & (y == 1)).sum()), int((b & (y == 0)).sum())
        fn = int((~b & (y == 1)).sum())
        ap = average_precision_score(y, p)
        # The AUPRC floor equals the prevalence, so months with more crashes
        # score higher for free. Lift = AUPRC / base rate is the only
        # cross-month-comparable form and must be read alongside AUPRC.
        base = float(y.mean())
        rows.append(dict(month=name, auprc=ap, base=base, lift=ap / max(base, 1e-12),
                         prec=tp / max(tp + fp, 1), rec=tp / max(tp + fn, 1)))
    if not rows:
        print("[fig] durability skipped (single-month predictions)"); return
    r = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(r.month, r.auprc, "-o", color=BLUE, lw=2.2, ms=7, label="AUPRC")
    ax.plot(r.month, r.prec, "-s", color=ORANGE, lw=2, ms=6, label="dispatch precision")
    ax.plot(r.month, r.rec, "-^", color=YELLOW, lw=2, ms=6, label="dispatch recall")
    # mark where forward validation begins (everything left of it is a backcast)
    fwd = [i for i, m in enumerate(r.month) if m.startswith("Jun")]
    if fwd:
        ax.axvspan(fwd[0] - 0.5, len(r) - 0.5, color=AQUA, alpha=0.07, zorder=0)
        ax.text(fwd[0] - 0.4, 0.04, "forward validation\n(after training)",
                fontsize=8, color=AQUA, va="bottom")
    # sits above the axes as a subtitle: at y=0.97 inside the axes it crossed
    # the dispatch-precision line, which runs between 0.82 and 0.97
    ax.text(0.0, 1.012, "model frozen; training months (Jun-2024, May-2025) excluded",
            transform=ax.transAxes, fontsize=8.5, color=MUTED, va="bottom")
    ax.set_ylim(0, 1); ax.set_ylabel("metric", color=INK)
    ax.set_title("Frozen-model performance across a full seasonal cycle", color=INK,
                 loc="left", fontsize=12, pad=26)

    # Lift on a secondary axis: AUPRC alone is not comparable across months
    # because its floor is the month's prevalence.
    ax2 = ax.twinx()
    ax2.plot(r.month, r.lift, "--D", color=PLUM, lw=1.6, ms=5, alpha=0.85,
             label="lift (AUPRC / base rate)")
    ax2.set_ylabel("lift over base rate (x)", color=PLUM)
    ax2.tick_params(axis="y", colors=PLUM, labelsize=9)
    ax2.set_ylim(0, max(r.lift) * 1.35)
    for s in ("top", "left", "bottom"):
        ax2.spines[s].set_visible(False)
    ax2.spines["right"].set_color(PLUM)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=9, labelcolor=INK,
              loc="lower left", ncol=2)
    _style(ax); fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}/results_durability.{ext}", dpi=250, facecolor="white")
    plt.close(fig)
    print(f"[fig] durability: {r.to_dict('records')} -> results_durability.png")
    _write_seasonal_cycle_table(r, t_disp)


def _write_seasonal_cycle_table(r, t_disp):
    """Emit Table 6 from the very numbers Figure 3 plots.

    Written here, from the same frame, so the table and the figure cannot
    disagree. The operating-point columns depend on the dispatch threshold,
    which is calibrated on the June frame - so a change in that frame moves
    precision and recall even when AUPRC and lift are untouched.
    """
    rows = []
    for _, x in r.iterrows():
        rows.append(
            f"{x['month'].replace(chr(10), ' ')} & {x['base'] * 100:.3f}\\% & "
            f"{x['auprc']:.3f} & {x['lift']:.0f}$\\times$ & "
            f"{x['prec']:.3f} & {x['rec']:.3f} \\\\")
    # separate the calendar years: the discontinuity is the point of the table
    split = next((i for i, x in enumerate(r.month) if "2025" in x), None)
    if split:
        rows.insert(split, r"\midrule")

    tex = (r"""% Frozen-model performance across a full seasonal cycle. Requires: booktabs.
% Generated by src/evaluation/plot_results_figures.py -- do not hand-edit.
\begin{table}[t]
\centering
\caption{Frozen-model performance across a full seasonal cycle. One model, trained once and never
updated, scored on seven bi-monthly slices with the dispatch threshold ("""
           + f"{t_disp:.4f}" + r""") read from the June-2025 precision-recall curve at a precision target of 0.80 and then held
fixed across every slice; the two training months are
excluded. Lift is AUPRC divided by each slice's own base rate, and is the only cross-month-comparable
form because AUPRC is bounded below by prevalence. Performance is tightly held across the 2025 slices
but degrades on the 2024 slices; the discontinuity follows the calendar year rather than the season
and is discussed in Sections 4.4 and 7.}
\label{tab:seasonal}
\begin{tabular}{l r r r r r}
\toprule
Slice & Base rate & AUPRC & Lift & Dispatch precision & Dispatch recall \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
    out = os.path.join(PROD_ROOT, os.path.join(str(_AP7_TABLES), "table3_seasonal_cycle.tex"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        fh.write(tex)
    print(f"[tbl] seasonal cycle (thr={t_disp:.4f}) -> {out}")


# ---------------------------------------------------------------- figure 3
def fig_tier_tradeoff(y, p, t_disp, t_adv, curve):
    prec, rec, thr = curve
    # vectorised: counts above each threshold via a single sort + searchsorted
    ps = np.sort(p)
    expo = (len(ps) - np.searchsorted(ps, thr, side="left")) * SEG_KM * BIN_H
    step = max(1, len(thr) // 2000)
    thr_s, prec_s, rec_s = thr[::step], prec[::step], rec[::step]
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.plot(thr_s, prec_s, color=BLUE, lw=2, label="precision")
    ax.plot(thr_s, rec_s, color=ORANGE, lw=2, label="recall")
    ax.set_xscale("log"); ax.set_xlabel("decision threshold", color=INK)
    ax.set_ylabel("precision / recall", color=INK); ax.set_ylim(0, 1)
    ax2 = ax.twiny(); ax2.axis("off")            # keep single y-scale rule: exposure annotated, not axis
    # The two thresholds sit close together (0.42 and 0.52), so centred labels
    # at a common height overlap. Anchor each to its own side of its line and
    # stagger the heights.
    for t, c, lab, ha, dy in ((t_adv, ORANGE, "advisory", "right", 1.015),
                              (t_disp, BLUE, "dispatch", "left", 1.105)):
        ax.axvline(t, color=c, ls="--", lw=1.4)
        km = (len(ps) - np.searchsorted(ps, t, side="left")) * SEG_KM * BIN_H
        pad = 0.93 if ha == "right" else 1.07
        ax.text(t * pad, dy, f"{lab}  {km:,.0f} km·h", color=c, fontsize=8.5,
                ha=ha, va="bottom")
    ax.set_title("Operating tiers on the June risk surface", color=INK, loc="left",
                 fontsize=12, pad=30)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="center left")
    _style(ax); fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}/results_tier_tradeoff.{ext}", dpi=250, facecolor="white")
    plt.close(fig)
    print("[fig] tier trade-off -> results_tier_tradeoff.png")


def main():
    os.makedirs(OUT, exist_ok=True)
    if not os.path.exists(os.path.join(PRED_DIR, "manifest.json")):
        sys.exit("Run archive_benchmark_predictions.py first (no prediction store found).")
    df = _load(PROPOSED)
    if df is None:
        sys.exit(f"Proposed model {PROPOSED} not in the prediction store.")
    jun = df[(df.dat >= "2025-06-01") & (df.dat < "2025-07-01")]
    y, p = jun.y.to_numpy(), jun.prob.to_numpy()
    t_disp, t_adv, curve = _thresholds(y, p)
    print(f"[cfg] dispatch thr={t_disp:.4f}  advisory thr={t_adv:.4f}")

    fig_worked_day(df, t_disp, t_adv)
    # The durability panel needs the 13-month seasonal-cycle simulation. Falling
    # back to the single-month June frame here would render a one-month result
    # under a "full seasonal cycle" title -- a mixed-frame figure of exactly the
    # kind that has to be caught before it ships. The cycle simulation was NOT
    # rebuilt after the 2026-08-12 leak/dedup fixes, so any cycle frame on disk
    # predates them and must not be combined with the rebuilt June panels.
    # Skip the figure entirely rather than fabricate it; restore it by re-running
    # the 13-month simulation on the fixed pipeline.
    cyc = _load(PROPOSED_CYCLE)
    if cyc is None:
        print(f"[SKIP] {PROPOSED_CYCLE} absent -> durability/seasonal-cycle figure "
              f"NOT produced (would otherwise be a one-month frame mislabelled as "
              f"a seasonal cycle). Re-run the 13-month simulation to restore it.")
    else:
        fig_durability(cyc, t_disp)
    fig_tier_tradeoff(y, p, t_disp, t_adv, curve)
    print("[DONE] result figures written to visualizations/")


if __name__ == "__main__":
    main()
