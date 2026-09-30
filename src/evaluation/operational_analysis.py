"""Operations section — dual-tier operating points, agency-unit yield, calibration.

Reads the proposed model's June simulation and reports, at two deployment tiers:
  * DISPATCH  (high precision): threshold for precision >= P_TARGET
  * ADVISORY  (high recall):    threshold for recall    >= R_TARGET
For each: precision, recall, F1, FAR, crashes captured, and flagged EXPOSURE in
km*h (500 m per PK-segment x 5-min bins) — the unit an agency acts on. Plus a
reliability (calibration) curve for the advisory tier, since dynamic-speed /
VMS actions need calibrated probabilities, not just ranking.

CPU-only. Outputs -> visualizations/operational_{tiers.md, calibration.png}
"""
from src.paths import (  # portable paths -- see src/paths.py
    PROJECT_ROOT_STR as _AP7_ROOT,
    EXPERIMENTS_ROOT_STR as _AP7_EXPERIMENTS,
    DATA_DIR_STR as _AP7_DATA,
    TABLES_DIR as _AP7_TABLES,
    FIGURES_DIR as _AP7_FIGS,
)

import os
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

from sklearn.metrics import precision_recall_curve

# Resolved through the benchmark registry rather than hardcoded: simulation
# folders are named after the model AND its training/simulation windows, so a
# fixed path silently rots when a model is re-simulated (and previously pointed
# at a 13-month frame while the tables assumed the one-month frame).
PROPOSED_KEY = "C2_C5_Focal"          # MLP+Focal on the GeoLSTM stage-1


def _proposed_csv():
    import sys
    root = _AP7_ROOT
    if root not in sys.path:
        sys.path.insert(0, root)
    from src.evaluation.benchmark_harvest import REGISTRY, discover_sim_csv, EXPERIMENTS_ROOT
    entry = next(e for e in REGISTRY if e["key"] == PROPOSED_KEY)
    csv = discover_sim_csv(entry, EXPERIMENTS_ROOT, "two_stage")
    if not csv:
        raise FileNotFoundError(f"no simulation found for {PROPOSED_KEY}")
    print(f"[cfg] {PROPOSED_KEY} -> {os.path.basename(os.path.dirname(csv))}")
    return csv
OUT = f"{_AP7_ROOT}/visualizations"
SEG_KM = 1.0           # 1 km per kilometre-post segment (PK 120-220 = 100 km)
BIN_H = 5.0 / 60.0     # 5-min bin in hours
P_TARGET, R_TARGET = 0.80, 0.50
EVAL_START, EVAL_END = "2025-06-01", "2025-07-01"


def tier_metrics(y, p, thr):
    b = (p >= thr).astype(int)
    tp = int(((b == 1) & (y == 1)).sum()); fp = int(((b == 1) & (y == 0)).sum())
    fn = int(((b == 0) & (y == 1)).sum())
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    far = fp / max((y == 0).sum(), 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    flagged = int((b == 1).sum())
    return dict(thr=thr, prec=prec, rec=rec, f1=f1, far=far, tp=tp, fp=fp,
                flagged=flagged, exposure_kmh=flagged * SEG_KM * BIN_H,
                crashes_captured=tp, crashes_total=int(y.sum()))


def main():
    os.makedirs(OUT, exist_ok=True)
    df = pd.read_csv(_proposed_csv(), sep=";", usecols=["pk", "sen", "dat", "accident_probability", "ACCIDENT_real"])
    df["dat"] = pd.to_datetime(df["dat"])
    df = df[(df["dat"] >= EVAL_START) & (df["dat"] < EVAL_END)]
    y = df["ACCIDENT_real"].astype(int).to_numpy()
    p = df["accident_probability"].astype(float).to_numpy()

    prec, rec, thr = precision_recall_curve(y, p)
    # thr has len-1 vs prec/rec; align
    prec, rec = prec[:-1], rec[:-1]
    # DISPATCH: highest recall attainable at or above the precision target
    ok_p = np.where(prec >= P_TARGET)[0]
    thr_dispatch = thr[ok_p[np.argmax(rec[ok_p])]] if len(ok_p) else thr[np.argmax(prec)]
    # ADVISORY: highest precision attainable at or above the recall target
    ok_r = np.where(rec >= R_TARGET)[0]
    thr_advisory = thr[ok_r[np.argmax(prec[ok_r])]] if len(ok_r) else thr[np.argmin(np.abs(rec - R_TARGET))]
    # BALANCED: the F1-optimal point, reported so the two tiers are read as
    # deliberate departures from the balance rather than as arbitrary cuts.
    f1c = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec + 1e-12), 0.0)
    thr_balanced = thr[int(np.argmax(f1c))]

    d = tier_metrics(y, p, thr_dispatch)
    a = tier_metrics(y, p, thr_advisory)
    b = tier_metrics(y, p, thr_balanced)
    total_kmh = len(y) * SEG_KM * BIN_H

    lines = [
        "# Operational analysis — proposed system (GeoLSTM + MLP+Focal), June 2025",
        "",
        f"Corridor exposure scored: {len(y):,} segment-intervals = "
        f"{total_kmh:,.0f} km*h; {int(y.sum())} crash intervals.",
        f"Segment = {SEG_KM} km, bin = 5 min.",
        "",
        "| Tier | Target | Thr | Precision | Recall | F1 | FAR | Crashes captured | Flagged exposure (km*h) | % of corridor flagged |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, tgt, m in (("DISPATCH (patrol/EMS)", f"P>={P_TARGET}", d),
                         ("BALANCED (max F1)", "max F1", b),
                         ("ADVISORY (VMS/VSL)", f"R>={R_TARGET}", a)):
        lines.append(
            f"| {name} | {tgt} | {m['thr']:.3f} | {m['prec']:.3f} | {m['rec']:.3f} "
            f"| {m['f1']:.3f} | {m['far']:.3f} | {m['crashes_captured']}/{m['crashes_total']} "
            f"| {m['exposure_kmh']:,.0f} | {100*m['flagged']/len(y):.2f}% |")
    lines += ["",
              f"**Dispatch reading:** flagging {100*d['flagged']/len(y):.2f}% of corridor-time "
              f"({d['exposure_kmh']:,.0f} km*h) captures {d['crashes_captured']} of "
              f"{d['crashes_total']} crashes at {d['prec']:.0%} precision "
              f"({d['crashes_captured']/max(d['exposure_kmh'],1)*100:.2f} crashes per 100 km*h flagged).",
              f"**Advisory reading:** catching {a['rec']:.0%} of crashes needs "
              f"{a['exposure_kmh']:,.0f} km*h ({100*a['flagged']/len(y):.1f}% of corridor-time)."]
    with open(os.path.join(OUT, "operational_tiers.md"), "w") as fh:
        fh.write("\n".join(lines))
    print("\n".join(lines))

    # Reliability / calibration curve (advisory tier needs calibrated prob).
    bins = np.linspace(0, 1, 11)
    idx = np.digitize(p, bins) - 1
    xs, ys, ns = [], [], []
    for b in range(10):
        m = idx == b
        if m.sum() > 0:
            xs.append(p[m].mean()); ys.append(y[m].mean()); ns.append(int(m.sum()))
    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.plot([0, 1], [0, 1], color="#8a8878", lw=1.2, ls="--", label="perfect calibration")
    ax.plot(xs, ys, "-o", color="#2a78d6", lw=2, ms=6, label="proposed model")
    ax.set_xlabel("Mean predicted probability", color="#1a1a19")
    ax.set_ylabel("Observed crash frequency", color="#1a1a19")
    ax.set_title("Reliability curve — June 2025", color="#1a1a19", loc="left")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    for _ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"operational_calibration.{_ext}"),
                    dpi=200, facecolor="white")
    print(f"\n[DONE] {OUT}/operational_tiers.md + operational_calibration.png")


if __name__ == "__main__":
    main()
