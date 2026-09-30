#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Table 4 — the benchmark across four rolling-origin months.

Replaces the single-arm rolling-origin table with the full classifier and
end-to-end panels evaluated on May, June, July and August 2025. Each month is
predicted by a model whose Stage-1 forecaster and Stage-2 classifier were
trained on windows rolled to end before it, so every cell is strictly
out-of-sample.

AUPRC is bounded below by prevalence, and the monthly base rate varies by more
than 40 % across these months, so every cell carries lift alongside AUPRC and
the ranking discussion in the manuscript is based on lift.
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
import os, sys, glob
import pandas as pd

ROOT = _AP7_ROOT
OUT_T = str(_AP7_TABLES)
OUT_R = str(_RESULTS_DIR)
MONTHS = [("May-2025", "May"), ("Jul-2025", "July"), ("Aug-2025", "Aug")]
JUNE_CSV = os.path.join(ROOT, "experiments/benchmark_results/benchmark_accident_3day_full.csv")
# arms shown in the multi-month table: the contrasts that carry H1 and H2
KEEP = ["C2_C5_Focal", "C2_C4_MLP2stage", "C2_C8_MLPF2", "C2_C11_LogRegPK",
        "C2_C0_LogReg", "C2_C10_HistRisk", "C2_LAG_MLPF", "H1_OBS_MLPF",
        "C3_E4_LSTM_cov", "C3_E5_Transf_cov"]


def _load(month_dir):
    hits = glob.glob(os.path.join(month_dir, "*full*.csv")) or \
           glob.glob(os.path.join(month_dir, "*.csv"))
    if not hits:
        return None
    return pd.read_csv(max(hits, key=os.path.getmtime))


def main() -> int:
    frames = {}
    j = pd.read_csv(JUNE_CSV)
    frames["June"] = j
    for mdir, label in MONTHS:
        d = _load(os.path.join(ROOT, "experiments/benchmark_results", f"rolling_{mdir}"))
        if d is None:
            print(f"[WARN] no harvest for {mdir} — column omitted")
            continue
        frames[label] = d

    order = ["May", "June", "July", "Aug"]
    cols = [c for c in order if c in frames]
    print(f"[cfg] months available: {cols}")

    rows = []
    for key in KEEP:
        rec = {"key": key}
        lab = None
        for c in cols:
            f = frames[c]
            m = f[f.key.astype(str) == key]
            if m.empty:
                rec[c] = None; continue
            r = m.iloc[0]
            lab = lab or str(r.get("label", key))
            ap = float(r["auprc"]); prev = float(r["prevalence"]) if "prevalence" in r else None
            rec[c] = (ap, ap / prev if prev else None)
        rec["label"] = lab or key
        rows.append(rec)

    os.makedirs(OUT_R, exist_ok=True); os.makedirs(OUT_T, exist_ok=True)
    flat = []
    for r in rows:
        o = {"key": r["key"], "label": r["label"]}
        for c in cols:
            v = r.get(c)
            o[f"{c}_auprc"] = None if v is None else round(v[0], 4)
            o[f"{c}_lift"] = None if v is None or v[1] is None else round(v[1], 1)
        flat.append(o)
    pd.DataFrame(flat).to_csv(os.path.join(OUT_R, "benchmark_multimonth.csv"), index=False)

    esc = lambda s: str(s).replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")
    L = [r"% Four-month rolling-origin benchmark. GENERATED -- do not hand-edit.",
         r"\begin{table*}[t]", r"\centering",
         r"\caption{Benchmark across four rolling-origin months. Each month is "
         r"predicted by a model trained on windows rolled to end before it. Cells "
         r"give AUPRC with lift over that month's own base rate in parentheses; "
         r"lift is the comparable quantity across months because AUPRC is bounded "
         r"below by prevalence.}",
         r"\label{tab:multimonth}",
         r"\begin{tabular}{l" + " c" * len(cols) + r"}", r"\toprule",
         "Configuration & " + " & ".join(cols) + r" \\", r"\midrule"]
    for r in rows:
        cells = []
        for c in cols:
            v = r.get(c)
            cells.append("--" if v is None else
                         (f"{v[0]:.3f} ({v[1]:.0f}$\\times$)" if v[1] else f"{v[0]:.3f}"))
        nm = esc(r["label"])
        if r["key"] == "C2_C5_Focal":
            nm = rf"\textbf{{{nm}}}"
        L.append(f"{nm} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    path = os.path.join(OUT_T, "table4_multimonth.tex")
    open(path, "w").write("\n".join(L))
    print(f"[DONE] {path}")
    print(f"[DONE] {OUT_R}/benchmark_multimonth.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
