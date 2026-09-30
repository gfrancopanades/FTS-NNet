#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
2nd-article benchmark harvester — Accident-prediction table (3-day regime)
==========================================================================

Walks the finished frozen-rolling simulations of every benchmark model and
assembles the headline ACCIDENT-PREDICTION comparison table for the paper,
in the *3-day-ahead* deployment regime (the `two_stage` / forecast-driven
simulation — NOT the real-time `no_gnn` ceiling, which is reported once as a
reference row per contrast).

It is built to be run REPEATEDLY: models whose simulation has not finished yet
are listed with status=PENDING and skipped;
re-running once their sims land fills those rows in automatically.

Three controlled contrasts are emitted:
  C1  Forecaster swap   — classifier fixed (v58 GBDT cascade), forecaster varies
  C2  Classifier swap   — forecaster fixed (proposed GNN-LSTM v17), classifier varies
  C3  End-to-end        — GeoLSTM + MLP-focal cascade vs end-to-end baselines

Metrics per model (all on the strictly-unseen eval window):
  * AUPRC  (primary; threshold-free)         + day-block bootstrap 95% CI
  * AUROC  (secondary; threshold-free)       + day-block bootstrap 95% CI
  * Brier
  * Frozen operating point (the shipped `accident_pred_binary`):
        Precision / Recall / F1 / F2 / FAR
  * Common-rule operating point (identical across models: max precision s.t.
        recall >= --recall-floor, re-derived on the eval PR curve):
        Precision / Recall / F1 / F2 / FAR + threshold
  * Significance vs the contrast's proposed row: paired day-block bootstrap on
        ΔAUPRC (and ΔAUROC), one-sided p that proposed > baseline.

Outputs (to --output-dir):
  benchmark_accident_3day_full.csv      one row per (contrast, model), all metrics
  benchmark_accident_3day_summary.md    formatted per-contrast tables for the paper
"""

from __future__ import annotations
from src.paths import (  # portable paths -- see src/paths.py
    PROJECT_ROOT_STR as _AP7_ROOT,
    EXPERIMENTS_ROOT_STR as _AP7_EXPERIMENTS,
    DATA_DIR_STR as _AP7_DATA,
    TABLES_DIR as _AP7_TABLES,
    FIGURES_DIR as _AP7_FIGS,
)

import argparse
import glob
import json
import os
import re
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROD_ENV_ROOT = _AP7_ROOT
EXPERIMENTS_ROOT = _AP7_EXPERIMENTS

# Running this file directly puts src/evaluation/ on sys.path, not the project
# root, so the shared package imports below need the root added explicitly.
if PROD_ENV_ROOT not in sys.path:
    sys.path.insert(0, PROD_ENV_ROOT)

from src.training.sim_dirname import parse_sim_dirname  # noqa: E402

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
# experiments/registry_final.json pins every arm to the runs it scores:
#   "cascade"   -> <root>/*_<gnn>/xgboost_<xgb>/simulation_two_stage_*/ *.csv
#   "benchmark" -> <root>/*_<gnn>/xgboost_<xgb>/simulation_benchmark_*/ *.csv
#   "e2e"       -> <root>/*_<run>/simulation_e2e_*/ *.csv
# A row with is_proposed=True is the reference the others in its contrast are
# tested against; proposed_key points each baseline at that reference. Every
# module that imports REGISTRY reads the same file.
_REGISTRY_JSON = os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), "experiments", "registry_final.json")
with open(_REGISTRY_JSON) as _f:
    REGISTRY = json.load(_f)


SCORE_COL = "accident_probability"
PRED_COL = "accident_pred_binary"
TRUE_COL = "ACCIDENT_real"

# Common evaluation window — the Layer-2 / cascade sim month. The Layer-3
# end-to-end sims roll over Jun–Oct 2025 (4 months); restricting every model to
# this single month makes the comparison apples-to-apples (same days scored).
EVAL_WINDOW_START = "2025-06-01"
EVAL_WINDOW_END = "2025-07-01"   # exclusive


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------
def _latest(paths):
    """Pick one simulation CSV from the candidates.

    Simulation folders are tagged with their evaluation window
    (``..._win<START><END>``) so that the same trained model simulated over
    different windows is preserved side by side rather than overwritten. The
    engineered features are estimated from a fraction of the assembled frame,
    so a longer simulation window changes the predictions for the very same
    rows; comparing models across different windows is therefore invalid.
    We resolve this by preferring the NARROWEST tagged window that still
    contains the evaluation period, and fall back to plain lexical order for
    legacy folders that carry no window tag.
    """
    paths = sorted(paths)
    if not paths:
        return None

    def _span(p):
        """Simulation window of the folder that holds this CSV, or None.

        Understands the canonical
        ``..._tr<s>-<e>_sim<s>-<e>`` form and the older ``_win<s><e>`` form;
        see src/training/sim_dirname.py.
        """
        for part in p.split(os.sep):
            info = parse_sim_dirname(part)
            if info:
                return info["sim_start"], info["sim_end"]
        return None

    want_s = EVAL_WINDOW_START.replace("-", "")
    want_e = EVAL_WINDOW_END.replace("-", "")
    covering = []
    for p in paths:
        sp = _span(p)
        if sp and sp[0] <= want_s and sp[1] >= want_e:
            covering.append((int(sp[1]) - int(sp[0]), p))
    if covering:
        return min(covering)[1]          # narrowest window covering the eval period
    return paths[-1]


def discover_sim_csv(entry, root, mode="two_stage"):
    """Return the simulation CSV path for a registry entry, or None if absent.

    mode: "two_stage" (forecast-driven, 3-day) or "no_gnn" (real-traffic ceiling).
    """
    kind = entry["kind"]
    if kind == "e2e":
        # end-to-end has no no_gnn twin
        if mode != "two_stage":
            return None
        pat = os.path.join(root, f"*_{entry['run']}", "simulation_e2e_*", "*.csv")
        return _latest(glob.glob(pat))

    gnn, xgb = entry["gnn"], entry["xgb"]
    if mode == "no_gnn":
        pat = os.path.join(root, f"*_{gnn}", f"xgboost_{xgb}",
                           "simulation_no_gnn_*", "*.csv")
        return _latest(glob.glob(pat))

    # Which subdirectory a run writes is a property of the TRAINING SCRIPT, not
    # of the registry's `kind` field, and the two drifted apart: every arm added
    # after the v73 sweep writes `simulation_benchmark_*` while carrying
    # kind="cascade", so keying the glob on `kind` silently reported fifteen of
    # sixteen Stage-1 arms as "sim not found" when their simulations existed.
    # Look for both layouts and take the newest match.
    hits = []
    for subdir in ("simulation_benchmark_*", "simulation_two_stage_*"):
        hits += glob.glob(os.path.join(root, f"*_{gnn}", f"xgboost_{xgb}",
                                       subdir, "*.csv"))
    return _latest(hits)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_scores(path, window=None):
    """Load (day, y_true, y_score, y_pred) from a simulation CSV (sep=';').

    window: optional (start, end) ISO date strings; rows are kept only for
    start <= day < end (end exclusive). Used to clip the 4-month e2e sims to
    the single Layer-2 month so every model is scored on the same days.
    """
    usecols = ["dat", SCORE_COL, PRED_COL, TRUE_COL]
    for enc in ("utf-8", "latin-1"):
        try:
            df = pd.read_csv(path, sep=";", usecols=usecols, decimal=".",
                             encoding=enc, low_memory=False)
            break
        except (UnicodeDecodeError, ValueError):
            continue
    else:
        raise IOError(f"Could not read {path}")
    df = df.dropna(subset=[SCORE_COL, TRUE_COL])
    out = pd.DataFrame({
        "day": df["dat"].astype(str).str.slice(0, 10),
        "y_true": (df[TRUE_COL].astype(float) > 0.5).astype(np.int8),
        "y_score": df[SCORE_COL].astype(np.float32),
        "y_pred": (df[PRED_COL].astype(float) > 0.5).astype(np.int8),
    })
    if window is not None:
        start, end = window
        out = out[(out["day"] >= start) & (out["day"] < end)].reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _prf(tp, fp, fn):
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    f2 = 5 * prec * rec / (4 * prec + rec) if (4 * prec + rec) else 0.0
    far = fp / (tp + fp) if (tp + fp) else 0.0  # = 1 - precision
    return prec, rec, f1, f2, far


def point_metrics(y_true, y_pred, prefix):
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    prec, rec, f1, f2, far = _prf(tp, fp, fn)
    return {f"{prefix}_precision": prec, f"{prefix}_recall": rec,
            f"{prefix}_f1": f1, f"{prefix}_f2": f2, f"{prefix}_far": far,
            f"{prefix}_tp": tp, f"{prefix}_fp": fp, f"{prefix}_fn": fn}


def common_operating_point(y_true, y_score, recall_floor):
    """Max-precision threshold subject to recall >= recall_floor (identical rule)."""
    from sklearn.metrics import precision_recall_curve
    prec, rec, thr = precision_recall_curve(y_true, y_score)
    # precision_recall_curve returns len(thr)+1 points; align to thresholds
    prec, rec = prec[:-1], rec[:-1]
    ok = rec >= recall_floor
    if not np.any(ok):
        # cannot reach the floor; take the highest-recall threshold
        idx = int(np.argmax(rec))
    else:
        cand = np.where(ok)[0]
        idx = cand[int(np.argmax(prec[cand]))]
    t = float(thr[idx])
    y_pred = (y_score >= t).astype(np.int8)
    m = point_metrics(y_true, y_pred, "common")
    m["common_threshold"] = t
    return m


def bootstrap_by_day(per_day, day_resamples):
    """Return arrays of AUPRC/AUROC over day-block bootstrap resamples.

    per_day: dict day -> (y_true, y_score) numpy arrays.
    day_resamples: list of lists of day keys (with replacement).
    """
    auprc, auroc = [], []
    for days in day_resamples:
        yt = np.concatenate([per_day[d][0] for d in days])
        ys = np.concatenate([per_day[d][1] for d in days])
        if yt.sum() == 0 or yt.sum() == len(yt):
            continue
        auprc.append(average_precision_score(yt, ys))
        auroc.append(roc_auc_score(yt, ys))
    return np.array(auprc), np.array(auroc)


def make_per_day(df):
    g = {}
    for day, sub in df.groupby("day"):
        g[day] = (sub["y_true"].to_numpy(), sub["y_score"].to_numpy())
    return g


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _safe_brier(y_true, y_score, key):
    """Brier score, tolerating scores that stray outside [0, 1].

    Balanced-bagging averages base-estimator outputs and can land marginally
    outside the unit interval (BL-C3 reaches -0.083 / 1.008). AUPRC and AUROC
    are rank statistics and are unaffected, but the Brier score is defined on
    probabilities, so sklearn refuses. Clipping keeps the arm in the table
    rather than failing the whole harvest, and the excursion is reported
    loudly because a model whose output is not a probability should not have
    its calibration read at face value.
    """
    import numpy as _np
    lo, hi = float(_np.nanmin(y_score)), float(_np.nanmax(y_score))
    if lo < 0.0 or hi > 1.0:
        print(f"[WARN] {key}: scores outside [0,1] (min={lo:.4f}, max={hi:.4f}); "
              f"clipped for the Brier score only -- treat its calibration as unreliable")
        y_score = _np.clip(y_score, 0.0, 1.0)
    return float(brier_score_loss(y_true, y_score))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiments-root", default=EXPERIMENTS_ROOT)
    ap.add_argument("--output-dir",
                    default=os.path.join(PROD_ENV_ROOT, "experiments", "benchmark_results"))
    ap.add_argument("--recall-floor", type=float, default=0.40,
                    help="Recall floor for the common operating-point rule.")
    ap.add_argument("--n-boot", type=int, default=1000,
                    help="Day-block bootstrap resamples (0 disables CIs/significance).")
    ap.add_argument("--eval-start", default=EVAL_WINDOW_START,
                    help="Eval window start (inclusive). Clips all models to the Layer-2 month.")
    ap.add_argument("--eval-end", default=EVAL_WINDOW_END,
                    help="Eval window end (exclusive).")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--registry-json", default=None,
                    help="Optional path to a JSON list of registry entries (same schema as the "
                         "module-level REGISTRY) that REPLACES it for this run. Use when "
                         "harvesting a parallel set of runs -- e.g. models trained on an "
                         "ACCIDENT-relabelled dataset -- without editing this file, so a "
                         "concurrently-queued harvest of the built-in REGISTRY is unaffected. "
                         "Omitted (default) = use the built-in REGISTRY unchanged.")
    args = ap.parse_args()
    window = (args.eval_start, args.eval_end)

    global REGISTRY
    if args.registry_json:
        with open(args.registry_json) as _f:
            REGISTRY = json.load(_f)
        print(f"[REGISTRY] Overridden from {args.registry_json}: {len(REGISTRY)} entries")

    os.makedirs(args.output_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # ---- pass 1: load every available model -------------------------------
    loaded = {}   # key -> dict(entry, df, status, csv)
    for e in REGISTRY:
        csv = discover_sim_csv(e, args.experiments_root, "two_stage")
        if csv is None:
            loaded[e["key"]] = dict(entry=e, df=None, status="PENDING", csv=None)
            print(f"[PENDING] {e['key']:18s} {e['label']:34s} (sim not found)")
            continue
        try:
            df = load_scores(csv, window=window)
            loaded[e["key"]] = dict(entry=e, df=df, status="OK", csv=csv)
            print(f"[OK]      {e['key']:18s} {e['label']:34s} "
                  f"n={len(df):>9,} pos={int(df['y_true'].sum()):>5} "
                  f"-> {os.path.basename(csv)}")
        except Exception as exc:  # noqa: BLE001
            loaded[e["key"]] = dict(entry=e, df=None, status=f"ERROR: {exc}", csv=csv)
            print(f"[ERROR]   {e['key']:18s} {exc}")

    # ---- per-contrast shared day-block bootstrap --------------------------
    rows = []
    boot_auprc = {}   # key -> array (for significance)
    boot_auroc = {}
    contrasts = sorted({e["contrast"] for e in REGISTRY})

    for c in contrasts:
        members = [k for k in loaded
                   if loaded[k]["entry"]["contrast"] == c and loaded[k]["status"] == "OK"]
        if not members:
            continue
        # common day set across present members of this contrast
        common_days = set.intersection(*[set(loaded[k]["df"]["day"].unique()) for k in members])
        common_days = sorted(common_days)
        day_resamples = []
        if args.n_boot > 0 and common_days:
            n = len(common_days)
            for _ in range(args.n_boot):
                idx = rng.integers(0, n, size=n)
                day_resamples.append([common_days[i] for i in idx])

        for k in members:
            df = loaded[k]["df"]
            yt = df["y_true"].to_numpy()
            ys = df["y_score"].to_numpy()
            yp = df["y_pred"].to_numpy()
            row = dict(contrast=c, key=k, label=loaded[k]["entry"]["label"],
                       version=loaded[k]["entry"]["version"], status="OK",
                       n=len(df), n_pos=int(yt.sum()),
                       prevalence=float(yt.mean()),
                       auprc=float(average_precision_score(yt, ys)),
                       auroc=float(roc_auc_score(yt, ys)),
                       brier=_safe_brier(yt, ys, k))
            row.update(point_metrics(yt, yp, "frozen"))
            row.update(common_operating_point(yt, ys, args.recall_floor))
            # restrict bootstrap to common days for comparability
            if day_resamples:
                pd_map = make_per_day(df[df["day"].isin(common_days)])
                ba, br = bootstrap_by_day(pd_map, day_resamples)
                boot_auprc[k], boot_auroc[k] = ba, br
                if len(ba):
                    row["auprc_ci_lo"], row["auprc_ci_hi"] = float(np.percentile(ba, 2.5)), float(np.percentile(ba, 97.5))
                    row["auroc_ci_lo"], row["auroc_ci_hi"] = float(np.percentile(br, 2.5)), float(np.percentile(br, 97.5))
            rows.append(row)

        # no_gnn ceiling row (from the contrast's proposed cascade)
        prop = next((k for k in members if loaded[k]["entry"].get("is_proposed")), None)
        if prop is not None:
            ceil_csv = discover_sim_csv(loaded[prop]["entry"], args.experiments_root, "no_gnn")
            if ceil_csv:
                try:
                    cdf = load_scores(ceil_csv, window=window)
                    yt, ys, yp = cdf["y_true"].to_numpy(), cdf["y_score"].to_numpy(), cdf["y_pred"].to_numpy()
                    crow = dict(contrast=c, key=f"{c}_ceiling", status="REFERENCE",
                                label="(reference) perfect-forecast ceiling [no_gnn, real traffic]",
                                version="-", n=len(cdf), n_pos=int(yt.sum()),
                                prevalence=float(yt.mean()),
                                auprc=float(average_precision_score(yt, ys)),
                                auroc=float(roc_auc_score(yt, ys)),
                                brier=float(brier_score_loss(yt, ys)))
                    crow.update(point_metrics(yt, yp, "frozen"))
                    crow.update(common_operating_point(yt, ys, args.recall_floor))
                    rows.append(crow)
                except Exception as exc:  # noqa: BLE001
                    print(f"[WARN] ceiling for {c}: {exc}")

    # pending/error rows (so the table shows what's still missing)
    for k, v in loaded.items():
        if v["status"] != "OK":
            e = v["entry"]
            rows.append(dict(contrast=e["contrast"], key=k, label=e["label"],
                             version=e["version"], status=v["status"]))

    # ---- significance vs proposed (paired day-block bootstrap) ------------
    for r in rows:
        e = next((x for x in REGISTRY if x["key"] == r.get("key")), None)
        if not e or e.get("is_proposed") or "proposed_key" not in e:
            continue
        pk = e["proposed_key"]
        if r.get("key") in boot_auprc and pk in boot_auprc:
            d = boot_auprc[pk] - boot_auprc[r["key"]]
            r["d_auprc_vs_proposed"] = float(np.mean(d))
            r["p_auprc_proposed_better"] = float(np.mean(d <= 0))
            d2 = boot_auroc[pk] - boot_auroc[r["key"]]
            r["p_auroc_proposed_better"] = float(np.mean(d2 <= 0))

    # ---- write outputs ----------------------------------------------------
    full = pd.DataFrame(rows)
    order = {c: i for i, c in enumerate(contrasts)}
    full["_o"] = full["contrast"].map(order).fillna(99)
    full = full.sort_values(["_o", "key"]).drop(columns="_o")
    csv_path = os.path.join(args.output_dir, "benchmark_accident_3day_full.csv")
    full.to_csv(csv_path, index=False)

    md_path = os.path.join(args.output_dir, "benchmark_accident_3day_summary.md")
    write_summary_md(full, md_path, args)

    print(f"\n[DONE] {csv_path}\n       {md_path}")


def _sig(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return ""
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def write_summary_md(full, path, args):
    titles = {
        "C1": "C1 — Forecaster swap (classifier fixed = v58 GBDT cascade)",
        "C2": "C2 — Classifier swap (forecaster fixed = GeoLSTM stage-1, v62)",
        "C3": "C3 — End-to-end (GeoLSTM + MLP-focal cascade vs end-to-end baselines)",
    }
    lines = [
        "# Accident-prediction benchmark — 3-day frozen-rolling regime",
        "",
        f"_Generated {datetime.now():%Y-%m-%d %H:%M}. "
        f"Eval window = {args.eval_start} → {args.eval_end} (all models clipped to "
        f"this single month so days scored are identical; the e2e sims natively "
        f"roll Jun–Oct and are clipped here). "
        f"Common operating point = max precision s.t. recall ≥ {args.recall_floor:.2f}. "
        f"Bootstrap = {args.n_boot} day-blocks. "
        f"Significance vs proposed: * p<0.05, ** p<0.01 (one-sided, ΔAUPRC)._",
        "",
        "> ℹ️ **Anchors:** C1/C3 use the GNN-LSTM stage-1 run `2628911`; C2 is "
        "anchored on the best Table-A stage-1 (GeoLSTM `2628913`), so its "
        "reference row is the GeoLSTM→GBDT cascade. Labels "
        "and the June-2025 window are identical everywhere.",
        "",
    ]
    head = ("| Model | Ver | AUPRC | AUROC | Prec | Recall | F1 | F2 | FAR | Status |\n"
            "|---|---|---|---|---|---|---|---|---|---|")
    for c in ["C1", "C2", "C3"]:
        sub = full[full["contrast"] == c]
        if sub.empty:
            continue
        lines += [f"## {titles.get(c, c)}", ""]
        # Row-population consistency: AUPRC is only comparable across models
        # evaluated on the SAME rows/labels. Flag mismatches (notably C3 e2e,
        # which scores on a denser grid than the cascade rows).
        evald = sub[sub["status"].isin(["OK", "REFERENCE"])]
        ns = evald["n"].dropna().to_numpy()
        # Relative spread: after clipping to the common month the cascade/classifier
        # rows match exactly; the e2e is slightly smaller (4h warm-up) — only flag a
        # genuinely large mismatch.
        if len(ns) > 1 and (ns.max() - ns.min()) / ns.max() > 0.05:
            pairs = ", ".join(f"{r['label'].split(' ')[0]}={int(r['n']):,}"
                              for _, r in evald.iterrows() if pd.notna(r["n"]))
            lines += [
                f"> ⚠️ **Row-population mismatch (>5%) — AUPRC/AUROC not directly comparable.** "
                f"Rows scored differ across models ({pairs}).",
                ""]
        if c == "C3":
            lines += [
                "> ℹ️ **C3 note:** end-to-end (BL-E0/E1) use the same labels and "
                "June window as the cascade. "
                "(e2e still drops a 4h warm-up at the window start → ~9.6k fewer rows.)",
                ""]
        lines.append(head)
        for _, r in sub.iterrows():
            if r.get("status") not in ("OK", "REFERENCE"):
                lines.append(f"| {r['label']} | {r.get('version','')} | — | — | — | — | — | — | — | {r.get('status','')} |")
                continue
            mark = _sig(r.get("p_auprc_proposed_better"))
            ci = ""
            if "auprc_ci_lo" in r and pd.notna(r.get("auprc_ci_lo")):
                ci = f" [{r['auprc_ci_lo']:.3f}–{r['auprc_ci_hi']:.3f}]"
            lab = f"**{r['label']}**" if r['key'].endswith("proposed") else r["label"]
            def f(x):
                return f"{x:.3f}" if pd.notna(x) else "—"
            lines.append(
                f"| {lab} | {r.get('version','')} | {f(r['auprc'])}{ci}{mark} | "
                f"{f(r['auroc'])} | {f(r.get('frozen_precision'))} | {f(r.get('frozen_recall'))} | "
                f"{f(r.get('frozen_f1'))} | {f(r.get('frozen_f2'))} | {f(r.get('frozen_far'))} | "
                f"{r.get('status','')} |")
        lines.append("")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


if __name__ == "__main__":
    sys.exit(main())
