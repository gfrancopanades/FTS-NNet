#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Section 6 — SHAP attribution for the benchmark's best crash classifier.

Complements the permutation importance of ``safety_insights.py``. Permutation
importance answers *how much does the model rely on this feature overall*; SHAP
additionally gives the **sign and per-instance contribution**, so a flagged
interval can be explained in terms of the conditions that raised its risk. That
is what an operator needs from a dispatch alert, and it is what lets Section 6
state which features drive machine-learned crash classification rather than
merely which ones the model happens to use.

Design decisions, and why:

* **PermutationExplainer, not KernelSHAP.** The classifier is a torch MLP behind
  a StandardScaler, wrapped in a project-specific class; there is no clean graph
  to hand DeepSHAP, and KernelSHAP on 52 features needs ~2^52 coalition
  sampling to converge. The permutation explainer is model-agnostic, treats
  ``predict_proba`` as a black box, and is exact in expectation under feature
  ordering randomisation.
* **Stratified explain set.** Crashes are 0.19 % of rows, so a uniform sample
  would contain almost no positives and the resulting summary would describe
  the negative class only. We explain ALL positives in the evaluation month plus
  a random sample of negatives, then report attributions separately so the
  positive-class signal is never averaged away.
* **Background set = the negative-class median regime.** SHAP values are defined
  against a reference; using a sample of crash-free intervals makes each value
  read as "how much did this feature push risk above a typical safe interval",
  which is the operationally meaningful contrast.

Outputs (figures dir):
    shap_beeswarm.{png,pdf}   direction + magnitude, top features
    shap_bar.{png,pdf}        mean |SHAP|, ranked
    shap_values.csv           per-feature mean |SHAP| and signed mean

Run::

    AP7_SHAP_XGB_DIR=<experiments>/<gnn_dir>/xgboost_<job> \\
    python -m src.evaluation.shap_analysis
"""

from __future__ import annotations
from src.paths import (  # portable paths -- see src/paths.py
    PROJECT_ROOT_STR as _AP7_ROOT,
    EXPERIMENTS_ROOT_STR as _AP7_EXPERIMENTS,
    DATA_DIR_STR as _AP7_DATA,
    TABLES_DIR as _AP7_TABLES,
    FIGURES_DIR as _AP7_FIGS,
)

import glob
import json
import os
import sys

import joblib
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

# Best configuration of the benchmark by AUPRC (two-stage MLP, 0.475).
FIG_DIR = f"{_AP7_ROOT}/visualizations"
RES_DIR = f"{_AP7_ROOT}/experiments/benchmark_results"

def _proposed_dirs():
    """Resolve the proposed arm's (stage-1, stage-2) directories from the registry.

    A hardcoded default silently points at whichever run was current when the
    file was written, which is how the permutation and SHAP columns of Table 7
    ended up describing two different classifiers.
    """
    import json
    root = _AP7_ROOT
    exp = _AP7_EXPERIMENTS
    reg = json.load(open(os.path.join(root, "experiments", "registry_final.json")))
    e = next(x for x in reg if x["key"] == "C2_C5_Focal")
    import glob as _g
    hits = _g.glob(os.path.join(exp, f"*_{e['gnn']}"))
    if not hits:                    # runs not on disk: main() explains
        return None, None
    return os.path.join(hits[0], f"xgboost_{e['xgb']}"), hits[0]


XGB_DIR = os.environ.get("AP7_SHAP_XGB_DIR") or _proposed_dirs()[0]
GNN_DIR = os.environ.get("AP7_SHAP_GNN_DIR") or _proposed_dirs()[1]
N_NEG = int(os.environ.get("AP7_SHAP_N_NEG", "4000"))
N_BACKGROUND = int(os.environ.get("AP7_SHAP_N_BACKGROUND", "200"))
TOP_K = int(os.environ.get("AP7_SHAP_TOP_K", "15"))
SEED = 12345

EVAL_START, EVAL_END = "2025-06-01", "2025-07-01"


def _load_classifier():
    manifests = sorted(glob.glob(
        os.path.join(XGB_DIR, "benchmark_classifier_manifest_*.json")))
    if not manifests:
        sys.exit(f"[FAIL] no classifier manifest in {XGB_DIR}")
    manifest = json.load(open(manifests[-1]))
    model_time = manifest["model_time"]
    path = os.path.join(XGB_DIR, f"benchmark-classifier_version={model_time}.joblib")
    clf = joblib.load(path)
    print(f"[model] token={manifest['classifier_token']} "
          f"job={os.path.basename(XGB_DIR)} test PR-AUC="
          f"{manifest['test_metrics_stage1']['pr_auc']:.4f}")
    return clf, manifest


def _build_matrix(features):
    """Rebuild the June feature matrix through the same stage-0 path the
    simulation uses, so explained rows are exactly the scored rows."""
    from src.training.ablation_study_v52_simulation_only import (
        prepare_simulation_dataframe_v52)
    df = prepare_simulation_dataframe_v52(GNN_DIR, {})
    df["dat"] = pd.to_datetime(df["dat"])
    df = df[(df.dat >= EVAL_START) & (df.dat < EVAL_END)]
    # The classifier was trained with V52's pk x hour interaction features, but
    # prepare_simulation_dataframe_v52(..., {}) is called with an empty
    # experiments dict and therefore does not replay them. Reindexing alone
    # would silently substitute all-zero columns for features the model relies
    # on, distorting every attribution. Replay them explicitly, exactly as the
    # training and simulation paths do.
    from src.training.ablation_study_v52_xgboost_only import add_pk_hour_interactions
    df, _added = add_pk_hour_interactions(df)
    X = df[features].apply(pd.to_numeric, errors="coerce").fillna(0)
    y = df["ACCIDENT"].fillna(0).astype(int).to_numpy()
    return X.reset_index(drop=True), y


def main() -> int:
    if not XGB_DIR or not GNN_DIR:
        sys.exit("[FAIL] the registry's proposed arm (C2_C5_Focal) has no run under "
                 "AP7_EXPERIMENTS_DIR; set AP7_SHAP_XGB_DIR and AP7_SHAP_GNN_DIR to its directories")
    import shap

    rng = np.random.default_rng(SEED)
    clf, manifest = _load_classifier()
    features = list(manifest["stage1_features"])

    X, y = _build_matrix(features)
    print(f"[data] June matrix {X.shape}, positives={int(y.sum()):,}")

    pos_idx = np.flatnonzero(y == 1)
    neg_idx = np.flatnonzero(y == 0)
    neg_sample = rng.choice(neg_idx, size=min(N_NEG, neg_idx.size), replace=False)
    explain_idx = np.concatenate([pos_idx, neg_sample])
    background = X.iloc[rng.choice(neg_idx, size=N_BACKGROUND, replace=False)]
    X_explain = X.iloc[explain_idx]
    is_pos = np.isin(explain_idx, pos_idx)
    print(f"[shap] explaining {len(X_explain):,} rows "
          f"({is_pos.sum():,} positive) against {N_BACKGROUND} background rows")

    def f(arr):
        return clf.predict_proba(pd.DataFrame(arr, columns=features))

    explainer = shap.PermutationExplainer(f, background.to_numpy())
    sv = explainer(X_explain.to_numpy())
    vals = sv.values if hasattr(sv, "values") else np.asarray(sv)
    if vals.ndim == 3:                      # (n, features, outputs)
        vals = vals[:, :, -1]

    mean_abs = np.abs(vals).mean(axis=0)
    mean_abs_pos = np.abs(vals[is_pos]).mean(axis=0)
    signed_pos = vals[is_pos].mean(axis=0)
    out = (pd.DataFrame({
        "feature": features,
        "mean_abs_shap": mean_abs,
        "mean_abs_shap_positives": mean_abs_pos,
        "signed_mean_shap_positives": signed_pos,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True))

    os.makedirs(RES_DIR, exist_ok=True)
    csv_path = os.path.join(RES_DIR, "shap_values.csv")
    out.to_csv(csv_path, index=False)
    print(f"\n[top {TOP_K} by mean |SHAP|]")
    print(out.head(TOP_K).to_string(index=False, float_format=lambda v: f"{v:.5f}"))

    os.makedirs(FIG_DIR, exist_ok=True)
    order = out.head(TOP_K)["feature"].tolist()
    cols = [features.index(c) for c in order]

    # --- ranked bar: overall vs positive-class reliance --------------------
    fig, ax = plt.subplots(figsize=(8.4, 0.42 * TOP_K + 1.6))
    ypos = np.arange(len(order))[::-1]
    ax.barh(ypos + 0.18, out.head(TOP_K)["mean_abs_shap"], height=0.36,
            color="#2a78d6", label="all explained rows")
    ax.barh(ypos - 0.18, out.head(TOP_K)["mean_abs_shap_positives"], height=0.36,
            color="#eb6834", label="crash intervals only")
    ax.set_yticks(ypos); ax.set_yticklabels(order, fontsize=9)
    ax.set_xlabel("mean |SHAP| (contribution to predicted crash probability)")
    ax.set_title("Feature attribution, proposed classifier (BL-C5, MLP + focal loss)",
                 loc="left", fontweight="semibold")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, axis="x", alpha=.3); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG_DIR, f"shap_bar.{ext}"), dpi=170)
    plt.close(fig)

    # --- beeswarm: direction of each association --------------------------
    try:
        plt.figure()
        shap.summary_plot(vals[:, cols], X_explain.iloc[:, cols],
                          feature_names=order, show=False, max_display=TOP_K)
        fig = plt.gcf(); fig.set_size_inches(8.4, 0.42 * TOP_K + 1.6)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(FIG_DIR, f"shap_beeswarm.{ext}"), dpi=170)
        plt.close(fig)
    except Exception as exc:                       # plotting must not lose the CSV
        print(f"[warn] beeswarm failed: {exc}")

    print(f"\n[DONE] {csv_path}")
    print(f"[DONE] {FIG_DIR}/shap_bar.{{png,pdf}}, shap_beeswarm.{{png,pdf}}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
