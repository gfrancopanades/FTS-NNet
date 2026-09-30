"""Section 6 — what the model learned (associational, not causal).

Reloads the trained Stage-2 classifier and rebuilds the June evaluation feature
matrix through the *same* stage-0 path the simulation uses, then reports:

  1. Permutation importance (model-agnostic; the only importance measure valid
     for both the focal-loss MLP and the tree models) on a stratified subsample,
     scored by the drop in AUPRC.
  2. One-dimensional partial dependence for the top-k features, showing the
     direction and shape of each association.

Both are ASSOCIATIONAL. Nothing here identifies causal effects: features are
correlated with each other and with unobserved conditions, and the model was
trained to predict, not to estimate effects.

Outputs (visualizations/):
    safety_importance.{png,pdf}, safety_pdp.{png,pdf}, safety_insights.md

Run (needs the full stage-0 data build, ~90 G):
    sbatch --mem=120G --cpus-per-task=8 -t 6:00:00 \
      -J safety_ins -o log_output/output_safety_ins_%j.log \
      -e log_error/error_safety_ins_%j.log \
      --wrap="cd <root> && export PYTHONPATH=<root> INCLUDE_WEATHER_1D=False && \
              python -u src/evaluation/safety_insights.py"
"""
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
from sklearn.metrics import average_precision_score

PROD_ROOT = _AP7_ROOT
if PROD_ROOT not in sys.path:
    sys.path.insert(0, PROD_ROOT)
os.environ.setdefault("INCLUDE_WEATHER_1D", "False")

OUT = f"{_AP7_ROOT}/visualizations"
# Proposed Stage-2: MLP+Focal trained on the GeoLSTM stage-1 forecasts.
# Stage-2 classifier whose learned associations are reported in Section 6.
# Overridable so the analysis can be pointed at any benchmarked run:
#   AP7_SAFETY_XGB_DIR=<experiments>/<gnn_dir>/xgboost_<job>
# Default is the leak-fixed proposed configuration (MLP+focal, job 2670046).
# NOTE: job 2663221 -- the previous default -- predates the 2026-08-12 fixes
# (June-label leakage through pk_crash_rate, duplicate segment-intervals,
# location-major FE baselines). Importances computed there describe a model
# whose top feature carried the contamination, and are not interpretable.
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


XGB_DIR = os.environ.get("AP7_SAFETY_XGB_DIR") or _proposed_dirs()[0]
GNN_DIR = os.environ.get("AP7_SAFETY_GNN_DIR") or _proposed_dirs()[1]
EVAL_START, EVAL_END = "2025-06-01", "2025-07-01"
N_SUB, N_REPEATS, TOP_K = 300_000, 3, 15
BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1a1a19", "#8a8878", "#e9e8e0"


def _style(ax):
    ax.grid(True, color=GRID, lw=0.7); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=9)


def build_eval_matrix():
    """June feature matrix + labels, via the simulation's own stage-0 path."""
    from src.training.ablation_study_v52_simulation_only import prepare_simulation_dataframe_v52

    # Labels exactly as the simulation pipeline produces them.
    df = prepare_simulation_dataframe_v52(GNN_DIR, {})
    df["dat"] = pd.to_datetime(df["dat"])
    df = df[(df.dat >= EVAL_START) & (df.dat < EVAL_END)].reset_index(drop=True)
    # The classifier was trained with V52's pk x hour interaction features, but
    # prepare_simulation_dataframe_v52(..., {}) is called with an empty
    # experiments dict and therefore does not replay them. Reindexing alone
    # would silently substitute all-zero columns for features the model relies
    # on, distorting every attribution. Replay them explicitly, exactly as the
    # training and simulation paths do.
    from src.training.ablation_study_v52_xgboost_only import add_pk_hour_interactions
    df, _added = add_pk_hour_interactions(df)
    print(f"[labels] June positives as produced by the simulation pipeline: "
          f"{int(df['ACCIDENT'].sum()):,}")
    return df


def main():
    if not XGB_DIR or not GNN_DIR:
        sys.exit("[FAIL] the registry's proposed arm (C2_C5_Focal) has no run under "
                 "AP7_EXPERIMENTS_DIR; set AP7_SAFETY_XGB_DIR and AP7_SAFETY_GNN_DIR to its directories")
    os.makedirs(OUT, exist_ok=True)
    manifest = sorted(glob.glob(os.path.join(XGB_DIR, "benchmark_classifier_manifest_*.json")))[-1]
    meta = json.load(open(manifest))
    feats = meta["stage1_features"]
    model_time = os.path.basename(manifest).replace(
        "benchmark_classifier_manifest_", "").replace(".json", "")
    clf = joblib.load(os.path.join(XGB_DIR, f"benchmark-classifier_version={model_time}.joblib"))
    print(f"[cfg] classifier={meta.get('classifier_token')} features={len(feats)}")

    df = build_eval_matrix()
    y = df["ACCIDENT"].astype(int).to_numpy()
    X = df.reindex(columns=feats).apply(pd.to_numeric, errors="coerce").fillna(0)
    print(f"[data] June matrix {X.shape}, positives={int(y.sum()):,}")

    # stratified subsample: all positives + random negatives (keeps AUPRC stable)
    rng = np.random.RandomState(0)
    pos = np.where(y == 1)[0]
    neg = rng.choice(np.where(y == 0)[0], size=min(N_SUB - len(pos), (y == 0).sum()),
                     replace=False)
    idx = np.sort(np.concatenate([pos, neg]))
    Xs, ys = X.iloc[idx].reset_index(drop=True), y[idx]
    base = average_precision_score(ys, clf.predict_proba(Xs))
    print(f"[base] subsample AUPRC={base:.4f} (n={len(ys):,}, pos={int(ys.sum()):,})")

    # ---- permutation importance -------------------------------------------
    rows = []
    for f in feats:
        drops = []
        for r in range(N_REPEATS):
            Xp = Xs.copy()
            Xp[f] = Xp[f].sample(frac=1.0, random_state=r).to_numpy()
            drops.append(base - average_precision_score(ys, clf.predict_proba(Xp)))
        rows.append(dict(feature=f, auprc_drop=float(np.mean(drops)), sd=float(np.std(drops))))
        print(f"  perm {f:32s} dAUPRC={np.mean(drops):+.4f}")
    imp = pd.DataFrame(rows).sort_values("auprc_drop", ascending=False).reset_index(drop=True)
    imp.to_csv(os.path.join(OUT, "safety_importance.csv"), index=False)

    top = imp.head(TOP_K).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.8, 0.34 * len(top) + 1.4))
    ax.barh(top.feature, top["auprc_drop"], xerr=top.sd, color=BLUE, height=0.62,
            error_kw=dict(ecolor=MUTED, lw=1))
    ax.set_xlabel("drop in AUPRC when permuted", color=INK)
    ax.set_title("Permutation importance (associational)", color=INK, loc="left", fontsize=12)
    _style(ax); fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}/safety_importance.{ext}", dpi=250, facecolor="white")
    plt.close(fig)

    # ---- 1-D partial dependence for the top features ----------------------
    top_feats = imp.head(6).feature.tolist()
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.4))
    for ax, f in zip(axes.ravel(), top_feats):
        qs = np.linspace(0.02, 0.98, 18)
        grid = np.unique(np.quantile(Xs[f].to_numpy(), qs))
        pd_vals = []
        Xtmp = Xs.sample(n=min(40_000, len(Xs)), random_state=1).reset_index(drop=True)
        for g in grid:
            Xg = Xtmp.copy(); Xg[f] = g
            pd_vals.append(float(clf.predict_proba(Xg).mean()))
        ax.plot(grid, pd_vals, color=ORANGE, lw=2.2)
        ax.set_title(f, color=INK, fontsize=10, loc="left")
        ax.set_ylabel("mean predicted risk", color=INK, fontsize=9)
        _style(ax)
    fig.suptitle("Partial dependence of the six most important features (associational)",
                 color=INK, x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}/safety_pdp.{ext}", dpi=250, facecolor="white")
    plt.close(fig)

    with open(os.path.join(OUT, "safety_insights.md"), "w") as fh:
        fh.write("# Safety insights (associational)\n\n"
                 f"Model: {meta.get('classifier_token')} on the GeoLSTM stage-1 forecast; "
                 f"June 2025; subsample n={len(ys):,} (all {int(ys.sum()):,} positives + "
                 f"random negatives); baseline AUPRC {base:.3f}.\n\n"
                 "Permutation importance = mean AUPRC drop over "
                 f"{N_REPEATS} shuffles.\n\n"
                 "| Feature | dAUPRC | sd |\n|---|---|---|\n")
        for _, r in imp.head(TOP_K).iterrows():
            fh.write(f"| {r.feature} | {r['auprc_drop']:.4f} | {r.sd:.4f} |\n")
        fh.write("\n**These associations are not causal effects.** Features are mutually "
                 "correlated and confounded with unobserved conditions; the model was fitted "
                 "for prediction, not for effect estimation.\n")
    print(f"[DONE] safety insights -> {OUT}/safety_{{importance,pdp}}.png + safety_insights.md")


if __name__ == "__main__":
    main()
