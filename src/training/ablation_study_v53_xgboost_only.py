#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V53 XGBoost-only: V52 + asymmetric Stage-1 / Stage-2 feature pruning
=====================================================================
Builds on V52 (`ablation_study_v52_xgboost_only.py`) by recognising that
Stage 1 and Stage 2 have orthogonal jobs and therefore deserve different
feature sets:

  * Stage 1 ("find candidates"):   high gain on features that separate
    positives from negatives.  Variance / volatility features
    (``speed_cv_2``, ``speed_std_4``, ``flow_regime``, …) dominate.
    These should stay.
  * Stage 2 ("filter false alarms"):  needs features that separate the
    two flavours of positives — TPs from FPs.  Variance features fire
    on BOTH and so waste capacity; the ``pk_crash_rate*`` family,
    ``pk_x_hor*``, ``hor_cos``, ``vol_cv_2``, ``vol_std_2`` are the
    actual discriminators (per the Stage-Z feature-space analysis on
    V52 ``xgboost_2569859`` — see notebooks/v51_v17_TPFPFN_analysis).

V53 therefore introduces two new module attributes:

  * ``STAGE1_DROP_FEATURES``: list of features stripped from Stage 1.
    Default = the gain≈0 / KS≈0 "dead weight" plus the two V52 features
    that under-performed (``is_pk_hot_zone``, ``pk_hot_x_evening_rush``).
  * ``STAGE2_FEATURES``: explicit Stage-2 whitelist (``s1_score`` is
    always appended automatically).  ``None`` falls back to V52
    behaviour (Stage-2 inherits Stage-1's set).

Everything else — pk × hour interactions (c), the
threshold sweep (a), the memorize-vs-holdout split (f) — is delegated
to V52.

Usage
-----
Drop-in replacement for V52's runner.  The bash runner exports the same
env vars (``PREVIOUS_EXPERIMENT_DIR``, ``XGB_RUN_ID``):

    python -m src.training.ablation_study_v53_xgboost_only \\
        --gnn-experiment-dir $AP7_EXPERIMENTS_DIR/v17_gnn_no-w1d_5min_<jobid>

CLI flags (all additive on top of V52's):
    --stage1-drop-feats f1,f2,...        Override Stage-1 drop list
    --stage1-no-prune                    Disable Stage-1 pruning entirely
    --stage2-feats f1,f2,...             Override Stage-2 whitelist
    --stage2-inherit-stage1              Use Stage-1's feature set for Stage-2
                                         (matches V52 behaviour)

Author: Gerard Franco
Date:   May 2026
"""

from __future__ import annotations
from src.paths import (  # portable paths -- see src/paths.py
    PROJECT_ROOT_STR as _AP7_ROOT,
    EXPERIMENTS_ROOT_STR as _AP7_EXPERIMENTS,
    DATA_DIR_STR as _AP7_DATA,
    TABLES_DIR as _AP7_TABLES,
    FIGURES_DIR as _AP7_FIGS,
)

import gc
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    fbeta_score,
    precision_recall_curve,
)
from sklearn.model_selection import StratifiedKFold

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import src.training.ablation_study_v51_xgboost_only as v51
import src.training.ablation_study_v52_xgboost_only as v52
from src.training.ablation_study_v51_xgboost_only import (
    create_balanced_subset,
    format_duration,
    stage0_load_data_with_gnn,
    stage1_feature_prep,
    stage2_train_ensemble,
    stage5_feature_importance_shap,
    stage6_feature_space_overlap,
    stage7b_shap_stage2,
    _S1_FLOAT_RANGES,
    _S1_INT_RANGES,
    _S2_FLOAT_RANGES,
    _S2_INT_RANGES,
    _wider_optuna_ranges,
    ENSEMBLE_CONFIG,
    N_THREADS,
    RANDOM_STATE,
)
from src.training.ablation_study_v52_xgboost_only import (
    add_pk_hour_interactions,
    sweep_thresholds_v52,
    EVENING_RUSH_HOURS,
    HOT_ZONE_PKS,
)


# =============================================================================
# V53 CONFIGURATION
# =============================================================================

PIPELINE_VERSION = "v53_xgboost_only"

# (V53 new) — features stripped from Stage 1 because they show gain≈0 AND
# KS(FP vs TN)≈0 in the V52 / no-w1d Stage-Z analysis.  Removing them is
# pure hygiene: Stage 1 doesn't use them, Stage 2 cannot use them either.
# The last two are V52 interaction features that under-performed (KS≈0.03).
STAGE1_DEAD_WEIGHT_DEFAULT: tuple[str, ...] = (
    "mes_sin", "mes_cos", "is_heavy_rain", "temp_near_freezing",
    "precip_change_1d_vs_3d", "cold_curve_risk",
    "wet_speed", "is_raining", "wet_descent_risk", "wet_curve_risk",
    "is_pk_hot_zone", "pk_hot_x_evening_rush",
)
STAGE1_DROP_FEATURES: list[str] = list(STAGE1_DEAD_WEIGHT_DEFAULT)

# (V53 new) — explicit Stage-2 whitelist.  Ranked by KS(TP vs FP) on V52
# `xgboost_2569859` Stage-Z output; threshold KS ≥ 0.17 OR |Cohen's d| ≥ 0.5
# (whichever fires first), excluding ``pk_crash_rate`` (near-duplicate of
# ``pk_crash_rate_log``).  ``s1_score`` is added automatically downstream.
STAGE2_CURATED_DEFAULT: tuple[str, ...] = (
    "pk_crash_rate_log",  # KS=0.430, d=+1.33
    "pk_crash_rate_mob",  # KS=0.415, d=+1.31
    "pk_x_hor",           # KS=0.292, d=+0.67
    "pk_x_hor_cos",       # KS=0.318, d=+0.64
    "hor_cos",            # KS=0.304, d=+0.62
    "vol_cv_2",           # KS=0.230, d=+0.60
    "vol_std_2",          # KS=0.220, d=+0.54
    "delta_vol_1",        # KS=0.248, d=-0.68
    "accel_speed",        # KS=0.191, d=-0.56
    "pk",                 # KS=0.209, d=+0.46
    "diaSem_cos",         # KS=0.177, d=+0.32
)
STAGE2_FEATURES: list[str] | None = list(STAGE2_CURATED_DEFAULT)

# Inherit V52 knobs so external callers can mutate the same names they used
# for V52.  ``_propagate_to_v52`` mirrors any writes back onto v52 + v51.
ADD_PK_HOUR_INTERACTIONS: bool = v52.ADD_PK_HOUR_INTERACTIONS
MIN_RECALL_CONSTRAINT: float = v52.MIN_RECALL_CONSTRAINT
MAX_FAR_TARGET: float = v52.MAX_FAR_TARGET
THRESHOLD_GRID_S1 = v52.THRESHOLD_GRID_S1
THRESHOLD_GRID_S2 = v52.THRESHOLD_GRID_S2
MEMORIZE_MODE: bool = bool(v51.MEMORIZE_MODE)


# =============================================================================
# FEATURE-SET RESOLUTION
# =============================================================================

def _resolve_stage1_features(available_features: list[str]) -> tuple[list[str], list[str]]:
    """Prune ``STAGE1_DROP_FEATURES`` from the V51-resolved available set.

    Returns (kept_features, dropped_features).  Features named in the
    drop list that aren't present are silently ignored — keeps the V53
    defaults usable even when the upstream feature pool shifts (e.g.,
    different ablation flags).
    """
    drop_set = set(STAGE1_DROP_FEATURES or [])
    present = set(available_features)
    dropped = [f for f in STAGE1_DROP_FEATURES if f in present]
    kept = [f for f in available_features if f not in drop_set]
    print("=" * 80)
    print("V53 STAGE-1 FEATURE PRUNING")
    print("=" * 80)
    print(f"  Available before prune : {len(available_features)}")
    print(f"  Dropped ({len(dropped)})         : {dropped}")
    print(f"  Missing from pool       : "
          f"{sorted(drop_set - present) if (drop_set - present) else '[]'}")
    print(f"  Kept for Stage 1        : {len(kept)}")
    return kept, dropped


def _resolve_stage2_features(stage1_features: list[str]) -> list[str]:
    """Resolve the Stage-2 feature whitelist.

    Semantics:
      * ``STAGE2_FEATURES is None`` → Stage 2 inherits Stage 1's features
        (V52 behaviour).  ``s1_score`` is still appended at training time.
      * ``STAGE2_FEATURES`` set → keep only those that exist in
        ``stage1_features`` (Stage 2 can't depend on a feature Stage 1
        didn't see).  Print a warning for anything we drop.
    """
    print("=" * 80)
    print("V53 STAGE-2 FEATURE SELECTION")
    print("=" * 80)
    if STAGE2_FEATURES is None:
        print("  STAGE2_FEATURES = None  →  Stage 2 inherits Stage 1's set "
              f"({len(stage1_features)} features) + s1_score")
        return list(stage1_features)

    s1_set = set(stage1_features)
    kept = [f for f in STAGE2_FEATURES if f in s1_set]
    dropped = [f for f in STAGE2_FEATURES if f not in s1_set]
    if dropped:
        print(f"  WARNING: {len(dropped)} requested Stage-2 features not in "
              f"Stage-1 pool, dropping: {dropped}")
    print(f"  Stage 2 whitelist ({len(kept)} features) + s1_score:")
    for f in kept:
        print(f"    - {f}")
    return kept


def _slice_for_stage2(X: pd.DataFrame, stage2_features: list[str]) -> pd.DataFrame:
    """Slice X down to ``stage2_features`` + ``s1_score`` (if present)."""
    keep = [c for c in stage2_features if c in X.columns]
    if "s1_score" in X.columns and "s1_score" not in keep:
        keep = keep + ["s1_score"]
    return X[keep]


# =============================================================================
# STAGE 2 TRAINING — INLINED FOR V53 (memorize + non-memorize branches)
# =============================================================================

def _train_stage2_memorize_v53(*, X_train_xgb, X_test_xgb, y_train_acc, y_test_acc,
                               base_models, ensemble_preds, stage2_features,
                               s1_best_params, model_time, output_dir):
    """V52-style memorize Stage 2 with Stage-2 feature subset support.

    Mirrors v52._train_stage2_memorize (lines 411-466) but slices the
    Stage-2 training pool to ``stage2_features + ['s1_score']`` before
    fitting, and slices the Stage-2 test pool the same way before scoring.
    """
    print("=" * 80)
    print("V53 STAGE 7 (MEMORIZE): Stage 2 with curated feature subset")
    print("=" * 80)

    s2_params = {
        **s1_best_params,
        "tree_method":  "hist",
        "objective":    "binary:logistic",
        "eval_metric":  "aucpr",
        "random_state": RANDOM_STATE + 1,
        "nthread":      N_THREADS,
    }
    neg_pos_ratio = ENSEMBLE_CONFIG["neg_pos_ratio"]

    X_all = pd.concat([X_train_xgb, X_test_xgb])
    y_all = pd.concat([y_train_acc,  y_test_acc])

    s1_score_all = np.zeros(len(X_all))
    d_all = xgb.DMatrix(X_all)
    for m in base_models:
        best_iter = getattr(m, "best_iteration", 500)
        s1_score_all += m.predict(d_all, iteration_range=(0, best_iter))
    s1_score_all /= max(len(base_models), 1)
    X_all = X_all.copy()
    X_all["s1_score"] = s1_score_all

    X_s2_pool, y_s2_pool = create_balanced_subset(
        X_all, y_all,
        neg_pos_ratio=neg_pos_ratio,
        random_state=RANDOM_STATE + 1,
    )

    X_s2_tr = _slice_for_stage2(X_s2_pool, stage2_features)
    n_tp_s2 = int(y_s2_pool.sum())
    n_fp_s2 = int((y_s2_pool == 0).sum())
    print(f"  Stage 2 (MEMORIZE) pool: {len(X_s2_tr):,} "
          f"(TP={n_tp_s2:,}, FP={n_fp_s2:,})")
    print(f"  Stage 2 columns        : {list(X_s2_tr.columns)}")

    dtrain_s2 = xgb.DMatrix(X_s2_tr, label=y_s2_pool)
    stage2_model = xgb.train(s2_params, dtrain_s2,
                             num_boost_round=500, verbose_eval=False)
    stage2_model.best_iteration = 500
    stage2_path = os.path.join(
        output_dir, f"xgboost-stage2-fp-filter_version={model_time}.json")
    stage2_model.save_model(stage2_path)
    print(f"  Stage 2 (MEMORIZE) model saved: {stage2_path}")

    X_test_r     = X_test_xgb.reset_index(drop=True)
    test_s1_flag = ensemble_preds >= 0.5
    flagged_idx  = np.where(test_s1_flag)[0]
    s2_test_probs = np.zeros(0)
    if test_s1_flag.any():
        X_te = X_test_r[test_s1_flag].copy()
        X_te["s1_score"] = ensemble_preds[test_s1_flag]
        X_te_s2 = _slice_for_stage2(X_te, stage2_features)
        s2_test_probs = stage2_model.predict(xgb.DMatrix(X_te_s2))

    y_test_np = y_test_acc.reset_index(drop=True).values
    return stage2_model, X_s2_tr, y_s2_pool, flagged_idx, s2_test_probs, y_test_np


def _train_stage2_holdout_v53(*, X_train_xgb, X_test_xgb, y_train_acc, y_test_acc,
                              base_models, ensemble_preds, stage2_features,
                              s1_best_params, model_time, output_dir):
    """Non-memorize Stage 2 training with curated feature subset.

    Refactor of v51.stage7_two_stage (lines 1127-1308) keeping the OOF
    Stage-1 scoring and Optuna search but slicing X_s2_tr / X_s2_te to
    ``stage2_features + ['s1_score']`` before fitting.  Plots and the
    comparison table are deliberately omitted to keep this function
    focused; if you want them, drop the Stage-2 model into v51's
    stage7b_shap_stage2 afterwards.
    """
    print("=" * 80)
    print("V53 STAGE 7A: OOF Stage-1 scoring (full Stage-1 feature set)")
    print("=" * 80)

    STAGE1_THRESHOLD = 0.5
    STAGE2_THRESHOLD = 0.5

    X_tr = X_train_xgb.reset_index(drop=True)
    y_tr = y_train_acc.reset_index(drop=True)
    stage1_features = list(X_tr.columns)
    neg_pos_ratio   = ENSEMBLE_CONFIG["neg_pos_ratio"]

    oof_n_folds  = 5
    oof_n_models = 5
    mean_iter   = int(np.mean([getattr(m, "best_iteration", 100) for m in base_models]))
    oof_rounds  = max(50, mean_iter)
    oof_params = {**s1_best_params, "tree_method": "hist",
                  "objective": "binary:logistic", "eval_metric": "aucpr",
                  "nthread": N_THREADS, "seed": RANDOM_STATE}

    oof_kf = StratifiedKFold(n_splits=oof_n_folds, shuffle=True,
                             random_state=RANDOM_STATE)
    s1_tr_probs = np.zeros(len(X_tr))
    rng = np.random.RandomState(RANDOM_STATE)
    print(f"OOF: {oof_n_folds} folds x {oof_n_models} bag models, "
          f"{oof_rounds} rounds each")
    for fi, (tr_idx, va_idx) in enumerate(oof_kf.split(X_tr.values, y_tr.values), 1):
        X_fold = X_tr.iloc[tr_idx]
        y_fold = y_tr.iloc[tr_idx]
        pos_ix = np.where(y_fold.values == 1)[0]
        neg_ix = np.where(y_fold.values == 0)[0]
        fold_probs = np.zeros(len(va_idx))
        for _ in range(oof_n_models):
            samp_neg = rng.choice(neg_ix, size=len(pos_ix) * neg_pos_ratio,
                                  replace=False)
            bag_idx  = np.concatenate([pos_ix, samp_neg])
            rng.shuffle(bag_idx)
            Xb = X_fold.iloc[bag_idx][stage1_features]
            yb = y_fold.iloc[bag_idx]
            m = xgb.train(oof_params, xgb.DMatrix(Xb, label=yb),
                          num_boost_round=oof_rounds, verbose_eval=False)
            fold_probs += m.predict(xgb.DMatrix(X_tr.iloc[va_idx][stage1_features]))
        s1_tr_probs[va_idx] = fold_probs / oof_n_models
        flagged = int((s1_tr_probs[va_idx] >= STAGE1_THRESHOLD).sum())
        print(f"  Fold {fi}/{oof_n_folds}: {len(va_idx):,} rows, flagged={flagged:,}")

    s2_mask = s1_tr_probs >= STAGE1_THRESHOLD
    print(f"\nStage-2 training pool: {int(s2_mask.sum()):,} "
          f"(TP={int(((s2_mask) & (y_tr == 1)).sum()):,}, "
          f"FP={int(((s2_mask) & (y_tr == 0)).sum()):,})")

    X_s2_full_tr = X_tr[s2_mask].copy()
    X_s2_full_tr["s1_score"] = s1_tr_probs[s2_mask]
    y_s2_tr      = y_tr[s2_mask].copy()

    X_test_r      = X_test_xgb.reset_index(drop=True)
    y_test_r      = y_test_acc.reset_index(drop=True)
    test_s1_flag  = ensemble_preds >= STAGE1_THRESHOLD
    X_s2_full_te  = X_test_r[test_s1_flag].copy()
    X_s2_full_te["s1_score"] = ensemble_preds[test_s1_flag]
    y_s2_te       = y_test_r[test_s1_flag].copy()

    X_s2_tr = _slice_for_stage2(X_s2_full_tr, stage2_features)
    X_s2_te = _slice_for_stage2(X_s2_full_te, stage2_features)

    n_tp_s2 = int(y_s2_tr.sum())
    n_fp_s2 = int((y_s2_tr == 0).sum())
    s2_spw  = n_fp_s2 / max(n_tp_s2, 1)
    print(f"  Stage 2 columns        : {list(X_s2_tr.columns)}")
    print(f"  Stage 2 scale_pos_weight (natural ratio) = {s2_spw:.2f}")

    print("=" * 80)
    print("V53 STAGE 7B: Optuna search on curated Stage-2 feature subset")
    print("=" * 80)

    s2_trials   = 30
    s2_cv_folds = 3

    def _s2_objective(trial: optuna.trial.Trial) -> float:
        params = {
            "max_depth":        trial.suggest_int("max_depth", 2, 8),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "gamma":            trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha":        trial.suggest_float("reg_alpha", 0.0, 5.0),
            "reg_lambda":       trial.suggest_float("reg_lambda", 0.0, 5.0),
            "scale_pos_weight": s2_spw,
            "tree_method":      "hist",
            "objective":        "binary:logistic",
            "eval_metric":      "aucpr",
            "nthread":          N_THREADS,
            "seed":             RANDOM_STATE,
        }
        skf = StratifiedKFold(n_splits=s2_cv_folds, shuffle=True,
                              random_state=RANDOM_STATE)
        X_arr = X_s2_tr.values
        y_arr = y_s2_tr.values
        scores = []
        for tr_idx, va_idx in skf.split(X_arr, y_arr):
            dtr = xgb.DMatrix(X_arr[tr_idx], label=y_arr[tr_idx])
            dva = xgb.DMatrix(X_arr[va_idx], label=y_arr[va_idx])
            mf  = xgb.train(params, dtr, num_boost_round=300,
                            evals=[(dva, "eval")], early_stopping_rounds=20,
                            verbose_eval=False)
            preds = mf.predict(dva, iteration_range=(0, mf.best_iteration))
            scores.append(fbeta_score(y_arr[va_idx],
                                      (preds >= 0.5).astype(int),
                                      beta=2.0, zero_division=0))
        return float(np.mean(scores)) if scores else 0.0

    s2_study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    print(f"Stage 2 Optuna search ({s2_trials} trials, {s2_cv_folds}-fold CV, "
          f"objective=F2 @ thr=0.5, train={len(X_s2_tr):,}, spw={s2_spw:.2f})")
    s2_study.optimize(_s2_objective, n_trials=s2_trials, show_progress_bar=True)
    print(f"Stage 2 best F2: {s2_study.best_value:.4f}")
    for k, v in s2_study.best_params.items():
        print(f"  {k}: {v}")

    s2_best_params = {**s2_study.best_params,
                      "scale_pos_weight": s2_spw,
                      "tree_method":  "hist",
                      "objective":    "binary:logistic",
                      "eval_metric":  "aucpr",
                      "nthread":      N_THREADS,
                      "seed":         RANDOM_STATE}
    dtr_s2 = xgb.DMatrix(X_s2_tr, label=y_s2_tr)
    dte_s2 = xgb.DMatrix(X_s2_te, label=y_s2_te)
    stage2_model = xgb.train(s2_best_params, dtr_s2,
                             num_boost_round=500,
                             early_stopping_rounds=30,
                             evals=[(dtr_s2, "train"), (dte_s2, "test")],
                             verbose_eval=50)
    stage2_path = os.path.join(
        output_dir, f"xgboost-stage2-fp-filter_version={model_time}.json")
    stage2_model.save_model(stage2_path)
    print(f"\nStage 2 model saved: {stage2_path}")

    s2_test_probs = stage2_model.predict(dte_s2)
    y_test_np     = y_test_r.values
    flagged_idx   = np.where(test_s1_flag)[0]

    s2_full_prob = np.zeros(len(y_test_np))
    s2_full_bin  = np.zeros(len(y_test_np), dtype=int)
    s2_full_prob[flagged_idx] = s2_test_probs
    s2_full_bin[flagged_idx]  = (s2_test_probs >= STAGE2_THRESHOLD).astype(int)

    s1_binary = (ensemble_preds >= STAGE1_THRESHOLD).astype(int)
    _print_two_stage_test_comparison(y_test_np, ensemble_preds,
                                     s1_binary, s2_full_prob, s2_full_bin)

    return stage2_model, X_s2_tr, y_s2_tr, flagged_idx, s2_test_probs, y_test_np


def _print_two_stage_test_comparison(y_test_np, ensemble_preds, s1_binary,
                                     s2_full_prob, s2_full_bin) -> None:
    """Compact Stage-1 vs Two-Stage test-set comparison (no plots)."""
    def _m(cm):
        tn, fp, fn, tp = cm.ravel()
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        f1 = 2 * p * r / max(p + r, 1e-12)
        return int(tp), int(fp), int(fn), float(p), float(r), float(f1)
    cm_s1 = confusion_matrix(y_test_np, s1_binary)
    cm_s2 = confusion_matrix(y_test_np, s2_full_bin)
    t1, f1, n1, p1, r1, f1_1 = _m(cm_s1)
    t2, f2, n2, p2, r2, f1_2 = _m(cm_s2)
    a1 = average_precision_score(y_test_np, ensemble_preds)
    a2 = average_precision_score(y_test_np, s2_full_prob)
    print("\n" + "=" * 55)
    print("V53 TEST-SET COMPARISON — Stage 1 alone vs Two-Stage")
    print("=" * 55)
    for label, v1, v2 in [("TP", t1, t2), ("FP", f1, f2), ("FN", n1, n2),
                          ("Precision", p1, p2), ("Recall", r1, r2),
                          ("F1", f1_1, f1_2), ("PR-AUC", a1, a2)]:
        if isinstance(v1, float):
            print(f"{label:22s}  {v1:>10.4f}  {v2:>10.4f}")
        else:
            print(f"{label:22s}  {v1:>10,}  {v2:>10,}")


# =============================================================================
# V53 RUN PIPELINE
# =============================================================================

def run_pipeline_v53(gnn_experiment_dir: str) -> bool:
    """V53 main pipeline.

    Flow:
        Stage 0 (V51)             load CSV + GNN + V21 FE
        V52 (c) pk × hour feats   if ADD_PK_HOUR_INTERACTIONS
        Stage 1 prep (V51)        feature prep + train/test split
        V53 prune                 drop STAGE1_DROP_FEATURES from Stage-1 pool
        Stage 2 ensemble (V51)    train Stage-1 ensemble on pruned features
        Stage 5/6 (V51)           SHAP + feature-space overlap diagnostics
        V53 Stage 7               inline Stage-2 training with curated subset
        V52 (a) sweep             threshold sweep (runs in both modes)
        V53 manifest              persist both feature lists
    """
    _propagate_to_v52()

    t_start = time.time()
    print("=" * 80)
    print("V53 XGBoost-only — V52 + asymmetric Stage-1/Stage-2 feature pruning")
    print(f"Training:        {v51.TRAIN_START_DATE} -> {v51.TRAIN_END_DATE}")
    print(f"Output dir:      {v51.OUTPUT_DIR}")
    print(f"GNN experiment:  {gnn_experiment_dir}")
    print(f"MEMORIZE_MODE:   {bool(v51.MEMORIZE_MODE)}")
    print("V53 experiment flags:")
    print(f"  (V53) Stage-1 drop count    : {len(STAGE1_DROP_FEATURES)} "
          f"(first 6: {STAGE1_DROP_FEATURES[:6]}...)")
    print(f"  (V53) Stage-2 whitelist     : "
          f"{'inherit Stage-1' if STAGE2_FEATURES is None else f'{len(STAGE2_FEATURES)} features'}")
    print(f"  (a)   threshold sweep        : "
          f"min_recall={MIN_RECALL_CONSTRAINT}, "
          f"S1 grid {THRESHOLD_GRID_S1[0]:.3f}-{THRESHOLD_GRID_S1[-1]:.3f}, "
          f"S2 grid {THRESHOLD_GRID_S2[0]:.3f}-{THRESHOLD_GRID_S2[-1]:.3f}")
    print(f"  (c)   pk × hour interactions : {ADD_PK_HOUR_INTERACTIONS}")
    print(f"  (f)   memorize + holdout     : both supported")
    print("=" * 80)

    os.makedirs(v51.OUTPUT_DIR, exist_ok=True)
    os.makedirs(v51.VIZ_DIR, exist_ok=True)

    df_full, fe_only_cols = stage0_load_data_with_gnn(gnn_experiment_dir)

    v52_extra_features: list[str] = []
    if ADD_PK_HOUR_INTERACTIONS:
        df_full, v52_extra_features = add_pk_hour_interactions(df_full)
        fe_only_cols = list(fe_only_cols) + [
            f for f in v52_extra_features if f not in fe_only_cols
        ]

    time_res_min = v51._get_time_resolution_minutes()

    (df_train, X_train_xgb, X_test_xgb, y_train_acc, y_test_acc,
     available_features, model_time) = stage1_feature_prep(df_full, fe_only_cols)

    if ADD_PK_HOUR_INTERACTIONS:
        missing_v52 = [f for f in v52_extra_features
                       if f not in available_features and f in X_train_xgb.columns]
        if missing_v52:
            print(f"[V53] Adding {len(missing_v52)} V52 interaction feats to available_features")
            available_features = list(available_features) + missing_v52

    del df_train
    gc.collect()

    stage1_features, stage1_dropped = _resolve_stage1_features(available_features)
    if stage1_dropped:
        X_train_xgb = X_train_xgb[stage1_features]
        X_test_xgb  = X_test_xgb[stage1_features]

    stage2_features = _resolve_stage2_features(stage1_features)

    print("[MEM] df_train released. Proceeding to Stage-1 ensemble training.")

    with _wider_optuna_ranges(_S1_INT_RANGES, _S1_FLOAT_RANGES):
        (base_models, ensemble_preds, _y_pred_binary, roc_auc, pr_auc,
         s1_best_params) = stage2_train_ensemble(
            X_train_xgb, X_test_xgb, y_train_acc, y_test_acc,
            stage1_features, model_time)
    gc.collect()

    stage5_feature_importance_shap(base_models, X_test_xgb, stage1_features)
    stage6_feature_space_overlap(
        X_train_xgb, X_test_xgb, y_train_acc, y_test_acc,
        stage1_features, ensemble_preds)

    if v51.MEMORIZE_MODE:
        (stage2_model, X_s2_tr, y_s2_tr, flagged_idx,
         s2_test_probs, y_test_np) = _train_stage2_memorize_v53(
            X_train_xgb=X_train_xgb, X_test_xgb=X_test_xgb,
            y_train_acc=y_train_acc, y_test_acc=y_test_acc,
            base_models=base_models, ensemble_preds=ensemble_preds,
            stage2_features=stage2_features,
            s1_best_params=s1_best_params,
            model_time=model_time, output_dir=v51.OUTPUT_DIR)
    else:
        with _wider_optuna_ranges(_S2_INT_RANGES, _S2_FLOAT_RANGES):
            (stage2_model, X_s2_tr, y_s2_tr, flagged_idx,
             s2_test_probs, y_test_np) = _train_stage2_holdout_v53(
                X_train_xgb=X_train_xgb, X_test_xgb=X_test_xgb,
                y_train_acc=y_train_acc, y_test_acc=y_test_acc,
                base_models=base_models, ensemble_preds=ensemble_preds,
                stage2_features=stage2_features,
                s1_best_params=s1_best_params,
                model_time=model_time, output_dir=v51.OUTPUT_DIR)
        stage7b_shap_stage2(stage2_model, X_s2_tr, y_s2_tr)

    S1_THR, S2_THR, sweep_summary = sweep_thresholds_v52(
        ensemble_preds=ensemble_preds,
        y_test_np=y_test_np,
        flagged_idx=flagged_idx,
        s2_test_probs=s2_test_probs,
        s1_grid=THRESHOLD_GRID_S1,
        s2_grid=THRESHOLD_GRID_S2,
        min_recall=MIN_RECALL_CONSTRAINT,
        max_far=MAX_FAR_TARGET,
        optimise_for="f2",
    )

    write_v53_manifests(
        gnn_experiment_dir=gnn_experiment_dir,
        model_time=model_time,
        stage1_features=stage1_features,
        stage1_dropped=stage1_dropped,
        stage2_features=stage2_features,
        s1_thr=S1_THR, s2_thr=S2_THR,
        roc_auc=roc_auc, pr_auc=pr_auc,
        sweep_summary=sweep_summary,
        v52_extra_features=v52_extra_features,
    )

    print("\n" + "=" * 80)
    print(f"V53 XGBoost training complete in {format_duration(time.time() - t_start)}")
    print(f"All plots saved to:  {v51.VIZ_DIR}")
    print(f"All models saved to: {v51.OUTPUT_DIR}")
    print(f"Model time: {model_time}")
    print("Next step: ablation_study_v53_simulation_only.py "
          f"--xgboost-experiment-dir {v51.OUTPUT_DIR} "
          f"--gnn-experiment-dir {gnn_experiment_dir}")
    print("=" * 80)
    return True


def write_v53_manifests(*, gnn_experiment_dir: str, model_time: str,
                        stage1_features: list[str], stage1_dropped: list[str],
                        stage2_features: list[str],
                        s1_thr: float, s2_thr: float,
                        roc_auc: float, pr_auc: float,
                        sweep_summary: dict,
                        v52_extra_features: list[str]) -> None:
    """Persist `v53_xgboost_manifest_*.json` and `experiment_config_xgb_v53.json`.

    Schema is a superset of V52's, adding ``experiments.stage1_dropped_features``
    and ``experiments.stage2_features``.  ``available_features`` reflects the
    POST-prune Stage-1 set so the sim runner builds the correct DMatrix.
    """
    experiments = {
        "pk_hour_interactions":      bool(ADD_PK_HOUR_INTERACTIONS),
        "v52_extra_features":        list(v52_extra_features),
        "stage1_dropped_features":   list(stage1_dropped),
        "stage1_drop_request":       list(STAGE1_DROP_FEATURES),
        "stage2_features":           list(stage2_features),
        "stage2_inherits_stage1":    STAGE2_FEATURES is None,
        "stage2_in_memorize":        True,
        "threshold_sweep": {
            "min_recall_constraint": float(MIN_RECALL_CONSTRAINT),
            "max_far_target":        float(MAX_FAR_TARGET),
            "s1_grid":               [float(x) for x in THRESHOLD_GRID_S1],
            "s2_grid":               [float(x) for x in THRESHOLD_GRID_S2],
            "optimise_for":          "f2",
            "best": {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer))
                         else v)
                     for k, v in sweep_summary["best"].items()},
        },
        "hot_zone_pks":       list(HOT_ZONE_PKS),
        "evening_rush_hours": list(EVENING_RUSH_HOURS),
    }

    manifest = {
        "pipeline_version":      PIPELINE_VERSION,
        "model_time":            model_time,
        "output_dir":            v51.OUTPUT_DIR,
        "gnn_experiment_dir":    gnn_experiment_dir,
        "available_features":    stage1_features,
        "stage1_features":       stage1_features,
        "stage2_features":       stage2_features,
        "stage1_threshold":      float(s1_thr),
        "stage2_threshold":      float(s2_thr),
        "min_recall_constraint": float(MIN_RECALL_CONSTRAINT),
        "max_far_target":        float(MAX_FAR_TARGET),
        "test_metrics_stage1":   {"roc_auc": float(roc_auc), "pr_auc": float(pr_auc)},
        "train_dates":           {"start": v51.TRAIN_START_DATE, "end": v51.TRAIN_END_DATE},
        "sim_dates":             {"start": v51.SIM_START_DATE,   "end": v51.SIM_END_DATE},
        "pk_range":              {"min": v51.PK_MIN, "max": v51.PK_MAX},
        "time_resolution":       v51.TIME_RESOLUTION,
        "ablation_settings":     v51.ABLATION_SETTINGS,
        "memorize_mode":         bool(v51.MEMORIZE_MODE),
        "stage2_disabled":       False,
        "experiments":           experiments,
        "based_on":              "ablation_study_v52_xgboost_only",
        "wider_tree_ranges": {
            "stage1_int":   _S1_INT_RANGES,
            "stage1_float": _S1_FLOAT_RANGES,
            "stage2_int":   _S2_INT_RANGES,
            "stage2_float": _S2_FLOAT_RANGES,
        },
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    manifest_path = os.path.join(
        v51.OUTPUT_DIR, f"v53_xgboost_manifest_{model_time}.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"[V53] Manifest saved: {manifest_path}")

    exp_config = {
        "model_time":         model_time,
        "test_mode":          v51.TEST_MODE,
        "memorize_mode":      bool(v51.MEMORIZE_MODE),
        "ablation_settings":  v51.ABLATION_SETTINGS,
        "pipeline_version":   PIPELINE_VERSION,
        "gnn_experiment_dir": gnn_experiment_dir,
        "stage1_threshold":   float(s1_thr),
        "stage2_threshold":   float(s2_thr),
        "available_features": stage1_features,
        "stage1_features":    stage1_features,
        "stage2_features":    stage2_features,
        "train_dates":        {"start": v51.TRAIN_START_DATE, "end": v51.TRAIN_END_DATE},
        "sim_dates":          {"start": v51.SIM_START_DATE,   "end": v51.SIM_END_DATE},
        "experiments":        experiments,
        "timestamp":          datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    exp_config_path = os.path.join(v51.OUTPUT_DIR, "experiment_config_xgb_v53.json")
    with open(exp_config_path, "w") as f:
        json.dump(exp_config, f, indent=2, default=str)
    print(f"[V53] Sim-runner config saved: {exp_config_path}")


# =============================================================================
# MODULE-ATTR PROPAGATION
# =============================================================================

def _propagate_to_v52() -> None:
    """Mirror V53 mutable module attrs onto V52 (which then mirrors onto V51).

    Any of V52's knobs that V53 inherits — pk×hour, sweep —
    can be set on either module; we keep them in sync so V51/V52/V34
    stage functions all see the same configuration.
    """
    v52.ADD_PK_HOUR_INTERACTIONS  = bool(ADD_PK_HOUR_INTERACTIONS)
    v52.MIN_RECALL_CONSTRAINT     = float(MIN_RECALL_CONSTRAINT)
    v52.MAX_FAR_TARGET            = float(MAX_FAR_TARGET)
    v52.THRESHOLD_GRID_S1         = THRESHOLD_GRID_S1
    v52.THRESHOLD_GRID_S2         = THRESHOLD_GRID_S2
    v52.MEMORIZE_MODE             = bool(MEMORIZE_MODE) or bool(v52.MEMORIZE_MODE)
    v51.MEMORIZE_MODE             = bool(MEMORIZE_MODE) or bool(v51.MEMORIZE_MODE)
    if hasattr(v52, "_propagate_to_v51"):
        try:
            v52._propagate_to_v51()
        except Exception as e:
            print(f"[V53] v52._propagate_to_v51() failed: {e}")


# =============================================================================
# ENTRY POINTS
# =============================================================================

def run_xgboost_only() -> bool:
    """Entry point invoked by ``bash_files/run_ablation_study_xgboost_only.sh``.

    Mirrors V52's signature: reads the GNN experiment dir from env and runs
    the V53 pipeline.  Also re-points ``v51.OUTPUT_DIR`` (and ``v51.VIZ_DIR``)
    to the per-XGBoost-run subfolder ``<gnn_dir>/xgboost_<XGB_RUN_ID>/``.
    """
    gnn_experiment_dir = (
        os.environ.get("PREVIOUS_EXPERIMENT_DIR", "")
        or os.environ.get("GNN_EXPERIMENT_DIR", "")
        or getattr(v51, "GNN_EXPERIMENT_DIR", "")
    )
    if not gnn_experiment_dir or not os.path.isdir(gnn_experiment_dir):
        print(f"[ERROR] PREVIOUS_EXPERIMENT_DIR not set or invalid: {gnn_experiment_dir!r}")
        sys.exit(1)

    xgb_run_id = (
        os.environ.get("XGB_RUN_ID", "")
        or os.environ.get("JOB_ID", "")
        or os.environ.get("SLURM_JOB_ID", "")
    )
    if not xgb_run_id:
        xgb_run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"[V53] XGB_RUN_ID not set; using timestamp fallback: {xgb_run_id}")

    v51.OUTPUT_DIR = os.path.join(gnn_experiment_dir, f"xgboost_{xgb_run_id}")
    v51.VIZ_DIR    = os.path.join(v51.OUTPUT_DIR, "visualizations")
    print(f"[V53] XGBoost artefacts will be saved under: {v51.OUTPUT_DIR}")

    return run_pipeline_v53(gnn_experiment_dir)


def _parse_csv_feats(s: str) -> list[str]:
    return [tok.strip() for tok in s.split(",") if tok.strip()]


def main() -> int:
    global MEMORIZE_MODE, ADD_PK_HOUR_INTERACTIONS
    global MIN_RECALL_CONSTRAINT, STAGE1_DROP_FEATURES, STAGE2_FEATURES

    import argparse
    parser = argparse.ArgumentParser(
        description="V53 XGBoost-only — V52 + asymmetric Stage-1/Stage-2 pruning."
    )
    parser.add_argument("--gnn-experiment-dir", required=True,
                        help="V14/V17 GNN experiment dir for GNN inference at training time.")
    parser.add_argument("--memorize-mode", action="store_true", default=False,
                        help="Force MEMORIZE_MODE on.")
    parser.add_argument("--no-pk-hour", action="store_true", default=False,
                        help="Disable V52 experiment (c) — pk × hour interactions.")
    parser.add_argument("--min-recall", type=float, default=MIN_RECALL_CONSTRAINT,
                        help=f"V52 experiment (a) — min recall constraint (default {MIN_RECALL_CONSTRAINT}).")
    parser.add_argument("--stage1-drop-feats", type=str, default=None,
                        help="Comma-separated list overriding STAGE1_DROP_FEATURES.")
    parser.add_argument("--stage1-no-prune", action="store_true", default=False,
                        help="Disable Stage-1 pruning entirely.")
    parser.add_argument("--stage2-feats", type=str, default=None,
                        help="Comma-separated list overriding STAGE2_FEATURES.")
    parser.add_argument("--stage2-inherit-stage1", action="store_true", default=False,
                        help="Use Stage-1's feature set for Stage-2 (matches V52).")

    args = parser.parse_args()

    if args.memorize_mode:
        MEMORIZE_MODE = True
        v51.MEMORIZE_MODE = True
    if args.no_pk_hour:
        ADD_PK_HOUR_INTERACTIONS = False
    MIN_RECALL_CONSTRAINT  = float(args.min_recall)

    if args.stage1_no_prune:
        STAGE1_DROP_FEATURES = []
    elif args.stage1_drop_feats is not None:
        STAGE1_DROP_FEATURES = _parse_csv_feats(args.stage1_drop_feats)

    if args.stage2_inherit_stage1:
        STAGE2_FEATURES = None
    elif args.stage2_feats is not None:
        STAGE2_FEATURES = _parse_csv_feats(args.stage2_feats)

    return 0 if run_pipeline_v53(args.gnn_experiment_dir) else 1


if __name__ == "__main__":
    sys.exit(main())
