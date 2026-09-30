#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V55 XGBoost-only: V54 + Stage-1 threshold sweep before Stage-2 training
========================================================================

In V54, Stage 2 is trained on rows where Stage-1 OOF probability ≥ 0.5
(hardcoded). The final threshold sweep at inference may then pick a very
different Stage-1 threshold (e.g. 0.95 for the recall-max run), so Stage 2
was trained on a different distribution from the one it sees at inference.

V55 closes that loop:

  1. Stage 1 trained with AUCPR Optuna (same as V54).
  2. NEW — Stage-1 OOF threshold sweep: pick the S1 threshold that
     maximises RECALL under min_recall ≥ 0.9 (cast a wide-but-bounded net).
  3. Use that picked threshold to filter Stage-2's training pool.
  4. Stage 2 trained with AUCPR Optuna (same as V54).
  5. Final test-time sweep: S1 LOCKED at the training-time pick; only S2
     is swept, with PRECISION maximisation under min_precision ≥ 0.9.

The Stage-1 sweep, the Stage-2 sweep, and both Optuna metrics are all
overridable via module attrs (or the bash runner's --optimise-for /
--min-recall / --min-precision / --stage1-* flags). The defaults below
encode the V55 recipe.

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
from sklearn.model_selection import StratifiedKFold

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import src.training.ablation_study_v51_xgboost_only as v51
import src.training.ablation_study_v52_xgboost_only as v52
import src.training.ablation_study_v53_xgboost_only as v53
import src.training.ablation_study_v54_xgboost_only as v54
from src.training.ablation_study_v51_xgboost_only import (
    create_balanced_subset,
    format_duration,
    stage0_load_data_with_gnn,
    stage1_feature_prep,
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
    THRESHOLD_GRID_S2,
)
from src.training.ablation_study_v54_xgboost_only import (
    stage2_train_ensemble_v54,
    _resolve_stage1_features,
    _resolve_stage2_features,
    _slice_for_stage2,
    _score_probs_for_optuna,
    _ALLOWED_OPTUNA_METRICS,
)


# =============================================================================
# V55 CONFIGURATION
# =============================================================================

PIPELINE_VERSION = "v55_xgboost_only"

# Inherit V53/V54 feature pruning + experimental knobs.
STAGE1_DROP_FEATURES: list[str] = list(v54.STAGE1_DROP_FEATURES)
STAGE2_FEATURES: list[str] | None = (
    list(v54.STAGE2_FEATURES) if v54.STAGE2_FEATURES is not None else None
)
ADD_PK_HOUR_INTERACTIONS: bool = v54.ADD_PK_HOUR_INTERACTIONS
MEMORIZE_MODE: bool = bool(v51.MEMORIZE_MODE)

# V55 recipe defaults — AUCPR for both Optuna searches.
OPTUNA_METRIC_STAGE1: str = "aucpr"
OPTUNA_METRIC_STAGE2: str = "aucpr"
OPTUNA_MIN_RECALL_STAGE1: float = 0.30  # unused when metric=aucpr, kept for compat
OPTUNA_MIN_RECALL_STAGE2: float = 0.30

# Stage-1 OOF threshold sweep (NEW in V55).
STAGE1_OPTIMISE_FOR: str = "recall"
STAGE1_MIN_RECALL: float = 0.90
STAGE1_MIN_PRECISION: float = 0.0

# Stage-2 (final) threshold sweep — S1 is LOCKED to the training-time pick.
OPTIMISE_FOR: str = "precision"
MIN_RECALL_CONSTRAINT: float = 0.0
MIN_PRECISION_CONSTRAINT: float = 0.90
MAX_FAR_TARGET: float = 1.0 - MIN_PRECISION_CONSTRAINT  # = 0.10

# Threshold grids.
# Stage-1 OOF sweep uses a fine grid (~0.005 steps in the tails, 0.025 in the middle).
STAGE1_OOF_GRID = np.unique(np.round(np.concatenate([
    np.arange(0.005, 0.05 + 1e-9, 0.005),
    np.arange(0.05, 0.50 + 1e-9, 0.025),
    np.arange(0.50, 1.00 + 1e-9, 0.025),
]), 4))
THRESHOLD_GRID_S2 = THRESHOLD_GRID_S2  # inherited


# =============================================================================
# Stage-1 OOF threshold sweep
# =============================================================================

def sweep_stage1_threshold(y_true: np.ndarray,
                           y_prob: np.ndarray,
                           *,
                           optimise_for: str = "recall",
                           min_recall: float = 0.0,
                           min_precision: float = 0.0,
                           grid: np.ndarray | None = None) -> tuple[float, dict]:
    """Pick a Stage-1 threshold on (y_true, y_prob).

    Maximises `optimise_for` (recall|precision|f1|f2) subject to
    `recall >= min_recall AND precision >= min_precision`. Falls back to
    max-F2 unconstrained if no threshold is feasible.

    Returns (best_threshold, summary_dict).
    """
    if grid is None:
        grid = STAGE1_OOF_GRID

    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    best = {"thr": 0.5, "score": -np.inf, "precision": 0.0, "recall": 0.0,
            "tp": 0, "fp": 0, "fn": 0, "f1": 0.0, "f2": 0.0, "feasible": False}
    rows: list[dict] = []

    for thr in grid:
        pred = (y_prob >= thr).astype(int)
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        fn = int(((pred == 0) & (y_true == 1)).sum())
        if (tp + fp) == 0:
            continue
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        f2 = 5 * precision * recall / max(4 * precision + recall, 1e-12)
        score = {"f2": f2, "f1": f1, "precision": precision,
                 "recall": recall}.get(optimise_for, f2)
        rows.append({"thr": float(thr), "tp": tp, "fp": fp, "fn": fn,
                     "precision": precision, "recall": recall,
                     "f1": f1, "f2": f2})
        if recall >= min_recall and precision >= min_precision and score > best["score"]:
            best.update({"thr": float(thr), "score": score,
                         "precision": precision, "recall": recall,
                         "tp": tp, "fp": fp, "fn": fn,
                         "f1": f1, "f2": f2, "feasible": True})

    if not best["feasible"]:
        # Graceful fallback: keep the user's objective, drop the floor(s).
        # If recall ≥ X is infeasible, return the threshold with the
        # *highest* `optimise_for` value available, not an arbitrary F2 peak.
        print(f"[V55-S1-SWEEP] WARNING: no threshold satisfies "
              f"recall ≥ {min_recall} AND precision ≥ {min_precision}. "
              f"Falling back to max-{optimise_for} unconstrained.")
        if rows:
            best_unc = max(rows, key=lambda r: r.get(optimise_for, r["f2"]))
            unc_score = float(best_unc.get(optimise_for, best_unc["f2"]))
            best.update({**best_unc, "score": unc_score, "feasible": False})

    print(f"[V55-S1-SWEEP] best ({optimise_for}, feasible={best['feasible']}): "
          f"thr={best['thr']:.4f}  P={best['precision']:.4f}  R={best['recall']:.4f}  "
          f"F1={best['f1']:.4f}  F2={best['f2']:.4f}  "
          f"TP={best['tp']}  FP={best['fp']}  FN={best['fn']}")

    return float(best["thr"]), best


# =============================================================================
# Stage-2 training (V55) — OOF + Stage-1 sweep + filter, then Stage-2 fit
# =============================================================================

def _train_stage2_holdout_v55(*, X_train_xgb, X_test_xgb, y_train_acc, y_test_acc,
                              base_models, ensemble_preds, stage2_features,
                              s1_best_params, model_time, output_dir,
                              optuna_metric: str, optuna_min_recall: float,
                              stage1_optimise_for: str,
                              stage1_min_recall: float,
                              stage1_min_precision: float):
    """V55 Stage-2 trainer.

    Differs from V54:
      - Inserts a Stage-1 OOF threshold sweep (`sweep_stage1_threshold`)
        between the OOF prob generation and the Stage-2 pool filter.
      - The picked Stage-1 threshold drives BOTH the train-pool filter
        (`s2_mask = s1_tr_probs >= picked`) AND the test-pool filter
        (`test_s1_flag = ensemble_preds >= picked`).
      - Returns the picked threshold and its sweep summary so the final
        test-time sweep can lock S1 to that value.
    """
    X_tr = X_train_xgb.reset_index(drop=True)
    y_tr = y_train_acc.reset_index(drop=True)
    stage1_features = list(X_tr.columns)
    neg_pos_ratio = ENSEMBLE_CONFIG["neg_pos_ratio"]

    # 1. OOF Stage-1 probabilities on the training set (unchanged from V54)
    oof_n_folds = 5
    oof_n_models = 5
    mean_iter = int(np.mean([getattr(m, "best_iteration", 100) for m in base_models]))
    oof_rounds = max(50, mean_iter)
    oof_params = {**s1_best_params, "tree_method": "hist",
                  "objective": "binary:logistic", "eval_metric": "aucpr",
                  "nthread": N_THREADS, "seed": RANDOM_STATE}

    oof_kf = StratifiedKFold(n_splits=oof_n_folds, shuffle=True, random_state=RANDOM_STATE)
    s1_tr_probs = np.zeros(len(X_tr))
    rng = np.random.RandomState(RANDOM_STATE)
    for fold_i, (tr_idx, va_idx) in enumerate(oof_kf.split(X_tr.values, y_tr.values), 1):
        X_fold = X_tr.iloc[tr_idx]
        y_fold = y_tr.iloc[tr_idx]
        pos_ix = np.where(y_fold.values == 1)[0]
        neg_ix = np.where(y_fold.values == 0)[0]
        fold_probs = np.zeros(len(va_idx))
        for _ in range(oof_n_models):
            samp_neg = rng.choice(neg_ix, size=len(pos_ix) * neg_pos_ratio, replace=False)
            bag_idx = np.concatenate([pos_ix, samp_neg])
            rng.shuffle(bag_idx)
            Xb = X_fold.iloc[bag_idx][stage1_features]
            yb = y_fold.iloc[bag_idx]
            m = xgb.train(oof_params, xgb.DMatrix(Xb, label=yb),
                          num_boost_round=oof_rounds, verbose_eval=False)
            fold_probs += m.predict(xgb.DMatrix(X_tr.iloc[va_idx][stage1_features]))
        s1_tr_probs[va_idx] = fold_probs / oof_n_models

    # 2. NEW IN V55 — Stage-1 OOF threshold sweep
    print("=" * 80)
    print("V55 STAGE 7A: STAGE-1 OOF THRESHOLD SWEEP")
    print("=" * 80)
    print(f"optimise_for={stage1_optimise_for}  "
          f"min_recall={stage1_min_recall}  min_precision={stage1_min_precision}")
    stage1_threshold, s1_sweep_summary = sweep_stage1_threshold(
        y_true=y_tr.values, y_prob=s1_tr_probs,
        optimise_for=stage1_optimise_for,
        min_recall=stage1_min_recall,
        min_precision=stage1_min_precision,
    )

    # 3. Filter both train and test pools at the picked Stage-1 threshold
    s2_mask = s1_tr_probs >= stage1_threshold
    X_s2_full_tr = X_tr[s2_mask].copy()
    X_s2_full_tr["s1_score"] = s1_tr_probs[s2_mask]
    y_s2_tr = y_tr[s2_mask].copy()

    X_test_r = X_test_xgb.reset_index(drop=True)
    y_test_r = y_test_acc.reset_index(drop=True)
    test_s1_flag = ensemble_preds >= stage1_threshold
    X_s2_full_te = X_test_r[test_s1_flag].copy()
    X_s2_full_te["s1_score"] = ensemble_preds[test_s1_flag]
    y_s2_te = y_test_r[test_s1_flag].copy()

    X_s2_tr = _slice_for_stage2(X_s2_full_tr, stage2_features)
    X_s2_te = _slice_for_stage2(X_s2_full_te, stage2_features)

    n_tp_s2 = int(y_s2_tr.sum())
    n_fp_s2 = int((y_s2_tr == 0).sum())
    s2_spw = n_fp_s2 / max(n_tp_s2, 1)
    print(f"[V55] Stage-2 training pool @ S1≥{stage1_threshold:.4f}: "
          f"{len(y_s2_tr):,} rows  (TP={n_tp_s2}, FP={n_fp_s2}, spw={s2_spw:.2f})")
    print(f"[V55] Stage-2 test pool     @ S1≥{stage1_threshold:.4f}: "
          f"{len(y_s2_te):,} rows  (TP={int(y_s2_te.sum())}, "
          f"FP={int((y_s2_te == 0).sum())})")

    # 4. Stage-2 Optuna (same structure as V54)
    s2_trials = 30
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
        skf = StratifiedKFold(n_splits=s2_cv_folds, shuffle=True, random_state=RANDOM_STATE)
        X_arr = X_s2_tr.values
        y_arr = y_s2_tr.values
        scores = []
        for tr_idx, va_idx in skf.split(X_arr, y_arr):
            dtr = xgb.DMatrix(X_arr[tr_idx], label=y_arr[tr_idx])
            dva = xgb.DMatrix(X_arr[va_idx], label=y_arr[va_idx])
            mf = xgb.train(params, dtr, num_boost_round=300,
                           evals=[(dva, "eval")], early_stopping_rounds=20,
                           verbose_eval=False)
            preds = mf.predict(dva, iteration_range=(0, mf.best_iteration))
            scores.append(_score_probs_for_optuna(
                y_true=y_arr[va_idx].astype(int),
                y_prob=preds.astype(float),
                metric=optuna_metric,
                min_recall=optuna_min_recall,
            ))
        return float(np.mean(scores)) if scores else 0.0

    print("=" * 80)
    print("V55 STAGE 7B: Stage-2 Optuna (configurable; defaults to AUCPR)")
    print("=" * 80)
    print(f"Optuna metric: {optuna_metric} (min_recall={optuna_min_recall})")
    s2_study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    s2_study.optimize(_s2_objective, n_trials=s2_trials, show_progress_bar=True)
    print(f"Stage 2 best Optuna score: {s2_study.best_value:.4f}")
    for k, v in s2_study.best_params.items():
        print(f"  {k}: {v}")

    s2_best_params = {
        **s2_study.best_params,
        "scale_pos_weight": s2_spw,
        "tree_method": "hist",
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "nthread": N_THREADS,
        "seed": RANDOM_STATE,
    }
    dtr_s2 = xgb.DMatrix(X_s2_tr, label=y_s2_tr)
    dte_s2 = xgb.DMatrix(X_s2_te, label=y_s2_te)
    stage2_model = xgb.train(
        s2_best_params,
        dtr_s2,
        num_boost_round=500,
        early_stopping_rounds=30,
        evals=[(dtr_s2, "train"), (dte_s2, "test")],
        verbose_eval=50,
    )
    stage2_path = os.path.join(output_dir, f"xgboost-stage2-fp-filter_version={model_time}.json")
    stage2_model.save_model(stage2_path)
    print(f"\nStage 2 model saved: {stage2_path}")

    s2_test_probs = stage2_model.predict(dte_s2)
    y_test_np = y_test_r.values
    flagged_idx = np.where(test_s1_flag)[0]
    return (stage2_model, X_s2_tr, y_s2_tr, flagged_idx, s2_test_probs, y_test_np,
            dict(s2_study.best_params), float(stage1_threshold), s1_sweep_summary)


# =============================================================================
# Manifests
# =============================================================================

def write_v55_manifests(*, gnn_experiment_dir: str, model_time: str,
                        stage1_features: list[str], stage1_dropped: list[str],
                        stage2_features: list[str],
                        s1_thr: float, s2_thr: float,
                        roc_auc: float, pr_auc: float,
                        sweep_summary: dict,
                        s1_sweep_summary: dict,
                        v52_extra_features: list[str],
                        s1_best_params: dict,
                        s2_best_params: dict) -> None:
    def _coerce(v):
        if isinstance(v, (np.floating, np.integer)):
            return float(v) if isinstance(v, np.floating) else int(v)
        return v

    experiments = {
        "pk_hour_interactions": bool(ADD_PK_HOUR_INTERACTIONS),
        "v52_extra_features": list(v52_extra_features),
        "stage1_dropped_features": list(stage1_dropped),
        "stage1_drop_request": list(STAGE1_DROP_FEATURES),
        "stage2_features": list(stage2_features),
        "stage2_inherits_stage1": STAGE2_FEATURES is None,
        "optuna_objective_stage1": {
            "metric": OPTUNA_METRIC_STAGE1,
            "min_recall": float(OPTUNA_MIN_RECALL_STAGE1),
        },
        "optuna_objective_stage2": {
            "metric": OPTUNA_METRIC_STAGE2,
            "min_recall": float(OPTUNA_MIN_RECALL_STAGE2),
        },
        "stage1_sweep": {
            "optimise_for": STAGE1_OPTIMISE_FOR,
            "min_recall_constraint": float(STAGE1_MIN_RECALL),
            "min_precision_constraint": float(STAGE1_MIN_PRECISION),
            "picked_threshold": float(s1_thr),
            "feasible": bool(s1_sweep_summary.get("feasible", False)),
            "summary": {k: _coerce(v) for k, v in s1_sweep_summary.items()
                        if k != "feasible"},
        },
        "threshold_sweep": {
            "optimise_for": OPTIMISE_FOR,
            "min_recall_constraint": float(MIN_RECALL_CONSTRAINT),
            "min_precision_constraint": float(MIN_PRECISION_CONSTRAINT),
            "max_far_target": float(MAX_FAR_TARGET),
            "s1_locked_to": float(s1_thr),
            "s2_grid": [float(x) for x in THRESHOLD_GRID_S2],
            "best": {k: _coerce(v) for k, v in sweep_summary["best"].items()},
        },
        "hot_zone_pks": list(HOT_ZONE_PKS),
        "evening_rush_hours": list(EVENING_RUSH_HOURS),
        "stage1_best_params": dict(s1_best_params),
        "stage2_best_params": dict(s2_best_params),
    }

    manifest = {
        "pipeline_version": PIPELINE_VERSION,
        "model_time": model_time,
        "output_dir": v51.OUTPUT_DIR,
        "gnn_experiment_dir": gnn_experiment_dir,
        "available_features": stage1_features,
        "stage1_features": stage1_features,
        "stage2_features": stage2_features,
        "stage1_threshold": float(s1_thr),
        "stage2_threshold": float(s2_thr),
        "min_recall_constraint": float(MIN_RECALL_CONSTRAINT),
        "min_precision_constraint": float(MIN_PRECISION_CONSTRAINT),
        "max_far_target": float(MAX_FAR_TARGET),
        "test_metrics_stage1": {"roc_auc": float(roc_auc), "pr_auc": float(pr_auc)},
        "train_dates": {"start": v51.TRAIN_START_DATE, "end": v51.TRAIN_END_DATE},
        "sim_dates": {"start": v51.SIM_START_DATE, "end": v51.SIM_END_DATE},
        "pk_range": {"min": v51.PK_MIN, "max": v51.PK_MAX},
        "time_resolution": v51.TIME_RESOLUTION,
        "ablation_settings": v51.ABLATION_SETTINGS,
        "memorize_mode": bool(v51.MEMORIZE_MODE),
        "stage2_disabled": False,
        "experiments": experiments,
        "based_on": "ablation_study_v54_xgboost_only",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    manifest_path = os.path.join(v51.OUTPUT_DIR, f"v55_xgboost_manifest_{model_time}.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"[V55] Manifest saved: {manifest_path}")

    exp_config = {
        "model_time": model_time,
        "test_mode": v51.TEST_MODE,
        "memorize_mode": bool(v51.MEMORIZE_MODE),
        "ablation_settings": v51.ABLATION_SETTINGS,
        "pipeline_version": PIPELINE_VERSION,
        "gnn_experiment_dir": gnn_experiment_dir,
        "stage1_threshold": float(s1_thr),
        "stage2_threshold": float(s2_thr),
        "available_features": stage1_features,
        "stage1_features": stage1_features,
        "stage2_features": stage2_features,
        "train_dates": {"start": v51.TRAIN_START_DATE, "end": v51.TRAIN_END_DATE},
        "sim_dates": {"start": v51.SIM_START_DATE, "end": v51.SIM_END_DATE},
        "experiments": experiments,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    exp_config_path = os.path.join(v51.OUTPUT_DIR, "experiment_config_xgb_v55.json")
    with open(exp_config_path, "w") as f:
        json.dump(exp_config, f, indent=2, default=str)
    print(f"[V55] Sim-runner config saved: {exp_config_path}")


# =============================================================================
# Pipeline
# =============================================================================

def _propagate_to_v54() -> None:
    v54.ADD_PK_HOUR_INTERACTIONS = bool(ADD_PK_HOUR_INTERACTIONS)
    v54.MEMORIZE_MODE = bool(MEMORIZE_MODE) or bool(v54.MEMORIZE_MODE)
    v51.MEMORIZE_MODE = bool(MEMORIZE_MODE) or bool(v51.MEMORIZE_MODE)
    if hasattr(v54, "_propagate_to_v52"):
        try:
            v54._propagate_to_v52()
        except Exception as e:
            print(f"[V55] v54._propagate_to_v52() failed: {e}")


def run_pipeline_v55(gnn_experiment_dir: str) -> bool:
    _propagate_to_v54()
    t_start = time.time()

    print("=" * 80)
    print("V55 XGBoost-only — V54 + Stage-1 threshold sweep before Stage-2 training")
    print(f"Output dir:      {v51.OUTPUT_DIR}")
    print(f"GNN experiment:  {gnn_experiment_dir}")
    print(f"MEMORIZE_MODE:   {bool(v51.MEMORIZE_MODE)}")
    print("Optuna objective config:")
    print(f"  Stage 1: metric={OPTUNA_METRIC_STAGE1}  min_recall={OPTUNA_MIN_RECALL_STAGE1}")
    print(f"  Stage 2: metric={OPTUNA_METRIC_STAGE2}  min_recall={OPTUNA_MIN_RECALL_STAGE2}")
    print("Stage-1 OOF threshold sweep config:")
    print(f"  optimise_for={STAGE1_OPTIMISE_FOR}  "
          f"min_recall={STAGE1_MIN_RECALL}  min_precision={STAGE1_MIN_PRECISION}")
    print("Stage-2 (final) threshold sweep config:")
    print(f"  optimise_for={OPTIMISE_FOR}  "
          f"min_recall={MIN_RECALL_CONSTRAINT}  "
          f"min_precision={MIN_PRECISION_CONSTRAINT} "
          f"(max_far={MAX_FAR_TARGET:.3f})")
    print("  S1 LOCKED at the training-time Stage-1 pick")
    print("=" * 80)

    if bool(v51.MEMORIZE_MODE):
        raise NotImplementedError("V55 does not yet implement MEMORIZE_MODE (no holdout pool).")

    os.makedirs(v51.OUTPUT_DIR, exist_ok=True)
    os.makedirs(v51.VIZ_DIR, exist_ok=True)

    df_full, fe_only_cols = stage0_load_data_with_gnn(gnn_experiment_dir)

    v52_extra_features: list[str] = []
    if ADD_PK_HOUR_INTERACTIONS:
        df_full, v52_extra_features = add_pk_hour_interactions(df_full)
        fe_only_cols = list(fe_only_cols) + [f for f in v52_extra_features if f not in fe_only_cols]

    time_res_min = v51._get_time_resolution_minutes()

    (df_train, X_train_xgb, X_test_xgb, y_train_acc, y_test_acc,
     available_features, model_time) = stage1_feature_prep(df_full, fe_only_cols)
    del df_train
    gc.collect()

    stage1_features, stage1_dropped = _resolve_stage1_features(available_features)
    if stage1_dropped:
        X_train_xgb = X_train_xgb[stage1_features]
        X_test_xgb = X_test_xgb[stage1_features]

    stage2_features = _resolve_stage2_features(stage1_features)

    # Stage-1 training (reused from V54 — AUCPR Optuna by default)
    with _wider_optuna_ranges(_S1_INT_RANGES, _S1_FLOAT_RANGES):
        (base_models, ensemble_preds, _y_pred_binary, roc_auc, pr_auc,
         s1_best_params) = stage2_train_ensemble_v54(
            X_train_xgb, X_test_xgb, y_train_acc, y_test_acc,
            stage1_features, model_time,
            optuna_metric=OPTUNA_METRIC_STAGE1,
            optuna_min_recall=float(OPTUNA_MIN_RECALL_STAGE1),
        )

    stage5_feature_importance_shap(base_models, X_test_xgb, stage1_features)
    stage6_feature_space_overlap(
        X_train_xgb, X_test_xgb, y_train_acc, y_test_acc,
        stage1_features, ensemble_preds,
    )

    # Stage-1 threshold sweep + Stage-2 training (V55-specific)
    with _wider_optuna_ranges(_S2_INT_RANGES, _S2_FLOAT_RANGES):
        (stage2_model, X_s2_tr, y_s2_tr, flagged_idx,
         s2_test_probs, y_test_np, s2_best_params,
         stage1_threshold_picked, s1_sweep_summary) = _train_stage2_holdout_v55(
            X_train_xgb=X_train_xgb, X_test_xgb=X_test_xgb,
            y_train_acc=y_train_acc, y_test_acc=y_test_acc,
            base_models=base_models, ensemble_preds=ensemble_preds,
            stage2_features=stage2_features,
            s1_best_params=s1_best_params,
            model_time=model_time, output_dir=v51.OUTPUT_DIR,
            optuna_metric=OPTUNA_METRIC_STAGE2,
            optuna_min_recall=float(OPTUNA_MIN_RECALL_STAGE2),
            stage1_optimise_for=STAGE1_OPTIMISE_FOR,
            stage1_min_recall=float(STAGE1_MIN_RECALL),
            stage1_min_precision=float(STAGE1_MIN_PRECISION),
        )
    stage7b_shap_stage2(stage2_model, X_s2_tr, y_s2_tr)

    # Final test-time sweep — S1 LOCKED, S2 swept
    print("=" * 80)
    print(f"V55 FINAL SWEEP — S1 locked at {stage1_threshold_picked:.4f}, S2 swept")
    print("=" * 80)
    S1_THR, S2_THR, sweep_summary = sweep_thresholds_v52(
        ensemble_preds=ensemble_preds,
        y_test_np=y_test_np,
        flagged_idx=flagged_idx,
        s2_test_probs=s2_test_probs,
        s1_grid=np.array([float(stage1_threshold_picked)]),
        s2_grid=THRESHOLD_GRID_S2,
        min_recall=float(MIN_RECALL_CONSTRAINT),
        max_far=float(MAX_FAR_TARGET),
        optimise_for=OPTIMISE_FOR,
    )

    write_v55_manifests(
        gnn_experiment_dir=gnn_experiment_dir,
        model_time=model_time,
        stage1_features=stage1_features,
        stage1_dropped=stage1_dropped,
        stage2_features=stage2_features,
        s1_thr=S1_THR, s2_thr=S2_THR,
        roc_auc=roc_auc, pr_auc=pr_auc,
        sweep_summary=sweep_summary,
        s1_sweep_summary=s1_sweep_summary,
        v52_extra_features=v52_extra_features,
        s1_best_params=s1_best_params,
        s2_best_params=s2_best_params,
    )

    print("\n" + "=" * 80)
    print(f"V55 XGBoost training complete in {format_duration(time.time() - t_start)}")
    print(f"All plots saved to:  {v51.VIZ_DIR}")
    print(f"All models saved to: {v51.OUTPUT_DIR}")
    print(f"Model time: {model_time}")
    print(f"S1 LOCKED at: {S1_THR:.4f}  |  S2 picked: {S2_THR:.4f}")
    print("Next step: ablation_study_v55_simulation_only.py "
          f"--xgboost-experiment-dir {v51.OUTPUT_DIR} "
          f"--gnn-experiment-dir {gnn_experiment_dir}")
    print("=" * 80)
    return True


def run_xgboost_only() -> bool:
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
        print(f"[V55] XGB_RUN_ID not set; using timestamp fallback: {xgb_run_id}")

    v51.OUTPUT_DIR = os.path.join(gnn_experiment_dir, f"xgboost_{xgb_run_id}")
    v51.VIZ_DIR = os.path.join(v51.OUTPUT_DIR, "visualizations")
    print(f"[V55] XGBoost artefacts will be saved under: {v51.OUTPUT_DIR}")
    return run_pipeline_v55(gnn_experiment_dir)


def main() -> int:
    global MEMORIZE_MODE, ADD_PK_HOUR_INTERACTIONS
    global OPTUNA_METRIC_STAGE1, OPTUNA_METRIC_STAGE2
    global OPTUNA_MIN_RECALL_STAGE1, OPTUNA_MIN_RECALL_STAGE2
    global STAGE1_OPTIMISE_FOR, STAGE1_MIN_RECALL, STAGE1_MIN_PRECISION
    global MIN_RECALL_CONSTRAINT, MIN_PRECISION_CONSTRAINT, MAX_FAR_TARGET, OPTIMISE_FOR

    import argparse
    p = argparse.ArgumentParser(description="V55 XGBoost-only — V54 + Stage-1 sweep.")
    p.add_argument("--gnn-experiment-dir", required=True)
    p.add_argument("--memorize-mode", action="store_true", default=False)
    p.add_argument("--no-pk-hour", action="store_true", default=False)

    # Optuna metric / floors
    p.add_argument("--optuna-metric-s1", type=str, default=OPTUNA_METRIC_STAGE1,
                   choices=sorted(_ALLOWED_OPTUNA_METRICS))
    p.add_argument("--optuna-metric-s2", type=str, default=OPTUNA_METRIC_STAGE2,
                   choices=sorted(_ALLOWED_OPTUNA_METRICS))
    p.add_argument("--optuna-min-recall-s1", type=float, default=OPTUNA_MIN_RECALL_STAGE1)
    p.add_argument("--optuna-min-recall-s2", type=float, default=OPTUNA_MIN_RECALL_STAGE2)

    # Stage-1 OOF threshold sweep (V55-specific)
    p.add_argument("--stage1-optimise-for", type=str, default=STAGE1_OPTIMISE_FOR,
                   choices=["precision", "recall", "f1", "f2"],
                   help="Stage-1 OOF sweep objective.")
    p.add_argument("--stage1-min-recall", type=float, default=STAGE1_MIN_RECALL)
    p.add_argument("--stage1-min-precision", type=float, default=STAGE1_MIN_PRECISION)

    # Stage-2 (final) threshold sweep
    p.add_argument("--min-recall", type=float, default=MIN_RECALL_CONSTRAINT)
    p.add_argument("--min-precision", type=float, default=MIN_PRECISION_CONSTRAINT)
    p.add_argument("--optimise-for", type=str, default=OPTIMISE_FOR,
                   choices=["precision", "recall", "f1", "f2"])

    args = p.parse_args()

    if args.memorize_mode:
        MEMORIZE_MODE = True
        v51.MEMORIZE_MODE = True
    if args.no_pk_hour:
        ADD_PK_HOUR_INTERACTIONS = False

    OPTUNA_METRIC_STAGE1 = str(args.optuna_metric_s1)
    OPTUNA_METRIC_STAGE2 = str(args.optuna_metric_s2)
    OPTUNA_MIN_RECALL_STAGE1 = float(args.optuna_min_recall_s1)
    OPTUNA_MIN_RECALL_STAGE2 = float(args.optuna_min_recall_s2)

    STAGE1_OPTIMISE_FOR = str(args.stage1_optimise_for)
    STAGE1_MIN_RECALL = float(args.stage1_min_recall)
    STAGE1_MIN_PRECISION = float(args.stage1_min_precision)

    MIN_RECALL_CONSTRAINT = float(args.min_recall)
    MIN_PRECISION_CONSTRAINT = float(args.min_precision)
    MAX_FAR_TARGET = float(1.0 - MIN_PRECISION_CONSTRAINT)
    OPTIMISE_FOR = str(args.optimise_for)

    return 0 if run_pipeline_v55(args.gnn_experiment_dir) else 1


if __name__ == "__main__":
    sys.exit(main())
