#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V54 XGBoost-only: V53 + configurable Optuna objective metrics
============================================================
V54 extends V53 by allowing you to choose *what Optuna optimizes* for both
Stage 1 and Stage 2 hyperparameter search.

Motivation
----------
V51/V52/V53 train with:
  - XGBoost loss:        binary:logistic
  - Early stopping:      AUCPR (eval_metric='aucpr')
  - Optuna objective:    F2 @ threshold=0.5

That is reasonable for recall-heavy operating points, but if your priority is
to **minimize false positives** (i.e., push precision extremely high), it is
more coherent to tune hyperparameters with an Optuna score aligned to that
goal, e.g.:

  - precision_at_recall: maximize precision achievable at recall >= R_min

This is the recommended configuration for "precision >= 0.9, recall can drop
to ~0.3".

New configuration knobs (module attrs / CLI)
--------------------------------------------
  OPTUNA_METRIC_STAGE1: str
  OPTUNA_METRIC_STAGE2: str
      One of:
        - 'precision_at_recall'   (recommended for low FP)
        - 'aucpr'
        - 'f2'
        - 'f1'

  OPTUNA_MIN_RECALL_STAGE1: float   (only used by precision_at_recall)
  OPTUNA_MIN_RECALL_STAGE2: float

V54 also defaults the threshold sweep to maximize precision under a recall
constraint, rather than maximize F2.

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
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit, StratifiedKFold

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import src.training.ablation_study_v51_xgboost_only as v51
import src.training.ablation_study_v52_xgboost_only as v52
import src.training.ablation_study_v53_xgboost_only as v53
from src.training.ablation_study_v51_xgboost_only import (
    accuracy_score,
    classification_report,
    create_balanced_subset,
    f1_score,
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
    OUTPUT_DIR,
    VIZ_DIR,
    ABLATION_SETTINGS,
    TEST_MODE,
)
from src.training.ablation_study_v52_xgboost_only import (
    add_pk_hour_interactions,
    sweep_thresholds_v52,
    EVENING_RUSH_HOURS,
    HOT_ZONE_PKS,
)


# =============================================================================
# V54 CONFIGURATION
# =============================================================================

PIPELINE_VERSION = "v54_xgboost_only"

# Inherit V53 pruning defaults (you can override via v54.* module attrs too).
#
# BENCH_NO_STAGE1_PRUNE=1 keeps every available feature instead. The default
# drop list is the "gain~=0 / KS~=0 dead weight" measured on the RETENTION
# label (V52 run xgboost_2569859), and 8 of its 12 entries are weather
# (is_raining, wet_curve_risk, wet_descent_risk, wet_speed, temp_near_freezing,
# cold_curve_risk, is_heavy_rain, precip_change_1d_vs_3d). That calibration
# does not transfer to the accident-PRECURSOR label: once a retention is
# underway the collapse itself is fully diagnostic and weather adds nothing,
# but a precursor window is by construction free-flowing traffic that only
# *looks* normal -- exactly where road-surface state should carry risk. Set
# this flag when training on a precursor-relabelled dataset to test whether
# the inherited pruning is discarding the signal that task depends on.
STAGE1_DROP_FEATURES: list[str] = (
    [] if os.environ.get("BENCH_NO_STAGE1_PRUNE", "0") == "1"
    else list(v53.STAGE1_DROP_FEATURES)
)
STAGE2_FEATURES: list[str] | None = list(v53.STAGE2_FEATURES) if v53.STAGE2_FEATURES is not None else None

# Inherit V52 experimental knobs.
ADD_PK_HOUR_INTERACTIONS: bool = v52.ADD_PK_HOUR_INTERACTIONS

# If your explicit goal is precision-first, lower the recall floor.
MIN_RECALL_CONSTRAINT: float = 0.30
# MAX_FAR_TARGET caps the false-alarm rate (FAR = FP/(TP+FP) = 1 - precision).
# `MAX_FAR_TARGET = 1 - MIN_PRECISION`, so leaving it at 1.0 means "no precision floor".
MAX_FAR_TARGET: float = 1.0
# What the final S1/S2 threshold sweep maximises: 'precision' (Goal A, default),
# 'recall' (Goal B, pair with MAX_FAR_TARGET < 1.0), 'f1', or 'f2'.
OPTIMISE_FOR: str = "precision"
THRESHOLD_GRID_S1 = v52.THRESHOLD_GRID_S1
THRESHOLD_GRID_S2 = v52.THRESHOLD_GRID_S2

MEMORIZE_MODE: bool = bool(v51.MEMORIZE_MODE)

# ── NEW: Optuna objective selection ──────────────────────────────────────────
OPTUNA_METRIC_STAGE1: str = "precision_at_recall"
OPTUNA_METRIC_STAGE2: str = "precision_at_recall"
OPTUNA_MIN_RECALL_STAGE1: float = 0.30
OPTUNA_MIN_RECALL_STAGE2: float = 0.30


# =============================================================================
# Optuna scoring utilities
# =============================================================================

_ALLOWED_OPTUNA_METRICS = {"precision_at_recall", "aucpr", "f2", "f1"}


def _score_probs_for_optuna(*,
                            y_true: np.ndarray,
                            y_prob: np.ndarray,
                            metric: str,
                            min_recall: float,
                            beta: float = 2.0) -> float:
    """Return a scalar score to MAXIMIZE for Optuna."""
    metric = (metric or "").strip().lower()
    if metric not in _ALLOWED_OPTUNA_METRICS:
        raise ValueError(f"Unknown OPTUNA metric {metric!r}. Allowed: {sorted(_ALLOWED_OPTUNA_METRICS)}")

    if metric == "aucpr":
        return float(average_precision_score(y_true, y_prob))

    if metric in {"f2", "f1"}:
        b = 2.0 if metric == "f2" else 1.0
        y_hat = (y_prob >= 0.5).astype(int)
        return float(fbeta_score(y_true, y_hat, beta=b, zero_division=0))

    # precision_at_recall
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    # precision_recall_curve returns len(prec)=len(rec)=len(thr)+1
    valid = rec >= float(min_recall)
    if not np.any(valid):
        return 0.0
    return float(np.max(prec[valid]))


# =============================================================================
# Stage 1 Optuna objective (V54)
# =============================================================================


class BalancedSubsetObjectiveV54:
    """Stage-1 Optuna objective with configurable metric.

    Mirrors V51's BalancedSubsetObjective but computes the fold score from
    predicted probabilities using `_score_probs_for_optuna`.
    """

    def __init__(self,
                 X_train: pd.DataFrame,
                 y_train: pd.Series,
                 available_features: list[str],
                 *,
                 neg_pos_ratio: int,
                 n_folds: int,
                 n_trials: int,
                 metric: str,
                 min_recall: float):
        self.X_train = X_train
        self.y_train = y_train
        self.available_features = list(available_features)
        self.neg_pos_ratio = int(neg_pos_ratio)
        self.n_folds = int(n_folds)
        self.n_trials = int(n_trials)
        self.metric = metric
        self.min_recall = float(min_recall)
        # true ratio for upper bound on scale_pos_weight (same idea as V51)
        self.true_ratio = int((y_train == 0).sum()) / max(int((y_train == 1).sum()), 1)

    def __call__(self, trial: optuna.trial.Trial) -> float:
        t0 = time.time()
        params = {
            "tree_method":      "hist",
            "objective":        "binary:logistic",
            "eval_metric":      "aucpr",
            "max_depth":        trial.suggest_int("max_depth", 3, 12),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 50),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "gamma":            trial.suggest_float("gamma", 0.0, 10.0),
            "reg_alpha":        trial.suggest_float("reg_alpha", 0.0, 10.0),
            "reg_lambda":       trial.suggest_float("reg_lambda", 0.0, 10.0),
            "scale_pos_weight": trial.suggest_float(
                "scale_pos_weight", 1.0, self.true_ratio / float(self.neg_pos_ratio)
            ),
            "random_state": RANDOM_STATE,
            "nthread":      N_THREADS,
        }

        tscv = TimeSeriesSplit(n_splits=self.n_folds)
        cv_scores: list[float] = []
        best_iters: list[int] = []

        for fold_i, (tr_idx, va_idx) in enumerate(tscv.split(self.X_train), 1):
            X_tr_full = self.X_train.iloc[tr_idx]
            y_tr_full = self.y_train.iloc[tr_idx]
            X_va = self.X_train.iloc[va_idx]
            y_va = self.y_train.iloc[va_idx]
            if int(y_va.sum()) == 0:
                continue

            X_tr_bal, y_tr_bal = create_balanced_subset(
                X_tr_full, y_tr_full,
                neg_pos_ratio=self.neg_pos_ratio,
                random_state=RANDOM_STATE + fold_i,
            )

            dtrain = xgb.DMatrix(X_tr_bal[self.available_features], label=y_tr_bal)
            dval = xgb.DMatrix(X_va[self.available_features], label=y_va)
            model = xgb.train(
                params,
                dtrain,
                num_boost_round=1000,
                evals=[(dval, "eval")],
                early_stopping_rounds=v51.XGBOOST_CONFIG["early_stopping"],
                verbose_eval=False,
            )
            preds = model.predict(dval, iteration_range=(0, model.best_iteration))
            score = _score_probs_for_optuna(
                y_true=y_va.to_numpy().astype(int),
                y_prob=preds.astype(float),
                metric=self.metric,
                min_recall=self.min_recall,
            )
            cv_scores.append(score)
            best_iters.append(int(model.best_iteration))

        if not cv_scores:
            return 0.0
        mean_score = float(np.mean(cv_scores))
        # lightweight logging (avoid spam)
        trial.set_user_attr("mean_best_iter", float(np.mean(best_iters)) if best_iters else None)
        trial.set_user_attr("metric", self.metric)
        trial.set_user_attr("min_recall", self.min_recall)
        trial.set_user_attr("elapsed_s", time.time() - t0)
        return mean_score


# =============================================================================
# Stage 1 ensemble training (V54)
# =============================================================================


def stage2_train_ensemble_v54(X_train_xgb: pd.DataFrame,
                              X_test_xgb: pd.DataFrame,
                              y_train_acc: pd.Series,
                              y_test_acc: pd.Series,
                              available_features: list[str],
                              model_time: str,
                              *,
                              optuna_metric: str,
                              optuna_min_recall: float):
    """V54 Stage-1 trainer: same as V51.stage2_train_ensemble but configurable Optuna score."""
    print("=" * 80)
    print("V54 STAGE 2B: ACCIDENT — Optuna hyperparameter search (configurable objective)")
    print("=" * 80)
    print(f"Optuna metric: {optuna_metric} (min_recall={optuna_min_recall})")

    target_col = "ACCIDENT"
    n_base = ENSEMBLE_CONFIG["n_base_models"]
    neg_pos_ratio = ENSEMBLE_CONFIG["neg_pos_ratio"]
    n_trials = ENSEMBLE_CONFIG["optuna_trials"]

    if v51.MEMORIZE_MODE:
        class _FakeStudy:
            best_params = {
                "max_depth": 12, "learning_rate": 0.3, "min_child_weight": 1,
                "subsample": 1.0, "colsample_bytree": 1.0, "gamma": 0.0,
                "reg_alpha": 0.0, "reg_lambda": 0.0, "scale_pos_weight": 1.0
            }
            best_value = 1.0
        study = _FakeStudy()
        n_base = 1
    else:
        objective = BalancedSubsetObjectiveV54(
            X_train=X_train_xgb,
            y_train=y_train_acc,
            available_features=available_features,
            neg_pos_ratio=neg_pos_ratio,
            n_folds=v51.XGBOOST_CONFIG["cv_folds"],
            n_trials=n_trials,
            metric=optuna_metric,
            min_recall=optuna_min_recall,
        )
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
        )
        t0 = time.time()
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        print(f"\nOptuna done in {format_duration(time.time() - t0)}")
        print(f"Best Optuna score: {study.best_value:.4f}")
        for k, v in study.best_params.items():
            print(f"  {k}: {v}")

    print("=" * 80)
    print("V54 STAGE 2C: Train ensemble base models")
    print("=" * 80)

    best_params = {
        **study.best_params,
        "tree_method": "hist",
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "random_state": RANDOM_STATE,
        "nthread": N_THREADS,
    }

    dval = xgb.DMatrix(X_test_xgb, label=y_test_acc)

    if v51.MEMORIZE_MODE:
        X_all_xgb = pd.concat([X_train_xgb, X_test_xgb])
        y_all_acc = pd.concat([y_train_acc, y_test_acc])
        early_stop, num_rounds = None, 500
    else:
        X_all_xgb = X_train_xgb
        y_all_acc = y_train_acc
        early_stop, num_rounds = 100, 2000

    base_models: list[xgb.Booster] = []
    base_model_paths: list[str] = []
    for i in range(n_base):
        t0 = time.time()
        X_sub, y_sub = create_balanced_subset(
            X_all_xgb, y_all_acc,
            neg_pos_ratio=neg_pos_ratio,
            random_state=RANDOM_STATE + i * 1000,
        )
        dtrain_sub = xgb.DMatrix(X_sub, label=y_sub)

        if v51.MEMORIZE_MODE:
            model_i = xgb.train(best_params, dtrain_sub, num_boost_round=num_rounds, verbose_eval=False)
            model_i.best_iteration = num_rounds
        else:
            model_i = xgb.train(
                best_params,
                dtrain_sub,
                num_boost_round=num_rounds,
                evals=[(dval, "eval")],
                early_stopping_rounds=early_stop,
                verbose_eval=False,
            )

        base_models.append(model_i)
        path_i = os.path.join(
            v51.OUTPUT_DIR,
            f"xgboost-ensemble-base{i}_target=accident_model=XGBoost-version={model_time}.json",
        )
        model_i.save_model(path_i)
        base_model_paths.append(path_i)

        best_iter_i = getattr(model_i, "best_iteration", num_rounds)
        y_pred_i = model_i.predict(dval, iteration_range=(0, best_iter_i))
        aucpr_i = average_precision_score(y_test_acc, y_pred_i)
        print(f"  Base {i+1}/{n_base}: train={len(X_sub):,}, best_iter={best_iter_i}, "
              f"AUCPR={aucpr_i:.4f}, time={time.time() - t0:.1f}s")

    ensemble_preds = np.zeros(len(X_test_xgb))
    for m in base_models:
        best_iter = getattr(m, "best_iteration", num_rounds)
        ensemble_preds += m.predict(dval, iteration_range=(0, best_iter))
    ensemble_preds /= max(n_base, 1)

    y_pred_binary = (ensemble_preds >= 0.5).astype(int)
    roc_auc = roc_auc_score(y_test_acc, ensemble_preds)
    pr_auc = average_precision_score(y_test_acc, ensemble_preds)
    print(f"\nEnsemble ROC-AUC: {roc_auc:.4f}")
    print(f"Ensemble PR-AUC:  {pr_auc:.4f}")

    # Compatibility models + ensemble metadata
    base_models[0].save_model(os.path.join(v51.OUTPUT_DIR, f"xgboost-classifier_model=XGBoost-version={model_time}.json"))
    model_path_accident = os.path.join(
        v51.OUTPUT_DIR,
        f"xgboost-classifier-target=accident_model=XGBoost-version={model_time}.json",
    )
    base_models[0].save_model(model_path_accident)

    ensemble_meta = {
        "type": "balanced_bagging_ensemble",
        "n_base_models": n_base,
        "neg_pos_ratio": neg_pos_ratio,
        "base_model_paths": base_model_paths,
        "best_hyperparameters": dict(study.best_params),
        "ensemble_roc_auc": roc_auc,
        "ensemble_pr_auc": pr_auc,
        "memorize_mode": bool(v51.MEMORIZE_MODE),
        "optuna_metric": optuna_metric,
        "optuna_min_recall": float(optuna_min_recall),
    }
    with open(os.path.join(
        v51.OUTPUT_DIR,
        f"xgboost-ensemble-metadata_target=accident_version={model_time}.json",
    ), "w") as f:
        json.dump(ensemble_meta, f, indent=2)

    metrics_by_target = {
        "ACCIDENT": {
            "target_col": "ACCIDENT",
            "accuracy": accuracy_score(y_test_acc, y_pred_binary),
            "f1_weighted": f1_score(y_test_acc, y_pred_binary, average="weighted", zero_division=0),
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "training_strategy": "balanced_bagging",
        }
    }
    print("ACCIDENT ENSEMBLE RESULTS")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC:  {pr_auc:.4f}")
    print(confusion_matrix(y_test_acc, y_pred_binary))
    print(classification_report(y_test_acc, y_pred_binary, target_names=["No Accident", "Accident"], zero_division=0))

    xgb_metadata = {
        "model_name": "XGBoost_Classifier",
        "model_time": model_time,
        "target_cols": ["ACCIDENT"],
        "primary_target": "ACCIDENT",
        "best_cv_score": float(getattr(study, "best_value", 0.0)),
        "test_metrics": metrics_by_target.get("ACCIDENT", {}),
        "best_hyperparameters": dict(getattr(study, "best_params", {})),
        "feature_cols": available_features,
        "models_by_target": {"ACCIDENT": os.path.basename(model_path_accident)},
        "metrics_by_target": metrics_by_target,
        "best_hyperparameters_by_target": {"ACCIDENT": dict(getattr(study, "best_params", {}))},
        "ablation_settings": ABLATION_SETTINGS,
        "test_mode": TEST_MODE,
        "optuna_objective": {"metric": optuna_metric, "min_recall": float(optuna_min_recall)},
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    xgb_meta_path = os.path.join(v51.OUTPUT_DIR, f"xgboost-metadata_model=XGBoost-version={model_time}.json")
    with open(xgb_meta_path, "w") as f:
        json.dump(xgb_metadata, f, indent=2)
    print(f"[SUCCESS] XGBoost metadata saved: {os.path.basename(xgb_meta_path)}")

    s1_best_params = dict(getattr(study, "best_params", {}))
    return base_models, ensemble_preds, y_pred_binary, roc_auc, pr_auc, s1_best_params


# =============================================================================
# Feature-set resolution (delegated to V53 helpers)
# =============================================================================


def _resolve_stage1_features(available_features: list[str]) -> tuple[list[str], list[str]]:
    drop_set = set(STAGE1_DROP_FEATURES or [])
    present = set(available_features)
    dropped = [f for f in STAGE1_DROP_FEATURES if f in present]
    kept = [f for f in available_features if f not in drop_set]
    print("=" * 80)
    print("V54 STAGE-1 FEATURE PRUNING")
    print("=" * 80)
    print(f"  Available before prune : {len(available_features)}")
    print(f"  Dropped ({len(dropped)})         : {dropped}")
    print(f"  Kept for Stage 1        : {len(kept)}")
    return kept, dropped


def _resolve_stage2_features(stage1_features: list[str]) -> list[str]:
    print("=" * 80)
    print("V54 STAGE-2 FEATURE SELECTION")
    print("=" * 80)
    if STAGE2_FEATURES is None:
        print("  STAGE2_FEATURES = None  →  Stage 2 inherits Stage 1's set "
              f"({len(stage1_features)} features) + s1_score")
        return list(stage1_features)
    s1_set = set(stage1_features)
    kept = [f for f in STAGE2_FEATURES if f in s1_set]
    dropped = [f for f in STAGE2_FEATURES if f not in s1_set]
    if dropped:
        print(f"  WARNING: dropping Stage-2 features not in Stage-1 pool: {dropped}")
    print(f"  Stage 2 whitelist ({len(kept)} features) + s1_score")
    return kept


def _slice_for_stage2(X: pd.DataFrame, stage2_features: list[str]) -> pd.DataFrame:
    keep = [c for c in stage2_features if c in X.columns]
    if "s1_score" in X.columns and "s1_score" not in keep:
        keep = keep + ["s1_score"]
    return X[keep]


# =============================================================================
# Stage 2 training (V54) — same structure as V53 but configurable Optuna score
# =============================================================================


def _train_stage2_holdout_v54(*, X_train_xgb, X_test_xgb, y_train_acc, y_test_acc,
                              base_models, ensemble_preds, stage2_features,
                              s1_best_params, model_time, output_dir,
                              optuna_metric: str, optuna_min_recall: float):
    """Non-memorize Stage 2 training with configurable Optuna objective."""
    STAGE1_THRESHOLD = 0.5
    STAGE2_THRESHOLD = 0.5

    X_tr = X_train_xgb.reset_index(drop=True)
    y_tr = y_train_acc.reset_index(drop=True)
    stage1_features = list(X_tr.columns)
    neg_pos_ratio = ENSEMBLE_CONFIG["neg_pos_ratio"]

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
            m = xgb.train(oof_params, xgb.DMatrix(Xb, label=yb), num_boost_round=oof_rounds, verbose_eval=False)
            fold_probs += m.predict(xgb.DMatrix(X_tr.iloc[va_idx][stage1_features]))
        s1_tr_probs[va_idx] = fold_probs / oof_n_models

    s2_mask = s1_tr_probs >= STAGE1_THRESHOLD
    X_s2_full_tr = X_tr[s2_mask].copy()
    X_s2_full_tr["s1_score"] = s1_tr_probs[s2_mask]
    y_s2_tr = y_tr[s2_mask].copy()

    X_test_r = X_test_xgb.reset_index(drop=True)
    y_test_r = y_test_acc.reset_index(drop=True)
    test_s1_flag = ensemble_preds >= STAGE1_THRESHOLD
    X_s2_full_te = X_test_r[test_s1_flag].copy()
    X_s2_full_te["s1_score"] = ensemble_preds[test_s1_flag]
    y_s2_te = y_test_r[test_s1_flag].copy()

    X_s2_tr = _slice_for_stage2(X_s2_full_tr, stage2_features)
    X_s2_te = _slice_for_stage2(X_s2_full_te, stage2_features)

    n_tp_s2 = int(y_s2_tr.sum())
    n_fp_s2 = int((y_s2_tr == 0).sum())
    s2_spw = n_fp_s2 / max(n_tp_s2, 1)

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
    print("V54 STAGE 7B: Stage-2 Optuna (configurable objective)")
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
    return stage2_model, X_s2_tr, y_s2_tr, flagged_idx, s2_test_probs, y_test_np, dict(s2_study.best_params)


# =============================================================================
# Manifests
# =============================================================================


def write_v54_manifests(*, gnn_experiment_dir: str, model_time: str,
                        stage1_features: list[str], stage1_dropped: list[str],
                        stage2_features: list[str],
                        s1_thr: float, s2_thr: float,
                        roc_auc: float, pr_auc: float,
                        sweep_summary: dict,
                        v52_extra_features: list[str],
                        s1_best_params: dict,
                        s2_best_params: dict) -> None:
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
        "threshold_sweep": {
            "min_recall_constraint": float(MIN_RECALL_CONSTRAINT),
            "max_far_target": float(MAX_FAR_TARGET),
            "s1_grid": [float(x) for x in THRESHOLD_GRID_S1],
            "s2_grid": [float(x) for x in THRESHOLD_GRID_S2],
            "optimise_for": OPTIMISE_FOR,
            "min_precision_constraint": float(1.0 - MAX_FAR_TARGET),
            "best": {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v)
                     for k, v in sweep_summary["best"].items()},
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
        "based_on": "ablation_study_v53_xgboost_only",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    manifest_path = os.path.join(v51.OUTPUT_DIR, f"v54_xgboost_manifest_{model_time}.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"[V54] Manifest saved: {manifest_path}")

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
    exp_config_path = os.path.join(v51.OUTPUT_DIR, "experiment_config_xgb_v54.json")
    with open(exp_config_path, "w") as f:
        json.dump(exp_config, f, indent=2, default=str)
    print(f"[V54] Sim-runner config saved: {exp_config_path}")


# =============================================================================
# Pipeline
# =============================================================================


def _propagate_to_v52() -> None:
    v52.ADD_PK_HOUR_INTERACTIONS = bool(ADD_PK_HOUR_INTERACTIONS)
    v52.MEMORIZE_MODE = bool(MEMORIZE_MODE) or bool(v52.MEMORIZE_MODE)
    v51.MEMORIZE_MODE = bool(MEMORIZE_MODE) or bool(v51.MEMORIZE_MODE)
    if hasattr(v52, "_propagate_to_v51"):
        try:
            v52._propagate_to_v51()
        except Exception as e:
            print(f"[V54] v52._propagate_to_v51() failed: {e}")


def run_pipeline_v54(gnn_experiment_dir: str) -> bool:
    _propagate_to_v52()
    t_start = time.time()

    print("=" * 80)
    print("V54 XGBoost-only — V53 + configurable Optuna objectives")
    print(f"Output dir:      {v51.OUTPUT_DIR}")
    print(f"GNN experiment:  {gnn_experiment_dir}")
    print(f"MEMORIZE_MODE:   {bool(v51.MEMORIZE_MODE)}")
    print("Optuna objective config:")
    print(f"  Stage 1: metric={OPTUNA_METRIC_STAGE1}  min_recall={OPTUNA_MIN_RECALL_STAGE1}")
    print(f"  Stage 2: metric={OPTUNA_METRIC_STAGE2}  min_recall={OPTUNA_MIN_RECALL_STAGE2}")
    print("Threshold sweep config:")
    print(f"  optimise_for={OPTIMISE_FOR}  "
          f"min_recall={MIN_RECALL_CONSTRAINT}  "
          f"min_precision={1.0 - MAX_FAR_TARGET:.3f} (max_far={MAX_FAR_TARGET:.3f})")
    print("=" * 80)

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

    if v51.MEMORIZE_MODE:
        # Reuse V53's memorize stage2 trainer; it doesn't have Optuna anyway.
        (stage2_model, X_s2_tr, y_s2_tr, flagged_idx,
         s2_test_probs, y_test_np) = v53._train_stage2_memorize_v53(  # type: ignore[attr-defined]
            X_train_xgb=X_train_xgb, X_test_xgb=X_test_xgb,
            y_train_acc=y_train_acc, y_test_acc=y_test_acc,
            base_models=base_models, ensemble_preds=ensemble_preds,
            stage2_features=stage2_features,
            s1_best_params=s1_best_params,
            model_time=model_time, output_dir=v51.OUTPUT_DIR,
        )
        s2_best_params = {}
    else:
        with _wider_optuna_ranges(_S2_INT_RANGES, _S2_FLOAT_RANGES):
            (stage2_model, X_s2_tr, y_s2_tr, flagged_idx,
             s2_test_probs, y_test_np, s2_best_params) = _train_stage2_holdout_v54(
                X_train_xgb=X_train_xgb, X_test_xgb=X_test_xgb,
                y_train_acc=y_train_acc, y_test_acc=y_test_acc,
                base_models=base_models, ensemble_preds=ensemble_preds,
                stage2_features=stage2_features,
                s1_best_params=s1_best_params,
                model_time=model_time, output_dir=v51.OUTPUT_DIR,
                optuna_metric=OPTUNA_METRIC_STAGE2,
                optuna_min_recall=float(OPTUNA_MIN_RECALL_STAGE2),
            )
        stage7b_shap_stage2(stage2_model, X_s2_tr, y_s2_tr)

    # Threshold sweep — `OPTIMISE_FOR` selects what to maximise; `MIN_RECALL_CONSTRAINT`
    # and `MAX_FAR_TARGET` (=1-min_precision) act as feasibility constraints.
    S1_THR, S2_THR, sweep_summary = sweep_thresholds_v52(
        ensemble_preds=ensemble_preds,
        y_test_np=y_test_np,
        flagged_idx=flagged_idx,
        s2_test_probs=s2_test_probs,
        s1_grid=THRESHOLD_GRID_S1,
        s2_grid=THRESHOLD_GRID_S2,
        min_recall=float(MIN_RECALL_CONSTRAINT),
        max_far=float(MAX_FAR_TARGET),
        optimise_for=OPTIMISE_FOR,
    )

    write_v54_manifests(
        gnn_experiment_dir=gnn_experiment_dir,
        model_time=model_time,
        stage1_features=stage1_features,
        stage1_dropped=stage1_dropped,
        stage2_features=stage2_features,
        s1_thr=S1_THR, s2_thr=S2_THR,
        roc_auc=roc_auc, pr_auc=pr_auc,
        sweep_summary=sweep_summary,
        v52_extra_features=v52_extra_features,
        s1_best_params=s1_best_params,
        s2_best_params=s2_best_params,
    )

    print("\n" + "=" * 80)
    print(f"V54 XGBoost training complete in {format_duration(time.time() - t_start)}")
    print(f"All plots saved to:  {v51.VIZ_DIR}")
    print(f"All models saved to: {v51.OUTPUT_DIR}")
    print(f"Model time: {model_time}")
    print("Next step: ablation_study_v54_simulation_only.py "
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
        print(f"[V54] XGB_RUN_ID not set; using timestamp fallback: {xgb_run_id}")

    v51.OUTPUT_DIR = os.path.join(gnn_experiment_dir, f"xgboost_{xgb_run_id}")
    v51.VIZ_DIR = os.path.join(v51.OUTPUT_DIR, "visualizations")
    print(f"[V54] XGBoost artefacts will be saved under: {v51.OUTPUT_DIR}")
    return run_pipeline_v54(gnn_experiment_dir)


def _parse_csv_feats(s: str) -> list[str]:
    return [tok.strip() for tok in s.split(",") if tok.strip()]


def main() -> int:
    global MEMORIZE_MODE, ADD_PK_HOUR_INTERACTIONS
    global OPTUNA_METRIC_STAGE1, OPTUNA_METRIC_STAGE2
    global OPTUNA_MIN_RECALL_STAGE1, OPTUNA_MIN_RECALL_STAGE2
    global MIN_RECALL_CONSTRAINT, MAX_FAR_TARGET, OPTIMISE_FOR
    global STAGE1_DROP_FEATURES, STAGE2_FEATURES

    import argparse
    p = argparse.ArgumentParser(description="V54 XGBoost-only — configurable Optuna objectives.")
    p.add_argument("--gnn-experiment-dir", required=True)
    p.add_argument("--memorize-mode", action="store_true", default=False)
    p.add_argument("--no-pk-hour", action="store_true", default=False)
    p.add_argument("--min-recall", type=float, default=MIN_RECALL_CONSTRAINT,
                   help="Recall floor for threshold sweep (precision-first).")
    p.add_argument("--min-precision", type=float, default=None,
                   help="Precision floor for threshold sweep (recall-first). "
                        "Sets MAX_FAR_TARGET=1-min_precision. Use with --optimise-for=recall.")
    p.add_argument("--optimise-for", type=str, default=OPTIMISE_FOR,
                   choices=["precision", "recall", "f1", "f2"],
                   help="What the final S1/S2 threshold sweep maximises.")
    p.add_argument("--optuna-metric-s1", type=str, default=OPTUNA_METRIC_STAGE1,
                   choices=sorted(_ALLOWED_OPTUNA_METRICS))
    p.add_argument("--optuna-metric-s2", type=str, default=OPTUNA_METRIC_STAGE2,
                   choices=sorted(_ALLOWED_OPTUNA_METRICS))
    p.add_argument("--optuna-min-recall-s1", type=float, default=OPTUNA_MIN_RECALL_STAGE1)
    p.add_argument("--optuna-min-recall-s2", type=float, default=OPTUNA_MIN_RECALL_STAGE2)
    p.add_argument("--stage1-drop-feats", type=str, default=None)
    p.add_argument("--stage1-no-prune", action="store_true", default=False)
    p.add_argument("--stage2-feats", type=str, default=None)
    p.add_argument("--stage2-inherit-stage1", action="store_true", default=False)

    args = p.parse_args()

    if args.memorize_mode:
        MEMORIZE_MODE = True
        v51.MEMORIZE_MODE = True
    if args.no_pk_hour:
        ADD_PK_HOUR_INTERACTIONS = False
    MIN_RECALL_CONSTRAINT = float(args.min_recall)
    OPTIMISE_FOR = str(args.optimise_for)
    if args.min_precision is not None:
        MAX_FAR_TARGET = float(1.0 - float(args.min_precision))

    OPTUNA_METRIC_STAGE1 = str(args.optuna_metric_s1)
    OPTUNA_METRIC_STAGE2 = str(args.optuna_metric_s2)
    OPTUNA_MIN_RECALL_STAGE1 = float(args.optuna_min_recall_s1)
    OPTUNA_MIN_RECALL_STAGE2 = float(args.optuna_min_recall_s2)

    if args.stage1_no_prune:
        STAGE1_DROP_FEATURES = []
    elif args.stage1_drop_feats is not None:
        STAGE1_DROP_FEATURES = _parse_csv_feats(args.stage1_drop_feats)

    if args.stage2_inherit_stage1:
        STAGE2_FEATURES = None
    elif args.stage2_feats is not None:
        STAGE2_FEATURES = _parse_csv_feats(args.stage2_feats)

    return 0 if run_pipeline_v54(args.gnn_experiment_dir) else 1


if __name__ == "__main__":
    sys.exit(main())

