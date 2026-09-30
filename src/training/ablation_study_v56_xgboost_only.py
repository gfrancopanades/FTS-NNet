#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V56 XGBoost-only: single-stage AUCPR pipeline (no Stage-2 FP filter)
=====================================================================

V56 strips the two-stage architecture down to its Stage-1 core:

  1. Stage 1 trained with AUCPR Optuna (same trainer as V54/V55).
  2. NO Stage 2 — the FP-filter model is not trained at all.
  3. Single test-set threshold sweep on the Stage-1 ensemble probabilities,
     maximising PRECISION subject to a min-recall floor (the floor keeps
     the sweep from degenerating into a 1-TP / precision=1.0 corner).

Rationale: V55 showed that the Stage-2 distribution-shift fix helps, but we
want a clean single-stage baseline trained and selected purely on AUCPR /
precision to quantify what Stage 2 actually buys us.

All knobs are overridable via module attrs (or the bash runner's
--optimise-for / --min-recall / --min-precision / --optuna-metric-s1 flags).

Based on: ablation_study_v55_xgboost_only (Stage-2 machinery removed).

Author: Gerard Franco
Date:   June 2026
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

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import src.training.ablation_study_v51_xgboost_only as v51
import src.training.ablation_study_v54_xgboost_only as v54
import src.training.ablation_study_v55_xgboost_only as v55
from src.training.ablation_study_v51_xgboost_only import (
    format_duration,
    stage0_load_data_with_gnn,
    stage1_feature_prep,
    stage5_feature_importance_shap,
    stage6_feature_space_overlap,
    _S1_FLOAT_RANGES,
    _S1_INT_RANGES,
    _wider_optuna_ranges,
)
from src.training.ablation_study_v52_xgboost_only import (
    add_pk_hour_interactions,
    EVENING_RUSH_HOURS,
    HOT_ZONE_PKS,
)
from src.training.ablation_study_v54_xgboost_only import (
    stage2_train_ensemble_v54,
    _resolve_stage1_features,
    _ALLOWED_OPTUNA_METRICS,
)
from src.training.ablation_study_v55_xgboost_only import sweep_stage1_threshold


# =============================================================================
# V56 CONFIGURATION
# =============================================================================

PIPELINE_VERSION = "v56_xgboost_only"

# Inherit V54/V55 feature pruning + experimental knobs.
STAGE1_DROP_FEATURES: list[str] = list(v54.STAGE1_DROP_FEATURES)
ADD_PK_HOUR_INTERACTIONS: bool = v54.ADD_PK_HOUR_INTERACTIONS
MEMORIZE_MODE: bool = bool(v51.MEMORIZE_MODE)

# V56 recipe — AUCPR Optuna for the (single) training stage.
OPTUNA_METRIC_STAGE1: str = "aucpr"
OPTUNA_MIN_RECALL_STAGE1: float = 0.30  # unused when metric=aucpr, kept for compat
# Kept so the bash runner's --optuna-metric-s2 flag is a harmless no-op.
OPTUNA_METRIC_STAGE2: str = "aucpr"
OPTUNA_MIN_RECALL_STAGE2: float = 0.30

# Single-stage (final) threshold sweep — maximise PRECISION under a
# min-recall floor. Without the floor, pure precision maximisation
# degenerates to the highest threshold with a single TP.
OPTIMISE_FOR: str = "precision"
MIN_RECALL_CONSTRAINT: float = 0.30
MIN_PRECISION_CONSTRAINT: float = 0.0
MAX_FAR_TARGET: float = 1.0 - MIN_PRECISION_CONSTRAINT  # kept for runner compat

# Fine sweep grid (same shape as V55's Stage-1 OOF grid: ~0.005 steps in the
# low tail, 0.025 elsewhere).
THRESHOLD_GRID = np.unique(np.round(np.concatenate([
    np.arange(0.005, 0.05 + 1e-9, 0.005),
    np.arange(0.05, 0.50 + 1e-9, 0.025),
    np.arange(0.50, 1.00 + 1e-9, 0.025),
]), 4))


# =============================================================================
# Manifests
# =============================================================================

def write_v56_manifests(*, gnn_experiment_dir: str, model_time: str,
                        stage1_features: list[str], stage1_dropped: list[str],
                        s1_thr: float,
                        roc_auc: float, pr_auc: float,
                        sweep_summary: dict,
                        v52_extra_features: list[str],
                        s1_best_params: dict) -> None:
    def _coerce(v):
        if isinstance(v, (np.floating, np.integer)):
            return float(v) if isinstance(v, np.floating) else int(v)
        return v

    experiments = {
        "pk_hour_interactions": bool(ADD_PK_HOUR_INTERACTIONS),
        "v52_extra_features": list(v52_extra_features),
        "stage1_dropped_features": list(stage1_dropped),
        "stage1_drop_request": list(STAGE1_DROP_FEATURES),
        "single_stage": True,
        "optuna_objective_stage1": {
            "metric": OPTUNA_METRIC_STAGE1,
            "min_recall": float(OPTUNA_MIN_RECALL_STAGE1),
        },
        "threshold_sweep": {
            "optimise_for": OPTIMISE_FOR,
            "min_recall_constraint": float(MIN_RECALL_CONSTRAINT),
            "min_precision_constraint": float(MIN_PRECISION_CONSTRAINT),
            "picked_threshold": float(s1_thr),
            "feasible": bool(sweep_summary.get("feasible", False)),
            "grid": [float(x) for x in THRESHOLD_GRID],
            "best": {k: _coerce(v) for k, v in sweep_summary.items()
                     if k != "feasible"},
        },
        "hot_zone_pks": list(HOT_ZONE_PKS),
        "evening_rush_hours": list(EVENING_RUSH_HOURS),
        "stage1_best_params": dict(s1_best_params),
    }

    manifest = {
        "pipeline_version": PIPELINE_VERSION,
        "model_time": model_time,
        "output_dir": v51.OUTPUT_DIR,
        "gnn_experiment_dir": gnn_experiment_dir,
        "available_features": stage1_features,
        "stage1_features": stage1_features,
        "stage2_features": [],
        "stage1_threshold": float(s1_thr),
        # Stage-2 threshold kept in the schema (sims read it) but inert.
        "stage2_threshold": 0.5,
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
        "stage2_disabled": True,
        "experiments": experiments,
        "based_on": "ablation_study_v55_xgboost_only",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    manifest_path = os.path.join(v51.OUTPUT_DIR, f"v56_xgboost_manifest_{model_time}.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"[V56] Manifest saved: {manifest_path}")

    exp_config = {
        "model_time": model_time,
        "test_mode": v51.TEST_MODE,
        "memorize_mode": bool(v51.MEMORIZE_MODE),
        "ablation_settings": v51.ABLATION_SETTINGS,
        "pipeline_version": PIPELINE_VERSION,
        "gnn_experiment_dir": gnn_experiment_dir,
        "stage1_threshold": float(s1_thr),
        "stage2_threshold": 0.5,
        "available_features": stage1_features,
        "stage1_features": stage1_features,
        "stage2_features": [],
        "train_dates": {"start": v51.TRAIN_START_DATE, "end": v51.TRAIN_END_DATE},
        "sim_dates": {"start": v51.SIM_START_DATE, "end": v51.SIM_END_DATE},
        "experiments": experiments,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    exp_config_path = os.path.join(v51.OUTPUT_DIR, "experiment_config_xgb_v56.json")
    with open(exp_config_path, "w") as f:
        json.dump(exp_config, f, indent=2, default=str)
    print(f"[V56] Sim-runner config saved: {exp_config_path}")


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
            print(f"[V56] v54._propagate_to_v52() failed: {e}")


def run_pipeline_v56(gnn_experiment_dir: str) -> bool:
    _propagate_to_v54()
    t_start = time.time()

    print("=" * 80)
    print("V56 XGBoost-only — single-stage AUCPR pipeline (no Stage-2 FP filter)")
    print(f"Output dir:      {v51.OUTPUT_DIR}")
    print(f"GNN experiment:  {gnn_experiment_dir}")
    print(f"MEMORIZE_MODE:   {bool(v51.MEMORIZE_MODE)}")
    print("Optuna objective config:")
    print(f"  Stage 1: metric={OPTUNA_METRIC_STAGE1}  min_recall={OPTUNA_MIN_RECALL_STAGE1}")
    print("Single-stage threshold sweep config:")
    print(f"  optimise_for={OPTIMISE_FOR}  "
          f"min_recall={MIN_RECALL_CONSTRAINT}  "
          f"min_precision={MIN_PRECISION_CONSTRAINT}")
    print("  Stage 2: DISABLED (single-stage pipeline)")
    print("=" * 80)

    if bool(v51.MEMORIZE_MODE):
        raise NotImplementedError("V56 does not yet implement MEMORIZE_MODE (no holdout pool).")

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

    # Single-stage threshold sweep on the test-set ensemble probabilities.
    # Reuses V55's constrained sweep helper (unchanged, imported).
    print("=" * 80)
    print("V56 FINAL SWEEP — single-stage threshold sweep for PRECISION")
    print("=" * 80)
    print(f"optimise_for={OPTIMISE_FOR}  "
          f"min_recall={MIN_RECALL_CONSTRAINT}  min_precision={MIN_PRECISION_CONSTRAINT}")
    y_test_np = y_test_acc.reset_index(drop=True).values
    S1_THR, sweep_summary = sweep_stage1_threshold(
        y_true=y_test_np, y_prob=ensemble_preds,
        optimise_for=OPTIMISE_FOR,
        min_recall=float(MIN_RECALL_CONSTRAINT),
        min_precision=float(MIN_PRECISION_CONSTRAINT),
        grid=THRESHOLD_GRID,
    )

    write_v56_manifests(
        gnn_experiment_dir=gnn_experiment_dir,
        model_time=model_time,
        stage1_features=stage1_features,
        stage1_dropped=stage1_dropped,
        s1_thr=S1_THR,
        roc_auc=roc_auc, pr_auc=pr_auc,
        sweep_summary=sweep_summary,
        v52_extra_features=v52_extra_features,
        s1_best_params=s1_best_params,
    )

    print("\n" + "=" * 80)
    print(f"V56 XGBoost training complete in {format_duration(time.time() - t_start)}")
    print(f"All plots saved to:  {v51.VIZ_DIR}")
    print(f"All models saved to: {v51.OUTPUT_DIR}")
    print(f"Model time: {model_time}")
    print(f"Single-stage threshold picked: {S1_THR:.4f}")
    print("Next step: ablation_study_v56_simulation_only.py "
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
        print(f"[V56] XGB_RUN_ID not set; using timestamp fallback: {xgb_run_id}")

    v51.OUTPUT_DIR = os.path.join(gnn_experiment_dir, f"xgboost_{xgb_run_id}")
    v51.VIZ_DIR = os.path.join(v51.OUTPUT_DIR, "visualizations")
    print(f"[V56] XGBoost artefacts will be saved under: {v51.OUTPUT_DIR}")
    return run_pipeline_v56(gnn_experiment_dir)


def main() -> int:
    global MEMORIZE_MODE, ADD_PK_HOUR_INTERACTIONS
    global OPTUNA_METRIC_STAGE1, OPTUNA_MIN_RECALL_STAGE1
    global MIN_RECALL_CONSTRAINT, MIN_PRECISION_CONSTRAINT, MAX_FAR_TARGET, OPTIMISE_FOR

    import argparse
    p = argparse.ArgumentParser(description="V56 XGBoost-only — single-stage AUCPR pipeline.")
    p.add_argument("--gnn-experiment-dir", required=True)
    p.add_argument("--memorize-mode", action="store_true", default=False)
    p.add_argument("--no-pk-hour", action="store_true", default=False)

    # Optuna metric / floors
    p.add_argument("--optuna-metric-s1", type=str, default=OPTUNA_METRIC_STAGE1,
                   choices=sorted(_ALLOWED_OPTUNA_METRICS))
    p.add_argument("--optuna-min-recall-s1", type=float, default=OPTUNA_MIN_RECALL_STAGE1)

    # Single-stage threshold sweep
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
    OPTUNA_MIN_RECALL_STAGE1 = float(args.optuna_min_recall_s1)

    MIN_RECALL_CONSTRAINT = float(args.min_recall)
    MIN_PRECISION_CONSTRAINT = float(args.min_precision)
    MAX_FAR_TARGET = float(1.0 - MIN_PRECISION_CONSTRAINT)
    OPTIMISE_FOR = str(args.optimise_for)

    return 0 if run_pipeline_v56(args.gnn_experiment_dir) else 1


if __name__ == "__main__":
    sys.exit(main())
