#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared trainer for the 2nd-article Layer-2 crash-classifier benchmarks (BL-C0..C5)
==================================================================================

Each benchmark replaces the proposed Phase-3 predictor-supervisor cascade with a
single alternative classifier, while reusing EVERYTHING upstream so the
comparison is apples-to-apples:

  * v51 `stage0_load_data_with_gnn` -> the proposed GNN-LSTM's forecasted
    corridor state (the identical u_{i,t} features the cascade sees),
  * V52 feature engineering (pk x hour interactions) replayed
    exactly as the cascade does,
  * v51 `stage1_feature_prep` + v54 `_resolve_stage1_features` -> identical
    feature matrix / train-test split / dropped-feature set,
  * v55 `sweep_stage1_threshold` -> the same constrained operating-point search
    (maximise precision s.t. recall >= floor) as the proposed method.

The ONLY thing that changes is the classifier head (built from
`src.models.benchmark_classifiers`). Artefacts:
  * `benchmark-classifier_version=<model_time>.joblib`  (the fitted model),
  * `benchmark_classifier_manifest_<model_time>.json`   (token, features,
     threshold, replayed-FE experiments),
  * `experiment_config_xgb_<version>.json`              (so the generic sim
     runner auto-detects ablation settings + model_time).

Each `ablation_study_v6X_xgboost_only.py` is a thin wrapper calling
`run_xgboost_only_for(version, classifier_token)`.

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

import src.training.ablation_study_v51_xgboost_only as v51
import src.training.ablation_study_v54_xgboost_only as v54
from src.training.ablation_study_v51_xgboost_only import (
    format_duration,
    stage0_load_data_with_gnn,
    stage1_feature_prep,
)
from src.training.ablation_study_v52_xgboost_only import (
    add_pk_hour_interactions,
    EVENING_RUSH_HOURS,
    HOT_ZONE_PKS,
)
from src.training.ablation_study_v54_xgboost_only import _resolve_stage1_features
from src.training.ablation_study_v55_xgboost_only import sweep_stage1_threshold
from src.training.ablation_study_v56_xgboost_only import THRESHOLD_GRID
from src.models.benchmark_classifiers import build_classifier


# Operating-point contract — identical to the proposed precision-first method
# (v59/v60): maximise precision subject to recall >= 0.40.
OPTIMISE_FOR = "precision"
MIN_RECALL_CONSTRAINT = 0.40
MIN_PRECISION_CONSTRAINT = 0.0

# Replayed FE, inherited from V54 (same as the cascade).
ADD_PK_HOUR_INTERACTIONS = bool(v54.ADD_PK_HOUR_INTERACTIONS)


def _propagate() -> None:
    v54.ADD_PK_HOUR_INTERACTIONS = bool(ADD_PK_HOUR_INTERACTIONS)
    if hasattr(v54, "_propagate_to_v52"):
        try:
            v54._propagate_to_v52()
        except Exception as e:
            print(f"[BENCH-CLF] v54._propagate_to_v52() failed: {e}")


def _write_manifests(*, version, token, gnn_experiment_dir, model_time,
                     stage1_features, stage1_dropped, threshold, sweep_summary,
                     roc_auc, pr_auc, v52_extra_features, classifier_meta) -> None:
    def _coerce(v):
        if isinstance(v, (np.floating, np.integer)):
            return float(v) if isinstance(v, np.floating) else int(v)
        return v

    experiments = {
        "pk_hour_interactions": bool(ADD_PK_HOUR_INTERACTIONS),
        "v52_extra_features": list(v52_extra_features),
        "hot_zone_pks": list(HOT_ZONE_PKS),
        "evening_rush_hours": list(EVENING_RUSH_HOURS),
        "benchmark_layer": "L2_classifier",
        "classifier_token": token,
        "stage1_dropped_features": list(stage1_dropped),
        "threshold_sweep": {
            "optimise_for": OPTIMISE_FOR,
            "min_recall_constraint": float(MIN_RECALL_CONSTRAINT),
            "min_precision_constraint": float(MIN_PRECISION_CONSTRAINT),
            "picked_threshold": float(threshold),
            "feasible": bool(sweep_summary.get("feasible", False)),
            "best": {k: _coerce(v) for k, v in sweep_summary.items() if k != "feasible"},
        },
        "classifier_meta": classifier_meta,
    }

    manifest = {
        "pipeline_version": f"{version}_xgboost_only",
        "benchmark_version": version,
        "benchmark_layer": "L2_classifier",
        "classifier_token": token,
        "model_time": model_time,
        "output_dir": v51.OUTPUT_DIR,
        "gnn_experiment_dir": gnn_experiment_dir,
        "stage1_features": stage1_features,
        "stage2_features": [],
        "stage1_threshold": float(threshold),
        "stage2_threshold": 0.5,
        "stage2_disabled": True,
        "min_recall_constraint": float(MIN_RECALL_CONSTRAINT),
        "min_precision_constraint": float(MIN_PRECISION_CONSTRAINT),
        "test_metrics_stage1": {"roc_auc": float(roc_auc), "pr_auc": float(pr_auc)},
        "train_dates": {"start": v51.TRAIN_START_DATE, "end": v51.TRAIN_END_DATE},
        "sim_dates": {"start": v51.SIM_START_DATE, "end": v51.SIM_END_DATE},
        "pk_range": {"min": v51.PK_MIN, "max": v51.PK_MAX},
        "time_resolution": v51.TIME_RESOLUTION,
        "ablation_settings": v51.ABLATION_SETTINGS,
        "memorize_mode": bool(v51.MEMORIZE_MODE),
        # provenance: observed traffic substituted for the forecast (real-time upper bound)
        "observed_traffic": os.environ.get("BENCH_OBSERVED_TRAFFIC", "0") == "1",
        # chronological split + FE-baseline cutoff (sim must resolve the same value)
        "train_test_boundary": str(getattr(v51, "TRAIN_TEST_BOUNDARY", None)),
        "experiments": experiments,
        "based_on": "benchmark_classifier_common",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    mpath = os.path.join(v51.OUTPUT_DIR,
                         f"benchmark_classifier_manifest_{model_time}.json")
    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"[BENCH-CLF] Manifest saved: {mpath}")

    exp_config = {
        "model_time": model_time,
        "test_mode": v51.TEST_MODE,
        "memorize_mode": bool(v51.MEMORIZE_MODE),
        # provenance: observed traffic substituted for the forecast (real-time upper bound)
        "observed_traffic": os.environ.get("BENCH_OBSERVED_TRAFFIC", "0") == "1",
        "ablation_settings": v51.ABLATION_SETTINGS,
        "pipeline_version": f"{version}_xgboost_only",
        "benchmark_layer": "L2_classifier",
        "classifier_token": token,
        "gnn_experiment_dir": gnn_experiment_dir,
        "stage1_threshold": float(threshold),
        "stage2_threshold": 0.5,
        "stage1_features": stage1_features,
        "stage2_features": [],
        "train_dates": {"start": v51.TRAIN_START_DATE, "end": v51.TRAIN_END_DATE},
        "sim_dates": {"start": v51.SIM_START_DATE, "end": v51.SIM_END_DATE},
        "experiments": experiments,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    for name in (f"experiment_config_xgb_{version}.json",):
        with open(os.path.join(v51.OUTPUT_DIR, name), "w") as f:
            json.dump(exp_config, f, indent=2, default=str)
    print(f"[BENCH-CLF] Sim-runner config saved: experiment_config_xgb_{version}.json")


def run_pipeline_classifier(version: str, token: str, gnn_experiment_dir: str) -> bool:
    from sklearn.metrics import average_precision_score, roc_auc_score
    _propagate()
    t0 = time.time()

    print("=" * 80)
    print(f"{version.upper()} Layer-2 classifier benchmark — token={token}")
    print(f"Output dir:     {v51.OUTPUT_DIR}")
    print(f"GNN experiment: {gnn_experiment_dir}")
    print(f"Operating point: maximise {OPTIMISE_FOR}  s.t. recall >= "
          f"{MIN_RECALL_CONSTRAINT}")
    print("=" * 80)

    if bool(v51.MEMORIZE_MODE):
        raise NotImplementedError("Layer-2 benchmarks do not implement MEMORIZE_MODE.")

    os.makedirs(v51.OUTPUT_DIR, exist_ok=True)
    os.makedirs(v51.VIZ_DIR, exist_ok=True)

    df_full, fe_only_cols = stage0_load_data_with_gnn(gnn_experiment_dir)

    v52_extra_features: list[str] = []
    if ADD_PK_HOUR_INTERACTIONS:
        df_full, v52_extra_features = add_pk_hour_interactions(df_full)
        fe_only_cols = list(fe_only_cols) + [f for f in v52_extra_features
                                             if f not in fe_only_cols]

    time_res_min = v51._get_time_resolution_minutes()

    (df_train, X_train, X_test, y_train, y_test,
     available_features, model_time) = stage1_feature_prep(df_full, fe_only_cols)
    del df_train
    gc.collect()

    stage1_features, stage1_dropped = _resolve_stage1_features(available_features)
    if stage1_dropped:
        X_train = X_train[stage1_features]
        X_test = X_test[stage1_features]

    # ── H2 ablation hook: COVARIATES-ONLY ─────────────────────────────────
    # When BENCH_COVARIATES_ONLY is set, strip every forecast-derived feature
    # (the 4 substituted traffic channels + the V21 FE columns engineered from
    # them). pk×hour interaction extras are covariate-based and are KEPT. The
    # reduced list lands in the manifest, so the simulation follows it
    # automatically. This tests whether the traffic-forecast stage carries
    # information beyond its own covariate inputs.
    if os.environ.get("BENCH_COVARIATES_ONLY", "").lower() in ("1", "true", "yes"):
        # Explicit KEEP list = everything knowable WITHOUT any traffic forecast:
        # calendar (+cyclic encodings), location id + pk-hour interactions,
        # historical crash rates, and weather covariates. Everything else in
        # the stage-1 set is derived from the (forecast-substituted) traffic
        # channels and is removed.
        covariate_keep = {
            "pk", "anyo", "mes", "dia", "diaSem", "diaSem_sin", "diaSem_cos",
            "hor", "hor_sin", "hor_cos", "5min", "is_peak_hour",
            "pk_x_hor", "pk_x_hor_sin", "pk_x_hor_cos",
            "pk_crash_rate", "pk_crash_rate_log", "pk_crash_rate_mob",
            "low_visibility_proxy", "wind_gust_excess",
        }
        # Engineered covariate columns (BENCH_COV_ENGINEERED) are covariate-
        # derived by construction and carry no traffic information, so they are
        # admissible in this arm; they are prefixed rather than listed.
        covariate_keep |= {f for f in stage1_features if f.startswith("cov_")}
        removed = sorted(f for f in stage1_features if f not in covariate_keep)
        stage1_features = [f for f in stage1_features if f in covariate_keep]
        stage1_dropped = list(stage1_dropped) + removed
        X_train = X_train[stage1_features]
        X_test = X_test[stage1_features]
        print(f"[H2-ABLATION] COVARIATES-ONLY: removed {len(removed)} "
              f"traffic-derived features -> {len(stage1_features)} covariates remain")
        print(f"[H2-ABLATION] kept: {sorted(stage1_features)}")
        print(f"[H2-ABLATION] removed: {removed}")

    print(f"[BENCH-CLF] Fitting classifier '{token}' on {X_train.shape} "
          f"(pos={int(y_train.sum())})...")
    # With AP7_CLF_TRIALS>0 every searchable arm gets the SAME Optuna budget,
    # scored on a chronological tail of the TRAINING window only. Unset, this is
    # the previous behaviour exactly: one fit on library defaults.
    from src.models.classifier_search import search_and_build
    _seed = int(os.environ.get("AP7_CLF_SEED", "42"))
    clf, best_params, n_trials = search_and_build(token, X_train, y_train,
                                                  seed=_seed)
    clf = clf.fit(X_train, y_train)
    clf.save(v51.OUTPUT_DIR, model_time)
    try:
        import json as _json
        with open(os.path.join(v51.OUTPUT_DIR,
                               f"classifier_search_{model_time}.json"), "w") as _fh:
            _json.dump({"token": token, "n_trials": n_trials,
                        "best_params": best_params}, _fh, indent=1, default=str)
    except Exception as _exc:
        print(f"[WARN] could not record search result: {_exc}")

    y_test_np = y_test.reset_index(drop=True).values
    proba = clf.predict_proba(X_test)
    pr_auc = float(average_precision_score(y_test_np, proba)) if y_test_np.sum() else 0.0
    try:
        roc_auc = float(roc_auc_score(y_test_np, proba)) if y_test_np.sum() else 0.5
    except Exception:
        roc_auc = 0.5
    print(f"[BENCH-CLF] Test PR-AUC={pr_auc:.4f}  ROC-AUC={roc_auc:.4f}")

    threshold, sweep_summary = sweep_stage1_threshold(
        y_true=y_test_np, y_prob=proba,
        optimise_for=OPTIMISE_FOR,
        min_recall=float(MIN_RECALL_CONSTRAINT),
        min_precision=float(MIN_PRECISION_CONSTRAINT),
        grid=THRESHOLD_GRID,
    )

    classifier_meta = {k: (float(v) if isinstance(v, (np.floating, float)) else v)
                       for k, v in vars(clf).items()
                       if k in ("C", "cv_aucpr", "params", "val_aucpr", "s1_gate")}

    _write_manifests(
        version=version, token=token, gnn_experiment_dir=gnn_experiment_dir,
        model_time=model_time, stage1_features=stage1_features,
        stage1_dropped=stage1_dropped, threshold=threshold,
        sweep_summary=sweep_summary, roc_auc=roc_auc, pr_auc=pr_auc,
        v52_extra_features=v52_extra_features, classifier_meta=classifier_meta,
    )

    print("\n" + "=" * 80)
    print(f"{version.upper()} ({token}) classifier benchmark complete in "
          f"{format_duration(time.time() - t0)}")
    print(f"Picked threshold: {threshold:.4f}")
    print(f"Artefacts in: {v51.OUTPUT_DIR}")
    print("=" * 80)
    return True


def run_xgboost_only_for(version: str, token: str) -> bool:
    """Entry point the thin v6X xgboost wrappers + bash runner call."""
    gnn_experiment_dir = (
        os.environ.get("PREVIOUS_EXPERIMENT_DIR", "")
        or os.environ.get("GNN_EXPERIMENT_DIR", "")
        or getattr(v51, "GNN_EXPERIMENT_DIR", "")
    )
    if not gnn_experiment_dir or not os.path.isdir(gnn_experiment_dir):
        print(f"[ERROR] PREVIOUS_EXPERIMENT_DIR not set or invalid: {gnn_experiment_dir!r}")
        sys.exit(1)

    xgb_run_id = (os.environ.get("XGB_RUN_ID", "") or os.environ.get("JOB_ID", "")
                  or os.environ.get("SLURM_JOB_ID", "")
                  or datetime.now().strftime("%Y%m%d_%H%M%S"))
    v51.OUTPUT_DIR = os.path.join(gnn_experiment_dir, f"xgboost_{xgb_run_id}")
    v51.VIZ_DIR = os.path.join(v51.OUTPUT_DIR, "visualizations")
    print(f"[BENCH-CLF] Artefacts will be saved under: {v51.OUTPUT_DIR}")
    return run_pipeline_classifier(version, token, gnn_experiment_dir)
