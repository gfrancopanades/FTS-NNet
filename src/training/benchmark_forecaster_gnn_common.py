#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Shared trainer for the 2nd-article Layer-1 forecasting benchmarks (BL-F1..F6)
=============================================================================

Every benchmark forecaster is trained with the SAME machinery as the main
GNN-LSTM so that downstream crash-prediction differences are attributable to
the forecaster, not the pipeline:

  * V17's tight disjoint training windows + window-aware data prep / sequence
    builder (imported, which monkeypatches `v5.prepare_gnn_data` /
    `v5.create_gnn_sequences` for us),
  * V14's RMSE-objective Optuna search and checkpointing
    (`v14.train_gnn_model`),
  * the same hyperparameter space (`v5.get_gnn_hyperparams`).

The ONLY swap is the model class: we point `v5.SpatioTemporalGNN_LSTM_V5`
(the symbol V14's objective instantiates) at the requested benchmark
`LightningModule`. After training we stamp the saved metadata with an
`architecture` token so the arch-aware `load_pretrained_gnn_model` rebuilds the
right class in the XGBoost and simulation stages.

Each `ablation_study_v6X_gnn_only.py` is a thin wrapper that calls
`run_benchmark_gnn(version=..., architecture=...)`.

Author: Gerard Franco
Date:   June 2026
"""

from __future__ import annotations

import gc
import json
import logging
import os
import sys
import time
import warnings
from datetime import datetime

import optuna
import torch

from src.training import ablation_study_v5_gnn_only as v5
from src.training import ablation_study_v14_gnn_only as v14
# Importing v17 applies its window-aware patches to v5 (prepare_gnn_data /
# create_gnn_sequences) and inherits V14's training config.
from src.training import ablation_study_v17_gnn_only as v17
from src.models.benchmark_forecasters import FORECASTER_REGISTRY


# Reuse V17's disjoint training windows so every benchmark sees the exact same
# data slice as the main V17 GNN run (fair comparison).
TRAIN_WINDOWS = list(v17.TRAIN_WINDOWS)


def build_experiment_name(version: str) -> str:
    prefix = v5.get_experiment_prefix()
    resolution = v5.get_time_resolution_token()
    job_id = v5.get_job_id()
    return f"{version}_gnn_{prefix}_{resolution}_{job_id}"


def _inject_architecture_metadata(output_dir: str, model_time: str,
                                  architecture: str, version: str,
                                  model_name: str) -> None:
    """Stamp the GNN metadata with the benchmark architecture token.

    `load_pretrained_gnn_model` reads `architecture` to pick the right class;
    without this the loader would default to the V5 GNN-LSTM and the state_dict
    load would fail.
    """
    path = os.path.join(output_dir,
                        f"best-model-metadata_model=GNN-version={model_time}.json")
    if not os.path.exists(path):
        print(f"[WARN] Metadata not found for architecture stamping: {path}")
        return
    with open(path) as f:
        meta = json.load(f)
    meta["architecture"] = architecture
    meta["model_name"] = model_name
    meta["benchmark_version"] = version
    meta["benchmark_paper"] = "2nd article (Layer-1 forecasting)"
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[ARCH] Stamped metadata with architecture={architecture} ({path})")


def run_benchmark_gnn(version: str, architecture: str, title: str) -> bool:
    """Train one Layer-1 benchmark forecaster end-to-end.

    version       e.g. "v62"
    architecture  registry token, e.g. "geolstm" (see FORECASTER_REGISTRY)
    title         human-readable banner string
    """
    if architecture not in FORECASTER_REGISTRY:
        raise KeyError(f"Unknown architecture {architecture!r}; "
                       f"known: {sorted(FORECASTER_REGISTRY)}")

    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
    logging.getLogger("lightning_fabric").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore")

    os.environ["LOCAL_RANK"] = "0"
    if torch.cuda.is_available():
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

    # Swap the model class V14's objective instantiates.
    model_cls = FORECASTER_REGISTRY[architecture]
    v5.SpatioTemporalGNN_LSTM_V5 = model_cls

    experiment_name = build_experiment_name(version)
    output_dir = os.path.join(v5.OUTPUT_BASE_DIR, experiment_name)
    os.makedirs(output_dir, exist_ok=True)
    model_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("\n" + "=" * 80)
    print(f"ABLATION STUDY {version.upper()} — {title}")
    print("=" * 80)
    print(f"Experiment:    {experiment_name}")
    print(f"Output:        {output_dir}")
    print(f"Architecture:  {architecture}  ({model_cls.__name__})")
    print(f"Test Mode:     {v5.TEST_MODE}")
    print(f"GPU Available: {torch.cuda.is_available()}")
    print(f"\n[NOTE] {version} trains on {len(TRAIN_WINDOWS)} disjoint windows "
          f"(same as V17):")
    for s, e in TRAIN_WINDOWS:
        print(f"         - {s} -> {e}")
    print("[NOTE] Same V14 RMSE objective / Optuna search / windows as the main "
          "GNN-LSTM; only the forecaster architecture differs.")
    print(f"       epochs={v5.GNN_CONFIG['epochs']} patience={v5.GNN_CONFIG['patience']} "
          f"trials={v5.GNN_CONFIG['trials']}")
    print("=" * 80)
    sys.stdout.flush()

    v5.print_ablation_config()

    config = {
        "experiment_name": experiment_name,
        "model_time": model_time,
        "architecture": architecture,
        "benchmark_version": version,
        "test_mode": v5.TEST_MODE,
        "ablation_settings": v5.ABLATION_SETTINGS,
        "gnn_config": v5.GNN_CONFIG,
        "train_windows": [{"start": s, "end": e} for s, e in TRAIN_WINDOWS],
        "hyperparameter_search_space": v5.get_gnn_hyperparams(),
        "pipeline_version": f"{version}_gnn_benchmark_{architecture}",
        "next_step": "Run an xgboost-only version with "
                     "PREVIOUS_EXPERIMENT_DIR pointing to this output",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    for name in ("experiment_config.json",
                 f"experiment_config_{version}.json"):
        with open(os.path.join(output_dir, name), "w") as f:
            json.dump(config, f, indent=2)

    total_start = time.time()
    stage_times = {}

    print("\n" + "=" * 80)
    print("STAGE 0: DATA LOADING")
    print("=" * 80)
    stage0_start = time.time()
    df_full, selected_features = v5.load_data()
    df_train = v17.filter_data_by_windows(df_full, TRAIN_WINDOWS)
    print(f"[INFO] Training data: {len(df_train):,} rows across "
          f"{len(TRAIN_WINDOWS)} windows")
    del df_full
    gc.collect()
    stage_times["stage0_data_loading"] = time.time() - stage0_start
    v5.print_step_timing("STAGE 0: Data Loading Complete", stage0_start)

    stage1_start = time.time()
    gnn_study, _ = v14.train_gnn_model(df_train, selected_features,
                                       output_dir, model_time)
    stage_times["stage1_gnn_training"] = time.time() - stage1_start
    v5.print_step_timing("STAGE 1: Benchmark Forecaster Training Complete",
                         stage1_start)

    _inject_architecture_metadata(output_dir, model_time, architecture, version,
                                  model_name=model_cls.__name__)

    total_duration = time.time() - total_start

    summary = {
        "experiment_name": experiment_name,
        "architecture": architecture,
        "benchmark_version": version,
        "total_duration_minutes": total_duration / 60,
        "stage_times_seconds": stage_times,
        "stage_times_formatted": {k: v5.format_duration(v)
                                  for k, v in stage_times.items()},
        "gnn_best_val_rmse": gnn_study.best_value if gnn_study.best_trial else None,
        "gnn_best_params": gnn_study.best_params if gnn_study.best_trial else None,
        "gnn_trials_completed": len(
            [t for t in gnn_study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
        "gnn_trials_pruned": len(
            [t for t in gnn_study.trials if t.state == optuna.trial.TrialState.PRUNED]),
        "gnn_trials_failed": len(
            [t for t in gnn_study.trials if t.state == optuna.trial.TrialState.FAIL]),
        "ablation_settings": v5.ABLATION_SETTINGS,
        "test_mode": v5.TEST_MODE,
        "train_windows": [{"start": s, "end": e} for s, e in TRAIN_WINDOWS],
        "completed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "next_step": f"Run xgboost-only with PREVIOUS_EXPERIMENT_DIR={output_dir}",
    }
    with open(os.path.join(output_dir, "experiment_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print(f"{version.upper()} ({architecture}) FORECASTER TRAINING COMPLETE")
    print(f"  TOTAL: {v5.format_duration(total_duration)}")
    print(f"Results saved to: {output_dir}")
    print("=" * 80)
    sys.stdout.flush()
    return True
