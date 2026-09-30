#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Ablation Study Pipeline V17: V14 RMSE objective + tight windowed training data
==============================================================================

V17 is the same design as V16 but with even tighter training windows (one
month each instead of V16's 4 + 2 months). Use V17 as the fastest-iteration
sibling of V16 — same modelling choices, smaller data slice.

V17 trains on:
    - 2024-06-01 → 2024-07-01  (prior-year same-season as the test period)
    - 2025-05-01 → 2025-06-01  (recent regime immediately before the test period)

Compared to V16 (4 + 2 months ≈ 6 months total):
    - V17 keeps just 2 months total (one prior-year + one recent)
    - Roughly 3× less training data than V16, ~9× less than V14
    - Same ablation logic: prior-year window matches summer test conditions,
      recent window hedges against year-over-year drift

Sequence-building note:
-----------------------
The two windows are non-contiguous (Jul 2024 → May 2025 gap, ~10 months).
V17 reuses V16's machinery: rows are tagged with `window_id`, and v5's
prepare_gnn_data / create_gnn_sequences are patched to group by
(via, sen, pk, window_id) so neither the 80/20 split nor any 12-step LSTM
sequence crosses the gap.

Author: Gerard Franco
Date: May 2026
Affiliation: Universitat Politècnica de Catalunya
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
import gc
import json
import logging
import os
import re
import sys
import time
import warnings
from datetime import datetime

import joblib
import numpy as np
import optuna
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler

from src.training import ablation_study_v5_gnn_only as v5
from src.training import ablation_study_v14_gnn_only as v14


PIPELINE_VERSION = "v17"

TRAIN_WINDOWS = [
    ("2024-06-01", "2024-07-01"),
    ("2025-05-01", "2025-06-01"),
]

# Inherit V14 training config (idempotent — v14 already mutated v5 on import).
# The AP7_GNN_* overrides exist only for the synthetic-data demo; unset, they
# leave the settings used for every reported run (150 / 40 / 30).
v5.GNN_CONFIG["epochs"] = int(os.environ.get("AP7_GNN_EPOCHS", 150))
v5.GNN_CONFIG["patience"] = int(os.environ.get("AP7_GNN_PATIENCE", 40))
v5.GNN_CONFIG["min_delta"] = 0.00001
v5.GNN_CONFIG["trials"] = int(os.environ.get("AP7_GNN_TRIALS", 30))  # benchmark suite: search converges well before 30 (job 2538897 peaked at trial 16); same budget for all models = fair


def filter_data_by_windows(df, windows):
    """Return rows inside the union of date windows, tagged with `window_id`.

    `window_id` is the grouping key that prevents sequences and 80/20 splits
    from straddling the gap between windows downstream.
    """
    parts = []
    for i, (start, end) in enumerate(windows):
        mask = (df["dat"] >= start) & (df["dat"] < end)
        sub = df[mask].copy()
        sub["window_id"] = i
        parts.append(sub)

    out = pd.concat(parts, axis=0).reset_index(drop=True)
    print(f"[WINDOWS] Filtered to {len(windows)} disjoint date windows:")
    for i, (start, end) in enumerate(windows):
        print(f"           - window {i}: {start} → {end}  ({len(parts[i]):,} rows)")
    print(f"[WINDOWS] Total rows after windowing: {len(out):,}")
    return out


# ----------------------------------------------------------------------------
# Window-aware overrides for v5 data prep / sequence creation.
#
# Same logic as V16 — without the window_id-aware grouping, both the 80/20
# temporal split and the sliding-window sequence builder would treat the two
# disjoint windows as one contiguous stream and produce LSTM inputs that span
# the Jul 2024 → May 2025 gap.
# ----------------------------------------------------------------------------

def _prepare_gnn_data_windowed(df, selected_features, model_time, output_dir):
    prepare_start = time.time()
    print("\n[GNN] Preparing training data (V17 window-aware)...")

    cols_targets = ["mean_speed", "intTot", "intP"]
    static_features = ["car", "segment", "ang_curv", "ang_pend_pos", "ang_pend_neg", "via", "sen", "pk"]
    static_features = [f for f in static_features if f in df.columns]

    time_cols = ["dat", "min", "anyo", "mes", "dia", "hor",
                 "5min", "10min", "15min", "30min", "45min", "60min"]
    temporal_features = [f for f in selected_features
                         if f not in static_features + cols_targets + time_cols]
    if v5.ABLATION_SETTINGS["include_temporal"]:
        temporal_features.extend(["diaSem", "anyo", "mes", "dia", "hor"])
    temporal_features = list(set([f for f in temporal_features if f in df.columns]))

    print(f"[FEATURES] Temporal features: {len(temporal_features)}")
    print(f"[FEATURES] Static features: {len(static_features)}")
    print(f"[FEATURES] Target columns: {cols_targets}")

    df = df.sort_values(by=["via", "sen", "pk", "window_id", "dat"]).reset_index(drop=True)
    print("[SORT] Data sorted by (via, sen, pk, window_id, dat) — sequences won't cross window gap")

    # window_id is a grouping key, not a feature — must be excluded from temporal_cols.
    excluded = cols_targets + static_features + ["dat", "min", "window_id"]
    temporal_cols = [col for col in df.columns if col not in excluded]
    temporal_cols = [c for c in temporal_cols if c in temporal_features or c in temporal_cols]

    temporal_features_df = df[temporal_cols].fillna(0)
    static_features_df = df[static_features].fillna(0)
    targets_df = df[cols_targets].fillna(0)

    print("[SCALE] Scaling features...")
    scaler_temporal = MinMaxScaler()
    scaler_static = MinMaxScaler()
    scaler_targets = MinMaxScaler()
    temporal_normalized = scaler_temporal.fit_transform(temporal_features_df)
    static_normalized = scaler_static.fit_transform(static_features_df)
    targets_normalized = scaler_targets.fit_transform(targets_df)
    print("[SUCCESS] Scaling completed")

    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(scaler_temporal, os.path.join(output_dir, f"scaler-temporal_model=GNN-version={model_time}.pkl"))
    joblib.dump(scaler_static, os.path.join(output_dir, f"scaler-static_model=GNN-version={model_time}.pkl"))
    joblib.dump(scaler_targets, os.path.join(output_dir, f"scaler-targets_model=GNN-version={model_time}.pkl"))

    print("[SPLIT] Performing temporal split per (PK, window)...")
    train_mask = np.ones(len(df), dtype=bool)
    location_groups = df.groupby(["via", "sen", "pk", "window_id"], sort=False)
    for _, group in location_groups:
        group_size = len(group)
        split_point = int(group_size * 0.8)
        val_positions = group.index[split_point:]
        train_mask[df.index.get_indexer(val_positions)] = False
    val_mask = ~train_mask

    temporal_train = temporal_normalized[train_mask]
    temporal_val = temporal_normalized[val_mask]
    static_train = static_normalized[train_mask]
    static_val = static_normalized[val_mask]
    targets_train = targets_normalized[train_mask]
    targets_val = targets_normalized[val_mask]
    del temporal_normalized, static_normalized, targets_normalized

    df_train = df.loc[df.index[train_mask]].copy()
    df_val = df.loc[df.index[val_mask]].copy()
    del df, train_mask, val_mask

    print(f"[INFO] Train samples: {len(temporal_train):,} | Val samples: {len(temporal_val):,}")
    v5.print_step_timing("GNN Data Preparation", prepare_start)

    return (temporal_train, static_train, targets_train,
            temporal_val, static_val, targets_val,
            scaler_temporal, scaler_static, scaler_targets,
            df_train, df_val, cols_targets, temporal_cols, static_features)


def _create_gnn_sequences_windowed(temporal_data, static_data, targets, df, sequence_length):
    from numpy.lib.stride_tricks import sliding_window_view

    print("[SEQUENCES] Creating sequences with vectorized operations (V17 window-aware)...")
    seq_start_time = time.time()

    unique_pks = sorted(df["pk"].unique())
    pk_to_id = {pk: idx for idx, pk in enumerate(unique_pks)}

    # window_id in the groupby key keeps each sliding window inside one date range.
    location_groups = df.groupby(["via", "sen", "pk", "window_id"], sort=False).size()

    all_sequences, all_static, all_targets, all_pk_ids, all_sen = [], [], [], [], []
    current_idx = 0
    n_features = temporal_data.shape[1]

    for (via, sen, pk, _wid), group_size in location_groups.items():
        if group_size <= sequence_length:
            current_idx += group_size
            continue

        location_temporal = temporal_data[current_idx:current_idx + group_size]
        location_static = static_data[current_idx:current_idx + group_size]
        location_targets = targets[current_idx:current_idx + group_size]

        n_sequences = group_size - sequence_length
        sequences = sliding_window_view(location_temporal, (sequence_length, n_features))
        sequences = sequences.squeeze(axis=1)[:n_sequences]

        seq_targets = location_targets[sequence_length:sequence_length + n_sequences]
        seq_static = location_static[sequence_length:sequence_length + n_sequences]

        pk_id = pk_to_id[pk]
        pk_ids = np.full(n_sequences, pk_id, dtype=np.int32)
        sen_ids = np.full(n_sequences, sen, dtype=np.int32)

        all_sequences.append(sequences.copy())
        all_static.append(seq_static)
        all_targets.append(seq_targets)
        all_pk_ids.append(pk_ids)
        all_sen.append(sen_ids)

        current_idx += group_size

    if not all_sequences:
        raise ValueError("No valid sequences created. Check sequence_length vs window sizes.")

    final_sequences = np.concatenate(all_sequences, axis=0).astype(np.float32)
    final_static = np.concatenate(all_static, axis=0).astype(np.float32)
    final_targets = np.concatenate(all_targets, axis=0).astype(np.float32)
    final_pk_ids = np.concatenate(all_pk_ids, axis=0)
    final_sen = np.concatenate(all_sen, axis=0)

    seq_duration = time.time() - seq_start_time
    print(f"[SEQUENCES] Created {len(final_sequences):,} sequences in {seq_duration:.1f}s")
    return (final_sequences, final_static, final_targets,
            final_pk_ids, final_sen, len(unique_pks))


# v14.train_gnn_model calls v5.prepare_gnn_data and v5.create_gnn_datasets,
# and v5.create_gnn_datasets resolves create_gnn_sequences via the module
# namespace — so patching v5's attributes here is enough.
v5.prepare_gnn_data = _prepare_gnn_data_windowed
v5.create_gnn_sequences = _create_gnn_sequences_windowed


def build_experiment_name() -> str:
    prefix = v5.get_experiment_prefix()
    resolution = v5.get_time_resolution_token()
    job_id = v5.get_job_id()
    return f"v17_gnn_{prefix}_{resolution}_{job_id}"


def _stamp_proposed_method(output_dir: str, model_time: str) -> None:
    """Label this V17 run as the proposed method in the 2nd-article benchmark.

    Adds `architecture` (the loader's default 'gnn_lstm_v5' token, so model
    reconstruction is byte-for-byte unchanged) plus benchmark-roster fields so
    the comparison treats V17 as a first-class peer of the BL-F0..F6
    forecasters. GeoLSTM (v62) is the same model minus the graph — the V17-vs-
    v62 gap isolates the corridor graph's contribution.
    """
    path = os.path.join(
        output_dir, f"best-model-metadata_model=GNN-version={model_time}.json")
    if not os.path.exists(path):
        print(f"[WARN] Metadata not found for proposed-method stamping: {path}")
        return
    with open(path) as f:
        meta = json.load(f)
    meta["architecture"] = "gnn_lstm_v5"
    meta["benchmark_version"] = "v17"
    meta["proposed_method"] = True
    meta["benchmark_label"] = "Proposed — GNN-LSTM (graph + LSTM)"
    meta["benchmark_paper"] = "2nd article (Layer-1 forecasting)"
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
    print("[ARCH] Stamped V17 metadata as proposed method "
          "(architecture=gnn_lstm_v5, proposed_method=True)")


def main():
    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
    logging.getLogger("lightning_fabric").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore")

    os.environ["LOCAL_RANK"] = "0"
    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    experiment_name = build_experiment_name()
    output_dir = os.path.join(v5.OUTPUT_BASE_DIR, experiment_name)
    os.makedirs(output_dir, exist_ok=True)
    model_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("\n" + "=" * 80)
    print("ABLATION STUDY V17 - GNN-LSTM (V14 RMSE OBJECTIVE + TIGHT WINDOWED TRAINING)")
    print("=" * 80)
    print(f"Experiment: {experiment_name}")
    print(f"Output: {output_dir}")
    print(f"Test Mode: {v5.TEST_MODE}")
    print(f"GPU Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
    print(f"\n[NOTE] V17 trains on {len(TRAIN_WINDOWS)} tight date windows (1 month each):")
    for s, e in TRAIN_WINDOWS:
        print(f"         - {s} → {e}")
    print("[NOTE] Goal: maximum training-time reduction; same modelling choices as V14/V16.")
    print("[NOTE] All training/Optuna config inherited from V14 (RMSE objective,")
    print(f"       {v5.GNN_CONFIG['epochs']} epochs, patience={v5.GNN_CONFIG['patience']}, {v5.GNN_CONFIG['trials']} trials).")
    print("=" * 80)
    sys.stdout.flush()

    v5.print_ablation_config()

    config = {
        "experiment_name": experiment_name,
        "model_time": model_time,
        "test_mode": v5.TEST_MODE,
        "ablation_settings": v5.ABLATION_SETTINGS,
        "gnn_config": v5.GNN_CONFIG,
        "train_windows": [{"start": s, "end": e} for s, e in TRAIN_WINDOWS],
        "hyperparameter_search_space": v5.get_gnn_hyperparams(),
        "pipeline_version": "v17_gnn_rmse_objective_tight_windowed",
        "goal_metric": {
            "name": "val_rmse",
            "description": "Pure RMSE — penalises large errors quadratically, no R² component",
        },
        "next_step": "Run v5_xgboost_only with PREVIOUS_EXPERIMENT_DIR pointing to this output",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    config_path = os.path.join(output_dir, "experiment_config.json")
    config_versioned_path = os.path.join(output_dir, f"experiment_config_{PIPELINE_VERSION}.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    with open(config_versioned_path, "w") as f:
        json.dump(config, f, indent=2)

    total_start = time.time()
    stage_times = {}

    stage0_start = time.time()
    print("\n" + "=" * 80)
    print("STAGE 0: DATA LOADING")
    print("=" * 80)

    df_full, selected_features = v5.load_data()
    df_train = filter_data_by_windows(df_full, TRAIN_WINDOWS)
    print(f"[INFO] Training data: {len(df_train):,} rows across {len(TRAIN_WINDOWS)} windows")
    sys.stdout.flush()

    del df_full
    gc.collect()

    stage_times["stage0_data_loading"] = time.time() - stage0_start
    v5.print_step_timing("STAGE 0: Data Loading Complete", stage0_start)

    stage1_start = time.time()
    gnn_study, _ = v14.train_gnn_model(df_train, selected_features, output_dir, model_time)
    stage_times["stage1_gnn_training"] = time.time() - stage1_start
    v5.print_step_timing("STAGE 1: GNN-LSTM Training Complete", stage1_start)

    _stamp_proposed_method(output_dir, model_time)

    total_duration = time.time() - total_start

    print("\n" + "=" * 80)
    print("GNN-LSTM TRAINING COMPLETE (V17)")
    print("=" * 80)
    print("\nTIMING BREAKDOWN:")
    print("-" * 60)
    print(f"  Stage 0 - Data Loading:      {v5.format_duration(stage_times['stage0_data_loading']):>15}")
    print(f"  Stage 1 - GNN-LSTM Training: {v5.format_duration(stage_times['stage1_gnn_training']):>15}")
    print("-" * 60)
    print(f"  TOTAL:                       {v5.format_duration(total_duration):>15}")
    print("=" * 80)
    sys.stdout.flush()

    print(f"\nResults saved to: {output_dir}")

    summary = {
        "experiment_name": experiment_name,
        "total_duration_minutes": total_duration / 60,
        "total_duration_hours": total_duration / 3600,
        "stage_times_seconds": stage_times,
        "stage_times_formatted": {k: v5.format_duration(v) for k, v in stage_times.items()},
        "gnn_best_val_rmse": gnn_study.best_value if gnn_study.best_trial else None,
        "gnn_best_params": gnn_study.best_params if gnn_study.best_trial else None,
        "gnn_trials_completed": len(
            [t for t in gnn_study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        ),
        "gnn_trials_pruned": len(
            [t for t in gnn_study.trials if t.state == optuna.trial.TrialState.PRUNED]
        ),
        "gnn_trials_failed": len(
            [t for t in gnn_study.trials if t.state == optuna.trial.TrialState.FAIL]
        ),
        "ablation_settings": v5.ABLATION_SETTINGS,
        "test_mode": v5.TEST_MODE,
        "train_windows": [{"start": s, "end": e} for s, e in TRAIN_WINDOWS],
        "completed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "next_step": f"Run v5_xgboost_only with PREVIOUS_EXPERIMENT_DIR={output_dir}",
    }

    summary_path = os.path.join(output_dir, "experiment_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print("SUCCESS!")
    print("=" * 80)
    sys.stdout.flush()

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ablation Study V17 - GNN-LSTM with RMSE objective + tight windowed training data"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--time-resolution",
        choices=v5.TIME_RESOLUTION_CHOICES,
        help="Time resolution token to use",
    )
    for token in v5.TIME_RESOLUTION_CHOICES:
        minutes = token.replace("min", "")
        group.add_argument(
            f"--{minutes}min",
            dest="time_resolution",
            action="store_const",
            const=token,
            help=argparse.SUPPRESS,
        )

    args = parser.parse_args()

    if getattr(args, "time_resolution", None):
        chosen = args.time_resolution
        new_data_file = re.sub(r"\d+min", chosen, v5.DATA_FILE, count=1)
        v5.DATA_FILE = new_data_file
        v5.DATA_PATH = os.path.join(_AP7_DATA, v5.DATA_FILE)
        print(f"[CONFIG] Using DATA_FILE={v5.DATA_FILE} for time resolution {chosen}")
        sys.stdout.flush()

    sys.exit(0 if main() else 1)
