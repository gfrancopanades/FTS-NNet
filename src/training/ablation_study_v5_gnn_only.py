#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Ablation Study Pipeline V5: GNN-LSTM Only Training
===================================================

VERSION 5.0 - GNN-LSTM ONLY TRAINING (IMPROVED)

Changes from V4:
- Uses SpatioTemporalGNN_LSTM_V5 model:
    * Removed final ReLU output activation
    * Added LayerNorm after LSTM output
    * Per-target weighted SmoothL1Loss (fixes intensity collapse)
    * LR scheduler patience reduced (10 -> 3)
- Narrowed hyperparameter search space (hidden_size [64,128])
- Increased dropout range (0.20, 0.40) for better regularization
- Wider weight_decay range (1e-5, 1e-2)
- Added target_weights as tunable hyperparameters
- Increased early stopping patience (5 -> 7) and decreased min_delta (0.001 -> 0.0005)
- Increased pruner warmup (5 -> 8 epochs)

Base module of every Stage-1 forecaster (data loading, training loop and
Optuna search). Run a forecaster with scripts/run_ablation_study_gnn_only.sh.

Author: Gerard Franco
Date: February 2026
Affiliation: Universitat Politècnica de Catalunya
"""
from src.paths import (  # portable paths -- see src/paths.py
    PROJECT_ROOT_STR as _AP7_ROOT,
    EXPERIMENTS_ROOT_STR as _AP7_EXPERIMENTS,
    DATA_DIR_STR as _AP7_DATA,
    TABLES_DIR as _AP7_TABLES,
    FIGURES_DIR as _AP7_FIGS,
)

import sys
import os
import time
import gc
import logging
import json
import joblib
import csv
import threading
import glob
import warnings
import re
import argparse
from datetime import datetime, timedelta
from math import ceil

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from optuna.integration import PyTorchLightningPruningCallback

import optuna

# Custom imports
from src.data_file_names import *
from src.training.io import *
from src.models.spatiotemporal_gnn_lstm_v5 import SpatioTemporalGNN_LSTM_V5


# ============================================================================
# CONFIGURATION - ABLATION STUDY SETTINGS
# ============================================================================

# TEST MODE: Use reduced data for quick testing
TEST_MODE = False  # Set to True for quick testing with 1 month training

# ABLATION STUDY: Feature groups to include (set to False to exclude)
ABLATION_SETTINGS = {
    # Weather features (1-day forecast)
    'include_weather_1d': True,
    # Weather features (3-day forecast)
    'include_weather_3d': True,
    # Traffic features (speed, intensity)
    'include_traffic': True,
    # Road geometry features (curvature, slopes)
    'include_geometry': True,
    # Mobility index
    'include_mobility': True,
    # Temporal features (hour, day of week, etc.)
    'include_temporal': True,
    # Imputation flags
    'include_imputation_flags': True,
}

# FEATURE DEFINITIONS BY GROUP
FEATURE_GROUPS = {
    'weather_1d': [
        '1d_fcst_temperature_2m', '1d_fcst_precipitation', '1d_fcst_snowfall',
        '1d_fcst_cloud_cover', '1d_fcst_wind_speed_10m', '1d_fcst_wind_gusts_10m',
        '1d_fcst_rain_binary'
    ],
    'weather_3d': [
        '3d_fcst_temperature_2m', '3d_fcst_precipitation', '3d_fcst_snowfall',
        '3d_fcst_cloud_cover', '3d_fcst_wind_speed_10m', '3d_fcst_wind_gusts_10m',
        '3d_fcst_rain_binary'
    ],
    'traffic': ['mean_speed', 'intTot', 'intP', 'car'],
    'geometry': ['ang_curv', 'ang_pend_pos', 'ang_pend_neg', 'segment'],
    'mobility': ['mob_esp'],
    'temporal': ['anyo', 'mes', 'dia', 'diaSem', 'hor', 'min'],
    'imputation': ['speed_imputation', 'intensity_imputation'],
    'static_core': ['pk', 'via', 'sen', 'car'],  # Always included for GNN
}

# GPU Configuration
USE_GPU = True
N_GPUS = int(os.environ.get('ABLATION_NUM_GPUS', 1))
N_THREADS = 16

# Training Configuration
RANDOM_STATE = 42

# Pipeline version tag (used to version output filenames)
PIPELINE_VERSION = "v5"

# GNN Configuration - Using current optimized settings from v2_with_sequence_loading
GNN_CONFIG = {
    # Env-overridable so an architecture whose per-trial cost is an outlier can
    # be capped explicitly and footnoted, rather than silently truncated by a
    # wall-clock timeout (which is what left STGCN at 4 of 30 trials).
    'trials': int(os.environ.get("AP7_GNN_TRIALS",
                                 "50" if not TEST_MODE else "5")),
    'epochs': 50 if not TEST_MODE else 10,
    'patience': 7 if not TEST_MODE else 3,      # V5: Increased from 5 (allow slow-converging models more time)
    'min_delta': 0.0005,                        # V5: Decreased from 0.001 (detect smaller improvements)
    'batch_size': 1024,
    'sequence_length': 12,                      # Default; adjusted adaptively in load_data()
    'gradient_clip': 0.5,
}

# Time-resolution options and sequence length selection criterion (stability-oriented):
# Explicit interval-to-sequence mapping used for consistency across runs:
#   5min -> 12, 10min -> 10, 15min -> 8, 30min -> 6, 45min -> 4, 60min -> 4
# If an unknown interval appears, fallback to GNN_CONFIG['sequence_length'].
TIME_RESOLUTION_CHOICES = ["5min", "10min", "15min", "30min", "45min", "60min"]
SEQUENCE_LENGTH_BY_INTERVAL = {
    5: 12,
    10: 10,
    15: 8,
    30: 6,
    45: 4,
    60: 4,
}

# Date Configuration
if TEST_MODE:
    TRAIN_START_DATE = '2024-04-01'
    TRAIN_END_DATE = '2024-05-01'  # 1 month
else:
    TRAIN_START_DATE = '2024-04-01'
    TRAIN_END_DATE = '2025-06-01'

# Data paths
DATA_FILE = 'CrashGNNLSTM_v1_vel-extinrix_int_geo_mob_wthr_5min_fund-propag-ltd_from_20240404_to_20251001.csv'
DATA_PATH = os.path.join(_AP7_DATA, DATA_FILE)

# Output directory
OUTPUT_BASE_DIR = _AP7_EXPERIMENTS
EXPERIMENT_NAME = None
OUTPUT_DIR = None


# ============================================================================
# UTILITY CLASSES AND FUNCTIONS
# ============================================================================

class ClearMemoryCallback(pl.Callback):
    """Callback to clear GPU memory between epochs"""
    def on_train_epoch_end(self, trainer, pl_module):
        torch.cuda.empty_cache()
        gc.collect()


def append_to_log(message, log_file):
    """Append message to log file"""
    with open(log_file, "a") as f:
        f.write(message + "\n")


class PerformanceLoggerCallback(pl.Callback):
    """Logs data loading vs training time per batch."""
    def __init__(self, log_file):
        self.log_file = log_file
        self._last_batch_end = None
        self._batch_start = None
        self._batch_count = 0
        self._data_time_total = 0.0
        self._train_time_total = 0.0

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        now = time.time()
        if self._last_batch_end is not None:
            self._data_time_total += (now - self._last_batch_end)
        self._batch_start = now

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        now = time.time()
        if self._batch_start is not None:
            self._train_time_total += (now - self._batch_start)
        self._last_batch_end = now
        self._batch_count += 1

    def on_train_epoch_end(self, trainer, pl_module):
        if self._batch_count == 0:
            return
        avg_data = self._data_time_total / self._batch_count
        avg_train = self._train_time_total / self._batch_count
        ratio = avg_data / (avg_train + 1e-9)
        msg = (
            f"[PERF] epoch={trainer.current_epoch + 1} "
            f"avg_data_time={avg_data:.4f}s "
            f"avg_train_time={avg_train:.4f}s "
            f"data_to_train_ratio={ratio:.3f}"
        )
        append_to_log(msg, self.log_file)
        self._batch_count = 0
        self._data_time_total = 0.0
        self._train_time_total = 0.0


def get_job_id():
    """Get the run id: an explicit --run-id (exported as JOB_ID by the
    launcher), else the scheduler job id, else a timestamp."""
    return (
        os.environ.get("JOB_ID")
        or os.environ.get("SLURM_JOB_ID")
        or os.environ.get("PBS_JOBID")
        or datetime.now().strftime('%Y%m%d_%H%M%S')
    )


def get_experiment_prefix():
    """Build prefix based on excluded feature groups."""
    excluded = []
    feature_codes = {
        'include_weather_1d': 'w1d',
        'include_weather_3d': 'w3d',
        'include_traffic': 'traf',
        'include_geometry': 'geo',
        'include_mobility': 'mob',
        'include_temporal': 'temp',
        'include_imputation_flags': 'imp',
    }
    for key, code in feature_codes.items():
        if not ABLATION_SETTINGS.get(key, True):
            excluded.append(code)
    return "full" if not excluded else f"no-{'-'.join(excluded)}"


def build_experiment_name():
    """Create experiment folder name using exclusions, resolution and job id."""
    prefix = get_experiment_prefix()
    resolution = get_time_resolution_token()
    job_id = get_job_id()
    return f"v5_gnn_{prefix}_{resolution}_{job_id}"


def get_time_resolution_token():
    """Extract time-resolution token (e.g. 5min, 15min) from DATA_FILE."""
    match = re.search(r'(\d+min)', DATA_FILE)
    return match.group(1) if match else "unknown-res"


def format_duration(seconds):
    """Format duration in human-readable format"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}min ({seconds:.0f}s)"
    else:
        hours = seconds / 3600
        minutes = (seconds % 3600) / 60
        return f"{hours:.1f}h ({int(hours)}h {int(minutes)}m)"


def print_step_timing(step_name, start_time, end_time=None):
    """Print timing information for a step"""
    if end_time is None:
        end_time = time.time()
    duration = end_time - start_time
    timestamp = datetime.now().strftime('%H:%M:%S')
    print(f"\n{'='*60}")
    print(f"⏱️  [{timestamp}] {step_name}")
    print(f"    Duration: {format_duration(duration)}")
    print(f"{'='*60}")
    sys.stdout.flush()


def get_selected_features():
    """Get list of features based on ablation settings"""
    selected = []
    
    if ABLATION_SETTINGS['include_weather_1d']:
        selected.extend(FEATURE_GROUPS['weather_1d'])
    if ABLATION_SETTINGS['include_weather_3d']:
        selected.extend(FEATURE_GROUPS['weather_3d'])
    if ABLATION_SETTINGS['include_traffic']:
        selected.extend(FEATURE_GROUPS['traffic'])
    if ABLATION_SETTINGS['include_geometry']:
        selected.extend(FEATURE_GROUPS['geometry'])
    if ABLATION_SETTINGS['include_mobility']:
        selected.extend(FEATURE_GROUPS['mobility'])
    if ABLATION_SETTINGS['include_temporal']:
        selected.extend(FEATURE_GROUPS['temporal'])
    if ABLATION_SETTINGS['include_imputation_flags']:
        selected.extend(FEATURE_GROUPS['imputation'])
    
    # Remove duplicates while preserving order
    seen = set()
    unique_features = []
    for f in selected:
        if f not in seen:
            seen.add(f)
            unique_features.append(f)
    
    return unique_features


def print_ablation_config():
    """Print the ablation study configuration"""
    print("\n" + "=" * 80)
    print("ABLATION STUDY CONFIGURATION")
    print("=" * 80)
    print(f"TEST_MODE: {TEST_MODE}")
    print(f"\nFeature groups:")
    for group, included in ABLATION_SETTINGS.items():
        status = "✓ INCLUDED" if included else "✗ EXCLUDED"
        group_name = group.replace('include_', '')
        features = FEATURE_GROUPS.get(group_name, [])
        print(f"  {status}: {group_name} ({len(features)} features)")
    
    selected = get_selected_features()
    print(f"\nTotal selected features: {len(selected)}")
    print("=" * 80)
    sys.stdout.flush()


# ============================================================================
# DATA LOADING
# ============================================================================

def get_interval_minutes_from_header(columns):
    """Infer interval minutes from CSV header."""
    for minutes in (5, 10, 15, 30, 45, 60):
        if f"{minutes}min" in columns:
            return minutes
    return None


def load_data():
    """Load and prepare the dataset with ablation feature filtering"""
    print("\n" + "=" * 80)
    print("LOADING DATA")
    print("=" * 80)
    sys.stdout.flush()
    
    print(f"[DATA] Loading training data from CSV file: {DATA_PATH}")
    start_time = time.time()
    
    print("[LOAD] Reading data from CSV file...")
    sys.stdout.flush()
    df = pd.read_csv(DATA_PATH, sep=";", decimal=".", encoding="latin-1")
    print(f"[SUCCESS] Dataset loaded: {df.shape}")
    sys.stdout.flush()

    interval_minutes = get_interval_minutes_from_header(df.columns)
    if interval_minutes:
        mapped_seq = SEQUENCE_LENGTH_BY_INTERVAL.get(interval_minutes, GNN_CONFIG['sequence_length'])
        GNN_CONFIG['sequence_length'] = mapped_seq
        print(
            f"[INFO] Detected interval column '{interval_minutes}min'; "
            f"mapped sequence_length={mapped_seq}"
        )
    else:
        print(
            "[WARNING] No interval column found in header (5min/10min/15min/30min/45min/60min). "
            f"Using default sequence_length={GNN_CONFIG['sequence_length']}."
        )
    
    # Encode categorical variables
    if 'Any' in df.columns:
        df.rename(columns={'Any': 'anyo'}, inplace=True)
    
    if 'via' in df.columns and df['via'].dtype == 'object':
        df['via'] = df['via'].map({'AP-7': 0}).fillna(0).astype(int)
    
    if 'sen' in df.columns and df['sen'].dtype == 'object':
        df['sen'] = df['sen'].map({'dec': 0, 'cre': 1}).fillna(1).astype(int)
    
    # Create datetime column
    # Check for various minute interval columns (5min, 10min, 15min, 30min, 45min, 60min)
    # and normalize to 'min' column for consistent processing
    
    # Create generic 'min' column from available interval columns
    # This handles 5min, 10min, 15min, 30min, 45min, 60min, etc.
    if 'min' not in df.columns:
        if '5min' in df.columns:
            df['min'] = df['5min']
        elif '10min' in df.columns:
            df['min'] = df['10min']
        elif '15min' in df.columns:
            df['min'] = df['15min']
        elif '30min' in df.columns:
            df['min'] = df['30min']
        elif '45min' in df.columns:
            df['min'] = df['45min']
        elif '60min' in df.columns:
            df['min'] = df['60min']
        elif 'dat' in df.columns:
            df['min'] = pd.to_datetime(df['dat']).dt.minute
        else:
            df['min'] = 0
            
    df['dat'] = pd.to_datetime(
        df['anyo'].astype(str) + '-' + 
        df['mes'].astype(str).str.zfill(2) + '-' + 
        df['dia'].astype(str).str.zfill(2) + ' ' + 
        df['hor'].astype(str).str.zfill(2) + ':' +
        df['min'].astype(str).str.zfill(2) + ':00'
    )
    
    # Sort data
    df.sort_values(by=['via', 'sen', 'pk', 'anyo', 'mes', 'dia', 'hor', 'min'], inplace=True)
    
    # Get selected features based on ablation settings
    selected_features = get_selected_features()
    available_features = [f for f in selected_features if f in df.columns]
    missing_features = [f for f in selected_features if f not in df.columns]
    
    if missing_features:
        print(f"[WARNING] Missing features: {missing_features}")
    
    print(f"[INFO] Using {len(available_features)} features for ablation study")
    print(f"[INFO] Date range: {df['dat'].min()} to {df['dat'].max()}")
    print(f"[COMPLETE] Data loading completed! Duration: {time.time() - start_time:.2f} seconds")
    sys.stdout.flush()
    
    return df, available_features


def filter_data_by_date(df, start_date, end_date):
    """Filter dataframe by date range"""
    mask = (df['dat'] >= start_date) & (df['dat'] < end_date)
    return df[mask].copy()


# ============================================================================
# GNN-LSTM TRAINING FUNCTIONS
# ============================================================================

def get_gnn_hyperparams():
    """
    GNN hyperparameter search space V5.
    
    Changes from V4:
    - hidden_size narrowed to [64, 128] (32 too small, 192/256 never won)
    - dropout increased to (0.20, 0.40) to combat overfitting
    - learning_rate upper bound widened to 1e-4
    - weight_decay range significantly widened (1e-5, 1e-2) for stronger regularization
    - NEW: target_weight_speed and target_weight_intensity for per-target loss weighting
    """
    return {
        "hidden_size": [64, 128],
        "num_layers": [2, 3],
        "dropout_prob_range": (0.20, 0.40),
        "learning_rate_range": (2e-5, 1e-4),
        "weight_decay_range": (1e-5, 1e-2),
        "pk_embed_dim": [16, 32, 64],
        "graph_hidden_dim": [32, 64],
        "num_graph_layers": [2, 3],
        "target_weight_speed_range": (0.5, 2.0),
        "target_weight_intensity_range": (1.0, 4.0),
    }


def prepare_gnn_data(df, selected_features, model_time, output_dir):
    """Prepare data for GNN-LSTM training"""
    prepare_start = time.time()
    print("\n[GNN] Preparing training data...")
    
    # Define target columns (always required)
    cols_targets = ['mean_speed', 'intTot', 'intP']
    
    # Static features (always included for GNN structure)
    static_features = ['car', 'segment', 'ang_curv', 'ang_pend_pos', 'ang_pend_neg', 'via', 'sen', 'pk']
    
    # Filter to available static features
    static_features = [f for f in static_features if f in df.columns]
    
    # Temporal features are everything in selected_features that's not static or target
    # Exclude raw time columns from normalization (unless explicitly included in temporal group)
    time_cols = ['dat', 'min', 'anyo', 'mes', 'dia', 'hor', '5min', '10min', '15min', '30min', '45min', '60min']
    temporal_features = [f for f in selected_features 
                        if f not in static_features + cols_targets + time_cols]
    
    # Add temporal components if included
    if ABLATION_SETTINGS['include_temporal']:
        temporal_features.extend(['diaSem', 'anyo', 'mes', 'dia', 'hor'])
    
    temporal_features = list(set([f for f in temporal_features if f in df.columns]))
    
    print(f"[FEATURES] Temporal features: {len(temporal_features)}")
    print(f"[FEATURES] Static features: {len(static_features)}")
    print(f"[FEATURES] Target columns: {cols_targets}")
    
    # CRITICAL: Sort by location and time to maintain chronological order
    df = df.sort_values(by=['via', 'sen', 'pk', 'dat']).reset_index(drop=True)
    print(f"[SORT] Data sorted by location and time (chronological order maintained)")
    
    # Prepare dataframes
    temporal_cols = [col for col in df.columns 
                    if col not in cols_targets + static_features + ['dat', 'min']]
    temporal_cols = [c for c in temporal_cols if c in temporal_features or c in temporal_cols]
    
    temporal_features_df = df[temporal_cols].fillna(0)
    static_features_df = df[static_features].fillna(0)
    targets_df = df[cols_targets].fillna(0)
    
    # Scale data
    print("[SCALE] Scaling features...")
    scaler_temporal = MinMaxScaler()
    scaler_static = MinMaxScaler()
    scaler_targets = MinMaxScaler()
    
    temporal_normalized = scaler_temporal.fit_transform(temporal_features_df)
    static_normalized = scaler_static.fit_transform(static_features_df)
    targets_normalized = scaler_targets.fit_transform(targets_df)
    
    print(f"[SUCCESS] Scaling completed")
    
    # Save scalers
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(scaler_temporal, os.path.join(output_dir, f'scaler-temporal_model=GNN-version={model_time}.pkl'))
    joblib.dump(scaler_static, os.path.join(output_dir, f'scaler-static_model=GNN-version={model_time}.pkl'))
    joblib.dump(scaler_targets, os.path.join(output_dir, f'scaler-targets_model=GNN-version={model_time}.pkl'))
    
    # Temporal split per PK (memory-efficient boolean mask approach)
    print("[SPLIT] Performing temporal split per PK...")
    train_mask = np.ones(len(df), dtype=bool)

    location_groups = df.groupby(['via', 'sen', 'pk'], sort=False)
    for _, group in location_groups:
        group_size = len(group)
        split_point = int(group_size * 0.8)
        # Mark validation rows as False in train_mask
        val_positions = group.index[split_point:]
        train_mask[df.index.get_indexer(val_positions)] = False

    val_mask = ~train_mask

    temporal_train = temporal_normalized[train_mask]
    temporal_val = temporal_normalized[val_mask]
    static_train = static_normalized[train_mask]
    static_val = static_normalized[val_mask]
    targets_train = targets_normalized[train_mask]
    targets_val = targets_normalized[val_mask]

    # Free full arrays now that splits are done
    del temporal_normalized, static_normalized, targets_normalized

    df_train = df.loc[df.index[train_mask]].copy()
    df_val = df.loc[df.index[val_mask]].copy()
    del df, train_mask, val_mask
    
    print(f"[INFO] Train samples: {len(temporal_train):,} | Val samples: {len(temporal_val):,}")
    print_step_timing("GNN Data Preparation", prepare_start)
    
    return (temporal_train, static_train, targets_train,
            temporal_val, static_val, targets_val,
            scaler_temporal, scaler_static, scaler_targets,
            df_train, df_val, cols_targets, temporal_cols, static_features)


def create_gnn_sequences(temporal_data, static_data, targets, df, sequence_length):
    """
    Create sequences for GNN training using vectorized numpy operations.
    
    OPTIMIZED: Uses numpy strided arrays for 10-50x faster sequence creation
    compared to Python loops. This is critical for datasets with 20M+ samples.
    """
    from numpy.lib.stride_tricks import sliding_window_view
    
    print(f"[SEQUENCES] Creating sequences with vectorized operations...")
    seq_start_time = time.time()
    
    unique_pks = sorted(df['pk'].unique())
    pk_to_id = {pk: idx for idx, pk in enumerate(unique_pks)}
    
    location_groups = df.groupby(['via', 'sen', 'pk']).size()
    
    # Pre-allocate lists for batch concatenation (more efficient than appending)
    all_sequences = []
    all_static = []
    all_targets = []
    all_pk_ids = []
    all_sen = []
    
    current_idx = 0
    n_features = temporal_data.shape[1]
    
    for (via, sen, pk), group_size in location_groups.items():
        if group_size <= sequence_length:
            current_idx += group_size
            continue
        
        # Extract location data slices
        location_temporal = temporal_data[current_idx:current_idx + group_size]
        location_static = static_data[current_idx:current_idx + group_size]
        location_targets = targets[current_idx:current_idx + group_size]
        
        # VECTORIZED: Create all sequences at once using sliding window view
        # This creates views without copying data - extremely fast!
        n_sequences = group_size - sequence_length
        
        # Use sliding_window_view for vectorized sequence creation
        # Shape: (n_sequences, sequence_length, n_features)
        sequences = sliding_window_view(location_temporal, (sequence_length, n_features))
        sequences = sequences.squeeze(axis=1)[:n_sequences]  # Remove extra dim and limit
        
        # Get corresponding targets and static features (after sequence)
        seq_targets = location_targets[sequence_length:sequence_length + n_sequences]
        seq_static = location_static[sequence_length:sequence_length + n_sequences]
        
        # Create PK and sen arrays for all sequences
        pk_id = pk_to_id[pk]
        pk_ids = np.full(n_sequences, pk_id, dtype=np.int32)
        sen_ids = np.full(n_sequences, sen, dtype=np.int32)
        
        # Append to batch lists
        all_sequences.append(sequences.copy())  # Copy to ensure contiguous memory
        all_static.append(seq_static)
        all_targets.append(seq_targets)
        all_pk_ids.append(pk_ids)
        all_sen.append(sen_ids)
        
        current_idx += group_size
    
    # Concatenate all batches at once (much faster than incremental appending)
    if not all_sequences:
        raise ValueError("No valid sequences created. Check sequence_length vs data size.")
    
    final_sequences = np.concatenate(all_sequences, axis=0).astype(np.float32)
    final_static = np.concatenate(all_static, axis=0).astype(np.float32)
    final_targets = np.concatenate(all_targets, axis=0).astype(np.float32)
    final_pk_ids = np.concatenate(all_pk_ids, axis=0)
    final_sen = np.concatenate(all_sen, axis=0)
    
    seq_duration = time.time() - seq_start_time
    print(f"[SEQUENCES] Created {len(final_sequences):,} sequences in {seq_duration:.1f}s")
    
    return (final_sequences, final_static, final_targets, 
            final_pk_ids, final_sen, len(unique_pks))


def create_gnn_datasets(temporal_train, static_train, targets_train,
                        temporal_val, static_val, targets_val,
                        df_train, df_val, batch_size, sequence_length):
    """Create TensorDatasets for GNN training with on-the-fly sequence generation.
    
    Returns datasets and metadata instead of DataLoaders, so that fresh
    DataLoaders can be created per Optuna trial. This prevents worker process
    corruption when trials are pruned or fail (persistent_workers + pruning
    causes workers to die and poison all subsequent trials sharing the same
    DataLoader).
    """
    dataloader_start = time.time()
    print("[TENSOR] Creating sequences and dataloaders for GNN...")
    
    # Generate sequences on the fly
    print(f"[INFO] Generating sequences on the fly...")
    train_seq, train_static, train_targets, train_pk_ids, train_sen, num_pks = create_gnn_sequences(
        temporal_train, static_train, targets_train, df_train, sequence_length)
    
    val_seq, val_static, val_targets, val_pk_ids, val_sen, _ = create_gnn_sequences(
        temporal_val, static_val, targets_val, df_val, sequence_length)
    
    print(f"[INFO] Created {len(train_seq):,} train / {len(val_seq):,} val sequences")
    print(f"[INFO] Number of unique PKs: {num_pks}")
    
    # Create tensors (in-memory)
    train_dataset = TensorDataset(
        torch.tensor(train_seq, dtype=torch.float32),
        torch.tensor(train_targets, dtype=torch.float32),
        torch.tensor(train_pk_ids, dtype=torch.long),
        torch.tensor(train_static, dtype=torch.float32),
        torch.tensor(train_sen, dtype=torch.long)
    )
    
    val_dataset = TensorDataset(
        torch.tensor(val_seq, dtype=torch.float32),
        torch.tensor(val_targets, dtype=torch.float32),
        torch.tensor(val_pk_ids, dtype=torch.long),
        torch.tensor(val_static, dtype=torch.float32),
        torch.tensor(val_sen, dtype=torch.long)
    )
    
    # Save dimensions before freeing memory
    input_size = train_seq.shape[2]
    static_size = train_static.shape[1]
    output_size = train_targets.shape[1]
    
    # Free numpy arrays to save memory
    del train_seq, train_static, train_targets, train_pk_ids, train_sen
    del val_seq, val_static, val_targets, val_pk_ids, val_sen
    gc.collect()
    
    # Compute num_workers for logging
    cpu_cores = os.cpu_count() or 1
    gpus_in_use = max(1, int(N_GPUS))
    cores_per_gpu = max(1, cpu_cores // gpus_in_use)
    num_workers = min(4, max(1, cores_per_gpu - 2))
    # Each persistent worker forks the parent, so the worker count is the main
    # host-RAM multiplier. GNN_NUM_WORKERS lets a run fit into a tight memory
    # slot; it only changes prefetch concurrency (the sampler runs in the main
    # process), so predictions are unaffected.
    _envw = os.environ.get("GNN_NUM_WORKERS", "")
    if _envw.strip().isdigit():
        num_workers = int(_envw)
        print(f"[DATALOADER] GNN_NUM_WORKERS override active: {num_workers}")
    
    print(f"[DATALOADER] Will use {num_workers} workers per trial with pin_memory=True, persistent_workers=True")
    
    dataloader_duration = time.time() - dataloader_start
    print(f"[DATALOADER] ⏱️ Sequences + Datasets created in {format_duration(dataloader_duration)}")
    
    return train_dataset, val_dataset, input_size, static_size, output_size, num_pks


def _create_trial_dataloaders(train_dataset, val_dataset, batch_size, num_workers_override=None):
    """Create fresh DataLoaders for a single Optuna trial.
    
    Supports overriding ``num_workers`` for runtime fallback when worker
    subprocesses become unstable (e.g. "DataLoader worker exited unexpectedly").
    """
    cpu_cores = os.cpu_count() or 1
    gpus_in_use = max(1, int(N_GPUS))
    cores_per_gpu = max(1, cpu_cores // gpus_in_use)
    default_workers = min(4, max(1, cores_per_gpu - 2))
    _envw = os.environ.get("GNN_NUM_WORKERS", "")
    if _envw.strip().isdigit():
        default_workers = int(_envw)
    num_workers = default_workers if num_workers_override is None else max(0, int(num_workers_override))
    use_persistent_workers = num_workers > 0
    
    common_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,  # maintain chronological order
        "num_workers": num_workers,
        "persistent_workers": use_persistent_workers,
        "pin_memory": True,
    }
    if num_workers > 0:
        common_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(
        train_dataset,
        **common_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        **common_kwargs,
    )
    
    return train_loader, val_loader


def _shutdown_dataloaders(*loaders):
    """Safely shut down DataLoader worker processes to prevent zombie workers."""
    for loader in loaders:
        if loader is None:
            continue
        try:
            # Force shutdown of the internal iterator and its workers
            if hasattr(loader, '_iterator') and loader._iterator is not None:
                loader._iterator._shutdown_workers()
                loader._iterator = None
        except Exception:
            pass


def _is_dataloader_worker_crash(error):
    """Return True when an exception matches known DataLoader worker crashes."""
    error_text = str(error).lower()
    return (
        "dataloader worker" in error_text
        or "worker exited unexpectedly" in error_text
        or "bus error" in error_text
        or "killed by signal" in error_text
        or "broken pipe" in error_text
    )


def create_gnn_objective(input_size, static_size, output_size, num_pks,
                         train_dataset, val_dataset, log_file, perf_log_file, model_time,
                         checkpoint_dir, output_dir, n_gpus=1, csv_path=None):
    """Create Optuna objective for GNN-LSTM optimization.
    
    Receives TensorDatasets (not DataLoaders) so that each trial can create
    its own fresh DataLoaders with clean worker processes.
    """
    
    HYPERPARAMS = get_gnn_hyperparams()
    best_tracker = {'val_loss': float('inf'), 'trial_number': -1, 'lock': threading.Lock()}
    dataloader_runtime = {'workers_override': None, 'lock': threading.Lock()}
    
    def objective(trial):
        gpu_id = trial.number % n_gpus
        train_loader = None
        val_loader = None
        
        # Stagger trial starts to avoid DataLoader race conditions
        # When multiple trials start simultaneously, they all try to iterate
        # the shared DataLoader at once, causing worker conflicts
        stagger_delay = gpu_id * 2.0  # 2 seconds between each GPU
        if stagger_delay > 0:
            time.sleep(stagger_delay)
        
        try:
            if torch.cuda.is_available():
                torch.cuda.set_device(gpu_id)
        except Exception as e:
            print(f"[ERROR] GPU setup failed: {e}")
            return 1e6
        
        try:
            # Sample hyperparameters
            hidden_size = trial.suggest_categorical("hidden_size", HYPERPARAMS["hidden_size"])
            num_layers = trial.suggest_categorical("num_layers", HYPERPARAMS["num_layers"])
            dropout_prob = trial.suggest_float("dropout_prob", *HYPERPARAMS["dropout_prob_range"], log=True)
            learning_rate = trial.suggest_float("learning_rate", *HYPERPARAMS["learning_rate_range"], log=True)
            weight_decay = trial.suggest_float("weight_decay", *HYPERPARAMS["weight_decay_range"], log=True)
            pk_embed_dim = trial.suggest_categorical("pk_embed_dim", HYPERPARAMS["pk_embed_dim"])
            graph_hidden_dim = trial.suggest_categorical("graph_hidden_dim", HYPERPARAMS["graph_hidden_dim"])
            num_graph_layers = trial.suggest_categorical("num_graph_layers", HYPERPARAMS["num_graph_layers"])
            
            # V5: Sample per-target loss weights
            target_weight_speed = trial.suggest_float("target_weight_speed", *HYPERPARAMS["target_weight_speed_range"])
            target_weight_intensity = trial.suggest_float("target_weight_intensity", *HYPERPARAMS["target_weight_intensity_range"])
            target_weights = [target_weight_speed, target_weight_intensity, target_weight_intensity]
            
            if torch.cuda.is_available():
                actual_gpu = torch.cuda.current_device()
                gpu_name = torch.cuda.get_device_name(actual_gpu)
                trial_start_msg = f"\n{'='*80}\n[TRIAL {trial.number + 1}/{GNN_CONFIG['trials']}] STARTING on GPU {gpu_id} (actual device: {actual_gpu}, name: {gpu_name})\n{'='*80}"
            else:
                trial_start_msg = f"\n{'='*80}\n[TRIAL {trial.number + 1}/{GNN_CONFIG['trials']}] STARTING on GPU {gpu_id} (CUDA not available)\n{'='*80}"
            print(trial_start_msg)
            append_to_log(trial_start_msg, log_file)
            
            # Print hyperparameters
            hyperparams_msg = (
                f"[HYPERPARAMS] Testing:\n"
                f"  LSTM: hidden_size={hidden_size}, num_layers={num_layers}, dropout={dropout_prob:.4f}\n"
                f"  Optimizer: lr={learning_rate:.6f}, weight_decay={weight_decay:.6f}\n"
                f"  GNN: pk_embed={pk_embed_dim}, graph_hidden={graph_hidden_dim}, graph_layers={num_graph_layers}\n"
                f"  V5 target_weights: speed={target_weight_speed:.3f}, intensity={target_weight_intensity:.3f}"
            )
            print(hyperparams_msg)
            append_to_log(hyperparams_msg, log_file)
            
            # Create fresh DataLoaders for this trial to avoid worker corruption
            # from previous pruned/failed trials (persistent_workers + pruning
            # causes workers to die and poison shared DataLoaders)
            with dataloader_runtime['lock']:
                workers_override = dataloader_runtime['workers_override']

            train_loader, val_loader = _create_trial_dataloaders(
                train_dataset, val_dataset, GNN_CONFIG['batch_size'], workers_override
            )
            
            # Create V5 model with per-target weights
            model = SpatioTemporalGNN_LSTM_V5(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                output_size=output_size,
                dropout_prob=dropout_prob,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                num_pks=num_pks,
                static_feature_dim=static_size,
                pk_embed_dim=pk_embed_dim,
                graph_hidden_dim=graph_hidden_dim,
                num_graph_layers=num_graph_layers,
                target_weights=target_weights
            )
            
            # Callbacks
            checkpoint_callback = ModelCheckpoint(
                dirpath=checkpoint_dir,
                filename=f'trial_{trial.number}_gnn-version={model_time}',
                save_top_k=1, monitor="val_loss", mode="min"
            )
            
            early_stopping = EarlyStopping(
                monitor="val_loss", patience=GNN_CONFIG['patience'],
                mode="min", min_delta=GNN_CONFIG['min_delta']
            )
            
            # Add Optuna pruning callback for early trial termination
            pruning_callback = PyTorchLightningPruningCallback(trial, monitor="val_loss")
            
            def _build_trainer():
                return pl.Trainer(
                    max_epochs=GNN_CONFIG['epochs'],
                    accelerator='gpu' if torch.cuda.is_available() else 'cpu',
                    devices=[gpu_id] if torch.cuda.is_available() else 1,
                    logger=False,
                    callbacks=[checkpoint_callback, early_stopping, ClearMemoryCallback(), pruning_callback,
                               PerformanceLoggerCallback(perf_log_file)],
                    enable_progress_bar=False,
                    enable_model_summary=False,
                    precision='16-mixed' if torch.cuda.is_available() else 32,
                    gradient_clip_val=GNN_CONFIG['gradient_clip']
                )

            trainer = _build_trainer()
            
            print(f"[TRAINING] Starting training for Trial {trial.number + 1}...")
            trial_train_start = time.time()
            fallback_used = False
            try:
                trainer.fit(model, train_loader, val_loader)
            except RuntimeError as fit_error:
                if not _is_dataloader_worker_crash(fit_error):
                    raise

                fallback_used = True
                fallback_msg = (
                    f"[WARN] Trial {trial.number + 1} hit a DataLoader worker crash "
                    f"({fit_error}). Retrying with num_workers=0 for stability."
                )
                print(fallback_msg)
                append_to_log(fallback_msg, log_file)

                with dataloader_runtime['lock']:
                    dataloader_runtime['workers_override'] = 0

                _shutdown_dataloaders(train_loader, val_loader)
                train_loader = None
                val_loader = None
                if 'trainer' in locals() and trainer is not None:
                    del trainer
                torch.cuda.empty_cache()
                gc.collect()

                train_loader, val_loader = _create_trial_dataloaders(
                    train_dataset, val_dataset, GNN_CONFIG['batch_size'], num_workers_override=0
                )
                trainer = _build_trainer()
                trainer.fit(model, train_loader, val_loader)
            trial_train_duration = time.time() - trial_train_start
            
            val_loss_tensor = trainer.callback_metrics.get("val_loss")
            if val_loss_tensor is None:
                raise ValueError("val_loss not found in callback_metrics - training may have failed")
            
            val_loss = val_loss_tensor.item()
            if not torch.isfinite(torch.tensor(val_loss)):
                error_msg = f"Trial {trial.number + 1} produced NaN/Inf val_loss: {val_loss}"
                print(f"[WARNING] {error_msg}")
                val_loss = 1e6
            
            train_loss_tensor = trainer.callback_metrics.get("train_loss")
            train_loss = train_loss_tensor.item() if train_loss_tensor is not None else 0.0
            if not torch.isfinite(torch.tensor(train_loss)):
                train_loss = 0.0
            
            val_r2_tensor = trainer.callback_metrics.get("val_r2")
            val_r2 = val_r2_tensor.item() if val_r2_tensor is not None else 0.0
            
            val_mae_tensor = trainer.callback_metrics.get("val_mae")
            val_mae = val_mae_tensor.item() if val_mae_tensor is not None else 0.0
            
            val_rmse_tensor = trainer.callback_metrics.get("val_rmse")
            val_rmse = val_rmse_tensor.item() if val_rmse_tensor is not None else 0.0
            
            generalization_gap = val_loss - train_loss
            
            minutes, seconds = divmod(trial_train_duration, 60)
            completion_msg = (
                f"\n[TRIAL {trial.number + 1}/{GNN_CONFIG['trials']}] COMPLETED\n"
                f"  Epochs trained: {trainer.current_epoch + 1}/{GNN_CONFIG['epochs']}\n"
                f"  Training time: {int(minutes)}m {seconds:.1f}s\n"
                f"  Final metrics:\n"
                f"    - val_loss: {val_loss:.6f}\n"
                f"    - val_r2: {val_r2:.4f}\n"
                f"    - val_mae: {val_mae:.4f}\n"
                f"    - val_rmse: {val_rmse:.4f}\n"
                f"    - train_loss: {train_loss:.6f}\n"
                f"    - generalization_gap: {generalization_gap:.6f}\n"
            )
            print(completion_msg)
            print("-" * 80)
            append_to_log(completion_msg, log_file)
            if fallback_used:
                append_to_log(
                    f"[INFO] Trial {trial.number + 1} completed after DataLoader fallback (num_workers=0).",
                    log_file,
                )
            
            trial_message = (
                f"Trial {trial.number + 1} completed with val_loss: {val_loss:.6f} in {trainer.current_epoch + 1} epochs | "
                f"Hyperparameters: hidden_size={hidden_size}, num_layers={num_layers}, "
                f"dropout_prob={dropout_prob:.6f}, learning_rate={learning_rate:.8f}, "
                f"weight_decay={weight_decay:.8f}, pk_embed_dim={pk_embed_dim}, "
                f"graph_hidden_dim={graph_hidden_dim}, num_graph_layers={num_graph_layers}, "
                f"target_w_speed={target_weight_speed:.3f}, target_w_intensity={target_weight_intensity:.3f} | "
                f"Metrics: val_r2={val_r2:.4f}, val_mae={val_mae:.4f}, val_rmse={val_rmse:.4f}, generalization_gap={generalization_gap:.6f}"
            )
            append_to_log(trial_message, log_file)
            
            # Save trial results to CSV
            if csv_path is not None:
                trial_results = {
                    'trial_number': trial.number + 1,
                    'gpu_id': gpu_id,
                    'hidden_size': hidden_size,
                    'num_layers': num_layers,
                    'dropout_prob': dropout_prob,
                    'learning_rate': learning_rate,
                    'weight_decay': weight_decay,
                    'pk_embed_dim': pk_embed_dim,
                    'graph_hidden_dim': graph_hidden_dim,
                    'num_graph_layers': num_graph_layers,
                    'target_weight_speed': target_weight_speed,
                    'target_weight_intensity': target_weight_intensity,
                    'epochs_trained': trainer.current_epoch + 1,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'val_r2': val_r2,
                    'val_mae': val_mae,
                    'val_rmse': val_rmse,
                    'generalization_gap': generalization_gap,
                    'training_time_seconds': trial_train_duration,
                    'is_nan_loss': val_loss >= 1e6,
                    'error_message': '',
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                
                # Write to CSV (thread-safe with file locking)
                file_exists = os.path.exists(csv_path)
                with best_tracker['lock']:
                    with open(csv_path, 'a', newline='') as csvfile:
                        writer = csv.DictWriter(csvfile, fieldnames=trial_results.keys())
                        if not file_exists:
                            writer.writeheader()
                        writer.writerow(trial_results)
                
                print(f"[CSV] Trial {trial.number + 1} results saved to: {csv_path}")
            
            # Save best model
            is_new_best = False
            with best_tracker['lock']:
                if val_loss < best_tracker['val_loss']:
                    best_tracker['val_loss'] = val_loss
                    best_tracker['trial_number'] = trial.number
                    is_new_best = True
            
            if is_new_best:
                best_model_path = os.path.join(output_dir, f"best-model_model=GNN-version={model_time}.pt")
                checkpoint_files = glob.glob(os.path.join(checkpoint_dir, f"trial_{trial.number}_gnn-version={model_time}*.ckpt"))
                if checkpoint_files:
                    checkpoint = torch.load(checkpoint_files[0], map_location='cpu')
                    state_dict = checkpoint.get('state_dict', checkpoint)
                    torch.save(state_dict, best_model_path)
                    
                    metadata = {
                        'model_name': 'SpatioTemporalGNN_LSTM_V5',
                        'model_version': 'v5',
                        'model_time': model_time,
                        'trial_number': trial.number + 1,
                        'input_size': input_size,
                        'static_size': static_size,
                        'output_size': output_size,
                        'num_pks': num_pks,
                        'best_val_loss': val_loss,
                        'hyperparameters': {
                            'hidden_size': hidden_size,
                            'num_layers': num_layers,
                            'dropout_prob': dropout_prob,
                            'learning_rate': learning_rate,
                            'weight_decay': weight_decay,
                            'pk_embed_dim': pk_embed_dim,
                            'graph_hidden_dim': graph_hidden_dim,
                            'num_graph_layers': num_graph_layers,
                            'target_weight_speed': target_weight_speed,
                            'target_weight_intensity': target_weight_intensity,
                            'target_weights': target_weights,
                        },
                        'ablation_settings': ABLATION_SETTINGS,
                        'gnn_config': GNN_CONFIG,
                        'test_mode': TEST_MODE,
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    }
                    with open(os.path.join(output_dir, f"best-model-metadata_model=GNN-version={model_time}.json"), 'w') as f:
                        json.dump(metadata, f, indent=2)
                    
                    best_msg = (
                        f"\n{'='*80}\n"
                        f"🏆 NEW BEST MODEL FOUND! Trial {trial.number + 1}\n"
                        f"   val_loss: {val_loss:.6f} | val_r2: {val_r2:.4f} | val_mae: {val_mae:.4f} | val_rmse: {val_rmse:.4f}\n"
                        f"   Model saved to: {best_model_path}\n"
                        f"   Metadata saved to: {os.path.join(output_dir, f'best-model-metadata_model=GNN-version={model_time}.json')}\n"
                        f"{'='*80}\n"
                    )
                    print(best_msg)
                    append_to_log(f"🏆 NEW BEST MODEL - Trial {trial.number + 1} - val_loss: {val_loss:.6f} - val_rmse: {val_rmse:.4f}", log_file)
            
            del model, trainer
            _shutdown_dataloaders(train_loader, val_loader)
            del train_loader, val_loader
            train_loader = None
            val_loader = None
            torch.cuda.empty_cache()
            gc.collect()
            
            return val_loss
            
        except optuna.TrialPruned:
            # Log pruned trial to CSV before re-raising
            trial_train_duration = time.time() - trial_train_start if 'trial_train_start' in dir() else 0
            epochs_trained = trainer.current_epoch + 1 if 'trainer' in dir() and trainer is not None else 0
            
            # Try to get metrics if available
            val_loss = None
            train_loss = None
            val_r2 = None
            val_mae = None
            val_rmse = None
            if 'trainer' in dir() and trainer is not None:
                val_loss_tensor = trainer.callback_metrics.get("val_loss")
                val_loss = val_loss_tensor.item() if val_loss_tensor is not None else None
                train_loss_tensor = trainer.callback_metrics.get("train_loss")
                train_loss = train_loss_tensor.item() if train_loss_tensor is not None else None
                val_r2_tensor = trainer.callback_metrics.get("val_r2")
                val_r2 = val_r2_tensor.item() if val_r2_tensor is not None else None
                val_mae_tensor = trainer.callback_metrics.get("val_mae")
                val_mae = val_mae_tensor.item() if val_mae_tensor is not None else None
                val_rmse_tensor = trainer.callback_metrics.get("val_rmse")
                val_rmse = val_rmse_tensor.item() if val_rmse_tensor is not None else None
            
            pruned_msg = f"[PRUNED] Trial {trial.number + 1} pruned at epoch {epochs_trained}"
            print(pruned_msg)
            append_to_log(pruned_msg, log_file)
            
            if csv_path is not None:
                pruned_results = {
                    'trial_number': trial.number + 1,
                    'gpu_id': gpu_id,
                    'hidden_size': hidden_size,
                    'num_layers': num_layers,
                    'dropout_prob': dropout_prob,
                    'learning_rate': learning_rate,
                    'weight_decay': weight_decay,
                    'pk_embed_dim': pk_embed_dim,
                    'graph_hidden_dim': graph_hidden_dim,
                    'num_graph_layers': num_graph_layers,
                    'target_weight_speed': target_weight_speed,
                    'target_weight_intensity': target_weight_intensity,
                    'epochs_trained': epochs_trained,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'val_r2': val_r2,
                    'val_mae': val_mae,
                    'val_rmse': val_rmse,
                    'generalization_gap': (val_loss - train_loss) if val_loss and train_loss else None,
                    'training_time_seconds': trial_train_duration,
                    'is_nan_loss': False,
                    'error_message': 'PRUNED',
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                
                file_exists = os.path.exists(csv_path)
                with best_tracker['lock']:
                    with open(csv_path, 'a', newline='') as csvfile:
                        writer = csv.DictWriter(csvfile, fieldnames=pruned_results.keys())
                        if not file_exists:
                            writer.writeheader()
                        writer.writerow(pruned_results)
                
                print(f"[CSV] Pruned trial {trial.number + 1} results saved to CSV")
            
            # Clean up - shut down DataLoaders BEFORE deleting model/trainer
            # to prevent worker processes from accessing freed CUDA memory
            _shutdown_dataloaders(train_loader, val_loader)
            train_loader = None
            val_loader = None
            if 'model' in dir():
                del model
            if 'trainer' in dir():
                del trainer
            torch.cuda.empty_cache()
            gc.collect()
            
            raise
        except Exception as e:
            error_msg = f"Trial {trial.number + 1} failed with error: {str(e)}"
            print(f"[ERROR] {error_msg}")
            import traceback
            traceback_str = traceback.format_exc()
            append_to_log(f"[ERROR] {error_msg}", log_file)
            append_to_log(f"Traceback: {traceback_str}", log_file)
            
            # Save error to CSV
            if csv_path is not None:
                error_results = {
                    'trial_number': trial.number + 1,
                    'gpu_id': gpu_id,
                    'hidden_size': None,
                    'num_layers': None,
                    'dropout_prob': None,
                    'learning_rate': None,
                    'weight_decay': None,
                    'pk_embed_dim': None,
                    'graph_hidden_dim': None,
                    'num_graph_layers': None,
                    'epochs_trained': 0,
                    'train_loss': None,
                    'val_loss': 1e6,
                    'val_r2': None,
                    'val_mae': None,
                    'val_rmse': None,
                    'generalization_gap': None,
                    'training_time_seconds': 0,
                    'is_nan_loss': True,
                    'error_message': str(e)[:500],  # Limit error message length
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                
                file_exists = os.path.exists(csv_path)
                with best_tracker['lock']:
                    with open(csv_path, 'a', newline='') as csvfile:
                        writer = csv.DictWriter(csvfile, fieldnames=error_results.keys())
                        if not file_exists:
                            writer.writeheader()
                        writer.writerow(error_results)
            
            # Clean up DataLoaders and GPU memory
            _shutdown_dataloaders(train_loader, val_loader)
            train_loader = None
            val_loader = None
            torch.cuda.empty_cache()
            gc.collect()
            
            return 1e6
    
    return objective


def train_gnn_model(df_train_data, selected_features, output_dir, model_time):
    """Train GNN-LSTM model with Optuna optimization"""
    print("\n" + "=" * 80)
    print("STAGE 1: GNN-LSTM HYPERPARAMETER OPTIMIZATION")
    print("=" * 80)
    sys.stdout.flush()
    
    log_file = os.path.join(output_dir, f'gnn_training_log_{model_time}.txt')
    perf_log_file = os.path.join(output_dir, f'gnn_perf_log_{model_time}.txt')
    checkpoint_dir = os.path.join(output_dir, 'gnn_checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Prepare data
    result = prepare_gnn_data(df_train_data, selected_features, model_time, output_dir)
    (temporal_train, static_train, targets_train,
     temporal_val, static_val, targets_val,
     scaler_temporal, scaler_static, scaler_targets,
     df_train, df_val, cols_targets, temporal_cols, static_features) = result
    
    # Create datasets (not DataLoaders - DataLoaders are created per trial to
    # prevent worker process corruption when trials are pruned or fail)
    train_dataset, val_dataset, input_size, static_size, output_size, num_pks = create_gnn_datasets(
        temporal_train, static_train, targets_train,
        temporal_val, static_val, targets_val,
        df_train, df_val, GNN_CONFIG['batch_size'], GNN_CONFIG['sequence_length']
    )
    
    print("\n" + "="*80)
    print("[OPTUNA] Starting hyperparameter optimization...")
    print(f"[INFO] Input size: {input_size}, Static size: {static_size}, Output size: {output_size}")
    print(f"[INFO] Number of PKs: {num_pks}")
    print("="*80 + "\n")
    sys.stdout.flush()
    
    # Create CSV file path for training evolution tracking
    csv_path = os.path.join(output_dir, f'training-evolution_model=GNN-version={model_time}.csv')
    print(f"[CSV] Training evolution will be saved to: {csv_path}")
    
    # Create and run optimization (pass datasets, not DataLoaders)
    objective = create_gnn_objective(
        input_size, static_size, output_size, num_pks,
        train_dataset, val_dataset, log_file, perf_log_file, model_time,
        checkpoint_dir, output_dir, N_GPUS, csv_path
    )
    
    # V5: Increased n_warmup_steps from 5 to 8 to avoid premature pruning
    # of trials with low learning rates that need more epochs to converge
    # The best MODEL is already checkpointed incrementally (see the
    # `is_new_best` block above), so a killed job never loses its best weights.
    # The SEARCH, however, lived only in memory: a resubmit restarted at trial 1
    # and redid work already paid for. Setting AP7_OPTUNA_DB to a path on shared
    # storage persists the study, so a requeued job resumes where it stopped.
    # Left unset, behaviour is exactly as before -- running jobs are unaffected.
    _study_kw = {}
    _db = os.environ.get("AP7_OPTUNA_DB", "").strip()
    if _db:
        os.makedirs(os.path.dirname(_db) or ".", exist_ok=True)
        _study_kw = dict(storage=f"sqlite:///{_db}",
                         study_name=os.environ.get("AP7_OPTUNA_STUDY", "gnn"),
                         load_if_exists=True)
        print(f"[OPTUNA] persistent study at {_db} (resumable)")
    study = optuna.create_study(
        direction="minimize",
        **_study_kw,
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,     # Run first 5 trials without pruning to establish baseline
            n_warmup_steps=8,       # V5: Don't prune until epoch 8 (was 5)
            interval_steps=1,       # Check for pruning every epoch
            n_min_trials=3          # Need at least 3 completed trials before pruning
        )
    )
    
    print(f"\n[OPTUNA] Starting {GNN_CONFIG['trials']} trials across {N_GPUS} GPUs...")
    print(f"[OPTUNA] Using MedianPruner for early trial termination")
    sys.stdout.flush()
    start_time = time.time()
    study.optimize(objective, n_trials=GNN_CONFIG['trials'], n_jobs=N_GPUS)
    duration = time.time() - start_time
    
    print('=' * 80)
    print('[COMPLETE] GNN OPTIMIZATION COMPLETED')
    print('=' * 80)
    print(f"[GNN] ⏱️ Optuna optimization: {format_duration(duration)}")
    print(f"[TRIALS] Completed: {len(study.trials)} trials")
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    failed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]
    print(f"[TRIALS] Breakdown: {len(completed_trials)} completed, {len(pruned_trials)} pruned, {len(failed_trials)} failed")
    
    if len(completed_trials) > 0:
        print("[BEST] Best hyperparameters found by Optuna:")
        for key, value in study.best_params.items():
            print(f"  - {key}: {value}")
        print(f"[RESULT] Best validation loss: {study.best_value:.6f}")
    else:
        print("[WARNING] No trials completed successfully - cannot determine best hyperparameters")
    print('=' * 80)
    sys.stdout.flush()
    
    return study, {
        'input_size': input_size,
        'static_size': static_size,
        'output_size': output_size,
        'num_pks': num_pks,
        'cols_targets': cols_targets,
        'temporal_cols': temporal_cols,
        'static_features': static_features
    }


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Main execution pipeline - GNN-LSTM Only Training"""
    
    # Suppress logs
    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
    logging.getLogger("lightning_fabric").setLevel(logging.ERROR)
    warnings.filterwarnings('ignore')
    
    os.environ["LOCAL_RANK"] = "0"
    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    # Build experiment paths
    global EXPERIMENT_NAME, OUTPUT_DIR
    EXPERIMENT_NAME = build_experiment_name()
    OUTPUT_DIR = os.path.join(OUTPUT_BASE_DIR, EXPERIMENT_NAME)

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Generate model timestamp
    model_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Print configuration
    print("\n" + "=" * 80)
    print("ABLATION STUDY V5 - GNN-LSTM ONLY TRAINING (IMPROVED)")
    print("=" * 80)
    print(f"Experiment: {EXPERIMENT_NAME}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Test Mode: {TEST_MODE}")
    print(f"GPU Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
    print("=" * 80)
    sys.stdout.flush()
    
    print_ablation_config()
    
    # Save configuration
    config = {
        'experiment_name': EXPERIMENT_NAME,
        'model_time': model_time,
        'test_mode': TEST_MODE,
        'ablation_settings': ABLATION_SETTINGS,
        'gnn_config': GNN_CONFIG,
        'train_dates': {'start': TRAIN_START_DATE, 'end': TRAIN_END_DATE},
        'hyperparameter_search_space': get_gnn_hyperparams(),
        'pipeline_version': 'v5_gnn_only',
        'next_step': 'Run v5_xgboost_only with PREVIOUS_EXPERIMENT_DIR pointing to this output',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Write generic and version-tagged config files
    config_path = os.path.join(OUTPUT_DIR, 'experiment_config.json')
    config_versioned_path = os.path.join(OUTPUT_DIR, f'experiment_config_{PIPELINE_VERSION}.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    with open(config_versioned_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    total_start = time.time()
    stage_times = {}  # Track timing for each stage
    
    # =========================================================================
    # STAGE 0: DATA LOADING
    # =========================================================================
    stage0_start = time.time()
    print("\n" + "=" * 80)
    print("STAGE 0: DATA LOADING")
    print("=" * 80)
    
    df_full, selected_features = load_data()
    
    # Filter for training
    df_train = filter_data_by_date(df_full, TRAIN_START_DATE, TRAIN_END_DATE)
    print(f"[INFO] Training data: {len(df_train):,} rows ({TRAIN_START_DATE} to {TRAIN_END_DATE})")
    sys.stdout.flush()
    
    # Free full dataframe - we only need training data for GNN
    del df_full
    gc.collect()
    
    stage_times['stage0_data_loading'] = time.time() - stage0_start
    print_step_timing("STAGE 0: Data Loading Complete", stage0_start)
    
    # =========================================================================
    # STAGE 1: GNN-LSTM TRAINING
    # =========================================================================
    stage1_start = time.time()
    gnn_study, gnn_info = train_gnn_model(df_train, selected_features, OUTPUT_DIR, model_time)
    stage_times['stage1_gnn_training'] = time.time() - stage1_start
    print_step_timing("STAGE 1: GNN-LSTM Training Complete", stage1_start)
    
    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    total_duration = time.time() - total_start
    
    print("\n" + "=" * 80)
    print("🎉 GNN-LSTM TRAINING COMPLETE")
    print("=" * 80)
    
    # Print timing breakdown
    print("\n📊 TIMING BREAKDOWN:")
    print("-" * 60)
    print(f"  Stage 0 - Data Loading:      {format_duration(stage_times['stage0_data_loading']):>15}")
    print(f"  Stage 1 - GNN-LSTM Training: {format_duration(stage_times['stage1_gnn_training']):>15}")
    print("-" * 60)
    print(f"  TOTAL:                       {format_duration(total_duration):>15}")
    print("=" * 80)
    sys.stdout.flush()
    
    print(f"\n📁 Results saved to: {OUTPUT_DIR}")
    print("\nFiles generated:")
    for f in os.listdir(OUTPUT_DIR):
        print(f"  - {f}")
    
    # Save final summary with timing details
    summary = {
        'experiment_name': EXPERIMENT_NAME,
        'total_duration_minutes': total_duration / 60,
        'total_duration_hours': total_duration / 3600,
        'stage_times_seconds': stage_times,
        'stage_times_formatted': {k: format_duration(v) for k, v in stage_times.items()},
        'gnn_best_val_loss': gnn_study.best_value if gnn_study.best_trial else None,
        'gnn_best_params': gnn_study.best_params if gnn_study.best_trial else None,
        'gnn_trials_completed': len([t for t in gnn_study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
        'gnn_trials_pruned': len([t for t in gnn_study.trials if t.state == optuna.trial.TrialState.PRUNED]),
        'gnn_trials_failed': len([t for t in gnn_study.trials if t.state == optuna.trial.TrialState.FAIL]),
        'ablation_settings': ABLATION_SETTINGS,
        'test_mode': TEST_MODE,
        'completed_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'next_step': f'Run v5_xgboost_only with PREVIOUS_EXPERIMENT_DIR={OUTPUT_DIR}'
    }
    
    # Write generic and version-tagged summary files
    summary_path = os.path.join(OUTPUT_DIR, 'experiment_summary.json')
    summary_versioned_path = os.path.join(OUTPUT_DIR, f'experiment_summary_{PIPELINE_VERSION}.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    with open(summary_versioned_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "=" * 80)
    print("NEXT STEPS:")
    print("=" * 80)
    print(f"To complete the pipeline with XGBoost training and simulation, run:")
    print(f"")
    print(f"  export PREVIOUS_EXPERIMENT_DIR=\"{OUTPUT_DIR}\"")
    print(f"  python src/training/ablation_study_v5_xgboost_only.py")
    print(f"")
    print("=" * 80)
    print("SUCCESS!")
    print("=" * 80)
    sys.stdout.flush()
    
    return True


if __name__ == "__main__":
    # Lightweight CLI layer to optionally select the time resolution dynamically.
    #
    # Supports either:
    #   - --time-resolution 5min|10min|15min|30min|45min|60min
    #   - Shorthand flags: --5min, --10min, --15min, --30min, --45min, --60min
    #
    # When provided, the chosen token replaces the interval embedded in DATA_FILE
    # (e.g. "..._15min_..." -> "..._10min_..."), so that 10min behaves exactly
    # like the other supported resolutions.
    parser = argparse.ArgumentParser(
        description="Ablation Study V5 - GNN-LSTM Only Training (Improved)"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--time-resolution",
        choices=TIME_RESOLUTION_CHOICES,
        help=(
            "Time resolution token to use (e.g. 5min, 10min, 15min, 30min, 45min, 60min). "
            "If omitted, uses the resolution embedded in DATA_FILE."
        ),
    )
    for token in TIME_RESOLUTION_CHOICES:
        minutes = token.replace("min", "")
        group.add_argument(
            f"--{minutes}min",
            dest="time_resolution",
            action="store_const",
            const=token,
            help=argparse.SUPPRESS,
        )

    args = parser.parse_args()

    # If a resolution is explicitly requested, rewrite DATA_FILE accordingly.
    if getattr(args, "time_resolution", None):
        chosen = args.time_resolution
        # Replace the first "<number>min" token in DATA_FILE, if present.
        new_data_file = re.sub(r'\d+min', chosen, DATA_FILE, count=1)
        DATA_FILE = new_data_file
        DATA_PATH = os.path.join(_AP7_DATA, DATA_FILE)
        print(f"[CONFIG] Using DATA_FILE={DATA_FILE} for time resolution {chosen}")
        sys.stdout.flush()

    success = main()
    sys.exit(0 if success else 1)
