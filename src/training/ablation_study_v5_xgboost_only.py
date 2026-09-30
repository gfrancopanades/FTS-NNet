#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Ablation Study Pipeline V5: XGBoost Training Only
=================================================

VERSION 5.0 - XGBOOST TRAINING ON GNN PREDICTIONS (IMPROVED)

Critical change from V4:
- XGBoost is now trained on GNN-predicted traffic features instead of ground truth.
  This fixes the cascading error problem: in V4, XGBoost learned from perfect traffic
  data but received imperfect GNN predictions at simulation time, causing ROC-AUC
  to collapse from 0.97 to 0.51 (random).
- C_NIVELL_AFECTACIO trials reduced from 100 to 50 (converged by trial 32 in V1).

This script expects the output directory from v5_gnn_only.py to be provided via
PREVIOUS_EXPERIMENT_DIR.

Author: Gerard Franco
Date: February 2026
Affiliation: Universitat Politecnica de Catalunya
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
import logging
import json
import joblib
import csv
import threading
import glob
import warnings
import shutil
import re
import argparse
from datetime import datetime

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score
)

import torch
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb
import optuna

# Custom imports
from src.models.spatiotemporal_gnn_lstm_v5 import SpatioTemporalGNN_LSTM_V5


# ============================================================================
# CONFIGURATION - RESUME FROM V4_GNN_ONLY OUTPUT
# ============================================================================

# Path to the v4_gnn_only experiment directory containing the trained GNN model
PREVIOUS_EXPERIMENT_DIR = os.environ.get(
    'PREVIOUS_EXPERIMENT_DIR',
    ''  # Must be set via environment variable
)

# Model timestamp from the previous experiment (auto-detected if not set)
PREVIOUS_MODEL_TIME = os.environ.get('PREVIOUS_MODEL_TIME', None)

# TEST MODE: Use reduced data for quick testing
TEST_MODE = False  # Set to True for quick testing

# ABLATION STUDY: Feature groups to include (set to False to exclude)
# Should match the settings used in v4_gnn_only.py
ABLATION_SETTINGS = {
    'include_weather_1d': True,
    'include_weather_3d': True,
    'include_traffic': True,
    'include_geometry': True,
    'include_mobility': True,
    'include_temporal': True,
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
    'temporal': ['anyo', 'mes', 'dia', 'diaSem', 'hor', '5min', '10min', '15min', '30min', '45min', '60min'],
    'imputation': ['speed_imputation', 'intensity_imputation'],
    'static_core': ['pk', 'via', 'sen', 'car'],
}

# GPU Configuration
USE_GPU = True
N_GPUS = int(os.environ.get('ABLATION_NUM_GPUS', 1))
N_THREADS = 16

# Training Configuration
RANDOM_STATE = 42

# Pipeline version tag (used to version output filenames)
PIPELINE_VERSION = "v5"

# GNN Configuration (for reference - model is pre-trained)
GNN_CONFIG = {
    'sequence_length': 12,  # Must match v4_gnn_only
}

# XGBoost Configuration
XGBOOST_CONFIG = {
    'trials_accident': 100 if not TEST_MODE else 10,
    'trials_secondary': 50 if not TEST_MODE else 10,  # V5: Reduced from 100 (converged early in V1)
    'cv_folds': 3,
    'early_stopping': 50 if not TEST_MODE else 10,
}

# Target/feature selection for XGBoost stage
XGBOOST_TARGET_COLUMNS = ['ACCIDENT', 'C_NIVELL_AFECTACIO']
XGBOOST_EXCLUDED_FEATURE_COLUMNS = ['F_TEMPS_AFECTACIO', 'F_LONG_AFECTACIO']

# Date Configuration
if TEST_MODE:
    TRAIN_START_DATE = '2024-04-01'
    TRAIN_END_DATE = '2024-05-01'
else:
    TRAIN_START_DATE = '2024-04-01'
    TRAIN_END_DATE = '2025-06-01'

# Data paths (default to 5min; can be overridden via CLI time resolution)
TIME_RESOLUTION_CHOICES = ["5min", "10min", "15min", "30min", "45min", "60min"]
DATA_FILE = 'CrashGNNLSTM_v1_vel-extinrix_int_geo_mob_wthr_5min_fund-propag-ltd_from_20240404_to_20251001.csv'
DATA_PATH = os.path.join(_AP7_DATA, DATA_FILE)

# Output directory
OUTPUT_BASE_DIR = _AP7_EXPERIMENTS
EXPERIMENT_NAME = None
OUTPUT_DIR = None


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def format_duration(seconds):
    """Format duration in human-readable format"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}min ({seconds:.0f}s)"
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
    print("\nFeature groups:")
    for group, included in ABLATION_SETTINGS.items():
        status = "✓ INCLUDED" if included else "✗ EXCLUDED"
        group_name = group.replace('include_', '')
        features = FEATURE_GROUPS.get(group_name, [])
        print(f"  {status}: {group_name} ({len(features)} features)")

    selected = get_selected_features()
    print(f"\nTotal selected features: {len(selected)}")
    print("=" * 80)
    sys.stdout.flush()


def detect_model_time(experiment_dir):
    """Auto-detect the model timestamp from the experiment directory"""
    metadata_files = glob.glob(os.path.join(experiment_dir, 'best-model-metadata_model=GNN-version=*.json'))
    if metadata_files:
        filename = os.path.basename(metadata_files[0])
        parts = filename.replace('.json', '').split('version=')
        if len(parts) > 1:
            return parts[1]

    scaler_files = glob.glob(os.path.join(experiment_dir, 'scaler-temporal_model=GNN-version=*.pkl'))
    if scaler_files:
        filename = os.path.basename(scaler_files[0])
        parts = filename.replace('.pkl', '').split('version=')
        if len(parts) > 1:
            return parts[1]

    raise ValueError(f"Could not auto-detect model timestamp from {experiment_dir}")


def get_job_id():
    """Get scheduler job id or fallback timestamp."""
    return (
        os.environ.get("SLURM_JOB_ID")
        or os.environ.get("JOB_ID")
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


def get_time_resolution_token():
    """Extract time-resolution token (e.g. 5min, 15min) from DATA_FILE."""
    match = re.search(r'(\d+min)', DATA_FILE)
    return match.group(1) if match else "unknown-res"


def get_time_resolution_minutes():
    """Extract numeric resolution in minutes from DATA_FILE."""
    token = get_time_resolution_token()
    match = re.match(r'(\d+)min', token)
    if match:
        return int(match.group(1))
    return 5


def get_time_resolution_feature():
    """Get temporal feature column name for current dataset resolution."""
    return f"{get_time_resolution_minutes()}min"


def build_experiment_name():
    """Create experiment folder name using exclusions, resolution and job id."""
    prefix = get_experiment_prefix()
    resolution = get_time_resolution_token()
    job_id = get_job_id()
    return f"v5_xgb_{prefix}_{resolution}_{job_id}"


# ============================================================================
# DATA LOADING
# ============================================================================

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

    if 'Any' in df.columns:
        df.rename(columns={'Any': 'anyo'}, inplace=True)

    if 'via' in df.columns and df['via'].dtype == 'object':
        df['via'] = df['via'].map({'AP-7': 0}).fillna(0).astype(int)

    if 'sen' in df.columns and df['sen'].dtype == 'object':
        df['sen'] = df['sen'].map({'dec': 0, 'cre': 1}).fillna(1).astype(int)

    if 'min' not in df.columns:
        for minutes in (5, 10, 15, 30, 45, 60):
            col = f"{minutes}min"
            if col in df.columns:
                df['min'] = df[col]
                break
        if 'min' not in df.columns and 'dat' in df.columns:
            df['min'] = pd.to_datetime(df['dat']).dt.minute
        elif 'min' not in df.columns:
            df['min'] = 0

    resolution_minutes = get_time_resolution_minutes()
    resolution_feature = get_time_resolution_feature()
    if resolution_feature not in df.columns:
        if resolution_minutes == 5:
            df[resolution_feature] = (df['min'] // 5).astype(int)
        else:
            df[resolution_feature] = ((df['min'] // resolution_minutes) * resolution_minutes).astype(int)

    df['dat'] = pd.to_datetime(
        df['anyo'].astype(str) + '-' +
        df['mes'].astype(str).str.zfill(2) + '-' +
        df['dia'].astype(str).str.zfill(2) + ' ' +
        df['hor'].astype(str).str.zfill(2) + ':' +
        df['min'].astype(str).str.zfill(2) + ':00'
    )

    df.sort_values(by=['via', 'sen', 'pk', 'anyo', 'mes', 'dia', 'hor', 'min'], inplace=True)

    if 'F_TEMPS_AFECTACIO' in df.columns and 'F_LONG_AFECTACIO' in df.columns:
        df['F_RETENCIO'] = df['F_TEMPS_AFECTACIO'] * df['F_LONG_AFECTACIO']
    else:
        df['F_RETENCIO'] = 0

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
# LOAD PRE-TRAINED GNN MODEL
# ============================================================================

def load_pretrained_gnn_model(experiment_dir, model_time):
    """Load pre-trained GNN model, scalers, and metadata from v4_gnn_only output"""
    print("\n" + "=" * 80)
    print("LOADING PRE-TRAINED GNN MODEL FROM V4_GNN_ONLY")
    print("=" * 80)
    print(f"[INFO] Loading from: {experiment_dir}")
    print(f"[INFO] Model timestamp: {model_time}")
    sys.stdout.flush()

    load_start = time.time()

    # Load GNN model metadata
    metadata_path = os.path.join(experiment_dir, f"best-model-metadata_model=GNN-version={model_time}.json")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"GNN metadata not found: {metadata_path}")

    with open(metadata_path, 'r') as f:
        gnn_metadata = json.load(f)

    # BL-F0 Historical Average: non-parametric, no torch model. Short-circuit
    # to the saved lookup artifact; downstream `generate_gnn_predictions...`
    # detects `is_historical_average` and fills the *_gnn columns by calendar
    # join instead of running inference.
    if gnn_metadata.get('architecture') == 'historical_average':
        from src.models.benchmark_forecasters import HistoricalAverageForecaster
        ha_path = os.path.join(
            experiment_dir, f"historical-average_model=GNN-version={model_time}.pkl")
        if not os.path.exists(ha_path):
            raise FileNotFoundError(f"Historical-Average artifact not found: {ha_path}")
        ha_model = HistoricalAverageForecaster.load(ha_path)
        print(f"[ARCH] Loaded Historical-Average (BL-F0) forecaster: {ha_path}")
        scaler_temporal = joblib.load(os.path.join(
            experiment_dir, f'scaler-temporal_model=GNN-version={model_time}.pkl'))
        scaler_static = joblib.load(os.path.join(
            experiment_dir, f'scaler-static_model=GNN-version={model_time}.pkl'))
        scaler_targets = joblib.load(os.path.join(
            experiment_dir, f'scaler-targets_model=GNN-version={model_time}.pkl'))
        gnn_info = {
            'input_size': gnn_metadata.get('input_size'),
            'static_size': gnn_metadata.get('static_size'),
            'output_size': gnn_metadata.get('output_size', 3),
            'num_pks': gnn_metadata.get('num_pks'),
            'cols_targets': ['mean_speed', 'intTot', 'intP'],
            'temporal_cols': None,
            'static_features': ['car', 'segment', 'ang_curv', 'ang_pend_pos',
                                'ang_pend_neg', 'via', 'sen', 'pk'],
        }
        return ha_model, gnn_metadata, gnn_info, scaler_temporal, scaler_static, scaler_targets

    print("[SUCCESS] GNN metadata loaded")
    best_val_loss = gnn_metadata.get('best_val_loss', None)
    best_val_goal_loss = gnn_metadata.get('best_val_goal_loss', None)

    # Support both v4/v5 (best_val_loss) and v6 (best_val_goal_loss / best_val_loss_task) metadata.
    if isinstance(best_val_goal_loss, (int, float)):
        print(f"  - Best validation goal loss (v6): {best_val_goal_loss:.6f}")
    if isinstance(best_val_loss, (int, float)):
        print(f"  - Best validation loss: {best_val_loss:.6f}")
    elif best_val_loss is not None:
        # Non-numeric value (e.g. "N/A") – print as-is without numeric formatting.
        print(f"  - Best validation loss: {best_val_loss}")
    print(f"  - Trial number: {gnn_metadata.get('trial_number', 'N/A')}")
    sys.stdout.flush()

    # Load GNN model
    model_path = os.path.join(experiment_dir, f"best-model_model=GNN-version={model_time}.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"GNN model not found: {model_path}")

    params = gnn_metadata['hyperparameters']
    
    # V5: Support both V4 (SpatioTemporalGNN_LSTM) and V5 model loading
    model_kwargs = dict(
        input_size=gnn_metadata['input_size'],
        hidden_size=params['hidden_size'],
        num_layers=params['num_layers'],
        output_size=gnn_metadata['output_size'],
        dropout_prob=params['dropout_prob'],
        learning_rate=params['learning_rate'],
        weight_decay=params['weight_decay'],
        num_pks=gnn_metadata['num_pks'],
        static_feature_dim=gnn_metadata['static_size'],
        pk_embed_dim=params['pk_embed_dim'],
        graph_hidden_dim=params['graph_hidden_dim'],
        num_graph_layers=params['num_graph_layers'],
    )
    
    # Add V5-specific target_weights if available
    if 'target_weights' in params:
        model_kwargs['target_weights'] = params['target_weights']

    # Architecture-aware reconstruction. The default (field absent or
    # 'gnn_lstm_v5') keeps the original V5 behaviour byte-for-byte; the
    # 2nd-article Layer-1 benchmarks (BL-F1..F6) record their own
    # `architecture` token in the metadata and are rebuilt from the shared
    # registry. They take the SAME model_kwargs, so nothing else changes
    # downstream (feature generation + simulation call the model identically).
    architecture = gnn_metadata.get('architecture', 'gnn_lstm_v5')
    if architecture and architecture != 'gnn_lstm_v5':
        from src.models.benchmark_forecasters import build_forecaster
        gnn_model = build_forecaster(architecture, **model_kwargs)
        print(f"[ARCH] Reconstructed benchmark forecaster: architecture={architecture}")
    else:
        gnn_model = SpatioTemporalGNN_LSTM_V5(**model_kwargs)

    state_dict = torch.load(model_path, map_location='cpu')
    gnn_model.load_state_dict(state_dict)
    gnn_model.eval()
    print("[SUCCESS] GNN model loaded")
    print(f"  - Hidden size: {params['hidden_size']}")
    print(f"  - Num layers: {params['num_layers']}")
    print(f"  - PK embed dim: {params['pk_embed_dim']}")
    print(f"  - Graph hidden dim: {params['graph_hidden_dim']}")
    sys.stdout.flush()

    # Load scalers
    scaler_temporal_path = os.path.join(experiment_dir, f'scaler-temporal_model=GNN-version={model_time}.pkl')
    scaler_static_path = os.path.join(experiment_dir, f'scaler-static_model=GNN-version={model_time}.pkl')
    scaler_targets_path = os.path.join(experiment_dir, f'scaler-targets_model=GNN-version={model_time}.pkl')

    if not os.path.exists(scaler_temporal_path):
        raise FileNotFoundError(f"Temporal scaler not found: {scaler_temporal_path}")
    if not os.path.exists(scaler_static_path):
        raise FileNotFoundError(f"Static scaler not found: {scaler_static_path}")
    if not os.path.exists(scaler_targets_path):
        raise FileNotFoundError(f"Targets scaler not found: {scaler_targets_path}")

    scaler_temporal = joblib.load(scaler_temporal_path)
    scaler_static = joblib.load(scaler_static_path)
    scaler_targets = joblib.load(scaler_targets_path)
    print("[SUCCESS] Scalers loaded")
    sys.stdout.flush()

    gnn_info = {
        'input_size': gnn_metadata['input_size'],
        'static_size': gnn_metadata['static_size'],
        'output_size': gnn_metadata['output_size'],
        'num_pks': gnn_metadata['num_pks'],
        'cols_targets': ['mean_speed', 'intTot', 'intP'],
        'temporal_cols': None,
        'static_features': ['car', 'segment', 'ang_curv', 'ang_pend_pos', 'ang_pend_neg', 'via', 'sen', 'pk'],
    }

    load_duration = time.time() - load_start
    print(f"\n[COMPLETE] Pre-trained GNN model loaded in {format_duration(load_duration)}")
    print("=" * 80)
    sys.stdout.flush()

    return gnn_model, gnn_metadata, gnn_info, scaler_temporal, scaler_static, scaler_targets


# ============================================================================
# V5: GNN INFERENCE ON TRAINING DATA
# ============================================================================

def _create_gnn_sequences_for_inference(temporal_data, static_data, df, sequence_length):
    """Create sequences for GNN inference (no targets needed).
    
    Returns sequences, static features, pk_ids, sen_ids, and the indices
    into the original DataFrame that each sequence maps to (for writing
    predictions back).
    """
    from numpy.lib.stride_tricks import sliding_window_view

    unique_pks = sorted(df['pk'].unique())
    pk_to_id = {pk: idx for idx, pk in enumerate(unique_pks)}

    location_groups = df.groupby(['via', 'sen', 'pk']).size()

    all_sequences = []
    all_static = []
    all_pk_ids = []
    all_sen = []
    all_target_indices = []

    current_idx = 0
    n_features = temporal_data.shape[1]

    for (via, sen, pk), group_size in location_groups.items():
        if group_size <= sequence_length:
            current_idx += group_size
            continue

        location_temporal = temporal_data[current_idx:current_idx + group_size]
        location_static = static_data[current_idx:current_idx + group_size]

        n_sequences = group_size - sequence_length
        sequences = sliding_window_view(location_temporal, (sequence_length, n_features))
        sequences = sequences.squeeze(axis=1)[:n_sequences]

        seq_static = location_static[sequence_length:sequence_length + n_sequences]

        pk_id = pk_to_id[pk]
        pk_ids = np.full(n_sequences, pk_id, dtype=np.int32)
        sen_ids = np.full(n_sequences, sen, dtype=np.int32)

        target_indices = np.arange(current_idx + sequence_length,
                                    current_idx + sequence_length + n_sequences)

        all_sequences.append(sequences.copy())
        all_static.append(seq_static)
        all_pk_ids.append(pk_ids)
        all_sen.append(sen_ids)
        all_target_indices.append(target_indices)

        current_idx += group_size

    if not all_sequences:
        raise ValueError("No valid sequences created for GNN inference.")

    return (
        np.concatenate(all_sequences, axis=0).astype(np.float32),
        np.concatenate(all_static, axis=0).astype(np.float32),
        np.concatenate(all_pk_ids, axis=0),
        np.concatenate(all_sen, axis=0),
        np.concatenate(all_target_indices, axis=0),
    )


def generate_gnn_predictions_on_training_data(
    gnn_model, df_train, scaler_temporal, scaler_static, scaler_targets,
    gnn_metadata, sequence_length=8, batch_size=2048, experiment_dir=None
):
    """Run GNN inference on training data to generate predicted traffic features.
    
    This is the critical V5 change: instead of training XGBoost on ground-truth
    traffic (mean_speed, intTot, intP), we train it on what the GNN actually
    predicts. This eliminates the distribution mismatch between training and
    simulation.
    
    Returns:
        df_with_preds: A copy of df_train with columns 'mean_speed_gnn',
                       'intTot_gnn', 'intP_gnn' containing GNN predictions.
                       Rows without predictions (first `sequence_length` per
                       location) keep NaN and are dropped by the caller.
    """
    # BL-F0 Historical Average: fill *_gnn by prior-year-weekday-profile lookup
    # (volume-corrected with the recent weeks); no sequence building / inference.
    if getattr(gnn_model, 'is_historical_average', False):
        print("\n" + "=" * 80)
        print("BL-F0: GENERATING HISTORICAL-AVERAGE PREDICTIONS (calendar lookup)")
        print("=" * 80)
        ha_start = time.time()
        df_sorted = df_train.sort_values(by=['via', 'sen', 'pk', 'dat']).reset_index(drop=True)
        preds = gnn_model.predict_dataframe(df_sorted)
        df_out = df_sorted.copy()
        df_out['mean_speed_gnn'] = preds[:, 0]
        df_out['intTot_gnn'] = preds[:, 1]
        df_out['intP_gnn'] = preds[:, 2]
        print(f"[HA-PRED] Filled {len(df_out):,} rows by calendar lookup in "
              f"{format_duration(time.time() - ha_start)}")
        return df_out

    print("\n" + "=" * 80)
    print("V5: GENERATING GNN PREDICTIONS ON TRAINING DATA")
    print("=" * 80)
    inference_start = time.time()

    cols_targets = ['mean_speed', 'intTot', 'intP']
    static_features = ['car', 'segment', 'ang_curv', 'ang_pend_pos', 'ang_pend_neg', 'via', 'sen', 'pk']
    static_features = [f for f in static_features if f in df_train.columns]

    selected_features = get_selected_features()
    time_cols = ['dat', 'min', 'anyo', 'mes', 'dia', 'hor', '5min', '10min', '15min', '30min', '45min', '60min']
    temporal_features = [f for f in selected_features
                         if f not in static_features + cols_targets + time_cols]
    if ABLATION_SETTINGS['include_temporal']:
        temporal_features.extend(['diaSem', 'anyo', 'mes', 'dia', 'hor'])
    temporal_features = list(set([f for f in temporal_features if f in df_train.columns]))

    df_sorted = df_train.sort_values(by=['via', 'sen', 'pk', 'dat']).reset_index(drop=True)

    temporal_cols = [col for col in df_sorted.columns
                     if col not in cols_targets + static_features + ['dat', 'min']]
    temporal_cols = [c for c in temporal_cols if c in temporal_features or c in temporal_cols]

    temporal_features_df = df_sorted[temporal_cols].fillna(0)
    static_features_df = df_sorted[static_features].fillna(0)

    # Align feature names with those used when fitting the temporal scaler.
    # This is important when reusing GNN runs across different resolutions
    # (e.g. 15min vs 5min) or when new engineered features like F_RETENCIO
    # are present in the current data but were absent at GNN training time.
    if hasattr(scaler_temporal, "feature_names_in_"):
        expected_cols = list(scaler_temporal.feature_names_in_)
        # Add any missing columns as zeros
        for col in expected_cols:
            if col not in temporal_features_df.columns:
                temporal_features_df[col] = 0.0
        # Restrict and reorder columns to exactly match the scaler's training schema
        temporal_features_df = temporal_features_df[expected_cols]
    temporal_norm = scaler_temporal.transform(temporal_features_df)
    static_norm = scaler_static.transform(static_features_df)

    print(f"[GNN-PRED] Creating sequences for {len(df_sorted):,} rows...")
    sys.stdout.flush()

    seqs, statics, pk_ids, sen_ids, target_indices = _create_gnn_sequences_for_inference(
        temporal_norm, static_norm, df_sorted, sequence_length
    )

    # A model trained under time-major batching learned to combine a signal from
    # its corridor neighbours. Inference sorts location-major, so a 2048-row
    # batch sits inside a single kilometre post and `build_highway_graph`
    # returns self-loops -- the neighbour signal the weights expect never
    # arrives. Permuting the SEQUENCES (rather than the frame) restores
    # co-temporal batches; `target_indices` travels with them, so predictions
    # still land on the right rows and the returned frame keeps its order.
    _tm = False
    try:
        from src.training.time_major_patch import is_time_major_run as _itm
        _tm = bool(experiment_dir) and _itm(experiment_dir)
    except Exception as _exc:                       # never fail inference on this
        print(f"[GNN-PRED] time-major check unavailable ({_exc}); assuming location-major")
    if os.environ.get("AP7_FORCE_TIME_MAJOR_INFERENCE") == "1":
        _tm = True
    if _tm:
        _ts = pd.to_datetime(df_sorted['dat']).values[np.asarray(target_indices)]
        _order = np.lexsort((np.asarray(pk_ids), _ts))
        seqs = seqs[_order]
        statics = statics[_order]
        pk_ids = np.asarray(pk_ids)[_order]
        sen_ids = np.asarray(sen_ids)[_order]
        target_indices = np.asarray(target_indices)[_order]
        print(f"[GNN-PRED] TIME-MAJOR inference: sequences reordered so each batch "
              f"holds co-temporal kilometre posts (model trained with an active graph)")

    print(f"[GNN-PRED] Created {len(seqs):,} sequences. Running inference...")
    sys.stdout.flush()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gnn_model.eval()
    gnn_model = gnn_model.to(device)

    dataset = TensorDataset(
        torch.tensor(seqs, dtype=torch.float32),
        torch.tensor(pk_ids, dtype=torch.long),
        torch.tensor(statics, dtype=torch.float32),
        torch.tensor(sen_ids, dtype=torch.long),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    all_preds = []
    with torch.inference_mode():
        for batch_idx, (seq_b, pk_b, stat_b, sen_b) in enumerate(loader):
            seq_b = seq_b.to(device)
            pk_b = pk_b.to(device)
            stat_b = stat_b.to(device)
            sen_b = sen_b.to(device)
            preds = gnn_model(seq_b, pk_b, stat_b, sen_b)
            all_preds.append(preds.cpu().numpy())

            if (batch_idx + 1) % 500 == 0:
                print(f"[GNN-PRED] Batch {batch_idx + 1}/{len(loader)}")
                sys.stdout.flush()

    gnn_model = gnn_model.cpu()
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    all_preds_np = np.concatenate(all_preds, axis=0)

    preds_original = scaler_targets.inverse_transform(all_preds_np)

    df_out = df_sorted.copy()
    df_out['mean_speed_gnn'] = np.nan
    df_out['intTot_gnn'] = np.nan
    df_out['intP_gnn'] = np.nan

    df_out.iloc[target_indices, df_out.columns.get_loc('mean_speed_gnn')] = preds_original[:, 0]
    df_out.iloc[target_indices, df_out.columns.get_loc('intTot_gnn')] = preds_original[:, 1]
    df_out.iloc[target_indices, df_out.columns.get_loc('intP_gnn')] = preds_original[:, 2]

    n_with_preds = df_out['mean_speed_gnn'].notna().sum()
    inference_duration = time.time() - inference_start

    print(f"[GNN-PRED] Done. {n_with_preds:,} / {len(df_out):,} rows have GNN predictions")
    print(f"[GNN-PRED] Duration: {format_duration(inference_duration)}")

    r2_speed = 1 - (((df_out.loc[df_out['mean_speed_gnn'].notna(), 'mean_speed'] -
                       df_out.loc[df_out['mean_speed_gnn'].notna(), 'mean_speed_gnn']) ** 2).mean() /
                     df_out.loc[df_out['mean_speed_gnn'].notna(), 'mean_speed'].var())
    r2_intTot = 1 - (((df_out.loc[df_out['intTot_gnn'].notna(), 'intTot'] -
                        df_out.loc[df_out['intTot_gnn'].notna(), 'intTot_gnn']) ** 2).mean() /
                      df_out.loc[df_out['intTot_gnn'].notna(), 'intTot'].var())

    print(f"[GNN-PRED] Training-set GNN quality: speed R2={r2_speed:.4f}, intTot R2={r2_intTot:.4f}")
    print("=" * 80)
    sys.stdout.flush()

    return df_out


# ============================================================================
# XGBOOST TRAINING FUNCTIONS
# ============================================================================

def get_xgboost_features(selected_features):
    """Get XGBoost feature columns based on ablation settings"""
    base_features = ['pk', 'anyo', 'mes', 'dia', 'diaSem', 'hor', get_time_resolution_feature()]
    xgb_features = base_features.copy()

    if ABLATION_SETTINGS['include_weather_1d']:
        xgb_features.extend(FEATURE_GROUPS['weather_1d'])
    if ABLATION_SETTINGS['include_weather_3d']:
        xgb_features.extend(FEATURE_GROUPS['weather_3d'])
    if ABLATION_SETTINGS['include_traffic']:
        xgb_features.extend([f for f in FEATURE_GROUPS['traffic'] if f not in xgb_features])
    if ABLATION_SETTINGS['include_geometry']:
        xgb_features.extend([f for f in FEATURE_GROUPS['geometry'] if f not in xgb_features])
    if ABLATION_SETTINGS['include_mobility']:
        xgb_features.extend(FEATURE_GROUPS['mobility'])
    if ABLATION_SETTINGS['include_imputation_flags']:
        xgb_features.extend(FEATURE_GROUPS['imputation'])

    seen = set()
    unique = []
    for f in xgb_features:
        if f not in seen:
            seen.add(f)
            unique.append(f)

    # Explicitly exclude accident-impact variables from features.
    return [f for f in unique if f not in XGBOOST_EXCLUDED_FEATURE_COLUMNS]


class XGBoostClassifierObjective:
    """Optuna objective for XGBoost classification."""

    def __init__(self, X_train, y_train, feature_cols, target_col, n_classes, n_folds=3, use_gpu=True, csv_path=None, n_trials=100):
        self.X_train = X_train
        self.y_train = y_train
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.n_classes = n_classes
        self.n_folds = n_folds
        self.use_gpu = use_gpu
        self.n_trials = n_trials
        self.scale_pos_weight = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)
        self.csv_path = csv_path
        self.lock = threading.Lock()

    def __call__(self, trial):
        trial_start_time = time.time()
        error_message = ''

        try:
            params = {
                'tree_method': 'hist',
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 100),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'gamma': trial.suggest_float('gamma', 0.0, 10.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
                'random_state': RANDOM_STATE
            }
            if self.n_classes <= 2:
                params['objective'] = 'binary:logistic'
                params['eval_metric'] = 'auc'
                params['scale_pos_weight'] = self.scale_pos_weight
            else:
                params['objective'] = 'multi:softprob'
                params['eval_metric'] = 'mlogloss'
                params['num_class'] = self.n_classes

            if self.use_gpu and torch.cuda.is_available():
                params['device'] = 'cuda:0'
            else:
                params['nthread'] = N_THREADS

            trial_start_msg = (
                f"\n{'='*80}\n"
                f"[TRIAL {trial.number + 1}/{self.n_trials}] STARTING - target={self.target_col}\n"
                f"{'='*80}"
            )
            print(trial_start_msg)
            print(
                "[HYPERPARAMS] Testing: "
                f"max_depth={params['max_depth']}, "
                f"lr={params['learning_rate']:.6f}, "
                f"min_child_weight={params['min_child_weight']}, "
                f"subsample={params['subsample']:.4f}, "
                f"colsample_bytree={params['colsample_bytree']:.4f}, "
                f"gamma={params['gamma']:.4f}, "
                f"reg_alpha={params['reg_alpha']:.4f}, "
                f"reg_lambda={params['reg_lambda']:.4f}"
            )
            sys.stdout.flush()

            tscv = TimeSeriesSplit(n_splits=self.n_folds)
            cv_scores = []
            best_iterations = []

            for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(self.X_train), start=1):
                X_fold_train = self.X_train.iloc[train_idx]
                y_fold_train = self.y_train.iloc[train_idx]
                X_fold_val = self.X_train.iloc[val_idx]
                y_fold_val = self.y_train.iloc[val_idx]

                dtrain = xgb.DMatrix(X_fold_train, label=y_fold_train)
                dval = xgb.DMatrix(X_fold_val, label=y_fold_val)

                model = xgb.train(
                    params, dtrain, num_boost_round=1000,
                    evals=[(dval, 'eval')],
                    early_stopping_rounds=XGBOOST_CONFIG['early_stopping'],
                    verbose_eval=False
                )

                y_pred = model.predict(dval, iteration_range=(0, model.best_iteration))
                if self.n_classes <= 2:
                    score = roc_auc_score(y_fold_val, y_pred)
                else:
                    y_pred_labels = np.argmax(y_pred, axis=1)
                    score = f1_score(y_fold_val, y_pred_labels, average='weighted', zero_division=0)
                cv_scores.append(score)
                best_iterations.append(model.best_iteration)
                print(
                    f"[CV] Fold {fold_idx}/{self.n_folds} | "
                    f"score={score:.4f} | best_iteration={model.best_iteration}"
                )
                sys.stdout.flush()

            mean_score = np.mean(cv_scores)
            trial_duration = time.time() - trial_start_time

            if self.csv_path is not None:
                trial_results = {
                    'trial_number': trial.number + 1,
                    'target_col': self.target_col,
                    'gpu_id': 0,
                    'max_depth': params['max_depth'],
                    'learning_rate': params['learning_rate'],
                    'min_child_weight': params['min_child_weight'],
                    'subsample': params['subsample'],
                    'colsample_bytree': params['colsample_bytree'],
                    'gamma': params['gamma'],
                    'reg_alpha': params['reg_alpha'],
                    'reg_lambda': params['reg_lambda'],
                    'scale_pos_weight': params.get('scale_pos_weight'),
                    'cv_roc_auc_mean': mean_score,
                    'cv_roc_auc_std': np.std(cv_scores),
                    'best_iterations_mean': np.mean(best_iterations),
                    'training_time_seconds': trial_duration,
                    'error_message': '',
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }

                file_exists = os.path.exists(self.csv_path)
                with self.lock:
                    with open(self.csv_path, 'a', newline='') as csvfile:
                        writer = csv.DictWriter(csvfile, fieldnames=trial_results.keys())
                        if not file_exists:
                            writer.writeheader()
                        writer.writerow(trial_results)

            print(
                f"\n[TRIAL {trial.number + 1}/{self.n_trials}] COMPLETED - target={self.target_col}\n"
                f"  CV score (mean): {mean_score:.4f}\n"
                f"  CV score (std):  {np.std(cv_scores):.4f}\n"
                f"  Avg best iter:   {np.mean(best_iterations):.1f}\n"
                f"  Training time:   {format_duration(trial_duration)}"
            )
            print("-" * 80)
            sys.stdout.flush()

            return mean_score

        except Exception as e:
            error_message = str(e)[:500]
            trial_duration = time.time() - trial_start_time
            print(
                f"[ERROR] Trial {trial.number + 1} failed for target={self.target_col}: {error_message}"
            )
            sys.stdout.flush()

            if self.csv_path is not None:
                error_results = {
                    'trial_number': trial.number + 1,
                    'target_col': self.target_col,
                    'gpu_id': 0,
                    'max_depth': None,
                    'learning_rate': None,
                    'min_child_weight': None,
                    'subsample': None,
                    'colsample_bytree': None,
                    'gamma': None,
                    'reg_alpha': None,
                    'reg_lambda': None,
                    'scale_pos_weight': None,
                    'cv_roc_auc_mean': 0.0,
                    'cv_roc_auc_std': None,
                    'best_iterations_mean': None,
                    'training_time_seconds': trial_duration,
                    'error_message': error_message,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }

                file_exists = os.path.exists(self.csv_path)
                with self.lock:
                    with open(self.csv_path, 'a', newline='') as csvfile:
                        writer = csv.DictWriter(csvfile, fieldnames=error_results.keys())
                        if not file_exists:
                            writer.writeheader()
                        writer.writerow(error_results)

            raise


def train_xgboost_model(df_train_data, output_dir, model_time):
    """Train XGBoost models for configured target columns.
    
    V5: The input df_train_data should already contain GNN prediction columns
    ('mean_speed_gnn', 'intTot_gnn', 'intP_gnn'). These are used as the
    traffic features instead of ground truth, so XGBoost learns from the
    same imperfect inputs it will receive at simulation time.
    """
    xgb_total_start = time.time()
    print("\n" + "=" * 80)
    print("STAGE 2: XGBOOST HYPERPARAMETER OPTIMIZATION (V5: ON GNN PREDICTIONS)")
    print("=" * 80)
    sys.stdout.flush()

    xgb_prep_start = time.time()
    feature_cols = get_xgboost_features(get_selected_features())
    available_features = [f for f in feature_cols if f in df_train_data.columns]

    print(f"[XGB] Using {len(available_features)} features")

    # V5: Replace ground truth traffic with GNN predictions where available
    has_gnn_preds = 'mean_speed_gnn' in df_train_data.columns
    if has_gnn_preds:
        df_train_data = df_train_data.dropna(subset=['mean_speed_gnn']).copy()
        print(f"[V5] Replacing traffic features with GNN predictions ({len(df_train_data):,} rows with predictions)")
        if 'mean_speed' in available_features:
            df_train_data['mean_speed'] = df_train_data['mean_speed_gnn']
        if 'intTot' in available_features:
            df_train_data['intTot'] = df_train_data['intTot_gnn']
        if 'intP' in available_features:
            df_train_data['intP'] = df_train_data['intP_gnn']
        if 'car' in available_features:
            df_train_data['car'] = df_train_data['intTot'] - df_train_data['intP']
    else:
        print("[WARNING] No GNN prediction columns found. Training on ground truth (V4 behavior).")
    sys.stdout.flush()

    X = df_train_data[available_features].fillna(0)
    split_idx = int(len(X) * 0.8)
    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]

    scaler = MinMaxScaler()
    scaler.fit(X_train)
    joblib.dump(scaler, os.path.join(output_dir, f'scaler-features_model=XGBoost-version={model_time}.pkl'))
    print(f"[XGB] Data preparation: {format_duration(time.time() - xgb_prep_start)}")
    sys.stdout.flush()

    target_cols = [c for c in XGBOOST_TARGET_COLUMNS if c in df_train_data.columns]
    missing_targets = [c for c in XGBOOST_TARGET_COLUMNS if c not in df_train_data.columns]
    if missing_targets:
        print(f"[WARNING] Missing target columns: {missing_targets}")
    if not target_cols:
        raise ValueError("No configured target columns found in dataset.")

    studies_by_target = {}
    models_by_target = {}
    metrics_by_target = {}
    params_by_target = {}

    for target_col in target_cols:
        print("\n" + "-" * 80)
        print(f"[TARGET] Training target: {target_col}")

        y_raw = df_train_data[target_col].fillna(0)
        classes = sorted(pd.Series(y_raw).astype(int).unique().tolist())
        if len(classes) < 2:
            print(f"[WARNING] Skipping {target_col}: only one class present ({classes}).")
            metrics_by_target[target_col] = {'status': 'skipped_single_class', 'classes': classes}
            continue

        class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        y_all = pd.Series(y_raw).astype(int).map(class_to_idx).astype(int)
        y_train = y_all.iloc[:split_idx]
        y_test = y_all.iloc[split_idx:]

        print(f"[INFO] Classes ({target_col}): {classes}")
        print(f"[INFO] Train: {len(X_train):,} | Test: {len(X_test):,}")
        sys.stdout.flush()

        csv_path = os.path.join(
            output_dir,
            f'training-evolution_model=XGBoost-target={target_col}-version={model_time}.csv'
        )
        print(f"[CSV] Training evolution will be saved to: {csv_path}")
        sys.stdout.flush()

        # V5: Differentiated trial counts per target
        n_trials = XGBOOST_CONFIG['trials_accident'] if target_col == 'ACCIDENT' else XGBOOST_CONFIG['trials_secondary']

        objective = XGBoostClassifierObjective(
            X_train, y_train, available_features,
            target_col=target_col,
            n_classes=len(classes),
            n_folds=XGBOOST_CONFIG['cv_folds'],
            use_gpu=USE_GPU,
            csv_path=csv_path,
            n_trials=n_trials
        )

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE)
        )
        print(f"[OPTUNA] Starting {n_trials} trials for {target_col}...")
        sys.stdout.flush()
        start_time = time.time()
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        duration = time.time() - start_time

        print('=' * 80)
        print(f"[COMPLETE] XGB OPTIMIZATION COMPLETED ({target_col})")
        print('=' * 80)
        print(f"[XGB] Optuna optimization: {format_duration(duration)}")
        print(f"[RESULT] Best score: {study.best_value:.4f}")
        print("[BEST] Best hyperparameters found by Optuna:")
        for key, value in study.best_params.items():
            print(f"  - {key}: {value}")
        print('=' * 80)
        sys.stdout.flush()

        print(f"[XGB] Training final model with best parameters ({target_col})...")
        sys.stdout.flush()

        best_params = {
            **study.best_params,
            'tree_method': 'hist',
            'random_state': RANDOM_STATE
        }
        if len(classes) <= 2:
            best_params['objective'] = 'binary:logistic'
            best_params['eval_metric'] = 'auc'
            best_params['scale_pos_weight'] = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)
        else:
            best_params['objective'] = 'multi:softprob'
            best_params['eval_metric'] = 'mlogloss'
            best_params['num_class'] = len(classes)

        if USE_GPU and torch.cuda.is_available():
            best_params['device'] = 'cuda:0'

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_test, label=y_test)

        final_model = xgb.train(
            best_params, dtrain, num_boost_round=2000,
            evals=[(dval, 'eval')],
            early_stopping_rounds=100,
            verbose_eval=False
        )

        y_pred_raw = final_model.predict(dval, iteration_range=(0, final_model.best_iteration))
        if len(classes) <= 2:
            y_pred = (y_pred_raw >= 0.5).astype(int)
            roc_auc = roc_auc_score(y_test, y_pred_raw)
        else:
            y_pred = np.argmax(y_pred_raw, axis=1)
            roc_auc = None

        test_metrics = {
            'target_col': target_col,
            'classes_original': classes,
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_weighted': f1_score(y_test, y_pred, average='weighted', zero_division=0),
            'precision_weighted': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall_weighted': recall_score(y_test, y_pred, average='weighted', zero_division=0),
        }
        if roc_auc is not None:
            test_metrics['roc_auc'] = roc_auc

        print(f"[INFO] Test metrics ({target_col}): acc={test_metrics['accuracy']:.4f}, f1_w={test_metrics['f1_weighted']:.4f}")
        if roc_auc is not None:
            print(f"[INFO] Test ROC-AUC ({target_col}): {roc_auc:.4f}")
        sys.stdout.flush()

        model_suffix = target_col.lower()
        model_path = os.path.join(output_dir, f'xgboost-classifier-target={model_suffix}_model=XGBoost-version={model_time}.json')
        final_model.save_model(model_path)
        if target_col == 'ACCIDENT':
            compat_model_path = os.path.join(output_dir, f'xgboost-classifier_model=XGBoost-version={model_time}.json')
            final_model.save_model(compat_model_path)

        print(f"[SUCCESS] Model saved: {model_path}")
        sys.stdout.flush()

        studies_by_target[target_col] = study
        models_by_target[target_col] = model_path
        metrics_by_target[target_col] = test_metrics
        params_by_target[target_col] = study.best_params

    primary_target = 'ACCIDENT' if 'ACCIDENT' in studies_by_target else (next(iter(studies_by_target)) if studies_by_target else None)
    primary_study = studies_by_target.get(primary_target) if primary_target else None

    metadata = {
        'model_name': 'XGBoost_Classifier',
        'model_time': model_time,
        'target_cols': target_cols,
        'primary_target': primary_target,
        'best_cv_score': primary_study.best_value if primary_study else None,
        'test_metrics': metrics_by_target.get(primary_target) if primary_target else {},
        'best_hyperparameters': primary_study.best_params if primary_study else {},
        'feature_cols': available_features,
        'excluded_feature_cols': XGBOOST_EXCLUDED_FEATURE_COLUMNS,
        'models_by_target': models_by_target,
        'metrics_by_target': metrics_by_target,
        'best_hyperparameters_by_target': params_by_target,
        'ablation_settings': ABLATION_SETTINGS,
        'test_mode': TEST_MODE,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    with open(os.path.join(output_dir, f'xgboost-metadata_model=XGBoost-version={model_time}.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"[SUCCESS] Target models saved: {len(models_by_target)}")
    print_step_timing("STAGE 2: XGBoost Hyperparameter Optimization Complete", xgb_total_start)
    sys.stdout.flush()

    return primary_study, models_by_target, available_features, metrics_by_target


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_xgboost_only():
    """Main execution pipeline - XGBoost Training Only"""

    # Suppress logs
    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
    logging.getLogger("lightning_fabric").setLevel(logging.ERROR)
    warnings.filterwarnings('ignore')

    os.environ["LOCAL_RANK"] = "0"
    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # Validate previous experiment directory
    if not PREVIOUS_EXPERIMENT_DIR:
        print("\n" + "=" * 80)
        print("ERROR: PREVIOUS_EXPERIMENT_DIR not set!")
        print("=" * 80)
        print("\nRun it through scripts/run_ablation_study_xgboost_only.sh, which")
        print("sets it to the Stage-1 run given by --gnn_vNN / --gnn-run-id.")
        print("=" * 80)
        sys.exit(1)

    if not os.path.exists(PREVIOUS_EXPERIMENT_DIR):
        raise FileNotFoundError(f"Previous experiment directory not found: {PREVIOUS_EXPERIMENT_DIR}")

    # Print header
    print("\n" + "=" * 80)
    print("ABLATION STUDY V5 - XGBOOST TRAINING ONLY (ON GNN PREDICTIONS)")
    print("=" * 80)
    print(f"GNN model from: {PREVIOUS_EXPERIMENT_DIR}")
    sys.stdout.flush()

    # Detect or use provided model timestamp
    global PREVIOUS_MODEL_TIME
    if PREVIOUS_MODEL_TIME is None:
        PREVIOUS_MODEL_TIME = detect_model_time(PREVIOUS_EXPERIMENT_DIR)
    print(f"GNN model timestamp: {PREVIOUS_MODEL_TIME}")

    # Reuse the existing GNN experiment folder so all stages share one directory.
    global EXPERIMENT_NAME, OUTPUT_DIR
    OUTPUT_DIR = PREVIOUS_EXPERIMENT_DIR
    EXPERIMENT_NAME = os.path.basename(OUTPUT_DIR.rstrip("/"))

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model_time = datetime.now().strftime('%Y%m%d_%H%M%S')

    print(f"\nUsing existing experiment: {EXPERIMENT_NAME}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"New model timestamp: {model_time}")
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
        'previous_experiment_dir': PREVIOUS_EXPERIMENT_DIR,
        'previous_model_time': PREVIOUS_MODEL_TIME,
        'test_mode': TEST_MODE,
        'ablation_settings': ABLATION_SETTINGS,
        'xgboost_config': XGBOOST_CONFIG,
        'train_dates': {'start': TRAIN_START_DATE, 'end': TRAIN_END_DATE},
        'pipeline_version': 'v5_xgboost_only',
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
    stage_times = {}

    # =========================================================================
    # STAGE 0: DATA LOADING + LOAD PRE-TRAINED GNN MODEL
    # =========================================================================
    stage0_start = time.time()
    print("\n" + "=" * 80)
    print("STAGE 0: DATA LOADING + LOADING GNN MODEL")
    print("=" * 80)
    sys.stdout.flush()

    df_full, selected_features = load_data()

    df_train = filter_data_by_date(df_full, TRAIN_START_DATE, TRAIN_END_DATE)
    print(f"[INFO] Training data: {len(df_train):,} rows ({TRAIN_START_DATE} to {TRAIN_END_DATE})")
    sys.stdout.flush()

    gnn_model, gnn_metadata, gnn_info, scaler_temporal, scaler_static, scaler_targets = load_pretrained_gnn_model(
        PREVIOUS_EXPERIMENT_DIR, PREVIOUS_MODEL_TIME
    )

    # Copy scalers to new output directory
    joblib.dump(scaler_temporal, os.path.join(OUTPUT_DIR, f'scaler-temporal_model=GNN-version={model_time}.pkl'))
    joblib.dump(scaler_static, os.path.join(OUTPUT_DIR, f'scaler-static_model=GNN-version={model_time}.pkl'))
    joblib.dump(scaler_targets, os.path.join(OUTPUT_DIR, f'scaler-targets_model=GNN-version={model_time}.pkl'))

    # Copy GNN model to new output directory
    src_model = os.path.join(PREVIOUS_EXPERIMENT_DIR, f"best-model_model=GNN-version={PREVIOUS_MODEL_TIME}.pt")
    dst_model = os.path.join(OUTPUT_DIR, f"best-model_model=GNN-version={model_time}.pt")
    shutil.copy2(src_model, dst_model)

    # Copy and update GNN metadata
    gnn_metadata_copy = gnn_metadata.copy()
    gnn_metadata_copy['original_experiment'] = PREVIOUS_EXPERIMENT_DIR
    gnn_metadata_copy['original_model_time'] = PREVIOUS_MODEL_TIME
    gnn_metadata_copy['copied_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(os.path.join(OUTPUT_DIR, f"best-model-metadata_model=GNN-version={model_time}.json"), 'w') as f:
        json.dump(gnn_metadata_copy, f, indent=2)

    print("\n[SUCCESS] Copied GNN model and scalers to new experiment directory")
    sys.stdout.flush()

    stage_times['stage0_data_gnn_loading'] = time.time() - stage0_start
    print_step_timing("STAGE 0: Data + GNN Loading Complete", stage0_start)

    # =========================================================================
    # STAGE 1: V5 - GENERATE GNN PREDICTIONS ON TRAINING DATA
    # =========================================================================
    stage1_start = time.time()
    print("\n" + "=" * 80)
    print("STAGE 1: V5 - GENERATING GNN PREDICTIONS ON TRAINING DATA")
    print("=" * 80)
    print(f"[INFO] GNN model loaded from: {PREVIOUS_EXPERIMENT_DIR}")
    print(f"[INFO] Best validation loss: {gnn_metadata.get('best_val_loss', 'N/A')}")
    sys.stdout.flush()

    # Detect sequence_length from data
    from src.training.ablation_study_v5_gnn_only import SEQUENCE_LENGTH_BY_INTERVAL
    interval_minutes = 15  # Default
    for minutes in [5, 10, 15, 30, 45, 60]:
        if f"{minutes}min" in df_train.columns:
            interval_minutes = minutes
            break
    seq_len = SEQUENCE_LENGTH_BY_INTERVAL.get(interval_minutes, 8)

    df_train = generate_gnn_predictions_on_training_data(
        gnn_model, df_train, scaler_temporal, scaler_static, scaler_targets,
        gnn_metadata, sequence_length=seq_len
    )

    stage_times['stage1_gnn_inference'] = time.time() - stage1_start
    print_step_timing("STAGE 1: GNN Inference Complete", stage1_start)

    # =========================================================================
    # STAGE 2: XGBOOST TRAINING (on GNN predictions)
    # =========================================================================
    stage2_start = time.time()
    xgb_study, xgb_models, xgb_features, xgb_metrics = train_xgboost_model(
        df_train, OUTPUT_DIR, model_time
    )
    stage_times['stage2_xgboost_training'] = time.time() - stage2_start
    print_step_timing("STAGE 2: XGBoost Training Complete", stage2_start)

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    total_duration = time.time() - total_start

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE (V5 XGBOOST ON GNN PREDICTIONS)")
    print("=" * 80)

    print("\nTIMING BREAKDOWN:")
    print("-" * 60)
    print(f"  Stage 0 - Data + GNN Loading:    {format_duration(stage_times['stage0_data_gnn_loading']):>15}")
    print(f"  Stage 1 - GNN Inference (V5):    {format_duration(stage_times['stage1_gnn_inference']):>15}")
    print(f"  Stage 2 - XGBoost Training:      {format_duration(stage_times['stage2_xgboost_training']):>15}")
    print("-" * 60)
    print(f"  TOTAL:                           {format_duration(total_duration):>15}")
    print("=" * 80)
    sys.stdout.flush()

    print(f"\nResults saved to: {OUTPUT_DIR}")
    print("\nFiles generated:")
    for f in os.listdir(OUTPUT_DIR):
        print(f"  - {f}")
    sys.stdout.flush()

    # Save final summary
    summary = {
        'experiment_name': EXPERIMENT_NAME,
        'gnn_experiment': PREVIOUS_EXPERIMENT_DIR,
        'gnn_model_time': PREVIOUS_MODEL_TIME,
        'total_duration_minutes': total_duration / 60,
        'total_duration_hours': total_duration / 3600,
        'stage_times_seconds': stage_times,
        'stage_times_formatted': {k: format_duration(v) for k, v in stage_times.items()},
        'gnn_best_val_loss': gnn_metadata.get('best_val_loss'),
        'xgb_best_score': xgb_study.best_value if xgb_study is not None else None,
        'xgb_model_paths': xgb_models,
        'xgb_test_metrics': xgb_metrics,
        'ablation_settings': ABLATION_SETTINGS,
        'test_mode': TEST_MODE,
        'completed_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    # Write generic and version-tagged summary files
    summary_path = os.path.join(OUTPUT_DIR, 'experiment_summary.json')
    summary_versioned_path = os.path.join(OUTPUT_DIR, f'experiment_summary_{PIPELINE_VERSION}.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    with open(summary_versioned_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print("SUCCESS!")
    print("=" * 80)
    sys.stdout.flush()


    return True


def main():
    # Lightweight CLI layer to optionally select the time resolution dynamically.
    #
    # Supports either:
    #   - --time-resolution 5min|10min|15min|30min|45min|60min
    #   - Shorthand flags: --5min, --10min, --15min, --30min, --45min, --60min
    #
    # When provided, the chosen token replaces the interval embedded in DATA_FILE
    # (e.g. "..._5min_..." -> "..._10min_..."), so that 10min behaves exactly
    # like the other supported resolutions.
    parser = argparse.ArgumentParser(
        description="Ablation Study V5 - XGBoost Training Only (on GNN Predictions)"
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

    if getattr(args, "time_resolution", None):
        global DATA_FILE, DATA_PATH  # type: ignore[global-variable-not-assigned]
        chosen = args.time_resolution
        new_data_file = re.sub(r'\d+min', chosen, DATA_FILE, count=1)
        DATA_FILE = new_data_file
        DATA_PATH = os.path.join(_AP7_DATA, DATA_FILE)
        print(f"[CONFIG] Using DATA_FILE={DATA_FILE} for time resolution {chosen}")
        sys.stdout.flush()

    return run_xgboost_only()


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
