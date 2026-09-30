#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Ablation Study Pipeline V21 XGBoost: Weather-Derived & Cyclical Feature Engineering
=====================================================================================

V21 = V20 (precision-optimised balanced-bagging + Platt calibration + Groups 1-10
literature-driven feature engineering) extended with three additional feature
groups aimed at the remaining signal gaps identified from the V19 post-mortem:

  1. The V19 Optuna search converged on very shallow trees (max_depth=4) and
     near-unit scale_pos_weight — model capacity was not the bottleneck, raw
     *signal* was. Groups 11-13 inject three signal families the current
     feature set does not express.

  2. Phase-2 base models converged at 150-230 rounds naturally (no iteration
     cap hit). The limit is what the features encode, not how many trees
     fit them.

New feature groups in V21 (on top of V20 Groups 1-10):

  Group 11 - Cyclical temporal encoding      (NEW)
     hor_sin, hor_cos, diaSem_sin, diaSem_cos, mes_sin, mes_cos
     Removes the artificial 23->0 / 6->0 discontinuities that hurt tree
     splits on hour-of-day and day-of-week near the wrap-around.

  Group 12 - Derived weather signals         (NEW)
     is_raining, is_heavy_rain, temp_near_freezing, low_visibility_proxy,
     wind_gust_excess, precip_change_1d_vs_3d
     V20 already includes the raw forecast columns, but trees have to
     discover thresholds (> 0.1 mm, T in [-2, 2] C) from scratch. Injecting
     operationally meaningful bins/flags makes the splits immediately
     available and interaction-ready.

  Group 13 - Weather x geometry interactions (NEW)
     wet_curve_risk, wet_descent_risk, wet_speed, cold_curve_risk
     Crash risk on wet pavement is non-linear in curvature and descent
     gradient; explicit products make that surface learnable with far
     fewer positive examples than pure tree splitting would need.

V20 groups (unchanged, kept here for completeness):

  Group 1 - Speed/volume variability         (Theofilatos 2019; Anik 2024)
     speed_std_2, speed_std_4, vol_std_2, speed_cv_2, vol_cv_2, speed_mean_4

  Group 2 - First differences (delta)        (Basso 2021; Anik 2024)
     delta_speed_1, delta_speed_2, delta_vol_1, accel_speed

  Group 3 - Lag features                     (Mehrannia 2021; Anik 2023)
     speed_lag1, speed_lag2, speed_lag4, vol_lag1, vol_lag2

  Group 4 - Historical crash precursor (SPF) (Anik 2024)
     pk_crash_rate, pk_crash_rate_log, pk_crash_rate_mob

  Group 5 - Upstream/downstream gradient     (Anik 2024; Basso 2021)
     speed_grad_up, speed_grad_dn, vol_grad_up

  Group 6 - Z-score from (pk, hor, diaSem)   (Theofilatos 2019; Anik 2024)
     speed_zscore, vol_zscore, speed_pct_below_normal

  Group 7 - Flow regime / occupancy proxy    (Mehrannia 2021)
     flow_regime, occ_proxy

  Group 8 - Heavy vehicle fraction           (Basso 2021)
     hv_fraction, hv_fraction_lag1

  Group 9 - Geometry x traffic interaction   (Wang 2022; Anik 2024)
     curv_x_speed, pend_x_speed, curv_x_delta_speed

  Group 10 - Temporal flags                  (Monsefi 2023)
     is_peak_hour

Leakage safety
--------------
All baseline-fitted features (pk_crash_rate, z-score mu/sigma, flow_regime V_ff)
are computed ONLY on the training slice (first 70 % of rows, matching V19's
70/10/20 split) and then applied to the full dataset. Lag/delta/rolling/gradient
features are strictly causal. Groups 11-13 are pure row-wise transforms of
columns already present at inference time (no history, no fitted statistics),
so they are leakage-safe by construction.

Author: Gerard Franco
Date: April 2026
Affiliation: Universitat Politecnica de Catalunya
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
import csv
import glob
import json
import joblib
import os
import re
import shutil
import sys
import threading
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import optuna
import torch
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# GNN/data-loading utilities — shared infrastructure, not version-specific
from src.training.ablation_study_v5_xgboost_only import (
    load_data as _v5_load_data,
    filter_data_by_date as _v5_filter_data_by_date,
    load_pretrained_gnn_model as _v5_load_pretrained_gnn_model,
    generate_gnn_predictions_on_training_data as _v5_generate_gnn_predictions,
)


# ============================================================================
# CONFIGURATION
# ============================================================================

PREVIOUS_EXPERIMENT_DIR = os.environ.get("PREVIOUS_EXPERIMENT_DIR", "")
PREVIOUS_MODEL_TIME = os.environ.get("PREVIOUS_MODEL_TIME", None)

TEST_MODE = False

ABLATION_SETTINGS = {
    'include_weather_1d': True,
    'include_weather_3d': True,
    'include_traffic': True,
    'include_geometry': True,
    'include_mobility': True,
    'include_temporal': True,
    'include_imputation_flags': True,
}

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
    'temporal': ['anyo', 'mes', 'dia', 'diaSem', 'hor', '5min', '10min', '15min',
                 '30min', '45min', '60min'],
    'imputation': ['speed_imputation', 'intensity_imputation'],
    'static_core': ['pk', 'via', 'sen', 'car'],
}

USE_GPU = True
N_GPUS = int(os.environ.get('ABLATION_NUM_GPUS', 1))
N_THREADS = 16
RANDOM_STATE = 42

PIPELINE_VERSION = "v21_xgboost_fe_weather_cyclical"

XGBOOST_CONFIG = {
    'trials_accident': 100 if not TEST_MODE else 10,
    'trials_secondary': 50 if not TEST_MODE else 10,
    'cv_folds': 3,
    'early_stopping': 50 if not TEST_MODE else 10,
}

XGBOOST_TARGET_COLUMNS = ['ACCIDENT', 'C_NIVELL_AFECTACIO']
XGBOOST_EXCLUDED_FEATURE_COLUMNS = ['F_TEMPS_AFECTACIO', 'F_LONG_AFECTACIO']

if TEST_MODE:
    TRAIN_START_DATE = '2024-04-01'
    TRAIN_END_DATE = '2024-05-01'
else:
    TRAIN_START_DATE = '2024-04-01'
    TRAIN_END_DATE = '2025-06-01'

TIME_RESOLUTION_CHOICES = ["5min", "10min", "15min", "30min", "45min", "60min"]
DATA_FILE = 'CrashGNNLSTM_v1_vel-extinrix_int_geo_mob_wthr_5min_fund-propag-ltd_from_20240404_to_20251001.csv'
DATA_PATH = os.path.join(_AP7_DATA, DATA_FILE)

OUTPUT_BASE_DIR = _AP7_EXPERIMENTS
EXPERIMENT_NAME = None
OUTPUT_DIR = None

ENSEMBLE_CONFIG = {
    "n_base_models": 10 if not TEST_MODE else 3,
    "neg_pos_ratio": 10,
    "optuna_trials": 80 if not TEST_MODE else 10,
}

THRESHOLD_CONFIG = {
    "min_recall": 0.40,
    "calibrate": True,
}

XGBOOST_UNCERTAINTY_CONFIG = {
    "enabled": True,
    "max_trees": 128,
    "max_train_samples": 250_000 if not TEST_MODE else 20_000,
    "max_test_samples": 250_000 if not TEST_MODE else 20_000,
}

SEQUENCE_LENGTH_BY_INTERVAL = {5: 12, 10: 10, 15: 8, 30: 6, 45: 4, 60: 4}

# ── V20 feature engineering configuration ────────────────────────────────────

FEATURE_ENGINEERING_CONFIG = {
    "speed_variability":    True,   # Group 1
    "first_differences":    True,   # Group 2
    "lag_features":         True,   # Group 3
    "crash_precursor":      True,   # Group 4
    "spatial_gradient":     True,   # Group 5
    "zscore_baseline":      True,   # Group 6
    "flow_regime":          True,   # Group 7
    "hv_fraction":          True,   # Group 8
    "geometry_interaction": True,   # Group 9
    "temporal_flags":       True,   # Group 10
    "cyclical_temporal":    True,   # Group 11 (V21)
    "weather_derived":      True,   # Group 12 (V21)
    "weather_interaction":  True,   # Group 13 (V21)
}

# Lag steps and rolling windows (in timesteps).
# At 15-min resolution: lag1=15 min, lag2=30 min, lag4=60 min.
LAG_STEPS = [1, 2, 4]
ROLL_WINDOWS = [2, 4]

_LOCATION_KEYS = ["via", "sen", "pk"]
_BASELINE_KEYS = ["pk", "hor", "diaSem"]
_EPS = 1e-3  # guard against division by near-zero

# ── Canonical frame layout: (via, sen, pk, dat) ──────────────────────────────
# LOCATION first, TIMESTAMP second, so consecutive rows inside a location are
# consecutive time steps. Every lag here — shift(), rolling(), the spatial
# gradient, hv_fraction_lag1 — is computed per (via, sen, pk) group and is only
# meaningful under that layout.
#
# Defined here, in the feature-engineering module, because this is the module
# that imposes the requirement, and because every consumer (v50, v51, the
# end-to-end benchmark, compute_feature_engineering) already imports it. Do not
# re-implement the sort at call sites: the pipeline previously had two competing
# location orders — (pk, sen, dat) in the window filters and (via, sen, pk, dat)
# inside the GNN — and code paths that skipped the GNN silently kept the wrong
# one. Both satisfy "location first, datetime second", but they order the
# location GROUPS differently, and downstream consumers that slice positionally
# then select entirely different rows.
LOCATION_SORT_KEYS = _LOCATION_KEYS + ["dat"]


def sort_location_major(df: pd.DataFrame) -> pd.DataFrame:
    """Sort into the canonical (via, sen, pk, dat) layout with a fresh index.

    Columns absent from ``df`` are skipped, so frames that predate a column
    still sort by whatever subset they do carry.
    """
    keys = [c for c in LOCATION_SORT_KEYS if c in df.columns]
    return df.sort_values(keys, kind="stable").reset_index(drop=True)


TRAIN_TEST_SPLIT_FRAC = 0.80


def resolve_train_test_boundary(windows, frac: float = TRAIN_TEST_SPLIT_FRAC,
                                override=None) -> pd.Timestamp:
    """Timestamp that splits the training windows `frac`/(1-frac) BY TIME.

    Derived analytically from the window date ranges rather than from row
    positions: the frame is on a uniform 5-min grid covering every location, so
    the row-count quantile and the duration quantile coincide, and the analytic
    form is reproducible without touching the data. Floored to midnight.

    Both the trainer and the simulation call this with the SAME window list, so
    they agree by construction — which they must, because the boundary also
    fixes `baseline_end_date`, and mismatched baselines mean the model is scored
    on features it was not trained on.

    `override` (e.g. $BENCH_TRAIN_TEST_BOUNDARY) short-circuits the derivation.
    """
    if override:
        return pd.Timestamp(override).normalize()
    spans = [(pd.Timestamp(a), pd.Timestamp(b)) for a, b in windows]
    spans.sort()
    total = sum(((b - a) for a, b in spans), pd.Timedelta(0))
    target = total * float(frac)
    acc = pd.Timedelta(0)
    for a, b in spans:
        if acc + (b - a) >= target:
            return (a + (target - acc)).normalize()
        acc += b - a
    return spans[-1][1].normalize()


def dedup_segment_intervals(df: pd.DataFrame, label: str = "V21") -> pd.DataFrame:
    """Collapse duplicate (via, sen, pk, dat) segment-intervals to one row each.

    The raw 5-min CSV carries ~1.26M duplicated keys (~5.3% of rows): the same
    segment-interval appears twice with identical speed, label, imputation
    flags and geometry but DIFFERENT intTot/intP — two parallel intensity
    series double-mapped to one location upstream of the CSV. Beyond double
    counting rows in every evaluation frame (June 2025: 1,797,320 rows vs
    1,728,000 real intervals, inflating AUPRC by ~0.01-0.03), each duplicate
    inserts a fake time step into its (via, sen, pk) group, corrupting every
    shift()/rolling() feature computed after it — so the damage reaches
    non-duplicated rows too.

    Resolution policy (owner decision 2026-08-12): keep ONE REAL ROW per key —
    no averaging — chosen by, in order:
      1. the row carrying ``ACCIDENT`` = 1, so no crash label can be dropped;
      2. the row with the highest ``intTot`` (highest risk signal).
    The owner's other requested tie-breaks — highest ``pk_crash_rate``,
    highest ``speed_lag`` — are provably no-ops here and are therefore not
    implemented: ``pk_crash_rate`` is keyed on (pk, hor, diaSem), which
    duplicate rows share by definition, and ``speed_lag`` derives from
    ``mean_speed``, which is identical within every duplicate group in the raw
    file (0 of 69,273 June groups differ on speed); both are also engineered
    AFTER this step. ``intTot``/``intP`` are the only columns that differ, so
    "highest risk" resolves to the higher-volume reading. Remaining ties fall
    back to canonical order (deterministic).

    Returns the frame in canonical `sort_location_major` order.
    """
    keys = [c for c in LOCATION_SORT_KEYS if c in df.columns]
    if "dat" not in keys:
        return df
    df = sort_location_major(df)
    n_before = len(df)
    if not df.duplicated(keys, keep=False).any():
        print(f"[{label}] No duplicate segment-intervals found.")
        return df
    pref = [c for c in ("ACCIDENT", "intTot") if c in df.columns]
    df = (df.sort_values(keys + pref,
                         ascending=[True] * len(keys) + [False] * len(pref),
                         kind="stable")
            .drop_duplicates(keys, keep="first"))
    df = sort_location_major(df)
    print(f"[{label}] Deduplicated segment-intervals: {n_before:,} -> {len(df):,} "
          f"rows ({n_before - len(df):,} removed; kept ACCIDENT row, then "
          f"highest intTot)")
    return df


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def format_duration(seconds):
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}min ({seconds:.0f}s)"
    hours = seconds / 3600
    minutes = (seconds % 3600) / 60
    return f"{hours:.1f}h ({int(hours)}h {int(minutes)}m)"


def print_step_timing(step_name, start_time, end_time=None):
    if end_time is None:
        end_time = time.time()
    duration = end_time - start_time
    timestamp = datetime.now().strftime('%H:%M:%S')
    print(f"\n{'='*60}")
    print(f"  [{timestamp}] {step_name}")
    print(f"    Duration: {format_duration(duration)}")
    print(f"{'='*60}")
    sys.stdout.flush()


def get_selected_features():
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
    unique = []
    for f in selected:
        if f not in seen:
            seen.add(f)
            unique.append(f)
    return unique


def print_ablation_config():
    print("\n" + "=" * 80)
    print("ABLATION STUDY CONFIGURATION")
    print("=" * 80)
    print(f"TEST_MODE: {TEST_MODE}")
    print("\nFeature groups:")
    for group, included in ABLATION_SETTINGS.items():
        status = "INCLUDED" if included else "EXCLUDED"
        group_name = group.replace('include_', '')
        features = FEATURE_GROUPS.get(group_name, [])
        print(f"  {status}: {group_name} ({len(features)} features)")
    selected = get_selected_features()
    print(f"\nTotal selected features: {len(selected)}")
    print("=" * 80)
    sys.stdout.flush()


def get_time_resolution_token():
    match = re.search(r'(\d+min)', DATA_FILE)
    return match.group(1) if match else "unknown-res"


def get_time_resolution_minutes():
    token = get_time_resolution_token()
    match = re.match(r'(\d+)min', token)
    return int(match.group(1)) if match else 5


def get_time_resolution_feature():
    return f"{get_time_resolution_minutes()}min"


def get_experiment_prefix():
    excluded = []
    feature_codes = {
        'include_weather_1d': 'w1d', 'include_weather_3d': 'w3d',
        'include_traffic': 'traf', 'include_geometry': 'geo',
        'include_mobility': 'mob', 'include_temporal': 'temp',
        'include_imputation_flags': 'imp',
    }
    for key, code in feature_codes.items():
        if not ABLATION_SETTINGS.get(key, True):
            excluded.append(code)
    return "full" if not excluded else f"no-{'-'.join(excluded)}"


def get_job_id():
    return (
        os.environ.get("SLURM_JOB_ID")
        or os.environ.get("JOB_ID")
        or os.environ.get("PBS_JOBID")
        or datetime.now().strftime('%Y%m%d_%H%M%S')
    )


def detect_model_time(experiment_dir):
    metadata_files = glob.glob(
        os.path.join(experiment_dir, 'best-model-metadata_model=GNN-version=*.json')
    )
    if metadata_files:
        parts = os.path.basename(metadata_files[0]).replace('.json', '').split('version=')
        if len(parts) > 1:
            return parts[1]
    scaler_files = glob.glob(
        os.path.join(experiment_dir, 'scaler-temporal_model=GNN-version=*.pkl')
    )
    if scaler_files:
        parts = os.path.basename(scaler_files[0]).replace('.pkl', '').split('version=')
        if len(parts) > 1:
            return parts[1]
    raise ValueError(f"Could not auto-detect model timestamp from {experiment_dir}")


def get_xgboost_features(selected_features):
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
    return [f for f in unique if f not in XGBOOST_EXCLUDED_FEATURE_COLUMNS]


# ============================================================================
# UNCERTAINTY: LEAF-DENSITY METRIC (inlined from v6)
# ============================================================================


def compute_leaf_density_uncertainty(
    model: xgb.Booster,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_pred_raw: np.ndarray,
    classes: list[int],
    target_col: str,
    output_dir: str,
    model_time: str,
    config: dict | None = None,
) -> dict | None:
    """Compute leaf-density based epistemic uncertainty metric for XGBoost."""
    if config is None:
        config = XGBOOST_UNCERTAINTY_CONFIG
    if not config.get("enabled", True):
        return None

    max_trees = int(config.get("max_trees", 128))
    max_train_samples = int(config.get("max_train_samples", 250_000))
    max_test_samples = int(config.get("max_test_samples", 250_000))

    train_idx = np.arange(len(X_train))
    test_idx = np.arange(len(X_test))
    if len(X_train) > max_train_samples:
        train_idx = np.linspace(0, len(X_train) - 1, max_train_samples, dtype=int)
    if len(X_test) > max_test_samples:
        test_idx = np.linspace(0, len(X_test) - 1, max_test_samples, dtype=int)

    X_train_sub = X_train.iloc[train_idx]
    y_train_sub = y_train.iloc[train_idx]
    X_test_sub = X_test.iloc[test_idx]
    y_test_sub = y_test.iloc[test_idx]

    dtrain_sub = xgb.DMatrix(X_train_sub, label=y_train_sub)
    dtest_sub = xgb.DMatrix(X_test_sub, label=y_test_sub)

    if hasattr(model, "best_iteration") and model.best_iteration is not None:
        total_trees = int(model.best_iteration)
    else:
        total_trees = int(model.num_boosted_rounds())
    n_trees = min(max_trees, total_trees)

    try:
        leaves_train = model.predict(dtrain_sub, pred_leaf=True, iteration_range=(0, n_trees))
        leaves_test = model.predict(dtest_sub, pred_leaf=True, iteration_range=(0, n_trees))
    except TypeError:
        leaves_train = model.predict(dtrain_sub, pred_leaf=True)
        leaves_test = model.predict(dtest_sub, pred_leaf=True)
        if leaves_train.shape[1] > n_trees:
            leaves_train = leaves_train[:, :n_trees]
            leaves_test = leaves_test[:, :n_trees]

    n_trees_used = leaves_train.shape[1]

    train_counts_per_tree: list[dict[int, int]] = []
    for j in range(n_trees_used):
        col = leaves_train[:, j].astype(int)
        unique, counts = np.unique(col, return_counts=True)
        train_counts_per_tree.append(dict(zip(unique.tolist(), counts.tolist())))

    test_support_sum = np.zeros(leaves_test.shape[0], dtype=np.float64)
    test_support_min = np.full(leaves_test.shape[0], np.inf, dtype=np.float64)

    for j in range(n_trees_used):
        tree_counts = train_counts_per_tree[j]
        leaf_ids = leaves_test[:, j].astype(int)
        counts = np.fromiter(
            (tree_counts.get(int(lid), 0) for lid in leaf_ids),
            dtype=np.float64, count=leaf_ids.shape[0],
        )
        test_support_sum += counts
        test_support_min = np.minimum(test_support_min, counts)

    leaf_support_mean = test_support_sum / float(n_trees_used)
    epistemic_uncertainty = 1.0 / (1.0 + np.log1p(leaf_support_mean))

    prob_scores = y_pred_raw[test_idx] if len(classes) <= 2 else y_pred_raw[test_idx, :].max(axis=1)

    df_uncert = pd.DataFrame({
        "sample_index_in_test": test_idx,
        "y_true": y_test_sub.values,
        "pred_score": prob_scores,
        "leaf_support_mean": leaf_support_mean,
        "leaf_support_min": test_support_min,
        "epistemic_uncertainty": epistemic_uncertainty,
    })

    csv_path = os.path.join(
        output_dir,
        f"xgboost-uncertainty-target={target_col}_model=XGBoost-version={model_time}.csv",
    )
    df_uncert.to_csv(csv_path, index=False)

    quantiles = [0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]
    summary = {
        "target_col": target_col,
        "n_trees_used": int(n_trees_used),
        "train_sample_count_for_uncertainty": int(len(train_idx)),
        "test_sample_count_for_uncertainty": int(len(test_idx)),
        "leaf_support_mean_quantiles": {
            str(q): float(np.quantile(leaf_support_mean, q)) for q in quantiles
        },
        "epistemic_uncertainty_quantiles": {
            str(q): float(np.quantile(epistemic_uncertainty, q)) for q in quantiles
        },
        "csv_path": csv_path,
    }
    print(f"[UNCERTAINTY] Saved leaf-density diagnostics for {target_col} to: {csv_path}")
    sys.stdout.flush()
    return summary


# ============================================================================
# BALANCED SUBSET CREATION
# ============================================================================


def create_balanced_subset(
    X: pd.DataFrame,
    y: pd.Series,
    neg_pos_ratio: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.Series]:
    """Create a balanced subset: all positives + undersampled negatives."""
    pos_mask = y == 1
    neg_mask = y == 0
    n_pos = pos_mask.sum()
    n_neg_target = int(n_pos * neg_pos_ratio)

    rng = np.random.RandomState(random_state)
    neg_indices = y[neg_mask].index.to_numpy()
    neg_sample = rng.choice(
        neg_indices,
        size=min(n_neg_target, len(neg_indices)),
        replace=False,
    )
    pos_indices = y[pos_mask].index.to_numpy()
    selected = np.sort(np.concatenate([pos_indices, neg_sample]))
    return X.loc[selected], y.loc[selected]


# ============================================================================
# V19 PRECISION-AT-RECALL + THRESHOLD + CALIBRATION
# ============================================================================


def precision_at_recall(y_true, y_scores, min_recall=0.40):
    """Best precision achievable at recall >= min_recall."""
    precisions, recalls, _ = precision_recall_curve(y_true, y_scores)
    valid = recalls >= min_recall
    if not valid.any():
        return precisions[0]
    return float(precisions[valid].max())


def find_optimal_threshold(y_true, y_scores, min_recall=0.40):
    """Decision threshold maximising precision subject to recall >= min_recall."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
    precisions = precisions[:-1]
    recalls = recalls[:-1]
    valid = recalls >= min_recall
    if valid.any():
        idx = np.where(valid)[0][np.argmax(precisions[valid])]
    else:
        idx = np.argmax(recalls)
    return float(thresholds[idx]), float(precisions[idx]), float(recalls[idx])


def calibrate_ensemble_probabilities(preds_calib, y_calib, preds_target):
    """Platt scaling: recalibrate raw ensemble probabilities."""
    lr = LogisticRegression(solver='lbfgs', max_iter=1000)
    lr.fit(preds_calib.reshape(-1, 1), y_calib)
    return lr.predict_proba(preds_target.reshape(-1, 1))[:, 1]


# ============================================================================
# OPTUNA OBJECTIVE: PRECISION-AT-RECALL (ACCIDENT, balanced subsets)
# ============================================================================


class BalancedSubsetObjective:
    """Optuna objective maximising precision at a target recall on balanced subsets."""

    def __init__(self, X_train, y_train, feature_cols, target_col,
                 neg_pos_ratio, min_recall=0.40, n_folds=3, use_gpu=True,
                 csv_path=None, n_trials=80):
        self.X_train = X_train
        self.y_train = y_train
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.neg_pos_ratio = neg_pos_ratio
        self.min_recall = min_recall
        self.n_folds = n_folds
        self.use_gpu = use_gpu
        self.n_trials = n_trials
        self.csv_path = csv_path
        self.lock = threading.Lock()

    def __call__(self, trial):
        trial_start_time = time.time()
        try:
            params = {
                'tree_method': 'hist',
                'objective': 'binary:logistic',
                'eval_metric': 'aucpr',
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 50),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'gamma': trial.suggest_float('gamma', 0.0, 10.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
                'scale_pos_weight': trial.suggest_float(
                    'scale_pos_weight', 1.0, float(self.neg_pos_ratio) * 2
                ),
                'random_state': RANDOM_STATE,
            }
            if self.use_gpu and torch.cuda.is_available():
                params['device'] = 'cuda:0'
            else:
                params['nthread'] = N_THREADS

            print(
                f"\n{'='*80}\n"
                f"[TRIAL {trial.number + 1}/{self.n_trials}] STARTING - "
                f"{self.target_col} (V20 PRECISION@RECALL)\n"
                f"{'='*80}"
            )
            print(
                f"[HYPERPARAMS] max_depth={params['max_depth']}, "
                f"lr={params['learning_rate']:.6f}, "
                f"spw={params['scale_pos_weight']:.2f}, "
                f"min_child_weight={params['min_child_weight']}"
            )
            sys.stdout.flush()

            tscv = TimeSeriesSplit(n_splits=self.n_folds)
            cv_scores, cv_precisions, cv_recalls, best_iterations = [], [], [], []

            for fold_idx, (train_idx, val_idx) in enumerate(
                tscv.split(self.X_train), start=1
            ):
                X_fold_full = self.X_train.iloc[train_idx]
                y_fold_full = self.y_train.iloc[train_idx]
                X_fold_val = self.X_train.iloc[val_idx]
                y_fold_val = self.y_train.iloc[val_idx]

                if y_fold_val.sum() == 0:
                    print(f"[CV] Fold {fold_idx}/{self.n_folds} | SKIPPED (no positives in val)")
                    continue

                X_fold_train, y_fold_train = create_balanced_subset(
                    X_fold_full, y_fold_full,
                    neg_pos_ratio=self.neg_pos_ratio,
                    random_state=RANDOM_STATE + fold_idx,
                )

                dtrain = xgb.DMatrix(X_fold_train, label=y_fold_train)
                dval = xgb.DMatrix(X_fold_val, label=y_fold_val)

                model = xgb.train(
                    params, dtrain, num_boost_round=1000,
                    evals=[(dval, 'eval')],
                    early_stopping_rounds=XGBOOST_CONFIG['early_stopping'],
                    verbose_eval=False,
                )
                y_pred = model.predict(dval, iteration_range=(0, model.best_iteration))
                score = precision_at_recall(y_fold_val, y_pred, min_recall=self.min_recall)
                _, prec_at_thr, rec_at_thr = find_optimal_threshold(
                    y_fold_val, y_pred, min_recall=self.min_recall,
                )
                cv_scores.append(score)
                cv_precisions.append(prec_at_thr)
                cv_recalls.append(rec_at_thr)
                best_iterations.append(model.best_iteration)
                print(
                    f"[CV] Fold {fold_idx}/{self.n_folds} | "
                    f"Prec@Recall>={self.min_recall:.0%}: {score:.4f} | "
                    f"(P={prec_at_thr:.4f}, R={rec_at_thr:.4f}) | "
                    f"train_size={len(X_fold_train):,} (balanced) | "
                    f"best_iter={model.best_iteration}"
                )
                sys.stdout.flush()

            if not cv_scores:
                return 0.0

            mean_score = float(np.mean(cv_scores))
            trial_duration = time.time() - trial_start_time

            print(
                f"\n[TRIAL {trial.number + 1}/{self.n_trials}] COMPLETED\n"
                f"  CV Prec@Recall (mean): {mean_score:.4f}\n"
                f"  CV Prec@Recall (std):  {np.std(cv_scores):.4f}\n"
                f"  CV Precision (mean):   {np.mean(cv_precisions):.4f}\n"
                f"  CV Recall (mean):      {np.mean(cv_recalls):.4f}\n"
                f"  Training time:         {format_duration(trial_duration)}"
            )
            sys.stdout.flush()

            if self.csv_path is not None:
                row = {
                    'trial_number': trial.number + 1,
                    'target_col': self.target_col,
                    'max_depth': params['max_depth'],
                    'learning_rate': params['learning_rate'],
                    'scale_pos_weight': params['scale_pos_weight'],
                    'min_child_weight': params['min_child_weight'],
                    'subsample': params['subsample'],
                    'colsample_bytree': params['colsample_bytree'],
                    'gamma': params['gamma'],
                    'reg_alpha': params['reg_alpha'],
                    'reg_lambda': params['reg_lambda'],
                    'cv_precision_at_recall_mean': mean_score,
                    'cv_precision_at_recall_std': np.std(cv_scores),
                    'cv_precision_mean': np.mean(cv_precisions),
                    'cv_recall_mean': np.mean(cv_recalls),
                    'best_iterations_mean': np.mean(best_iterations),
                    'training_time_seconds': trial_duration,
                }
                with self.lock:
                    file_exists = (
                        os.path.exists(self.csv_path)
                        and os.path.getsize(self.csv_path) > 0
                    )
                    with open(self.csv_path, 'a', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=row.keys())
                        if not file_exists:
                            writer.writeheader()
                        writer.writerow(row)
            return mean_score

        except Exception as e:
            print(f"[ERROR] Trial {trial.number + 1} failed: {e}")
            sys.stdout.flush()
            return 0.0


# ============================================================================
# OPTUNA OBJECTIVE: SECONDARY TARGETS (e.g. C_NIVELL_AFECTACIO)
# ============================================================================


class XGBoostClassifierObjective:
    """Optuna objective for non-ACCIDENT XGBoost targets."""

    def __init__(self, X_train, y_train, feature_cols, target_col, n_classes,
                 n_folds=3, use_gpu=True, csv_path=None, n_trials=100):
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
                'random_state': RANDOM_STATE,
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

            print(
                f"\n{'='*80}\n"
                f"[TRIAL {trial.number + 1}/{self.n_trials}] STARTING - "
                f"target={self.target_col}\n"
                f"{'='*80}"
            )
            sys.stdout.flush()

            tscv = TimeSeriesSplit(n_splits=self.n_folds)
            cv_scores, best_iterations = [], []

            for fold_idx, (train_idx, val_idx) in enumerate(
                tscv.split(self.X_train), start=1
            ):
                dtrain = xgb.DMatrix(
                    self.X_train.iloc[train_idx], label=self.y_train.iloc[train_idx]
                )
                dval = xgb.DMatrix(
                    self.X_train.iloc[val_idx], label=self.y_train.iloc[val_idx]
                )
                model = xgb.train(
                    params, dtrain, num_boost_round=1000,
                    evals=[(dval, 'eval')],
                    early_stopping_rounds=XGBOOST_CONFIG['early_stopping'],
                    verbose_eval=False,
                )
                y_pred = model.predict(dval, iteration_range=(0, model.best_iteration))
                if self.n_classes <= 2:
                    score = roc_auc_score(self.y_train.iloc[val_idx], y_pred)
                else:
                    score = f1_score(
                        self.y_train.iloc[val_idx],
                        np.argmax(y_pred, axis=1),
                        average='weighted', zero_division=0,
                    )
                cv_scores.append(score)
                best_iterations.append(model.best_iteration)
                print(
                    f"[CV] Fold {fold_idx}/{self.n_folds} | "
                    f"score={score:.4f} | best_iteration={model.best_iteration}"
                )
                sys.stdout.flush()

            mean_score = float(np.mean(cv_scores))
            trial_duration = time.time() - trial_start_time

            if self.csv_path is not None:
                row = {
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
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                }
                file_exists = os.path.exists(self.csv_path)
                with self.lock:
                    with open(self.csv_path, 'a', newline='') as csvfile:
                        writer = csv.DictWriter(csvfile, fieldnames=row.keys())
                        if not file_exists:
                            writer.writeheader()
                        writer.writerow(row)

            print(
                f"\n[TRIAL {trial.number + 1}/{self.n_trials}] COMPLETED - "
                f"target={self.target_col}\n"
                f"  CV score (mean): {mean_score:.4f}\n"
                f"  CV score (std):  {np.std(cv_scores):.4f}\n"
                f"  Avg best iter:   {np.mean(best_iterations):.1f}\n"
                f"  Training time:   {format_duration(trial_duration)}"
            )
            sys.stdout.flush()
            return mean_score

        except Exception as e:
            print(
                f"[ERROR] Trial {trial.number + 1} failed for "
                f"target={self.target_col}: {str(e)[:500]}"
            )
            sys.stdout.flush()
            raise


# ============================================================================
# V20 FEATURE ENGINEERING
# ============================================================================


# ---------------------------------------------------------------------------
# Covariate-side feature engineering (BENCH_COV_ENGINEERED)
#
# The H2 arms of Section 5.3(a) receive twenty plain covariates while the
# decomposed arm receives 52, of which 25 are lags, rolling statistics and
# z-scores of the traffic channels. Part of the two-order-of-magnitude gap
# could therefore be engineering rather than decomposition. This builds the
# same engineering families on the covariates a 72-hour issuance may use --
# the 3-day-lead weather forecast and the calendar -- so the direct arm is
# given a comparable budget rather than a plain feature vector.
#
# Only 3d_fcst_* enters: the 1-day-lead forecast is not admissible at H = 72 h.
# Every column is prefixed `cov_` so the covariates-only keep-list can admit
# them without naming each one.
# ---------------------------------------------------------------------------

_COV_BASE = ["3d_fcst_temperature_2m", "3d_fcst_precipitation",
             "3d_fcst_wind_gusts_10m", "3d_fcst_cloud_cover"]


def build_covariate_engineered_features(df, baseline_end_date=None):
    """Lags, rolling statistics and per-(pk, hour, weekday) z-scores of the
    72-hour-admissible covariates. Returns (df, new_column_names)."""
    import numpy as np
    import pandas as pd

    df = df.reset_index(drop=True).copy()
    loc = [c for c in _LOCATION_KEYS if c in df.columns]
    base = [c for c in _COV_BASE if c in df.columns]
    if not loc or not base:
        print("[COV-FE] nothing to build (missing location keys or covariates)")
        return df, []

    new = []
    grp = df.groupby(loc, sort=False)
    for c in base:
        short = "cov_" + c.replace("3d_fcst_", "")
        for lag in (1, 2, 4):
            k = f"{short}_lag{lag}"
            df[k] = grp[c].shift(lag)
            new.append(k)
        k = f"{short}_d1"
        df[k] = df[c] - df[f"{short}_lag1"]
        new.append(k)
        for w in (4, 12):
            k = f"{short}_mean_{w}"
            df[k] = (grp[c].rolling(window=w, min_periods=1).mean()
                          .reset_index(level=loc, drop=True))
            new.append(k)
            k = f"{short}_std_{w}"
            df[k] = (grp[c].rolling(window=w, min_periods=2).std()
                          .reset_index(level=loc, drop=True))
            new.append(k)

    # z-scores against the same (pk, hour, weekday) expectation the traffic
    # features use, estimated on the pre-boundary slice only.
    if set(_BASELINE_KEYS).issubset(df.columns):
        if baseline_end_date is not None and "dat" in df.columns:
            sl = pd.to_datetime(df["dat"]) < pd.Timestamp(baseline_end_date)
        else:
            sl = pd.Series(True, index=df.index)
        agg = (df[sl].groupby(_BASELINE_KEYS, observed=True)[base]
                     .agg(["mean", "std"]))
        agg.columns = [f"{a}__{b}" for a, b in agg.columns]
        df = df.merge(agg.reset_index(), on=_BASELINE_KEYS, how="left")
        for c in base:
            short = "cov_" + c.replace("3d_fcst_", "")
            mu, sd = f"{c}__mean", f"{c}__std"
            if mu in df.columns:
                s = df[sd].fillna(_EPS).clip(lower=_EPS)
                k = f"{short}_zscore"
                df[k] = (df[c] - df[mu]) / s
                new.append(k)
        df = df.drop(columns=[c for c in df.columns if c.endswith(("__mean", "__std"))])

    # calendar shape the traffic features get for free through the traffic
    if "dia" in df.columns and "mes" in df.columns:
        doy = (pd.to_datetime(df["dat"]).dt.dayofyear
               if "dat" in df.columns else df["dia"])
        df["cov_doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
        df["cov_doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
        new += ["cov_doy_sin", "cov_doy_cos"]
    if "diaSem" in df.columns:
        df["cov_is_weekend"] = (df["diaSem"] >= 5).astype("int8")
        new.append("cov_is_weekend")

    df[new] = df[new].astype("float32")
    print(f"[COV-FE] built {len(new)} engineered covariate features")
    return df, new


def build_engineered_features(
    df: pd.DataFrame,
    train_cutoff_idx: int,
    baseline_end_date=None,
) -> tuple[pd.DataFrame, list[str]]:
    """Inject literature-motivated features derived from the GNN traffic outputs.

    All per-(pk, hor, diaSem) baselines (z-score mu/sigma, crash rate, free-flow
    speed) are estimated on a *training-only* slice, so they do not leak into the
    calibration / test / simulation splits.

    Which rows form that slice is controlled by ``baseline_end_date``:

    * ``baseline_end_date`` given (**preferred**) — the slice is the CHRONOLOGICAL
      set ``df["dat"] < baseline_end_date``. Pass the first timestamp of the
      evaluation window (e.g. ``SIM_START_DATE``).
    * ``baseline_end_date=None`` — legacy positional fallback, ``df.iloc[:train_cutoff_idx]``.

    The positional form is only equivalent to a chronological one when row
    position tracks time, and in this pipeline it never does: the frame is sorted
    location-major (via, sen, pk, dat), so a prefix of ROWS is a prefix of
    LOCATIONS. That produced two distinct defects, both measured on the real data:

      * layout-dependence — under a pk-major frame (the GNN-bypass branches) the
        70% prefix covered only PKs 120-188, leaving 29.8% of rows with
        ``speed_zscore`` NaN->0 and ``pk_crash_rate`` = 0.0, while the sen-major
        frame (the GNN branch) covered all 100 PKs. Identical inputs, different
        features, purely because of sort order;
      * label leakage — under either sort the prefix ran to 2025-06-30 and so
        contained 901 of the 1,072 June-2025 accident labels. ``pk_crash_rate``
        is ``mean(ACCIDENT)`` per (pk, hor, diaSem), so the model was handed the
        June crash rate of the very cell it was asked to predict.

    A date cutoff removes both at once: it is invariant to row order, so every
    arm gets byte-identical baselines regardless of layout, and it cannot reach
    into the evaluation window. Prefer it; the positional path is kept only so
    legacy callers keep working.

    The input ``df`` must have a fresh contiguous RangeIndex and be sorted by
    (via, sen, pk, dat) as produced by ``generate_gnn_predictions_on_training_data``.
    Within each (via, sen, pk) group rows are in chronological order, which is
    the requirement for shift() and rolling() to produce meaningful lags.

    Returns (df_with_new_cols, list_of_new_column_names).
    """
    fe_start = time.time()
    cfg = FEATURE_ENGINEERING_CONFIG

    df = df.reset_index(drop=True).copy()
    n = len(df)
    if train_cutoff_idx <= 0 or train_cutoff_idx > n:
        train_cutoff_idx = int(n * 0.70)

    # Positional boolean selector for the baseline slice. Held as a numpy array
    # (not an index) because the merges below rebuild df's index; a left-merge on
    # unique aggregate keys preserves both row order and length, so positions
    # stay valid throughout.
    if baseline_end_date is not None:
        if "dat" not in df.columns:
            raise ValueError(
                "build_engineered_features: baseline_end_date given but the frame "
                "has no 'dat' column to apply it to."
            )
        _cut_ts = pd.Timestamp(baseline_end_date)
        baseline_slice = (pd.to_datetime(df["dat"]) < _cut_ts).to_numpy()
        if not baseline_slice.any():
            raise ValueError(
                f"build_engineered_features: no rows before baseline_end_date "
                f"{_cut_ts} — baselines would be empty."
            )
        print(
            f"[V21-FE] Engineering features: {n:,} rows, "
            f"baseline slice = dat < {_cut_ts} "
            f"({baseline_slice.sum():,} rows, {baseline_slice.mean()*100:.1f}%) "
            f"[chronological, order-invariant]"
        )
    else:
        baseline_slice = np.zeros(n, dtype=bool)
        baseline_slice[:train_cutoff_idx] = True
        print(
            f"[V21-FE] Engineering features: {n:,} rows, "
            f"train slice = [0:{train_cutoff_idx:,}]  "
            f"[LEGACY POSITIONAL — layout-dependent, may leak; "
            f"pass baseline_end_date instead]"
        )
    sys.stdout.flush()

    # Positional column so we can restore row order after any merges
    df["__row_order"] = np.arange(n, dtype=np.int64)

    new_features: list[str] = []

    has_speed = "mean_speed_gnn" in df.columns
    has_vol = "intTot_gnn" in df.columns
    has_intp = "intP_gnn" in df.columns

    if not has_speed:
        print("[V21-FE] WARNING: mean_speed_gnn missing — most FE groups skipped.")
        sys.stdout.flush()

    loc_cols = [c for c in _LOCATION_KEYS if c in df.columns]
    grp_loc = df.groupby(loc_cols, sort=False) if loc_cols else None

    # ── Group 3: Lag features ─────────────────────────────────────────────
    if cfg["lag_features"] and has_speed and grp_loc is not None:
        for lag in LAG_STEPS:
            col = f"speed_lag{lag}"
            df[col] = grp_loc["mean_speed_gnn"].shift(lag)
            new_features.append(col)
        if has_vol:
            for lag in [1, 2]:
                col = f"vol_lag{lag}"
                df[col] = grp_loc["intTot_gnn"].shift(lag)
                new_features.append(col)

    # ── Group 2: First differences ────────────────────────────────────────
    if cfg["first_differences"] and has_speed and grp_loc is not None:
        lag1 = (
            df["speed_lag1"] if "speed_lag1" in df.columns
            else grp_loc["mean_speed_gnn"].shift(1)
        )
        lag2 = (
            df["speed_lag2"] if "speed_lag2" in df.columns
            else grp_loc["mean_speed_gnn"].shift(2)
        )
        df["delta_speed_1"] = df["mean_speed_gnn"] - lag1
        df["delta_speed_2"] = df["mean_speed_gnn"] - lag2
        new_features.extend(["delta_speed_1", "delta_speed_2"])
        if has_vol:
            vlag1 = (
                df["vol_lag1"] if "vol_lag1" in df.columns
                else grp_loc["intTot_gnn"].shift(1)
            )
            df["delta_vol_1"] = df["intTot_gnn"] - vlag1
            new_features.append("delta_vol_1")
        df["accel_speed"] = df["delta_speed_1"] / max(get_time_resolution_minutes(), 1)
        new_features.append("accel_speed")

    # ── Group 1: Rolling variability ──────────────────────────────────────
    if cfg["speed_variability"] and has_speed and grp_loc is not None:
        for w in ROLL_WINDOWS:
            col = f"speed_std_{w}"
            df[col] = (
                grp_loc["mean_speed_gnn"]
                .rolling(window=w, min_periods=2)
                .std()
                .reset_index(level=loc_cols, drop=True)
            )
            new_features.append(col)
        df["speed_mean_4"] = (
            grp_loc["mean_speed_gnn"]
            .rolling(window=4, min_periods=1)
            .mean()
            .reset_index(level=loc_cols, drop=True)
        )
        new_features.append("speed_mean_4")
        if has_vol:
            df["vol_std_2"] = (
                grp_loc["intTot_gnn"]
                .rolling(window=2, min_periods=2)
                .std()
                .reset_index(level=loc_cols, drop=True)
            )
            new_features.append("vol_std_2")
        speed_denom = df["mean_speed_gnn"].abs().clip(lower=_EPS)
        if "speed_std_2" in df.columns:
            df["speed_cv_2"] = df["speed_std_2"] / speed_denom
            new_features.append("speed_cv_2")
        if has_vol and "vol_std_2" in df.columns:
            df["vol_cv_2"] = df["vol_std_2"] / df["intTot_gnn"].abs().clip(lower=_EPS)
            new_features.append("vol_cv_2")

    # ── Group 5: Spatial gradient (upstream / downstream PK) ─────────────
    if cfg["spatial_gradient"] and has_speed and {"dat", "pk"}.issubset(df.columns):
        sort_cols = [c for c in ["dat", "via", "sen", "pk"] if c in df.columns]
        grp_cols = [c for c in ["dat", "via", "sen"] if c in df.columns]
        tmp = df.sort_values(sort_cols, kind="stable")
        grp_t = tmp.groupby(grp_cols, sort=False)

        up_spd = grp_t["mean_speed_gnn"].shift(1).reindex(df.index)
        dn_spd = grp_t["mean_speed_gnn"].shift(-1).reindex(df.index)
        df["speed_grad_up"] = df["mean_speed_gnn"] - up_spd
        df["speed_grad_dn"] = dn_spd - df["mean_speed_gnn"]
        new_features.extend(["speed_grad_up", "speed_grad_dn"])
        if has_vol:
            df["vol_grad_up"] = (
                df["intTot_gnn"] - grp_t["intTot_gnn"].shift(1).reindex(df.index)
            )
            new_features.append("vol_grad_up")
        del tmp, grp_t

    # ── Group 6: Z-score from (pk, hor, diaSem) baseline ─────────────────
    if (
        cfg["zscore_baseline"]
        and has_speed
        and set(_BASELINE_KEYS).issubset(df.columns)
    ):
        agg_cols = ["mean_speed_gnn"] + (["intTot_gnn"] if has_vol else [])
        base = (
            df[baseline_slice]
            .groupby(_BASELINE_KEYS, observed=True)[agg_cols]
            .agg(["mean", "std"])
        )
        base.columns = [f"{a}_{b}" for a, b in base.columns]
        base = base.reset_index()
        df = df.merge(base, on=_BASELINE_KEYS, how="left")

        sigma_spd = df["mean_speed_gnn_std"].fillna(_EPS).clip(lower=_EPS)
        df["speed_zscore"] = (df["mean_speed_gnn"] - df["mean_speed_gnn_mean"]) / sigma_spd
        df["speed_pct_below_normal"] = (df["speed_zscore"] < -1.5).astype("int8")
        new_features.extend(["speed_zscore", "speed_pct_below_normal"])
        drop_cols = ["mean_speed_gnn_mean", "mean_speed_gnn_std"]
        if has_vol:
            sigma_vol = df["intTot_gnn_std"].fillna(_EPS).clip(lower=_EPS)
            df["vol_zscore"] = (df["intTot_gnn"] - df["intTot_gnn_mean"]) / sigma_vol
            new_features.append("vol_zscore")
            drop_cols += ["intTot_gnn_mean", "intTot_gnn_std"]
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # ── Group 4: Historical crash-rate precursor (SPF analogue) ──────────
    if (
        cfg["crash_precursor"]
        and "ACCIDENT" in df.columns
        and set(_BASELINE_KEYS).issubset(df.columns)
    ):
        ts = df[baseline_slice]
        cr = (
            ts.groupby(_BASELINE_KEYS, observed=True)["ACCIDENT"]
            .mean()
            .reset_index()
            .rename(columns={"ACCIDENT": "pk_crash_rate"})
        )
        df = df.merge(cr, on=_BASELINE_KEYS, how="left")
        df["pk_crash_rate"] = df["pk_crash_rate"].fillna(0.0)
        df["pk_crash_rate_log"] = np.log1p(df["pk_crash_rate"])
        new_features.extend(["pk_crash_rate", "pk_crash_rate_log"])
        if "mob_esp" in df.columns:
            cr_mob = (
                ts.groupby(_BASELINE_KEYS + ["mob_esp"], observed=True)["ACCIDENT"]
                .mean()
                .reset_index()
                .rename(columns={"ACCIDENT": "pk_crash_rate_mob"})
            )
            df = df.merge(cr_mob, on=_BASELINE_KEYS + ["mob_esp"], how="left")
            df["pk_crash_rate_mob"] = df["pk_crash_rate_mob"].fillna(0.0)
            new_features.append("pk_crash_rate_mob")

    # ── Group 7: Flow regime + occupancy proxy ────────────────────────────
    if cfg["flow_regime"] and has_speed and "pk" in df.columns:
        vff = (
            df[baseline_slice]
            .groupby("pk", observed=True)["mean_speed_gnn"]
            .quantile(0.85)
            .rename("__v_ff")
            .reset_index()
        )
        df = df.merge(vff, on="pk", how="left")
        vff_vals = df["__v_ff"].fillna(float(df["mean_speed_gnn"].median()))
        df["flow_regime"] = np.where(
            df["mean_speed_gnn"] > 0.85 * vff_vals, 0,
            np.where(df["mean_speed_gnn"] > 0.70 * vff_vals, 1, 2),
        ).astype("int8")
        df = df.drop(columns=["__v_ff"])
        new_features.append("flow_regime")
        if has_vol:
            df["occ_proxy"] = df["intTot_gnn"] / df["mean_speed_gnn"].abs().clip(lower=_EPS)
            new_features.append("occ_proxy")

    # ── Group 8: Heavy vehicle fraction ──────────────────────────────────
    if cfg["hv_fraction"] and has_vol and has_intp:
        df["hv_fraction"] = (
            (df["intTot_gnn"] - df["intP_gnn"]) /
            df["intTot_gnn"].abs().clip(lower=_EPS)
        )
        new_features.append("hv_fraction")
        if loc_cols:
            df["hv_fraction_lag1"] = (
                df.groupby(loc_cols, sort=False)["hv_fraction"].shift(1)
            )
            new_features.append("hv_fraction_lag1")

    # ── Group 9: Geometry x traffic interaction ───────────────────────────
    if cfg["geometry_interaction"] and has_speed and "ang_curv" in df.columns:
        df["curv_x_speed"] = df["ang_curv"].astype(float) * df["mean_speed_gnn"]
        new_features.append("curv_x_speed")
        pend_cols = [c for c in ("ang_pend_pos", "ang_pend_neg") if c in df.columns]
        if pend_cols:
            df["pend_x_speed"] = (
                df[pend_cols].abs().max(axis=1) * df["mean_speed_gnn"]
            )
            new_features.append("pend_x_speed")
        if "delta_speed_1" in df.columns:
            df["curv_x_delta_speed"] = (
                df["ang_curv"].astype(float) * df["delta_speed_1"]
            )
            new_features.append("curv_x_delta_speed")

    # ── Group 10: Temporal flags ──────────────────────────────────────────
    if cfg["temporal_flags"] and "hor" in df.columns:
        df["is_peak_hour"] = df["hor"].isin([7, 8, 9, 17, 18, 19, 20]).astype("int8")
        new_features.append("is_peak_hour")

    # ── Group 11: Cyclical temporal encoding (V21) ────────────────────────
    if cfg.get("cyclical_temporal", False):
        two_pi = 2.0 * np.pi
        if "hor" in df.columns:
            h = df["hor"].astype(float)
            df["hor_sin"] = np.sin(two_pi * h / 24.0).astype("float32")
            df["hor_cos"] = np.cos(two_pi * h / 24.0).astype("float32")
            new_features += ["hor_sin", "hor_cos"]
        if "diaSem" in df.columns:
            d = df["diaSem"].astype(float)
            df["diaSem_sin"] = np.sin(two_pi * d / 7.0).astype("float32")
            df["diaSem_cos"] = np.cos(two_pi * d / 7.0).astype("float32")
            new_features += ["diaSem_sin", "diaSem_cos"]
        if "mes" in df.columns:
            m = df["mes"].astype(float)
            df["mes_sin"] = np.sin(two_pi * m / 12.0).astype("float32")
            df["mes_cos"] = np.cos(two_pi * m / 12.0).astype("float32")
            new_features += ["mes_sin", "mes_cos"]

    # ── Group 12: Derived weather signals (V21) ───────────────────────────
    if cfg.get("weather_derived", False):
        if "1d_fcst_precipitation" in df.columns:
            precip = df["1d_fcst_precipitation"].astype(float)
            df["is_raining"] = (precip > 0.1).astype("int8")
            df["is_heavy_rain"] = (precip > 2.5).astype("int8")
            new_features += ["is_raining", "is_heavy_rain"]
        if "1d_fcst_temperature_2m" in df.columns:
            t = df["1d_fcst_temperature_2m"].astype(float)
            df["temp_near_freezing"] = ((t >= -2.0) & (t <= 2.0)).astype("int8")
            new_features.append("temp_near_freezing")
        if ("1d_fcst_cloud_cover" in df.columns
                and "1d_fcst_precipitation" in df.columns):
            df["low_visibility_proxy"] = (
                (df["1d_fcst_cloud_cover"].astype(float) > 80.0)
                & (df["1d_fcst_precipitation"].astype(float) > 0.1)
            ).astype("int8")
            new_features.append("low_visibility_proxy")
        if ("1d_fcst_wind_gusts_10m" in df.columns
                and "1d_fcst_wind_speed_10m" in df.columns):
            df["wind_gust_excess"] = (
                df["1d_fcst_wind_gusts_10m"].astype(float)
                - df["1d_fcst_wind_speed_10m"].astype(float)
            ).astype("float32")
            new_features.append("wind_gust_excess")
        if ("1d_fcst_precipitation" in df.columns
                and "3d_fcst_precipitation" in df.columns):
            df["precip_change_1d_vs_3d"] = (
                df["1d_fcst_precipitation"].astype(float)
                - df["3d_fcst_precipitation"].astype(float)
            ).astype("float32")
            new_features.append("precip_change_1d_vs_3d")

    # ── Group 13: Weather × geometry/traffic interactions (V21) ───────────
    if cfg.get("weather_interaction", False):
        if "is_raining" in df.columns:
            if "ang_curv" in df.columns:
                df["wet_curve_risk"] = (
                    df["is_raining"].astype("float32")
                    * df["ang_curv"].astype(float).abs()
                ).astype("float32")
                new_features.append("wet_curve_risk")
            if "ang_pend_neg" in df.columns:
                df["wet_descent_risk"] = (
                    df["is_raining"].astype("float32")
                    * df["ang_pend_neg"].astype(float).abs()
                ).astype("float32")
                new_features.append("wet_descent_risk")
            if "mean_speed_gnn" in df.columns:
                df["wet_speed"] = (
                    df["is_raining"].astype("float32")
                    * df["mean_speed_gnn"].astype(float)
                ).astype("float32")
                new_features.append("wet_speed")
        if ("temp_near_freezing" in df.columns
                and "ang_curv" in df.columns):
            df["cold_curve_risk"] = (
                df["temp_near_freezing"].astype("float32")
                * df["ang_curv"].astype(float).abs()
            ).astype("float32")
            new_features.append("cold_curve_risk")

    # Restore original row order (survives any merges via __row_order column)
    df = df.sort_values("__row_order", kind="stable").reset_index(drop=True)
    df = df.drop(columns=["__row_order"])

    # Deduplicate while preserving insertion order; guard against missing cols
    seen: set[str] = set()
    unique_new: list[str] = []
    for f in new_features:
        if f not in seen and f in df.columns:
            seen.add(f)
            unique_new.append(f)

    print(
        f"[V21-FE] Added {len(unique_new)} engineered features "
        f"in {format_duration(time.time() - fe_start)}:"
    )
    for i in range(0, len(unique_new), 6):
        print(f"  {', '.join(unique_new[i:i+6])}")
    sys.stdout.flush()

    return df, unique_new


# ============================================================================
# XGBOOST TRAINING (V20: V19 logic + feature engineering)
# ============================================================================


def train_xgboost_balanced_bagging(
    df_train_data: pd.DataFrame,
    output_dir: str,
    model_time: str,
):
    """Train XGBoost balanced-bagging ensemble (V20: adds feature engineering).

    Identical to V19 except that engineered features (Groups 1–10) are injected
    on the GNN-predicted traffic columns immediately after GNN replacement and
    before the 70/10/20 split. Baselines for leakage-sensitive features are
    fitted only on the training slice.
    """
    xgb_total_start = time.time()
    print("\n" + "=" * 80)
    print("STAGE 2: XGBOOST PRECISION-OPTIMIZED BALANCED BAGGING (V20)")
    print("=" * 80)
    print(f"[V20] Ensemble config: {ENSEMBLE_CONFIG}")
    print(f"[V20] Threshold config: {THRESHOLD_CONFIG}")
    sys.stdout.flush()

    xgb_prep_start = time.time()
    feature_cols = get_xgboost_features(get_selected_features())
    available_features = [f for f in feature_cols if f in df_train_data.columns]
    print(f"[XGB] Base features available: {len(available_features)}")

    # ── GNN prediction replacement (identical to V19) ─────────────────────
    has_gnn_preds = "mean_speed_gnn" in df_train_data.columns
    if has_gnn_preds:
        df_train_data = df_train_data.dropna(subset=["mean_speed_gnn"]).copy()
        print(
            f"[V20] Replacing traffic features with GNN predictions "
            f"({len(df_train_data):,} rows)"
        )
        if "mean_speed" in available_features:
            df_train_data["mean_speed"] = df_train_data["mean_speed_gnn"]
        if "intTot" in available_features:
            df_train_data["intTot"] = df_train_data["intTot_gnn"]
        if "intP" in available_features:
            df_train_data["intP"] = df_train_data["intP_gnn"]
        # REMOVED 2026-08-11: `car` is the LANE COUNT (2-5, constant per pk), a
        # static geometry feature. It was overwritten here with intTot - intP
        # (light-vehicle count) — the 9th such site, missed by the first sweep
        # because this one lives in v21's own legacy pipeline rather than the
        # v50/v51 chain. Raw `car` is preserved.
    else:
        print("[WARNING] No GNN prediction columns found.")
    sys.stdout.flush()

    # ── V20: Feature engineering on GNN outputs ───────────────────────────
    # train_cutoff_idx matches the split_train that will be used below.
    train_cutoff_for_fe = int(len(df_train_data) * 0.70)
    df_train_data, new_fe_features = build_engineered_features(
        df_train_data, train_cutoff_for_fe
    )
    fe_set = set(available_features)
    new_avail = [f for f in new_fe_features
                 if f in df_train_data.columns and f not in fe_set]
    available_features = available_features + new_avail
    print(
        f"[V20] Feature engineering added {len(new_avail)} features. "
        f"Total: {len(available_features)}"
    )
    sys.stdout.flush()

    X = df_train_data[available_features].fillna(0)

    # 70 / 10 / 20 split (same as V19)
    split_train = int(len(X) * 0.70)
    split_calib = int(len(X) * 0.80)
    X_train = X.iloc[:split_train]
    X_calib = X.iloc[split_train:split_calib]
    X_test = X.iloc[split_calib:]

    scaler = MinMaxScaler()
    scaler.fit(X_train)
    joblib.dump(
        scaler,
        os.path.join(output_dir, f"scaler-features_model=XGBoost-version={model_time}.pkl"),
    )

    print(
        f"[V20] Data split: train={len(X_train):,} | calib={len(X_calib):,} | "
        f"test={len(X_test):,}"
    )
    print(f"[XGB] Data preparation: {format_duration(time.time() - xgb_prep_start)}")
    sys.stdout.flush()

    target_cols = [c for c in XGBOOST_TARGET_COLUMNS if c in df_train_data.columns]
    if not target_cols:
        raise ValueError("No configured target columns found in dataset.")

    studies_by_target: dict[str, optuna.Study] = {}
    models_by_target: dict[str, str] = {}
    metrics_by_target: dict[str, dict] = {}
    params_by_target: dict[str, dict] = {}

    min_recall = THRESHOLD_CONFIG["min_recall"]

    for target_col in target_cols:
        print("\n" + "-" * 80)
        print(f"[TARGET] Training target: {target_col}")

        y_raw = df_train_data[target_col].fillna(0)
        classes = sorted(pd.Series(y_raw).astype(int).unique().tolist())
        if len(classes) < 2:
            print(f"[WARNING] Skipping {target_col}: only one class.")
            metrics_by_target[target_col] = {"status": "skipped_single_class"}
            continue

        class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        y_all = pd.Series(y_raw).astype(int).map(class_to_idx).astype(int)
        y_train = y_all.iloc[:split_train]
        y_calib = y_all.iloc[split_train:split_calib]
        y_test = y_all.iloc[split_calib:]

        pos_rate = y_train.sum() / len(y_train) * 100
        print(f"[INFO] Classes ({target_col}): {classes}")
        print(f"[INFO] Train: {len(X_train):,} | Calib: {len(X_calib):,} | Test: {len(X_test):,}")
        print(
            f"[INFO] Positive rate (train): {pos_rate:.4f}% "
            f"({int(y_train.sum()):,} positives)"
        )
        sys.stdout.flush()

        csv_path = os.path.join(
            output_dir,
            f"training-evolution_model=XGBoost-target={target_col}-version={model_time}.csv",
        )

        use_ensemble = (len(classes) <= 2) and (target_col == "ACCIDENT")
        roc_auc = None
        pr_auc = None
        model_path = None
        optimal_threshold = 0.5
        test_prec = test_rec = test_f1 = test_f2 = fdr = 0.0
        y_pred_raw: np.ndarray
        final_model = None
        base_models: list[xgb.Booster] = []

        if use_ensemble:
            n_base = ENSEMBLE_CONFIG["n_base_models"]
            neg_pos_ratio = ENSEMBLE_CONFIG["neg_pos_ratio"]
            n_trials = ENSEMBLE_CONFIG["optuna_trials"]

            print(f"[V20] Training precision-optimized ensemble for {target_col}")
            print(f"[V20] {n_base} base models, neg:pos ratio = {neg_pos_ratio}:1")
            print(f"[V20] Optuna objective: precision @ recall >= {min_recall:.0%}")
            sys.stdout.flush()

            objective = BalancedSubsetObjective(
                X_train, y_train, available_features,
                target_col=target_col, neg_pos_ratio=neg_pos_ratio,
                min_recall=min_recall,
                n_folds=XGBOOST_CONFIG['cv_folds'],
                use_gpu=USE_GPU, csv_path=csv_path, n_trials=n_trials,
            )
            study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
            )
            t0 = time.time()
            study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
            print(f"[COMPLETE] Hyp search done ({target_col})")
            print(f"[RESULT] Best Precision@Recall: {study.best_value:.4f}")
            print(f"[TIME] {format_duration(time.time() - t0)}")
            sys.stdout.flush()

            best_params = {
                **study.best_params,
                'tree_method': 'hist',
                'objective': 'binary:logistic',
                'eval_metric': 'aucpr',
                'random_state': RANDOM_STATE,
            }
            if USE_GPU and torch.cuda.is_available():
                best_params['device'] = 'cuda:0'
            else:
                best_params['nthread'] = N_THREADS

            dcalib = xgb.DMatrix(X_calib, label=y_calib)
            base_model_paths = []

            for i in range(n_base):
                t_i = time.time()
                X_sub, y_sub = create_balanced_subset(
                    X_train, y_train,
                    neg_pos_ratio=neg_pos_ratio,
                    random_state=RANDOM_STATE + i * 1000,
                )
                model = xgb.train(
                    best_params, xgb.DMatrix(X_sub, label=y_sub),
                    num_boost_round=2000,
                    evals=[(dcalib, 'eval')],
                    early_stopping_rounds=100,
                    verbose_eval=False,
                )
                base_models.append(model)
                path = os.path.join(
                    output_dir,
                    f"xgboost-ensemble-base{i}_target={target_col.lower()}"
                    f"_model=XGBoost-version={model_time}.json",
                )
                model.save_model(path)
                base_model_paths.append(path)
                aucpr_i = average_precision_score(
                    y_calib,
                    model.predict(dcalib, iteration_range=(0, model.best_iteration)),
                )
                print(
                    f"[V20] Base model {i+1}/{n_base}: "
                    f"train_size={len(X_sub):,}, best_iter={model.best_iteration}, "
                    f"AUCPR(calib)={aucpr_i:.4f}, time={format_duration(time.time()-t_i)}"
                )
                sys.stdout.flush()

            def _ensemble_predict(dmat):
                preds = np.zeros(dmat.num_row())
                for m in base_models:
                    preds += m.predict(dmat, iteration_range=(0, m.best_iteration))
                return preds / n_base

            ens_calib = _ensemble_predict(dcalib)
            dtest = xgb.DMatrix(X_test, label=y_test)
            ens_test = _ensemble_predict(dtest)

            if THRESHOLD_CONFIG["calibrate"]:
                cal_calib = calibrate_ensemble_probabilities(ens_calib, y_calib, ens_calib)
                cal_test = calibrate_ensemble_probabilities(ens_calib, y_calib, ens_test)
                preds_for_thr = cal_calib
                preds_for_eval = cal_test
            else:
                preds_for_thr = ens_calib
                preds_for_eval = ens_test

            optimal_threshold, calib_precision, calib_recall = find_optimal_threshold(
                y_calib, preds_for_thr, min_recall=min_recall,
            )
            print(f"[V20] Optimal threshold: {optimal_threshold:.4f}")
            print(
                f"[V20] Calibration: precision={calib_precision:.4f}, "
                f"recall={calib_recall:.4f}"
            )
            sys.stdout.flush()

            y_pred_raw = preds_for_eval
            y_pred = (preds_for_eval >= optimal_threshold).astype(int)
            roc_auc = roc_auc_score(y_test, preds_for_eval)
            pr_auc = average_precision_score(y_test, preds_for_eval)
            test_prec = precision_score(y_test, y_pred, zero_division=0)
            test_rec = recall_score(y_test, y_pred, zero_division=0)
            test_f1 = f1_score(y_test, y_pred, zero_division=0)
            test_f2 = fbeta_score(y_test, y_pred, beta=2, zero_division=0)
            tp = int(((y_pred == 1) & (y_test == 1)).sum())
            fp = int(((y_pred == 1) & (y_test == 0)).sum())
            fn = int(((y_pred == 0) & (y_test == 1)).sum())
            tn = int(((y_pred == 0) & (y_test == 0)).sum())
            fdr = fp / max(tp + fp, 1)

            print(f"\n{'='*80}")
            print(
                f"[V20] FINAL TEST RESULTS ({target_col}) "
                f"@ threshold={optimal_threshold:.4f}"
            )
            print(f"{'='*80}")
            print(f"  TP={tp:,}  FP={fp:,}  FN={fn:,}  TN={tn:,}")
            print(f"  Precision:  {test_prec:.4f}  ({test_prec:.2%})")
            print(f"  Recall:     {test_rec:.4f}  ({test_rec:.2%})")
            print(f"  F1:         {test_f1:.4f}")
            print(f"  F2:         {test_f2:.4f}")
            print(f"  FDR:        {fdr:.4f}  ({fdr:.2%})")
            print(f"  ROC-AUC:    {roc_auc:.4f}")
            print(f"  PR-AUC:     {pr_auc:.4f}")
            print(f"{'='*80}")
            sys.stdout.flush()

            compat_path = os.path.join(
                output_dir,
                f"xgboost-classifier_model=XGBoost-version={model_time}.json",
            )
            base_models[0].save_model(compat_path)

            model_path = os.path.join(
                output_dir,
                f"xgboost-classifier-target={target_col.lower()}"
                f"_model=XGBoost-version={model_time}.json",
            )
            base_models[0].save_model(model_path)

            ensemble_meta = {
                "type": "balanced_bagging_ensemble_v20",
                "n_base_models": n_base,
                "neg_pos_ratio": neg_pos_ratio,
                "base_model_paths": base_model_paths,
                "best_hyperparameters": study.best_params,
                "ensemble_roc_auc": roc_auc,
                "ensemble_pr_auc": pr_auc,
                "optimal_threshold": optimal_threshold,
                "calibration_enabled": THRESHOLD_CONFIG["calibrate"],
                "min_recall_target": min_recall,
                "test_precision": test_prec,
                "test_recall": test_rec,
                "test_f1": test_f1,
                "test_f2": test_f2,
                "test_fdr": fdr,
                "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
                "engineered_features": new_avail,
            }
            with open(
                os.path.join(
                    output_dir,
                    f"xgboost-ensemble-metadata_target={target_col.lower()}"
                    f"_version={model_time}.json",
                ),
                "w",
            ) as f:
                json.dump(ensemble_meta, f, indent=2)

            if THRESHOLD_CONFIG["calibrate"]:
                calibrator = LogisticRegression(solver='lbfgs', max_iter=1000)
                calibrator.fit(ens_calib.reshape(-1, 1), y_calib)
                cal_path = os.path.join(
                    output_dir,
                    f"xgboost-calibrator_target={target_col.lower()}"
                    f"_version={model_time}.pkl",
                )
                joblib.dump(calibrator, cal_path)
                print(f"[V20] Calibrator saved: {cal_path}")

            with open(
                os.path.join(
                    output_dir,
                    f"xgboost-threshold_target={target_col.lower()}"
                    f"_version={model_time}.json",
                ),
                "w",
            ) as f:
                json.dump(
                    {
                        "optimal_threshold": optimal_threshold,
                        "min_recall_target": min_recall,
                        "calibration_precision": calib_precision,
                        "calibration_recall": calib_recall,
                        "test_precision": test_prec,
                        "test_recall": test_rec,
                    },
                    f,
                    indent=2,
                )
            sys.stdout.flush()

        else:
            # Non-ACCIDENT: standard approach
            n_trials = XGBOOST_CONFIG['trials_secondary']
            study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
            )
            study.optimize(
                XGBoostClassifierObjective(
                    X_train, y_train, available_features,
                    target_col=target_col, n_classes=len(classes),
                    n_folds=XGBOOST_CONFIG['cv_folds'],
                    use_gpu=USE_GPU, csv_path=csv_path, n_trials=n_trials,
                ),
                n_trials=n_trials,
                show_progress_bar=True,
            )
            best_params = {**study.best_params, 'tree_method': 'hist',
                          'random_state': RANDOM_STATE}
            if len(classes) <= 2:
                best_params['objective'] = 'binary:logistic'
                best_params['eval_metric'] = 'auc'
                best_params['scale_pos_weight'] = (
                    (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)
                )
            else:
                best_params['objective'] = 'multi:softprob'
                best_params['eval_metric'] = 'mlogloss'
                best_params['num_class'] = len(classes)
            if USE_GPU and torch.cuda.is_available():
                best_params['device'] = 'cuda:0'
            else:
                best_params['nthread'] = N_THREADS

            final_model = xgb.train(
                best_params,
                xgb.DMatrix(X_train, label=y_train),
                num_boost_round=2000,
                evals=[(xgb.DMatrix(X_test, label=y_test), 'eval')],
                early_stopping_rounds=100,
                verbose_eval=False,
            )
            y_pred_raw = final_model.predict(
                xgb.DMatrix(X_test, label=y_test),
                iteration_range=(0, final_model.best_iteration),
            )
            if len(classes) <= 2:
                y_pred = (y_pred_raw >= 0.5).astype(int)
                roc_auc = roc_auc_score(y_test, y_pred_raw)
                pr_auc = average_precision_score(y_test, y_pred_raw)
            else:
                y_pred = np.argmax(y_pred_raw, axis=1)

            model_path = os.path.join(
                output_dir,
                f"xgboost-classifier-target={target_col.lower()}"
                f"_model=XGBoost-version={model_time}.json",
            )
            final_model.save_model(model_path)

        # Common metrics
        test_metrics = {
            "target_col": target_col,
            "classes_original": classes,
            "accuracy": accuracy_score(y_test, y_pred),
            "f1_weighted": f1_score(y_test, y_pred, average="weighted", zero_division=0),
            "precision_weighted": precision_score(
                y_test, y_pred, average="weighted", zero_division=0
            ),
            "recall_weighted": recall_score(
                y_test, y_pred, average="weighted", zero_division=0
            ),
            "training_strategy": (
                "balanced_bagging_v20" if use_ensemble else "standard"
            ),
        }
        if roc_auc is not None:
            test_metrics["roc_auc"] = roc_auc
        if pr_auc is not None:
            test_metrics["pr_auc"] = pr_auc
        if use_ensemble:
            test_metrics.update({
                "optimal_threshold": optimal_threshold,
                "precision_at_threshold": test_prec,
                "recall_at_threshold": test_rec,
                "f1_at_threshold": test_f1,
                "f2_at_threshold": test_f2,
                "fdr_at_threshold": fdr,
            })

        print(
            f"[INFO] Test metrics ({target_col}): "
            f"acc={test_metrics['accuracy']:.4f}, "
            f"f1_w={test_metrics['f1_weighted']:.4f}"
        )
        if roc_auc is not None:
            print(f"[INFO] Test ROC-AUC ({target_col}): {roc_auc:.4f}")
        if pr_auc is not None:
            print(f"[INFO] Test PR-AUC ({target_col}): {pr_auc:.4f}")
        sys.stdout.flush()

        uncertainty_model = base_models[0] if use_ensemble else final_model
        uncertainty_summary = compute_leaf_density_uncertainty(
            uncertainty_model, X_train, y_train, X_test, y_test,
            y_pred_raw, classes, target_col, output_dir, model_time,
            config=XGBOOST_UNCERTAINTY_CONFIG,
        )
        if uncertainty_summary is not None:
            test_metrics["leaf_density_uncertainty"] = uncertainty_summary

        print(f"[SUCCESS] Model saved: {model_path}")
        sys.stdout.flush()

        studies_by_target[target_col] = study
        models_by_target[target_col] = model_path
        metrics_by_target[target_col] = test_metrics
        params_by_target[target_col] = study.best_params

    primary_target = "ACCIDENT" if "ACCIDENT" in studies_by_target else (
        next(iter(studies_by_target)) if studies_by_target else None
    )
    primary_study = studies_by_target.get(primary_target) if primary_target else None

    metadata = {
        "model_name": "XGBoost_FeatureEngineered_BalancedBagging",
        "model_time": model_time,
        "target_cols": target_cols,
        "primary_target": primary_target,
        "best_cv_score": primary_study.best_value if primary_study else None,
        "test_metrics": metrics_by_target.get(primary_target) if primary_target else {},
        "best_hyperparameters": primary_study.best_params if primary_study else {},
        "feature_cols": available_features,
        "base_feature_cols": [f for f in available_features if f not in set(new_avail)],
        "engineered_feature_cols": new_avail,
        "models_by_target": models_by_target,
        "metrics_by_target": metrics_by_target,
        "ensemble_config": ENSEMBLE_CONFIG,
        "threshold_config": THRESHOLD_CONFIG,
        "feature_engineering_config": FEATURE_ENGINEERING_CONFIG,
        "ablation_settings": ABLATION_SETTINGS,
        "test_mode": TEST_MODE,
        "pipeline_version": PIPELINE_VERSION,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(
        os.path.join(
            output_dir,
            f"xgboost-metadata_model=XGBoost-version={model_time}.json",
        ),
        "w",
    ) as f:
        json.dump(metadata, f, indent=2)

    print(f"[SUCCESS] Target models saved: {len(models_by_target)}")
    print_step_timing(
        "STAGE 2: XGBoost Feature-Engineered Training Complete (V20)", xgb_total_start
    )
    sys.stdout.flush()

    return primary_study, models_by_target, available_features, metrics_by_target


# ============================================================================
# MAIN PIPELINE (V20)
# ============================================================================


def run_xgboost_only() -> bool:
    """Main execution pipeline - V20 Feature-Engineered XGBoost."""
    global PREVIOUS_MODEL_TIME, OUTPUT_DIR, EXPERIMENT_NAME, DATA_FILE, DATA_PATH

    import logging as _logging
    _logging.getLogger("pytorch_lightning").setLevel(_logging.ERROR)
    _logging.getLogger("lightning_fabric").setLevel(_logging.ERROR)
    warnings.filterwarnings("ignore")

    os.environ["LOCAL_RANK"] = "0"
    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    if not PREVIOUS_EXPERIMENT_DIR:
        print("\n" + "=" * 80)
        print("ERROR: PREVIOUS_EXPERIMENT_DIR not set!")
        print("=" * 80)
        sys.exit(1)
    if not os.path.exists(PREVIOUS_EXPERIMENT_DIR):
        raise FileNotFoundError(
            f"Previous experiment directory not found: {PREVIOUS_EXPERIMENT_DIR}"
        )

    print("\n" + "=" * 80)
    print("ABLATION STUDY V20 - XGBOOST FEATURE-ENGINEERED BALANCED BAGGING")
    print("=" * 80)
    print(f"GNN model from: {PREVIOUS_EXPERIMENT_DIR}")
    print("[V20] Feature engineering groups:")
    for k, on in FEATURE_ENGINEERING_CONFIG.items():
        print(f"    [{'ON ' if on else 'off'}] {k}")
    sys.stdout.flush()

    if PREVIOUS_MODEL_TIME is None:
        PREVIOUS_MODEL_TIME = detect_model_time(PREVIOUS_EXPERIMENT_DIR)
    print(f"GNN model timestamp: {PREVIOUS_MODEL_TIME}")

    OUTPUT_DIR = PREVIOUS_EXPERIMENT_DIR
    EXPERIMENT_NAME = os.path.basename(OUTPUT_DIR.rstrip("/"))
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"\nUsing existing experiment: {EXPERIMENT_NAME}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"New XGBoost model timestamp (V20): {model_time}")
    print(f"Test Mode: {TEST_MODE}")
    print(
        f"Ensemble: {ENSEMBLE_CONFIG['n_base_models']} base models, "
        f"ratio={ENSEMBLE_CONFIG['neg_pos_ratio']}:1"
    )
    print(f"Threshold: precision @ recall >= {THRESHOLD_CONFIG['min_recall']:.0%}")
    print(
        f"Calibration: "
        f"{'Platt scaling' if THRESHOLD_CONFIG['calibrate'] else 'disabled'}"
    )
    print("=" * 80)
    sys.stdout.flush()

    print_ablation_config()

    config = {
        "experiment_name": EXPERIMENT_NAME,
        "model_time": model_time,
        "previous_experiment_dir": PREVIOUS_EXPERIMENT_DIR,
        "previous_model_time": PREVIOUS_MODEL_TIME,
        "test_mode": TEST_MODE,
        "ablation_settings": ABLATION_SETTINGS,
        "xgboost_config": XGBOOST_CONFIG,
        "ensemble_config": ENSEMBLE_CONFIG,
        "threshold_config": THRESHOLD_CONFIG,
        "feature_engineering_config": FEATURE_ENGINEERING_CONFIG,
        "lag_steps": LAG_STEPS,
        "roll_windows": ROLL_WINDOWS,
        "train_dates": {"start": TRAIN_START_DATE, "end": TRAIN_END_DATE},
        "pipeline_version": PIPELINE_VERSION,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(OUTPUT_DIR, "experiment_config_xgb_v20.json"), "w") as f:
        json.dump(config, f, indent=2)

    total_start = time.time()
    stage_times: dict[str, float] = {}

    # ── STAGE 0: Data + GNN loading ──────────────────────────────────────
    stage0_start = time.time()
    print("\n" + "=" * 80)
    print("STAGE 0: DATA LOADING + LOADING GNN MODEL (V20)")
    print("=" * 80)
    sys.stdout.flush()

    df_full, selected_features = _v5_load_data()
    df_train = _v5_filter_data_by_date(df_full, TRAIN_START_DATE, TRAIN_END_DATE)
    print(
        f"[INFO] Training data: {len(df_train):,} rows "
        f"({TRAIN_START_DATE} to {TRAIN_END_DATE})"
    )

    gnn_model, gnn_metadata, gnn_info, scaler_temporal, scaler_static, scaler_targets = (
        _v5_load_pretrained_gnn_model(PREVIOUS_EXPERIMENT_DIR, PREVIOUS_MODEL_TIME)
    )

    joblib.dump(
        scaler_temporal,
        os.path.join(OUTPUT_DIR, f"scaler-temporal_model=GNN-version={model_time}.pkl"),
    )
    joblib.dump(
        scaler_static,
        os.path.join(OUTPUT_DIR, f"scaler-static_model=GNN-version={model_time}.pkl"),
    )
    joblib.dump(
        scaler_targets,
        os.path.join(OUTPUT_DIR, f"scaler-targets_model=GNN-version={model_time}.pkl"),
    )

    src_model = os.path.join(
        PREVIOUS_EXPERIMENT_DIR,
        f"best-model_model=GNN-version={PREVIOUS_MODEL_TIME}.pt",
    )
    dst_model = os.path.join(OUTPUT_DIR, f"best-model_model=GNN-version={model_time}.pt")
    if os.path.exists(src_model):
        shutil.copy2(src_model, dst_model)

    gnn_meta_copy = gnn_metadata.copy()
    gnn_meta_copy["original_experiment"] = PREVIOUS_EXPERIMENT_DIR
    gnn_meta_copy["original_model_time"] = PREVIOUS_MODEL_TIME
    gnn_meta_copy["copied_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(
        os.path.join(OUTPUT_DIR, f"best-model-metadata_model=GNN-version={model_time}.json"),
        "w",
    ) as f:
        json.dump(gnn_meta_copy, f, indent=2)

    stage_times["stage0_data_gnn_loading"] = time.time() - stage0_start
    print_step_timing("STAGE 0: Data + GNN Loading Complete (V20)", stage0_start)

    # ── STAGE 1: GNN inference ────────────────────────────────────────────
    stage1_start = time.time()
    print("\n" + "=" * 80)
    print("STAGE 1: GENERATING GNN PREDICTIONS ON TRAINING DATA (V20)")
    print("=" * 80)
    sys.stdout.flush()

    interval_minutes = 15
    for minutes in [5, 10, 15, 30, 45, 60]:
        if f"{minutes}min" in df_train.columns:
            interval_minutes = minutes
            break
    seq_len = SEQUENCE_LENGTH_BY_INTERVAL.get(interval_minutes, 8)

    df_train = _v5_generate_gnn_predictions(
        gnn_model, df_train,
        scaler_temporal, scaler_static, scaler_targets,
        gnn_metadata, sequence_length=seq_len,
    )

    stage_times["stage1_gnn_inference"] = time.time() - stage1_start
    print_step_timing("STAGE 1: GNN Inference Complete (V20)", stage1_start)

    # ── STAGE 2: XGBoost training (includes FE internally) ───────────────
    stage2_start = time.time()
    xgb_study, xgb_models, xgb_features, xgb_metrics = train_xgboost_balanced_bagging(
        df_train, OUTPUT_DIR, model_time,
    )
    stage_times["stage2_xgboost_training"] = time.time() - stage2_start
    print_step_timing("STAGE 2: XGBoost Training Complete (V20)", stage2_start)

    total_duration = time.time() - total_start

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE (V20 XGBOOST FEATURE-ENGINEERED BALANCED BAGGING)")
    print("=" * 80)
    print("\nTIMING BREAKDOWN:")
    print("-" * 60)
    print(
        f"  Stage 0 - Data + GNN Loading:  "
        f"{format_duration(stage_times['stage0_data_gnn_loading']):>15}"
    )
    print(
        f"  Stage 1 - GNN Inference:       "
        f"{format_duration(stage_times['stage1_gnn_inference']):>15}"
    )
    print(
        f"  Stage 2 - XGBoost Training:    "
        f"{format_duration(stage_times['stage2_xgboost_training']):>15}"
    )
    print("-" * 60)
    print(f"  TOTAL:                         {format_duration(total_duration):>15}")
    print("=" * 80)
    sys.stdout.flush()

    summary = {
        "experiment_name": EXPERIMENT_NAME,
        "gnn_experiment": PREVIOUS_EXPERIMENT_DIR,
        "gnn_model_time": PREVIOUS_MODEL_TIME,
        "xgb_model_time": model_time,
        "total_duration_minutes": total_duration / 60,
        "stage_times_seconds": stage_times,
        "stage_times_formatted": {k: format_duration(v) for k, v in stage_times.items()},
        "xgb_best_score": xgb_study.best_value if xgb_study is not None else None,
        "xgb_model_paths": xgb_models,
        "xgb_test_metrics": xgb_metrics,
        "feature_engineering_config": FEATURE_ENGINEERING_CONFIG,
        "ensemble_config": ENSEMBLE_CONFIG,
        "threshold_config": THRESHOLD_CONFIG,
        "ablation_settings": ABLATION_SETTINGS,
        "test_mode": TEST_MODE,
        "pipeline_version": PIPELINE_VERSION,
        "completed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(OUTPUT_DIR, "experiment_summary_xgb_v20.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {OUTPUT_DIR}")
    print("\n" + "=" * 80)
    print("SUCCESS!")
    print("=" * 80)
    sys.stdout.flush()
    return True


def main() -> bool:
    parser = argparse.ArgumentParser(
        description="Ablation Study V20 - XGBoost Feature-Engineered Balanced Bagging"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--time-resolution", choices=TIME_RESOLUTION_CHOICES)
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
        global DATA_FILE, DATA_PATH
        chosen = args.time_resolution
        DATA_FILE = re.sub(r"\d+min", chosen, DATA_FILE, count=1)
        DATA_PATH = os.path.join(_AP7_DATA, DATA_FILE)
        print(f"[CONFIG] Using DATA_FILE={DATA_FILE}")
        sys.stdout.flush()

    return run_xgboost_only()


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
