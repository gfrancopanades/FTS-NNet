#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V51 XGBoost-only: GNN-fed Two-Stage XGBoost Training (OOF) with WIDER trees
============================================================================
Identical pipeline to V50 (`ablation_study_v50_xgboost_only.py`) — V14 GNN
inference → V21 feature engineering → V34 two-stage XGBoost training — but
with widened tree-shape Optuna search spaces in BOTH Stage 1 and Stage 2.

Motivation
----------
With ~60 engineered features fed to Stage 1 and balanced-bagging subsets
of only ~23K rows per base model (≈ n_pos × (1 + neg_pos_ratio)), V34's
defaults left Optuna pressed against several boundaries when the V50
run was inspected (job 2541146). Round 2 of V51 retunes the search
space using those empirical hits as a guide:

  Boundary hits in V50 run 2541146 (drove the round-2 changes)
  -------------------------------------------------------------
  Stage 1 reg_lambda    chose 9.95   on a 0–10 range  (CEILING)
  Stage 1 learning_rate chose 0.156  on a 0.01–0.2 range (near ceiling)
  Stage 2 max_depth     chose 8      on a 2–8 range   (CEILING)
  Stage 2 gamma         chose 1.17   on a 0–5 range   (above V51-r1's
                                                       too-tight 0–1 cap)
  Stage 2 min_child_weight  chose 14 on a 1–20 range  (near ceiling)

Widening ranges (V51 round 2 vs V34/V50)
----------------------------------------
Stage 1  (`v34.BalancedSubsetObjective`):
  max_depth         3 → 12       becomes   6 → 12
  min_child_weight  1 → 50       becomes   5 → 30
  gamma             0.0 → 10.0   becomes   0.0 → 3.0
  learning_rate     0.01 → 0.2   becomes   0.01 → 0.3   (log=True)
  reg_lambda        0.0 → 10.0   becomes   0.0 → 30.0   (key change)
  subsample         0.6 → 1.0    becomes   0.7 → 1.0
  colsample_bytree  0.6 → 1.0    becomes   0.7 → 1.0

Stage 2  (`_s2_objective` closure inside `v34.stage7_two_stage`):
  max_depth         2 → 8        becomes   4 → 12   (key change)
  min_child_weight  1 → 20       becomes   3 → 25
  gamma             0.0 → 5.0    becomes   0.0 → 3.0
  reg_alpha         0.0 → 5.0    becomes   0.0 → 10.0
  reg_lambda        0.0 → 5.0    becomes   0.0 → 10.0
  subsample         0.6 → 1.0    becomes   0.7 → 1.0
  colsample_bytree  0.6 → 1.0    becomes   0.7 → 1.0

Untouched
---------
- Stage 1 `scale_pos_weight`: V50 picked 1.18 (floor 1.0) → balanced
  bagging is already neutralising class imbalance, no need to expand.
- Stage 1 `reg_alpha`: V50 picked 1.96 on 0–10 → comfortable interior.
- Stage 2 `learning_rate`: V50 picked 0.04 on 0.01–0.3 (log) → fine.

Why widen `reg_lambda` to 30 (Stage 1)
--------------------------------------
The CV folds in V50 showed strong temporal drift (fold 1 F2≈0.3,
folds 2–3 F2≈0.05–0.1). Optuna pinned `reg_lambda` at the ceiling,
i.e. the model is asking for substantially more L2 to push back
against this drift. Without widening, the search cannot reach the
true optimum.

Implementation note
-------------------
V34 hard-codes its hyperparameter search ranges inside the trial objectives,
including a closure inside `stage7_two_stage`. To avoid forking those
70-line objectives, V51 widens them at runtime by monkey-patching
`optuna.trial.Trial.suggest_int` / `suggest_float` for the duration of
each stage's Optuna run. The patch maps known parameter names to V51
ranges; all other suggestions pass through unchanged.

Author: Gerard Franco
Date:   May 2026
"""
import os  # noqa: E402  (portable paths)
from src.paths import (  # portable paths -- see src/paths.py
    PROJECT_ROOT_STR as _AP7_ROOT,
    EXPERIMENTS_ROOT_STR as _AP7_EXPERIMENTS,
    DATA_DIR_STR as _AP7_DATA,
    TABLES_DIR as _AP7_TABLES,
    FIGURES_DIR as _AP7_FIGS,
)


import os, sys, json, time, glob, warnings, gc, re, threading, csv
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path

# Non-interactive matplotlib backend — must be set before any pyplot import
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import duckdb
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
import optuna
from optuna.trial import Trial
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, fbeta_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, classification_report,
    precision_recall_curve,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from scipy.stats import ks_2samp
from matplotlib.cm import RdBu_r
from matplotlib.patches import Patch

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

import src.training.ablation_study_v5_xgboost_only as _v5_mod
from src.training.ablation_study_v5_xgboost_only import (
    detect_model_time as _v5_detect_model_time,
    load_pretrained_gnn_model as _v5_load_pretrained_gnn_model,
    generate_gnn_predictions_on_training_data as _v5_generate_gnn_predictions,
    get_time_resolution_minutes as _v5_get_time_resolution_minutes,
)

# ── V21 (feature-engineering pipeline used by compute_feature_engineering.py) ─
import src.training.ablation_study_v21_xgboost_only as _v21_mod
from src.training.ablation_study_v21_xgboost_only import (
    build_engineered_features,
    FEATURE_ENGINEERING_CONFIG,
    SEQUENCE_LENGTH_BY_INTERVAL,
)

_SCRIPT_DIR  = Path(__file__).resolve().parent   # src/training/
PROJECT_ROOT = _SCRIPT_DIR.parent.parent          # prod_environment/

# =============================================================================
# CONFIGURATION  (mirrors notebook cell 2)
# =============================================================================

VERSION          = "v34"
OUTPUT_DIR       = f"{_AP7_EXPERIMENTS}/v34_xgb_no_gnn"
PK_MIN           = 120
PK_MAX           = 220
TRAIN_START_DATE = "2024-04-01"
TRAIN_END_DATE   = "2025-06-01"
SIM_START_DATE   = "2025-06-01"
SIM_END_DATE     = "2025-10-01"
NEG_POS_RATIO    = 10
TIME_RESOLUTION  = "5min"
USE_REAL_DATA    = True
INCLUDE_WEATHER_1D = False
INCLUDE_WEATHER_3D = True
INCLUDE_TRAFFIC    = True
INCLUDE_GEOMETRY   = True
INCLUDE_MOBILITY   = True
INCLUDE_TEMPORAL   = True
INCLUDE_IMPUTATION = True
TEST_MODE      = False
MEMORIZE_MODE  = False
RANDOM_STATE   = 42
N_THREADS      = 8
USE_GPU        = False

ABLATION_SETTINGS = {
    'include_weather_1d':      INCLUDE_WEATHER_1D,
    'include_weather_3d':      INCLUDE_WEATHER_3D,
    'include_traffic':          INCLUDE_TRAFFIC,
    'include_geometry':         INCLUDE_GEOMETRY,
    'include_mobility':         INCLUDE_MOBILITY,
    'include_temporal':         INCLUDE_TEMPORAL,
    'include_imputation_flags': INCLUDE_IMPUTATION,
}

FEATURE_GROUPS = {
    'weather_1d':  ['1d_fcst_temperature_2m', '1d_fcst_precipitation', '1d_fcst_snowfall',
                    '1d_fcst_cloud_cover', '1d_fcst_wind_speed_10m', '1d_fcst_wind_gusts_10m',
                    '1d_fcst_rain_binary'],
    'weather_3d':  ['3d_fcst_temperature_2m', '3d_fcst_precipitation', '3d_fcst_snowfall',
                    '3d_fcst_cloud_cover', '3d_fcst_wind_speed_10m', '3d_fcst_wind_gusts_10m',
                    '3d_fcst_rain_binary'],
    'traffic':     ['mean_speed', 'intTot', 'intP', 'car'],
    'geometry':    ['ang_curv', 'ang_pend_pos', 'ang_pend_neg', 'segment'],
    'mobility':    ['mob_esp'],
    'temporal':    ['anyo', 'mes', 'dia', 'diaSem', 'hor',
                    '5min', '10min', '15min', '30min', '45min', '60min'],
    'imputation':  ['speed_imputation', 'intensity_imputation'],
    'static_core': ['pk', 'via', 'sen', 'car'],
}

XGBOOST_CONFIG = {
    'trials_accident': 80 if not TEST_MODE else 10,
    'cv_folds':        3,
    'early_stopping':  50 if not TEST_MODE else 10,
}

ENSEMBLE_CONFIG = {
    'n_base_models': 10 if not TEST_MODE else 3,
    'neg_pos_ratio': NEG_POS_RATIO,
    'optuna_trials': 80 if not TEST_MODE else 10,
}

DATA_FILE_MAP = {
    "5min":  "CrashGNNLSTM_v1_vel-extinrix_int_geo_mob_wthr_5min_fund-propag-ltd_from_20240404_to_20251001.csv",
    "10min": "CrashGNNLSTM_v1_vel-extinrix_int_geo_mob_wthr_10min_fund-propag-ltd_from_20240404_to_20251001.csv",
    "15min": "CrashGNNLSTM_v1_vel-extinrix_int_geo_mob_wthr_15min_fund-propag-ltd_from_20240404_to_20251001.csv",
    "30min": "CrashGNNLSTM_v1_vel-extinrix_int_geo_mob_wthr_30min_fund-propag-ltd_from_20240404_to_20251001.csv",
    "45min": "CrashGNNLSTM_v1_vel-extinrix_int_geo_mob_wthr_45min_fund-propag-ltd_from_20240404_to_20251001.csv",
    "60min": "CrashGNNLSTM_v1_vel-extinrix_int_geo_mob_wthr_1h_fund-propag-ltd_from_20240404_to_20251001.csv",
}

DATA_FILE = DATA_FILE_MAP[TIME_RESOLUTION]
DATA_PATH  = Path(_AP7_DATA) / DATA_FILE
FE_FILE    = (
    f"CrashGNNLSTM_feature_engineering_v1_{TIME_RESOLUTION}_real_data.csv"
    if USE_REAL_DATA else
    f"CrashGNNLSTM_feature_engineering_v1_{TIME_RESOLUTION}.csv"
)
FE_PATH = Path(_AP7_DATA) / FE_FILE

# =============================================================================
# VISUALIZATIONS DIRECTORY
# =============================================================================

VIZ_DIR = os.path.join(OUTPUT_DIR, "visualizations")


def _savefig(fig, name: str) -> None:
    os.makedirs(VIZ_DIR, exist_ok=True)
    job_id = (
        os.environ.get("SLURM_JOB_ID", "")
        or os.environ.get("XGB_RUN_ID", "")
        or os.environ.get("JOB_ID", "")
    )
    suffix = f"_job{job_id}" if job_id else ""
    path = os.path.join(VIZ_DIR, f"{name}{suffix}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] Saved: {path}")


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def format_duration(seconds):
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{seconds/60:.1f}min ({seconds:.0f}s)"
    h, m = seconds / 3600, (seconds % 3600) / 60
    return f"{h:.1f}h ({int(h)}h {int(m)}m)"


def get_selected_features():
    group_key_map = {
        'include_weather_1d':      'weather_1d',
        'include_weather_3d':      'weather_3d',
        'include_traffic':          'traffic',
        'include_geometry':         'geometry',
        'include_mobility':         'mobility',
        'include_temporal':         'temporal',
        'include_imputation_flags': 'imputation',
    }
    selected, seen = [], set()
    for key, grp in group_key_map.items():
        if ABLATION_SETTINGS.get(key, False):
            selected.extend(FEATURE_GROUPS.get(grp, []))
    return [f for f in selected if not (f in seen or seen.add(f))]


def _get_time_resolution_minutes():
    m = re.search(r'(\d+)min', DATA_FILE)
    return int(m.group(1)) if m else 5


def load_data(pk_min=None, pk_max=None, date_min=None, date_max=None):
    print("\n" + "=" * 80)
    print("LOADING DATA")
    print("=" * 80)
    sys.stdout.flush()
    t0 = time.time()
    print(f"[DATA] Reading: {DATA_PATH}")

    with open(str(DATA_PATH), 'r', encoding='latin-1') as fh:
        _header = fh.readline().strip().split(';')
    year_col = 'Any' if 'Any' in _header else 'anyo'

    _filters = []
    if pk_min is not None:
        _filters.append(f"pk >= {pk_min} AND pk <= {pk_max}")
    if date_min is not None:
        _filters.append(
            f"make_date(CAST(\"{year_col}\" AS INTEGER), CAST(mes AS INTEGER), CAST(dia AS INTEGER))"
            f" >= DATE '{date_min.date()}'"
        )
        _filters.append(
            f"make_date(CAST(\"{year_col}\" AS INTEGER), CAST(mes AS INTEGER), CAST(dia AS INTEGER))"
            f" < DATE '{date_max.date()}'"
        )

    if _filters:
        _where = "WHERE " + " AND ".join(_filters)
        _query = f"SELECT * FROM read_csv('{DATA_PATH}', delim=';', header=true, ignore_errors=true) {_where}"
        df = duckdb.query(_query).df()
    else:
        df = pd.read_csv(DATA_PATH, sep=";", decimal=".", encoding="latin-1")
    print(f"[SUCCESS] Loaded: {df.shape}")

    if 'Any' in df.columns:
        df.rename(columns={'Any': 'anyo'}, inplace=True)
    if 'via' in df.columns and df['via'].dtype == 'object':
        df['via'] = df['via'].map({'AP-7': 0}).fillna(0).astype(int)
    if 'sen' in df.columns and df['sen'].dtype == 'object':
        df['sen'] = df['sen'].map({'dec': 0, 'cre': 1}).fillna(1).astype(int)

    if 'min' not in df.columns:
        for m in (5, 10, 15, 30, 45, 60):
            col = f"{m}min"
            if col in df.columns:
                df['min'] = df[col]; break
        if 'min' not in df.columns:
            df['min'] = pd.to_datetime(df['dat']).dt.minute if 'dat' in df.columns else 0

    res_min  = _get_time_resolution_minutes()
    res_feat = f"{res_min}min"
    if res_feat not in df.columns:
        df[res_feat] = ((df['min'] // res_min) * res_min).astype(int)

    df['dat'] = pd.to_datetime(
        df['anyo'].astype(str) + '-' + df['mes'].astype(str).str.zfill(2) + '-' +
        df['dia'].astype(str).str.zfill(2) + ' ' + df['hor'].astype(str).str.zfill(2) + ':' +
        df['min'].astype(str).str.zfill(2) + ':00'
    )
    df.sort_values(['via', 'sen', 'pk', 'anyo', 'mes', 'dia', 'hor', 'min'], inplace=True)

    if 'F_TEMPS_AFECTACIO' in df.columns and 'F_LONG_AFECTACIO' in df.columns:
        df['F_RETENCIO'] = df['F_TEMPS_AFECTACIO'] * df['F_LONG_AFECTACIO']
    else:
        df['F_RETENCIO'] = 0

    selected  = get_selected_features()
    available = [f for f in selected if f in df.columns]
    missing   = [f for f in selected if f not in df.columns]
    if missing:
        print(f"[WARN] Missing features: {missing}")
    print(f"[INFO] Using {len(available)} features")
    print(f"[COMPLETE] Loaded in {time.time()-t0:.1f}s")
    sys.stdout.flush()
    return df, available


def filter_data_by_date(df, start_date, end_date):
    return df[(df['dat'] >= start_date) & (df['dat'] < end_date)].copy()


def create_balanced_subset(X, y, neg_pos_ratio, random_state):
    pos_mask, neg_mask = y == 1, y == 0
    n_pos        = pos_mask.sum()
    n_neg_target = int(n_pos * neg_pos_ratio)
    rng          = np.random.RandomState(random_state)
    neg_indices  = y[neg_mask].index.to_numpy()
    neg_sample   = rng.choice(neg_indices, size=min(n_neg_target, len(neg_indices)), replace=False)
    pos_indices  = y[pos_mask].index.to_numpy()
    selected     = np.sort(np.concatenate([pos_indices, neg_sample]))
    return X.loc[selected], y.loc[selected]


def _metrics(cm):
    tp, fp, fn = cm[1, 1], cm[0, 1], cm[1, 0]
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return tp, fp, fn, prec, rec, f1


# =============================================================================
# BALANCED SUBSET OBJECTIVE (Optuna)
# =============================================================================

class BalancedSubsetObjective:
    def __init__(self, X_train, y_train, feature_cols, target_col,
                 neg_pos_ratio, n_folds=3, use_gpu=False, csv_path=None, n_trials=80,
                 true_ratio=None):
        self.X_train       = X_train
        self.y_train       = y_train
        self.feature_cols  = feature_cols
        self.target_col    = target_col
        self.neg_pos_ratio = neg_pos_ratio
        self.n_folds       = n_folds
        self.use_gpu       = use_gpu
        self.n_trials      = n_trials
        self.csv_path      = csv_path
        self.lock          = threading.Lock()
        self.true_ratio    = (true_ratio if true_ratio is not None
                              else int((y_train == 0).sum()) / max(int((y_train == 1).sum()), 1))

    def __call__(self, trial):
        t0 = time.time()
        try:
            params = {
                'tree_method':       'hist',
                'objective':         'binary:logistic',
                'eval_metric':       'aucpr',
                'max_depth':         trial.suggest_int('max_depth', 3, 12),
                'learning_rate':     trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'min_child_weight':  trial.suggest_int('min_child_weight', 1, 50),
                'subsample':         trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree':  trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'gamma':             trial.suggest_float('gamma', 0.0, 10.0),
                'reg_alpha':         trial.suggest_float('reg_alpha', 0.0, 10.0),
                'reg_lambda':        trial.suggest_float('reg_lambda', 0.0, 10.0),
                'scale_pos_weight':  trial.suggest_float('scale_pos_weight', 1.0,
                                         self.true_ratio / float(self.neg_pos_ratio)),
                'random_state':      RANDOM_STATE,
                'nthread':           N_THREADS,
            }
            print(f"[TRIAL {trial.number+1}/{self.n_trials}] "
                  f"depth={params['max_depth']} lr={params['learning_rate']:.4f} "
                  f"spw={params['scale_pos_weight']:.1f}")
            sys.stdout.flush()

            tscv       = TimeSeriesSplit(n_splits=self.n_folds)
            cv_scores  = []
            best_iters = []

            for fold_i, (tr_idx, va_idx) in enumerate(tscv.split(self.X_train), 1):
                X_tr_full = self.X_train.iloc[tr_idx]
                y_tr_full = self.y_train.iloc[tr_idx]
                X_va      = self.X_train.iloc[va_idx]
                y_va      = self.y_train.iloc[va_idx]
                if y_va.sum() == 0:
                    continue
                X_tr_bal, y_tr_bal = create_balanced_subset(
                    X_tr_full, y_tr_full,
                    neg_pos_ratio=self.neg_pos_ratio,
                    random_state=RANDOM_STATE + fold_i,
                )
                dtrain = xgb.DMatrix(X_tr_bal, label=y_tr_bal)
                dval_f = xgb.DMatrix(X_va, label=y_va)
                model  = xgb.train(
                    params, dtrain, num_boost_round=1000,
                    evals=[(dval_f, 'eval')],
                    early_stopping_rounds=XGBOOST_CONFIG['early_stopping'],
                    verbose_eval=False,
                )
                preds = model.predict(dval_f, iteration_range=(0, model.best_iteration))
                score = fbeta_score(y_va, (preds >= 0.5).astype(int), beta=2.0, zero_division=0)
                cv_scores.append(score)
                best_iters.append(model.best_iteration)
                print(f"  Fold {fold_i}/{self.n_folds} F2={score:.4f} iter={model.best_iteration}")
                sys.stdout.flush()

            if not cv_scores:
                return 0.0
            mean_score = float(np.mean(cv_scores))
            print(f"  => mean F2={mean_score:.4f}  ({format_duration(time.time()-t0)})")
            sys.stdout.flush()

            if self.csv_path:
                row = {
                    'trial_number': trial.number + 1, 'target_col': self.target_col,
                    **{k: params[k] for k in ['max_depth', 'learning_rate', 'min_child_weight',
                       'subsample', 'colsample_bytree', 'gamma', 'reg_alpha', 'reg_lambda',
                       'scale_pos_weight']},
                    'cv_f2_mean': mean_score, 'cv_f2_std': float(np.std(cv_scores)),
                    'best_iterations_mean': float(np.mean(best_iters)),
                    'training_time_seconds': time.time() - t0,
                }
                with self.lock:
                    exists = os.path.exists(self.csv_path) and os.path.getsize(self.csv_path) > 0
                    with open(self.csv_path, 'a', newline='') as fh:
                        w = csv.DictWriter(fh, fieldnames=row.keys())
                        if not exists:
                            w.writeheader()
                        w.writerow(row)
            return mean_score
        except Exception as e:
            print(f"[ERROR] Trial {trial.number+1}: {e}")
            return 0.0


# =============================================================================
# STAGE 0: DATA LOADING
# =============================================================================

def stage0_load_data():
    print('=' * 80)
    print('STAGE 0A: DATA LOADING')
    print('=' * 80)

    _date_min = min(pd.Timestamp(TRAIN_START_DATE), pd.Timestamp(SIM_START_DATE))
    _date_max = max(pd.Timestamp(TRAIN_END_DATE),   pd.Timestamp(SIM_END_DATE))

    t0 = time.time()
    df_full, selected_features = load_data(
        pk_min=PK_MIN, pk_max=PK_MAX,
        date_min=_date_min, date_max=_date_max,
    )
    print(f'Loaded in {time.time()-t0:.1f}s')
    print(f'[V34] PK=[{PK_MIN},{PK_MAX}] Date=[{_date_min.date()},{_date_max.date()}): {len(df_full):,} rows')

    print('=' * 80)
    print('STAGE 0A-FE: FEATURE ENGINEERING MERGE')
    print('=' * 80)

    with open(str(FE_PATH), 'r', encoding='latin-1') as fh:
        _fe_header = fh.readline().strip().split(';')
    main_cols   = set(df_full.columns)
    fe_only_cols = [c for c in _fe_header if c not in main_cols]
    print(f'New FE columns ({len(fe_only_cols)}): {fe_only_cols}')

    _fe_cols     = ['dat', 'pk', 'sen', 'ang_pend_pos', 'ang_pend_neg'] + fe_only_cols
    _fe_cols_sql = ', '.join(f'"{c}"' for c in _fe_cols if c in _fe_header)
    _fe_query    = f"""
        SELECT {_fe_cols_sql}
        FROM read_csv('{FE_PATH}', delim=';', header=true, ignore_errors=true)
        WHERE pk >= {PK_MIN} AND pk <= {PK_MAX}
        AND CAST(dat AS DATE) >= DATE '{_date_min.date()}'
        AND CAST(dat AS DATE) <  DATE '{_date_max.date()}'
    """
    df_fe = duckdb.query(_fe_query).df()
    print(f'Feature engineering loaded: {df_fe.shape[0]:,} rows x {df_fe.shape[1]} columns')

    df_fe['dat'] = pd.to_datetime(df_fe['dat'])
    ref_pk   = df_fe['pk'].iloc[0]
    geo_fe   = df_fe[df_fe['pk'] == ref_pk].groupby('sen')[['ang_pend_pos', 'ang_pend_neg']].first()
    geo_main = df_full.groupby(['pk', 'sen'])[['ang_pend_pos', 'ang_pend_neg']].first().loc[ref_pk]

    sen_map = {}
    for fe_sen, fe_row in geo_fe.iterrows():
        for main_sen, main_row in geo_main.iterrows():
            if abs(fe_row['ang_pend_pos'] - main_row['ang_pend_pos']) < 0.01:
                sen_map[fe_sen] = main_sen; break
    print(f'Sen mapping (FE -> main): {sen_map}')
    df_fe['sen'] = df_fe['sen'].map(sen_map)

    df_full = df_full.merge(df_fe[['dat', 'pk', 'sen'] + fe_only_cols],
                            on=['dat', 'pk', 'sen'], how='left')
    del df_fe
    print(f'Merged shape: {df_full.shape[0]:,} rows x {df_full.shape[1]} columns')

    print(f'Shape: {df_full.shape}')
    print(f'Date range: {df_full["dat"].min()} to {df_full["dat"].max()}')
    print(f'PKs present: {sorted(df_full["pk"].unique())}')
    print(f'ACCIDENT distribution:\n{df_full["ACCIDENT"].value_counts()}')

    return df_full, fe_only_cols


# =============================================================================
# STAGE 1: FEATURE PREP + TRAIN/TEST SPLIT
# =============================================================================

def stage1_feature_prep(df_full, fe_only_cols):
    print('=' * 80)
    print('STAGE 2A: FEATURE PREPARATION + TRAIN/TEST SPLIT')
    print('=' * 80)

    df_train = filter_data_by_date(df_full, TRAIN_START_DATE, TRAIN_END_DATE)
    print(f'Training data ({TRAIN_START_DATE} -> {TRAIN_END_DATE}): {len(df_train):,} rows')
    n_pos  = int(df_train['ACCIDENT'].fillna(0).astype(int).sum())
    n_total = len(df_train)
    print(f'ACCIDENT: {n_pos:,} pos / {n_total - n_pos:,} neg  ({n_pos/n_total*100:.4f}%)')

    model_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    print(f'XGBoost model timestamp: {model_time}')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    available_features = ['pk', 'anyo', 'mes', 'dia', 'diaSem', 'hor', TIME_RESOLUTION,
                          'mean_speed', 'intTot', 'intP', 'car']
    available_features += [f for f in fe_only_cols if f in df_train.columns]
    available_features  = [f for f in dict.fromkeys(available_features) if f in df_train.columns]

    if 'car' not in df_train.columns and 'intTot' in df_train.columns and 'intP' in df_train.columns:
        df_train['car'] = df_train['intTot'] - df_train['intP']
        if 'car' not in available_features:
            available_features.append('car')

    print(f'Available features ({len(available_features)}): {available_features}')

    X = df_train[available_features].apply(pd.to_numeric, errors='coerce').fillna(0)

    # ── TRAIN / TEST SPLIT: chronological, by date mask ───────────────────
    # Was `X.iloc[:int(len(X)*0.8)]` — a POSITIONAL cut on a LOCATION-major
    # frame, i.e. a spatial holdout (one carriageway of the upper PKs). Three
    # things were wrong with it, all measured on the real frame:
    #   * its base rate was 0.0565% against June's 0.1876% — a 3.3x mismatch,
    #     so its PR-AUC was not on the same scale as the metric it was meant to
    #     predict, and the threshold swept on it (see `sweep_stage1_threshold`,
    #     whose pick is frozen into the simulation) was calibrated for the
    #     wrong prevalence;
    #   * it held only 397 positives vs 1,331 chronologically — twice the noise;
    #   * its meaning depended on an upstream sort.
    # A date mask selects by value, so it is immune to row order and states the
    # intent in the code. The boundary also bounds the FE baselines (stage 0E),
    # keeping `pk_crash_rate` out of the rows it is scored on.
    _boundary = globals().get("TRAIN_TEST_BOUNDARY") or _v21_mod.resolve_train_test_boundary(
        GNN_TRAIN_WINDOWS, override=os.environ.get("BENCH_TRAIN_TEST_BOUNDARY"))
    if os.environ.get("BENCH_POSITIONAL_SPLIT", "0") == "1":
        # Escape hatch for reproducing pre-fix runs; not for new results.
        is_train = np.zeros(len(X), dtype=bool)
        is_train[:int(len(X) * 0.8)] = True
        print(f'[SPLIT] LEGACY POSITIONAL 80/20 (BENCH_POSITIONAL_SPLIT=1) — '
              f'spatial holdout, base-rate-mismatched; for reproduction only')
    else:
        is_train = (pd.to_datetime(df_train['dat']) < _boundary).to_numpy()
        print(f'[SPLIT] Chronological: train dat < {_boundary} | test dat >= {_boundary}')

    X_train_xgb = X[is_train]
    X_test_xgb  = X[~is_train]
    if len(X_test_xgb) == 0 or len(X_train_xgb) == 0:
        raise ValueError(
            f'Train/test split at {_boundary} left an empty side '
            f'(train={len(X_train_xgb):,}, test={len(X_test_xgb):,}). '
            f'Check BENCH_TRAIN_TEST_BOUNDARY against the training windows.')

    scaler_features = MinMaxScaler()
    scaler_features.fit(X_train_xgb)
    joblib.dump(scaler_features, os.path.join(
        OUTPUT_DIR, f'scaler-features_model=XGBoost-version={model_time}.pkl'))

    print(f'Train: {X_train_xgb.shape}  |  Test: {X_test_xgb.shape}')

    y_raw       = df_train['ACCIDENT'].fillna(0)
    classes     = sorted(y_raw.astype(int).unique().tolist())
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    y_all       = y_raw.astype(int).map(class_to_idx).astype(int)
    # Same mask as X — y must never be split by a different rule than X.
    y_train_acc = y_all[is_train]
    y_test_acc  = y_all[~is_train]
    print(f'[SPLIT] positives: train={int(y_train_acc.sum()):,} '
          f'test={int(y_test_acc.sum()):,} '
          f'(test rate {y_test_acc.mean()*100:.4f}%)')

    print('X_train descriptive stats:')
    print(X_train_xgb.describe().T[['count', 'mean', 'std', 'min', 'max']])
    nan_counts = X_train_xgb.isnull().sum()
    inf_count  = np.isinf(X_train_xgb.select_dtypes(include="float").values).sum()
    print(f'NaN total: {nan_counts.sum()}, Inf total: {inf_count}')

    return df_train, X_train_xgb, X_test_xgb, y_train_acc, y_test_acc, available_features, model_time


# =============================================================================
# STAGE 2: OPTUNA + ENSEMBLE TRAINING
# =============================================================================

def stage2_train_ensemble(X_train_xgb, X_test_xgb, y_train_acc, y_test_acc,
                          available_features, model_time):
    print('=' * 80)
    print('STAGE 2B: ACCIDENT — Optuna hyperparameter search')
    print('=' * 80)

    target_col    = 'ACCIDENT'
    n_base        = ENSEMBLE_CONFIG['n_base_models']
    neg_pos_ratio = ENSEMBLE_CONFIG['neg_pos_ratio']
    n_trials      = ENSEMBLE_CONFIG['optuna_trials']

    csv_path = os.path.join(OUTPUT_DIR,
        f'training-evolution_model=XGBoost-target={target_col}-version={model_time}.csv')

    if MEMORIZE_MODE:
        class _FakeStudy:
            best_params = {'max_depth': 12, 'learning_rate': 0.3, 'min_child_weight': 1,
                           'subsample': 1.0, 'colsample_bytree': 1.0, 'gamma': 0.0,
                           'reg_alpha': 0.0, 'reg_lambda': 0.0, 'scale_pos_weight': 1.0}
            best_value = 1.0
        study = _FakeStudy()
        n_base = 1
    else:
        print(f'Ensemble: {n_base} base models, neg_pos_ratio={neg_pos_ratio}, trials={n_trials}')
        objective = BalancedSubsetObjective(
            X_train_xgb, y_train_acc, available_features,
            target_col=target_col, neg_pos_ratio=neg_pos_ratio,
            n_folds=XGBOOST_CONFIG['cv_folds'],
            use_gpu=USE_GPU, csv_path=csv_path, n_trials=n_trials,
        )
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
        )
        t0 = time.time()
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        print(f'\nOptuna done in {format_duration(time.time()-t0)}')
        print(f'Best F2: {study.best_value:.4f}')
        for k, v in study.best_params.items():
            print(f'  {k}: {v}')

    print('=' * 80)
    print('STAGE 2C: Train ensemble base models')
    print('=' * 80)

    best_params = {**study.best_params, 'tree_method': 'hist',
                   'objective': 'binary:logistic', 'eval_metric': 'aucpr',
                   'random_state': RANDOM_STATE, 'nthread': N_THREADS}

    dval = xgb.DMatrix(X_test_xgb, label=y_test_acc)

    if MEMORIZE_MODE:
        X_all_xgb = pd.concat([X_train_xgb, X_test_xgb])
        y_all_acc  = pd.concat([y_train_acc,  y_test_acc])
        _early_stop, _num_rounds = None, 500
    else:
        X_all_xgb  = X_train_xgb
        y_all_acc  = y_train_acc
        _early_stop, _num_rounds = 100, 2000

    base_models, base_model_paths = [], []
    for i in range(n_base):
        t0 = time.time()
        X_sub, y_sub = create_balanced_subset(
            X_all_xgb, y_all_acc,
            neg_pos_ratio=neg_pos_ratio, random_state=RANDOM_STATE + i * 1000)
        dtrain_sub = xgb.DMatrix(X_sub, label=y_sub)

        if MEMORIZE_MODE:
            model_i = xgb.train(best_params, dtrain_sub, num_boost_round=_num_rounds,
                                verbose_eval=False)
            model_i.best_iteration = _num_rounds
        else:
            model_i = xgb.train(best_params, dtrain_sub, num_boost_round=_num_rounds,
                                evals=[(dval, 'eval')],
                                early_stopping_rounds=_early_stop, verbose_eval=False)

        base_models.append(model_i)
        path_i = os.path.join(OUTPUT_DIR,
            f'xgboost-ensemble-base{i}_target=accident_model=XGBoost-version={model_time}.json')
        model_i.save_model(path_i)
        base_model_paths.append(path_i)

        best_iter_i = getattr(model_i, 'best_iteration', _num_rounds)
        y_pred_i    = model_i.predict(dval, iteration_range=(0, best_iter_i))
        aucpr_i     = average_precision_score(y_test_acc, y_pred_i)
        print(f'  Base {i+1}/{n_base}: train={len(X_sub):,}, '
              f'best_iter={best_iter_i}, AUCPR={aucpr_i:.4f}, time={time.time()-t0:.1f}s')

    ensemble_preds = np.zeros(len(X_test_xgb))
    for m in base_models:
        best_iter = getattr(m, 'best_iteration', _num_rounds)
        ensemble_preds += m.predict(dval, iteration_range=(0, best_iter))
    ensemble_preds /= n_base

    y_pred_binary = (ensemble_preds >= 0.5).astype(int)
    roc_auc = roc_auc_score(y_test_acc, ensemble_preds)
    pr_auc  = average_precision_score(y_test_acc, ensemble_preds)
    print(f'\nEnsemble ROC-AUC: {roc_auc:.4f}')
    print(f'Ensemble PR-AUC:  {pr_auc:.4f}')

    # Save compatibility models + ensemble metadata
    base_models[0].save_model(os.path.join(OUTPUT_DIR,
        f'xgboost-classifier_model=XGBoost-version={model_time}.json'))
    model_path_accident = os.path.join(OUTPUT_DIR,
        f'xgboost-classifier-target=accident_model=XGBoost-version={model_time}.json')
    base_models[0].save_model(model_path_accident)

    ensemble_meta = {
        'type': 'balanced_bagging_ensemble', 'n_base_models': n_base,
        'neg_pos_ratio': neg_pos_ratio, 'base_model_paths': base_model_paths,
        'best_hyperparameters': study.best_params,
        'ensemble_roc_auc': roc_auc, 'ensemble_pr_auc': pr_auc,
        'memorize_mode': MEMORIZE_MODE,
    }
    with open(os.path.join(OUTPUT_DIR,
              f'xgboost-ensemble-metadata_target=accident_version={model_time}.json'), 'w') as f:
        json.dump(ensemble_meta, f, indent=2)

    # Stage 2D/E: metrics + metadata
    metrics_by_target = {'ACCIDENT': {
        'target_col': 'ACCIDENT', 'accuracy': accuracy_score(y_test_acc, y_pred_binary),
        'f1_weighted': f1_score(y_test_acc, y_pred_binary, average='weighted', zero_division=0),
        'roc_auc': roc_auc, 'pr_auc': pr_auc, 'training_strategy': 'balanced_bagging',
    }}
    print('ACCIDENT ENSEMBLE RESULTS')
    print(f'ROC-AUC: {roc_auc:.4f}')
    print(f'PR-AUC:  {pr_auc:.4f}')
    print(confusion_matrix(y_test_acc, y_pred_binary))
    print(classification_report(y_test_acc, y_pred_binary,
          target_names=['No Accident', 'Accident'], zero_division=0))

    xgb_metadata = {
        'model_name': 'XGBoost_Classifier', 'model_time': model_time,
        'target_cols': ['ACCIDENT'], 'primary_target': 'ACCIDENT',
        'best_cv_score': study.best_value, 'test_metrics': metrics_by_target.get('ACCIDENT', {}),
        'best_hyperparameters': study.best_params, 'feature_cols': available_features,
        'models_by_target': {'ACCIDENT': os.path.basename(model_path_accident)},
        'metrics_by_target': metrics_by_target,
        'best_hyperparameters_by_target': {'ACCIDENT': study.best_params},
        'ablation_settings': ABLATION_SETTINGS, 'test_mode': TEST_MODE,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }
    xgb_meta_path = os.path.join(OUTPUT_DIR,
        f'xgboost-metadata_model=XGBoost-version={model_time}.json')
    with open(xgb_meta_path, 'w') as f:
        json.dump(xgb_metadata, f, indent=2)
    print(f'[SUCCESS] XGBoost metadata saved: {os.path.basename(xgb_meta_path)}')

    # Extract best_params as a plain dict so the caller can del study immediately.
    s1_best_params = dict(study.best_params)
    return base_models, ensemble_preds, y_pred_binary, roc_auc, pr_auc, s1_best_params


# =============================================================================
# STAGE 3: ROLLING SIMULATION (Stage 1 only)
# =============================================================================

def stage3_simulation(df_full, base_models, available_features, model_time):
    print('=' * 80)
    print('STAGE 3: ROLLING SIMULATION (NO-GNN, REAL TRAFFIC VALUES)')
    print('=' * 80)

    def run_no_gnn_simulation(df_full, base_models, available_features,
                               sim_start, sim_end, pk_min, pk_max, model_time, output_dir):
        start = pd.Timestamp(sim_start)
        end   = pd.Timestamp(sim_end)
        df_sim = df_full[
            (df_full['pk'] >= pk_min) & (df_full['pk'] <= pk_max) &
            (df_full['dat'] >= start) & (df_full['dat'] < end)
        ].copy()
        if len(df_sim) == 0:
            print(f'[WARN] No data for sim period {sim_start} -> {sim_end}')
            return None
        print(f'Sim period:  {sim_start} -> {sim_end}')
        print(f'Rows:        {len(df_sim):,}')
        print(f'PKs:         {sorted(df_sim["pk"].unique())}')

        for col in available_features:
            if col not in df_sim.columns:
                df_sim[col] = 0
        X_sim  = df_sim[available_features].apply(pd.to_numeric, errors='coerce').fillna(0)
        d_sim  = xgb.DMatrix(X_sim)
        all_preds = np.zeros(len(X_sim))
        for m in base_models:
            best_iter = getattr(m, 'best_iteration', None)
            all_preds += m.predict(d_sim, iteration_range=(0, best_iter)) if best_iter else m.predict(d_sim)
        all_preds /= len(base_models)

        results = df_sim[['via', 'sen', 'pk', 'anyo', 'mes', 'dia', 'diaSem',
                           'hor', 'min', 'dat']].copy()
        results['accident_probability'] = all_preds
        results['accident_pred_binary'] = (all_preds >= 0.5).astype(int)
        if 'ACCIDENT' in df_sim.columns:
            results['ACCIDENT_real'] = df_sim['ACCIDENT'].fillna(0).astype(int).values
        if 'mean_speed' in df_sim.columns:
            results['mean_speed_real'] = df_sim['mean_speed'].values
        if 'intTot' in df_sim.columns:
            results['intTot_real'] = df_sim['intTot'].values
        if 'intP' in df_sim.columns:
            results['intP_real'] = df_sim['intP'].values
        results['model_time'] = model_time
        results['simulation_mode'] = 'no_gnn_real_traffic'
        return results

    t0 = time.time()
    sim_results = run_no_gnn_simulation(
        df_full=df_full, base_models=base_models, available_features=available_features,
        sim_start=SIM_START_DATE, sim_end=SIM_END_DATE,
        pk_min=PK_MIN, pk_max=PK_MAX, model_time=model_time, output_dir=OUTPUT_DIR,
    )
    print(f'Simulation completed in {format_duration(time.time()-t0)}')

    if sim_results is not None:
        sim_results['date'] = pd.to_datetime(sim_results['dat']).dt.date
        daily = sim_results.groupby('date').agg(
            rows=('accident_probability', 'count'),
            mean_prob=('accident_probability', 'mean'),
            max_prob=('accident_probability', 'max'),
            pred_accidents=('accident_pred_binary', 'sum'),
        )
        if 'ACCIDENT_real' in sim_results.columns:
            daily['real_accidents'] = sim_results.groupby('date')['ACCIDENT_real'].sum()
        print(f'\nSimulation period: {SIM_START_DATE} -> {SIM_END_DATE}')
        print(f'Total predictions: {len(sim_results):,}')
        print('\nDaily summary:')
        print(daily.to_string())
        print('\nProbability distribution:')
        print(sim_results['accident_probability'].describe())

        sim_dir  = os.path.join(OUTPUT_DIR, f'simulation_no_gnn_{model_time}')
        os.makedirs(sim_dir, exist_ok=True)
        sim_path = os.path.join(sim_dir, f'simulation_results_{model_time}.csv')
        sim_results.to_csv(sim_path, sep=';', index=False)
        print(f'\n[SUCCESS] Simulation results saved: {sim_path}')

    return sim_results


# =============================================================================
# STAGE 4: SIMULATION EVALUATION (Stage 1)
# =============================================================================

def stage4_evaluate_simulation(sim_results):
    print('=' * 80)
    print('STAGE 4: SIMULATION EVALUATION')
    print('=' * 80)

    if sim_results is None or len(sim_results) == 0:
        print('[WARN] No simulation results to evaluate.')
        return

    df = sim_results.dropna(subset=["ACCIDENT_real", "accident_probability"]).copy()
    df["ACCIDENT_real"]        = (df["ACCIDENT_real"].astype(float) > 0).astype(int)
    df["accident_probability"] = df["accident_probability"].astype(float).clip(0, 1)
    df["accident_pred_binary"] = (df["accident_probability"] >= 0.5).astype(int)

    print(f"Rows used for evaluation: {len(df):,}")
    print(f"Accident prevalence:      {df['ACCIDENT_real'].mean():.6f}")
    print(f"Sim period:               {SIM_START_DATE} -> {SIM_END_DATE}")

    TP = int(((df["accident_pred_binary"] == 1) & (df["ACCIDENT_real"] == 1)).sum())
    TN = int(((df["accident_pred_binary"] == 0) & (df["ACCIDENT_real"] == 0)).sum())
    FP = int(((df["accident_pred_binary"] == 1) & (df["ACCIDENT_real"] == 0)).sum())
    FN = int(((df["accident_pred_binary"] == 0) & (df["ACCIDENT_real"] == 1)).sum())
    cm = np.array([[TP, FP], [FN, TN]])

    # Plot 1: Confusion Matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=False, cmap="Blues", ax=ax,
                xticklabels=["Real Positive", "Real Negative"],
                yticklabels=["Predicted Positive", "Predicted Negative"],
                linewidths=0.5, square=True, vmin=0, vmax=cm.max() * 3)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j + 0.5, i + 0.5, f"{cm[i, j]:,}",
                    ha="center", va="center", fontsize=14, fontweight="bold", color="black")
    ax.set_title(f"Confusion Matrix | {VERSION} | {SIM_START_DATE} -> {SIM_END_DATE}",
                 fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("REAL", fontsize=12); ax.set_ylabel("PREDICTED", fontsize=12)
    plt.tight_layout()
    _savefig(fig, "stage4_confusion_matrix")

    # Plot 2: Precision-Recall Curve
    yt    = df["ACCIDENT_real"].to_numpy()
    yprob = df["accident_probability"].to_numpy()
    prec_val = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    rec_val  = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1_val   = 2 * prec_val * rec_val / (prec_val + rec_val) if (prec_val + rec_val) > 0 else 0.0
    beta     = 2
    f2_val   = ((1 + beta**2) * prec_val * rec_val / (beta**2 * prec_val + rec_val)
                if (beta**2 * prec_val + rec_val) > 0 else 0.0)
    aucpr    = average_precision_score(yt, yprob)
    prec_c, rec_c, _ = precision_recall_curve(yt, yprob)
    prevalence = yt.mean()

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.axhline(prevalence, color="grey", linestyle=":", alpha=0.6,
               label=f"Random baseline (prevalence={prevalence:.5f})")
    ax.plot(rec_c, prec_c, label=f"{VERSION}  (AUCPR={aucpr:.4f})")
    ax.scatter([rec_val], [prec_val], s=120, zorder=5, edgecolors="black", linewidths=1.5,
               marker="*", label=f"t=0.5: P={prec_val:.4f}, R={rec_val:.4f}")
    ax.set_xlabel("Recall (Sensitivity)", fontsize=12); ax.set_ylabel("Precision", fontsize=12)
    ax.set_title(f"Precision-Recall Curve | {VERSION} | {SIM_START_DATE} -> {SIM_END_DATE}",
                 fontsize=14, fontweight="bold")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.legend(fontsize=10, loc="upper right")
    plt.tight_layout()
    _savefig(fig, "stage4_precision_recall_curve")

    summary = pd.DataFrame([{"version": VERSION,
        "sim_period": f"{SIM_START_DATE} -> {SIM_END_DATE}", "n_rows": len(df),
        "TP": TP, "FP": FP, "FN": FN, "TN": TN,
        "Precision": round(prec_val, 4), "Recall": round(rec_val, 4),
        "F1": round(f1_val, 4), "F2": round(f2_val, 4), "AUCPR": round(aucpr, 4),
    }]).set_index("version")
    print("\nSimulation evaluation metrics @ threshold=0.5:")
    print(summary.to_string())


# =============================================================================
# STAGE 5: FEATURE IMPORTANCE + SHAP (Stage 1)
# =============================================================================

def stage5_feature_importance_shap(base_models, X_test_xgb, available_features):
    print('=' * 80)
    print('STAGE 5A: XGBoost built-in feature importance')
    print('=' * 80)

    importance_types = ['gain', 'weight', 'cover']
    scores = {t: {} for t in importance_types}
    for m in base_models:
        for imp_type in importance_types:
            raw = m.get_score(importance_type=imp_type)
            for k, v in raw.items():
                fname = available_features[int(k[1:])] if k.startswith('f') and k[1:].isdigit() else k
                scores[imp_type][fname] = scores[imp_type].get(fname, 0.0) + v
    n_models = len(base_models)
    for t in importance_types:
        for k in scores[t]:
            scores[t][k] /= n_models

    df_imp = pd.DataFrame(scores).fillna(0).sort_values('gain', ascending=False).head(20)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    colors = ['#4C72B0', '#DD8452', '#55A868']
    for ax, (imp_type, color) in zip(axes, zip(importance_types, colors)):
        vals = df_imp[imp_type].sort_values()
        ax.barh(vals.index, vals.values, color=color, edgecolor='white', linewidth=0.5)
        ax.set_title(f'Feature Importance — {imp_type.upper()}', fontsize=13, fontweight='bold')
        ax.set_xlabel(imp_type.capitalize(), fontsize=11)
        ax.tick_params(axis='y', labelsize=9)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.suptitle(f'XGBoost Ensemble Feature Importance (top 20 by gain) | {VERSION}',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    _savefig(fig, "stage5a_feature_importance")
    print("Feature importance (top 20 by gain):")
    print(df_imp.sort_values('gain', ascending=False).to_string())

    print('=' * 80)
    print('STAGE 5B: SHAP analysis (Stage 1)')
    print('=' * 80)

    MAX_SHAP_ROWS = 5_000
    X_shap = X_test_xgb.reset_index(drop=True)
    if len(X_shap) > MAX_SHAP_ROWS:
        rng   = np.random.default_rng(42)
        idx   = rng.choice(len(X_shap), MAX_SHAP_ROWS, replace=False)
        X_shap = X_shap.iloc[idx].reset_index(drop=True)

    dshap          = xgb.DMatrix(X_shap)
    shap_vals_list = []
    print(f"Computing SHAP values on {len(X_shap):,} samples across {len(base_models)} base model(s)...")
    for i, m in enumerate(base_models):
        contribs = m.predict(dshap, pred_contribs=True)
        shap_vals_list.append(contribs[:, :-1])
        print(f"  Base model {i+1}/{len(base_models)} done.")
    shap_vals_avg = np.mean(shap_vals_list, axis=0)
    print(f"SHAP matrix shape: {shap_vals_avg.shape}")

    mean_abs_shap = np.abs(shap_vals_avg).mean(axis=0)
    shap_imp      = pd.Series(mean_abs_shap, index=available_features).sort_values(ascending=False)
    top_n_shap    = 20

    # SHAP bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    shap_imp.head(top_n_shap).sort_values().plot.barh(
        ax=ax, color='#4C72B0', edgecolor='white', linewidth=0.5)
    ax.set_xlabel('Mean |SHAP value|', fontsize=11)
    ax.set_title(f'Mean Absolute SHAP Value (top {top_n_shap}) | {VERSION}',
                 fontsize=13, fontweight='bold')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    _savefig(fig, "stage5b_shap_bar")

    # SHAP beeswarm
    top_feats = shap_imp.head(top_n_shap).index.tolist()
    feat_idx  = [available_features.index(f) for f in top_feats]
    sv_top    = shap_vals_avg[:, feat_idx]
    x_top     = X_shap[top_feats].to_numpy()

    fig, ax = plt.subplots(figsize=(11, 7))
    for rank, (fi, fname) in enumerate(zip(feat_idx, top_feats)):
        sv_col  = sv_top[:, rank]
        x_col   = x_top[:, rank]
        x_norm  = (x_col - x_col.min()) / (x_col.ptp() + 1e-9)
        jitter  = np.random.default_rng(rank).uniform(-0.3, 0.3, len(sv_col))
        ax.scatter(sv_col, rank + jitter, c=x_norm, cmap='coolwarm',
                   alpha=0.4, s=6, linewidths=0)
    ax.set_yticks(range(top_n_shap)); ax.set_yticklabels(top_feats, fontsize=9)
    ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_xlabel('SHAP value (impact on log-odds)', fontsize=11)
    ax.set_title(f'SHAP Beeswarm (top {top_n_shap} features) | {VERSION}\n'
                 'Colour: feature value  (blue=low, red=high)',
                 fontsize=13, fontweight='bold')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    _savefig(fig, "stage5b_shap_beeswarm")

    print("\nTop 20 features by mean |SHAP|:")
    print(shap_imp.head(top_n_shap).rename('mean_abs_shap').to_frame().to_string())


# =============================================================================
# STAGE 6: FEATURE-SPACE OVERLAP ANALYSIS
# =============================================================================

def stage6_feature_space_overlap(X_train_xgb, X_test_xgb, y_train_acc, y_test_acc,
                                  available_features, ensemble_preds):
    print('=' * 80)
    print('STAGE 6: FEATURE-SPACE OVERLAP ANALYSIS')
    print('=' * 80)

    X_all = pd.concat([X_train_xgb, X_test_xgb]).reset_index(drop=True)
    y_all = pd.concat([y_train_acc, y_test_acc]).reset_index(drop=True)

    scaler_std = StandardScaler()
    X_scaled   = scaler_std.fit_transform(X_all)
    pos_idx    = np.where(y_all == 1)[0]
    neg_idx    = np.where(y_all == 0)[0]
    print(f"Total samples: {len(y_all):,}  Positives: {len(pos_idx):,}  Negatives: {len(neg_idx):,}")

    K = 5
    nn_neg = NearestNeighbors(n_neighbors=K, metric='euclidean', n_jobs=-1)
    nn_neg.fit(X_scaled[neg_idx])
    dists_pos2neg, _ = nn_neg.kneighbors(X_scaled[pos_idx])

    rng = np.random.default_rng(42)
    sample_neg_idx  = rng.choice(neg_idx, size=min(len(pos_idx) * 5, len(neg_idx)), replace=False)
    nn_neg_self     = NearestNeighbors(n_neighbors=2, metric='euclidean', n_jobs=-1)
    nn_neg_self.fit(X_scaled[neg_idx])
    dists_neg2neg, _ = nn_neg_self.kneighbors(X_scaled[sample_neg_idx])
    dists_neg2neg_1nn = dists_neg2neg[:, 1]

    print(f"\nPos->nearest Neg 1-NN: median={np.median(dists_pos2neg[:,0]):.4f}")
    print(f"Neg->nearest Neg 1-NN: median={np.median(dists_neg2neg_1nn):.4f}")

    nn_pos = NearestNeighbors(n_neighbors=1, metric='euclidean', n_jobs=-1)
    nn_pos.fit(X_scaled[pos_idx])
    dists_neg2pos, _ = nn_pos.kneighbors(X_scaled[neg_idx])
    dists_neg2pos    = dists_neg2pos.ravel()

    test_start      = len(X_train_xgb)
    test_mask       = np.arange(len(X_all)) >= test_start
    test_neg_global = neg_idx[np.isin(neg_idx, np.where(test_mask)[0])]
    test_neg_local  = test_neg_global - test_start
    dists_test_neg, _ = nn_pos.kneighbors(X_scaled[test_neg_global])
    dists_test_neg  = dists_test_neg.ravel()

    fp_mask = (ensemble_preds >= 0.5) & (y_test_acc.values == 0)
    tn_mask = (ensemble_preds <  0.5) & (y_test_acc.values == 0)
    fp_local_idx = np.where(fp_mask)[0]
    tn_local_idx = np.where(tn_mask)[0]
    local_to_dist = dict(zip(test_neg_local, dists_test_neg))
    fp_dists = np.array([local_to_dist[i] for i in fp_local_idx if i in local_to_dist])
    tn_dists = np.array([local_to_dist[i] for i in tn_local_idx if i in local_to_dist])
    print(f"\nFalse Positives ({len(fp_dists):,}): median={np.median(fp_dists):.4f}")
    print(f"True  Negatives ({len(tn_dists):,}): median={np.median(tn_dists):.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    bins = np.linspace(0, 8, 80)
    axes[0].hist(dists_pos2neg[:, 0], bins=bins, alpha=0.6, density=True,
                 label='Pos->nearest Neg', color='crimson')
    axes[0].hist(dists_neg2neg_1nn, bins=bins, alpha=0.6, density=True,
                 label='Neg->nearest Neg', color='steelblue')
    axes[0].set_xlabel('Euclidean distance (standardised)', fontsize=10)
    axes[0].set_title('1-NN Distance Distributions', fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=9)
    axes[0].spines['top'].set_visible(False); axes[0].spines['right'].set_visible(False)

    bins2 = np.linspace(0, max(np.percentile(tn_dists, 99), np.percentile(fp_dists, 99)), 60)
    axes[1].hist(fp_dists, bins=bins2, alpha=0.65, density=True,
                 label=f'FP (n={len(fp_dists):,})', color='orangered')
    axes[1].hist(tn_dists, bins=bins2, alpha=0.45, density=True,
                 label=f'TN (n={len(tn_dists):,})', color='forestgreen')
    axes[1].set_xlabel('Distance to nearest positive', fontsize=10)
    axes[1].set_title('Test Negatives: Distance to Nearest Positive', fontsize=12, fontweight='bold')
    axes[1].legend(fontsize=9)
    axes[1].spines['top'].set_visible(False); axes[1].spines['right'].set_visible(False)

    sorted_d = np.sort(dists_neg2pos)
    cum_pct  = np.arange(1, len(sorted_d) + 1) / len(sorted_d) * 100
    step     = max(1, len(sorted_d) // 5000)
    axes[2].plot(sorted_d[::step], cum_pct[::step], color='darkorange', linewidth=1.5)
    axes[2].set_xlabel('Distance to nearest positive', fontsize=10)
    axes[2].set_ylabel('Cumulative % of negatives', fontsize=10)
    axes[2].set_title('Negatives Overlapping Positive Regions', fontsize=12, fontweight='bold')
    axes[2].axhline(1, color='gray', linestyle='--', linewidth=0.8, label='1%')
    axes[2].axhline(5, color='gray', linestyle=':', linewidth=0.8, label='5%')
    axes[2].legend(fontsize=9)
    axes[2].set_xlim(0, np.percentile(sorted_d, 99.5))
    axes[2].spines['top'].set_visible(False); axes[2].spines['right'].set_visible(False)

    plt.suptitle('Feature-Space Overlap Analysis', fontsize=14, fontweight='bold', y=1.03)
    plt.tight_layout()
    _savefig(fig, "stage6_feature_space_overlap")

    return fp_dists, tn_dists


# =============================================================================
# STAGE 7: TWO-STAGE ARCHITECTURE (OOF + Stage 2 Optuna)
# =============================================================================

def stage7_two_stage(X_train_xgb, X_test_xgb, y_train_acc, y_test_acc,
                     available_features, ensemble_preds, base_models, model_time, s1_best_params):
    print('=' * 80)
    print('STAGE 7A: Build Stage 2 training data (OOF Stage 1 scores)')
    print('=' * 80)

    STAGE1_THRESHOLD = 0.5
    STAGE2_THRESHOLD = 0.5
    MIN_RECALL_CONSTRAINT = 0.70
    MAX_FAR_TARGET        = 0.30

    X_tr = X_train_xgb.reset_index(drop=True)
    y_tr = y_train_acc.reset_index(drop=True)
    neg_pos_ratio = ENSEMBLE_CONFIG['neg_pos_ratio']

    OOF_N_FOLDS  = 5
    OOF_N_MODELS = 5
    _mean_iter   = int(np.mean([getattr(m, 'best_iteration', 100) for m in base_models]))
    OOF_ROUNDS   = max(50, _mean_iter)

    _oof_kf = StratifiedKFold(n_splits=OOF_N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    s1_tr_probs = np.zeros(len(X_tr))
    _oof_params = {**s1_best_params, 'tree_method': 'hist',
                   'objective': 'binary:logistic', 'eval_metric': 'aucpr',
                   'nthread': N_THREADS, 'seed': RANDOM_STATE}
    _rng = np.random.RandomState(RANDOM_STATE)

    print(f'Generating OOF Stage 1 scores ({OOF_N_FOLDS} folds x {OOF_N_MODELS} models, {OOF_ROUNDS} rounds each)...')
    for _fold_i, (_tr_idx, _va_idx) in enumerate(_oof_kf.split(X_tr.values, y_tr.values), 1):
        _X_fold = X_tr.iloc[_tr_idx]
        _y_fold = y_tr.iloc[_tr_idx]
        _pos_ix = np.where(_y_fold.values == 1)[0]
        _neg_ix = np.where(_y_fold.values == 0)[0]
        _fold_probs = np.zeros(len(_va_idx))
        for _ in range(OOF_N_MODELS):
            _samp_neg = _rng.choice(_neg_ix, size=len(_pos_ix) * neg_pos_ratio, replace=False)
            _bag_idx  = np.concatenate([_pos_ix, _samp_neg])
            _rng.shuffle(_bag_idx)
            _Xb = _X_fold.iloc[_bag_idx][available_features]
            _yb = _y_fold.iloc[_bag_idx]
            _dm = xgb.DMatrix(_Xb, label=_yb)
            _m  = xgb.train(_oof_params, _dm, num_boost_round=OOF_ROUNDS, verbose_eval=False)
            _fold_probs += _m.predict(xgb.DMatrix(X_tr.iloc[_va_idx][available_features]))
        s1_tr_probs[_va_idx] = _fold_probs / OOF_N_MODELS
        _flagged = (s1_tr_probs[_va_idx] >= STAGE1_THRESHOLD).sum()
        print(f'  Fold {_fold_i}/{OOF_N_FOLDS}: {len(_va_idx):,} rows, flagged={_flagged:,}')

    print(f'\nOOF Stage 1 on train set ({len(X_tr):,} rows):')
    print(f'  Flagged (>= {STAGE1_THRESHOLD}): {(s1_tr_probs >= STAGE1_THRESHOLD).sum():,}')
    print(f'  TP in flagged: {int(((s1_tr_probs >= STAGE1_THRESHOLD) & (y_tr == 1)).sum()):,}')
    print(f'  FP in flagged: {int(((s1_tr_probs >= STAGE1_THRESHOLD) & (y_tr == 0)).sum()):,}')

    s2_mask   = s1_tr_probs >= STAGE1_THRESHOLD
    X_s2_tr   = X_tr[s2_mask].copy()
    y_s2_tr   = y_tr[s2_mask].copy()
    X_s2_tr['s1_score'] = s1_tr_probs[s2_mask]
    n_tp_s2   = int(y_s2_tr.sum())
    n_fp_s2   = int((y_s2_tr == 0).sum())
    s2_spw    = n_fp_s2 / max(n_tp_s2, 1)
    print(f'\nStage 2 training pool: {len(X_s2_tr):,}  TP={n_tp_s2:,}  FP={n_fp_s2:,}  ratio={s2_spw:.1f}')

    # Stage 2 test set
    X_test_r     = X_test_xgb.reset_index(drop=True)
    y_test_r     = y_test_acc.reset_index(drop=True)
    test_s1_flag = ensemble_preds >= STAGE1_THRESHOLD
    X_s2_te      = X_test_r[test_s1_flag].copy()
    y_s2_te      = y_test_r[test_s1_flag].copy()
    X_s2_te['s1_score'] = ensemble_preds[test_s1_flag]

    # Stage 2 Optuna
    S2_OPTUNA_TRIALS = 30
    S2_CV_FOLDS      = 3

    def _s2_objective(trial):
        params = {
            'max_depth':        trial.suggest_int('max_depth', 2, 8),
            'learning_rate':    trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
            'subsample':        trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'gamma':            trial.suggest_float('gamma', 0.0, 5.0),
            'reg_alpha':        trial.suggest_float('reg_alpha', 0.0, 5.0),
            'reg_lambda':       trial.suggest_float('reg_lambda', 0.0, 5.0),
            'scale_pos_weight': s2_spw,
            'tree_method': 'hist', 'objective': 'binary:logistic',
            'eval_metric': 'aucpr', 'nthread': N_THREADS, 'seed': RANDOM_STATE,
        }
        skf     = StratifiedKFold(n_splits=S2_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        X_arr   = X_s2_tr.values
        y_arr   = y_s2_tr.values
        cv_scores = []
        for tr_idx, va_idx in skf.split(X_arr, y_arr):
            dtrain_f = xgb.DMatrix(X_arr[tr_idx], label=y_arr[tr_idx])
            dval_f   = xgb.DMatrix(X_arr[va_idx],  label=y_arr[va_idx])
            model_f  = xgb.train(params, dtrain_f, num_boost_round=300,
                                 evals=[(dval_f, 'eval')], early_stopping_rounds=20,
                                 verbose_eval=False)
            preds_f  = model_f.predict(dval_f, iteration_range=(0, model_f.best_iteration))
            cv_scores.append(fbeta_score(y_arr[va_idx], (preds_f >= 0.5).astype(int),
                                         beta=2.0, zero_division=0))
        return float(np.mean(cv_scores)) if cv_scores else 0.0

    s2_study = optuna.create_study(direction='maximize',
                                   sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    print(f'Stage 2 Optuna search ({S2_OPTUNA_TRIALS} trials, {S2_CV_FOLDS}-fold CV, '
          f'objective=F2, train={len(X_s2_tr):,}, spw={s2_spw:.2f})...')
    s2_study.optimize(_s2_objective, n_trials=S2_OPTUNA_TRIALS, show_progress_bar=True)
    print(f'Stage 2 best F2 : {s2_study.best_value:.4f}')
    for k, v in s2_study.best_params.items():
        print(f'  {k}: {v}')

    s2_best_params = {**s2_study.best_params, 'scale_pos_weight': s2_spw,
                      'tree_method': 'hist', 'objective': 'binary:logistic',
                      'eval_metric': 'aucpr', 'nthread': N_THREADS, 'seed': RANDOM_STATE}
    dtrain_s2 = xgb.DMatrix(X_s2_tr, label=y_s2_tr)
    dtest_s2  = xgb.DMatrix(X_s2_te, label=y_s2_te)
    print(f'\nTraining final Stage 2 model (train={len(X_s2_tr):,}, scale_pos_weight={s2_spw:.2f})...')
    stage2_model = xgb.train(s2_best_params, dtrain_s2, num_boost_round=500,
                             early_stopping_rounds=30,
                             evals=[(dtrain_s2, 'train'), (dtest_s2, 'test')],
                             verbose_eval=50)
    stage2_path = os.path.join(OUTPUT_DIR, f'xgboost-stage2-fp-filter_version={model_time}.json')
    stage2_model.save_model(stage2_path)
    print(f'\nStage 2 model saved: {stage2_path}')

    # Stage 7C: Test set comparison
    s2_test_probs = stage2_model.predict(dtest_s2)
    y_test_np     = y_test_r.values
    s2_full_prob  = np.zeros(len(y_test_np))
    s2_full_bin   = np.zeros(len(y_test_np), dtype=int)
    flagged_idx   = np.where(test_s1_flag)[0]
    s2_full_prob[flagged_idx] = s2_test_probs
    s2_full_bin[flagged_idx]  = (s2_test_probs >= STAGE2_THRESHOLD).astype(int)

    s1_binary = (ensemble_preds >= STAGE1_THRESHOLD).astype(int)
    cm_s1     = confusion_matrix(y_test_np, s1_binary)
    cm_s2     = confusion_matrix(y_test_np, s2_full_bin)
    tp1, fp1, fn1, p1, r1, f1_1 = _metrics(cm_s1)
    tp2, fp2, fn2, p2, r2, f1_2 = _metrics(cm_s2)
    auc_s1 = average_precision_score(y_test_np, ensemble_preds)
    auc_s2 = average_precision_score(y_test_np, s2_full_prob)

    print('\n' + '=' * 55)
    print('COMPARISON — Test set: Stage 1 alone vs Two-Stage')
    print('=' * 55)
    for label, v1, v2 in [('TP', tp1, tp2), ('FP', fp1, fp2), ('FN', fn1, fn2),
                           ('Precision', p1, p2), ('Recall', r1, r2),
                           ('F1', f1_1, f1_2), ('PR-AUC', auc_s1, auc_s2)]:
        if isinstance(v1, float):
            print(f'{label:22s}  {v1:>10.4f}  {v2:>10.4f}')
        else:
            print(f'{label:22s}  {v1:>10,}  {v2:>10,}')

    fig = plt.figure(figsize=(18, 5))
    gs  = gridspec.GridSpec(1, 3, figure=fig)
    for ax_i, (title, cm) in enumerate([('Stage 1 alone', cm_s1), ('Stage 1 + Stage 2', cm_s2)]):
        ax = fig.add_subplot(gs[0, ax_i])
        sns.heatmap(cm, annot=True, fmt=',d', cmap='Blues', ax=ax,
                    xticklabels=['Pred NEG', 'Pred POS'],
                    yticklabels=['Real NEG', 'Real POS'], cbar=False)
        ax.set_title(title, fontsize=12, fontweight='bold')
    ax3 = fig.add_subplot(gs[0, 2])
    prec1_c, rec1_c, _ = precision_recall_curve(y_test_np, ensemble_preds)
    prec2_c, rec2_c, _ = precision_recall_curve(y_test_np, s2_full_prob)
    ax3.plot(rec1_c, prec1_c, label=f'Stage 1   (AUCPR={auc_s1:.4f})', color='steelblue')
    ax3.plot(rec2_c, prec2_c, label=f'Two-Stage (AUCPR={auc_s2:.4f})', color='crimson')
    ax3.scatter([r1], [p1], s=100, c='steelblue', zorder=5, marker='*')
    ax3.scatter([r2], [p2], s=100, c='crimson', zorder=5, marker='*')
    ax3.axhline(y_test_np.mean(), color='gray', linestyle=':', alpha=0.6,
                label=f'Baseline (prev={y_test_np.mean():.5f})')
    ax3.set_xlabel('Recall', fontsize=10); ax3.set_ylabel('Precision', fontsize=10)
    ax3.set_title('Precision-Recall Curve', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9); ax3.set_xlim(0, 1); ax3.set_ylim(0, 1)
    ax3.spines['top'].set_visible(False); ax3.spines['right'].set_visible(False)
    plt.suptitle(f'Two-Stage Architecture — Test Set Evaluation | {VERSION}',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    _savefig(fig, "stage7_two_stage_test_comparison")

    return (stage2_model, X_s2_tr, y_s2_tr, s2_full_prob, s2_full_bin,
            y_test_np, flagged_idx, s2_test_probs, STAGE1_THRESHOLD, STAGE2_THRESHOLD,
            MIN_RECALL_CONSTRAINT, MAX_FAR_TARGET)


# =============================================================================
# STAGE 7B: STAGE 2 SHAP ANALYSIS
# =============================================================================

def stage7b_shap_stage2(stage2_model, X_s2_tr, y_s2_tr):
    print('=' * 80)
    print('STAGE 7B-SHAP: Stage 2 Model Explainability')
    print('=' * 80)

    N_SHAP_SAMPLES = min(3000, len(X_s2_tr))
    rng      = np.random.default_rng(42)
    shap_idx = rng.choice(len(X_s2_tr), size=N_SHAP_SAMPLES, replace=False)
    X_s2_arr = X_s2_tr.values[shap_idx]
    y_s2_arr = y_s2_tr.values[shap_idx]
    s2_feats = list(X_s2_tr.columns)

    d_shap    = xgb.DMatrix(X_s2_arr, feature_names=s2_feats)
    print(f'Computing Stage 2 SHAP values on {N_SHAP_SAMPLES} samples ...')
    shap_raw  = stage2_model.predict(d_shap, pred_contribs=True)
    shap_vals = shap_raw[:, :-1]
    shap_feats = np.array(s2_feats)
    mean_abs   = np.abs(shap_vals).mean(axis=0)
    order      = np.argsort(mean_abs)[::-1]

    print(f'Stage 2 SHAP matrix: {shap_vals.shape}')
    print(f'\nTop 20 features by mean |SHAP|:')
    df_imp = pd.DataFrame({'feature': shap_feats[order[:20]], 'mean_abs_shap': mean_abs[order[:20]]})
    print(df_imp.to_string(index=False))

    # Cap plots to the actual feature count (Stage-2 may be trained on a curated subset).
    n_feats = int(shap_vals.shape[1])
    TOP_N, TOP_BEE = min(20, n_feats), min(15, n_feats)
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    colors = ['crimson' if f == 's1_score' else 'steelblue' for f in shap_feats[order[:TOP_N]]]
    axes[0].barh(shap_feats[order[:TOP_N]][::-1], mean_abs[order[:TOP_N]][::-1], color=colors[::-1])
    axes[0].set_xlabel('Mean |SHAP value|', fontsize=11)
    axes[0].set_title(f'Stage 2 -- Feature Importance (top {TOP_N})', fontweight='bold')
    axes[0].legend(handles=[
        Patch(color='crimson',   label='s1_score (Stage 1 output)'),
        Patch(color='steelblue', label='Traffic / geometry features'),
    ], fontsize=9, loc='lower right')

    bee_order = order[:TOP_BEE]
    bee_feats = shap_feats[bee_order]
    bee_shap  = shap_vals[:, bee_order]
    bee_vals  = X_s2_arr[:, bee_order]
    bee_norm  = np.zeros_like(bee_vals, dtype=float)
    for fi in range(TOP_BEE):
        col = bee_vals[:, fi]; lo, hi = np.percentile(col, [5, 95])
        bee_norm[:, fi] = np.clip((col - lo) / (hi - lo + 1e-9), 0, 1)
    for fi in range(TOP_BEE - 1, -1, -1):
        ypos = np.full(N_SHAP_SAMPLES, fi) + rng.uniform(-0.3, 0.3, N_SHAP_SAMPLES)
        axes[1].scatter(bee_shap[:, fi], ypos, c=RdBu_r(1 - bee_norm[:, fi]),
                        s=4, alpha=0.5, linewidths=0)
    axes[1].set_yticks(range(TOP_BEE)); axes[1].set_yticklabels(bee_feats, fontsize=9)
    axes[1].axvline(0, color='black', linewidth=0.8, linestyle='--')
    axes[1].set_xlabel('SHAP value (impact on Stage 2 accept probability)', fontsize=10)
    axes[1].set_title(f'Stage 2 -- SHAP Beeswarm (top {TOP_BEE})', fontweight='bold')
    sm = plt.cm.ScalarMappable(cmap=RdBu_r, norm=plt.Normalize(0, 1)); sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes[1], pad=0.01)
    cbar.set_label('Feature value\n(blue=low, red=high)', fontsize=8)
    cbar.set_ticks([0, 1]); cbar.set_ticklabels(['Low', 'High'])
    plt.suptitle('Stage 2 FP-Filter -- SHAP Explainability', fontsize=13, fontweight='bold')
    plt.tight_layout()
    _savefig(fig, "stage7b_shap_stage2_bar_beeswarm")

    tp_mask_s2 = y_s2_arr == 1
    fp_mask_s2 = y_s2_arr == 0
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    for ax, fi in zip(axes.flatten(), order[:8]):
        feat    = shap_feats[fi]
        tp_shap = shap_vals[tp_mask_s2, fi]; fp_shap = shap_vals[fp_mask_s2, fi]
        lo = min(tp_shap.min(), fp_shap.min()); hi = max(tp_shap.max(), fp_shap.max())
        bins = np.linspace(lo, hi, 30)
        ax.hist(fp_shap, bins=bins, alpha=0.6, density=True, color='orange', label=f'FP ({fp_mask_s2.sum()})')
        ax.hist(tp_shap, bins=bins, alpha=0.7, density=True, color='steelblue', label=f'TP ({tp_mask_s2.sum()})')
        ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
        ax.set_title(feat, fontsize=9, fontweight='bold' if feat == 's1_score' else 'normal')
        ax.legend(fontsize=7); ax.set_yticks([])
    plt.suptitle('Stage 2 SHAP -- TP vs FP per feature\n'
                 '(positive SHAP = pushes toward accepting the flag)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    _savefig(fig, "stage7b_shap_stage2_tp_fp_distributions")

    print('\n=== Stage 2 SHAP summary ===')
    print(f'{"Feature":30s}  {"mean|SHAP|":>10s}  {"SHAP(TP)":>10s}  {"SHAP(FP)":>10s}  {"Diff":>8s}')
    print('-' * 75)
    for fi in order[:15]:
        feat  = shap_feats[fi]
        m_tp  = shap_vals[tp_mask_s2, fi].mean(); m_fp = shap_vals[fp_mask_s2, fi].mean()
        mark  = ' <-- Stage 1 output' if feat == 's1_score' else ''
        print(f'{feat:30s}  {mean_abs[fi]:10.4f}  {m_tp:10.4f}  {m_fp:10.4f}  {m_tp-m_fp:8.4f}{mark}')


# =============================================================================
# STAGE 7D: THRESHOLD SWEEP
# =============================================================================

def stage7d_threshold_sweep(ensemble_preds, y_test_np, flagged_idx, s2_test_probs,
                             MIN_RECALL_CONSTRAINT, MAX_FAR_TARGET):
    print('=' * 80)
    print('STAGE 7D: Threshold sweep')
    print('=' * 80)

    s1_thresholds = np.arange(0.10, 0.55, 0.05)
    s2_thresholds = np.arange(0.10, 0.95, 0.05)
    best_result   = {'s1': 0.5, 's2': 0.5, 'precision': 0.0, 'recall': 0.0, 'far': 1.0, 'f1': 0.0}

    s2_test_probs_full = np.zeros(len(y_test_np))
    s2_test_probs_full[flagged_idx] = s2_test_probs

    for s1_t in s1_thresholds:
        s1_flags = ensemble_preds >= s1_t
        if s1_flags.sum() == 0:
            continue
        for s2_t in s2_thresholds:
            s1_flag_idx      = np.where(s1_flags)[0]
            s2_scores_flagged = s2_test_probs_full[s1_flag_idx]
            final_pred        = np.zeros(len(y_test_np), dtype=int)
            final_pred[s1_flag_idx[s2_scores_flagged >= s2_t]] = 1
            tp = int(((final_pred == 1) & (y_test_np == 1)).sum())
            fp = int(((final_pred == 1) & (y_test_np == 0)).sum())
            fn = int(((final_pred == 0) & (y_test_np == 1)).sum())
            recall    = tp / max(tp + fn, 1)
            precision = tp / max(tp + fp, 1)
            far       = fp / max(tp + fp, 1)
            f1        = 2 * precision * recall / max(precision + recall, 1e-9)
            if recall >= MIN_RECALL_CONSTRAINT and far <= MAX_FAR_TARGET:
                if precision > best_result['precision']:
                    best_result = {'s1': float(s1_t), 's2': float(s2_t),
                                   'precision': precision, 'recall': recall,
                                   'far': far, 'f1': f1, 'tp': tp, 'fp': fp, 'fn': fn}

    S1_THR, S2_THR = 0.5, 0.5
    if best_result['precision'] > 0:
        print(f"Best threshold pair found: S1={best_result['s1']:.2f}  S2={best_result['s2']:.2f}")
        print(f"  Precision={best_result['precision']:.4f}  Recall={best_result['recall']:.4f}  FAR={best_result['far']:.4f}")
        S1_THR, S2_THR = best_result['s1'], best_result['s2']
    else:
        print(f"WARNING: No threshold pair achieves recall >= {MIN_RECALL_CONSTRAINT} AND "
              f"FAR <= {MAX_FAR_TARGET}. Using defaults (0.5, 0.5).")

    return S1_THR, S2_THR


# =============================================================================
# STAGE 8: TWO-STAGE ROLLING SIMULATION
# =============================================================================

def stage8_two_stage_simulation(df_full, base_models, stage2_model, available_features,
                                 model_time, S1_THR, S2_THR):
    print('=' * 80)
    print('STAGE 8: Two-Stage Rolling Simulation')
    print('=' * 80)

    def run_two_stage_simulation(df_full, base_models, stage2_model, available_features,
                                  sim_start, sim_end, pk_min, pk_max, model_time, output_dir,
                                  stage1_threshold=0.5, stage2_threshold=0.5):
        start  = pd.Timestamp(sim_start); end = pd.Timestamp(sim_end)
        df_sim = df_full[(df_full['pk'] >= pk_min) & (df_full['pk'] <= pk_max) &
                         (df_full['dat'] >= start) & (df_full['dat'] < end)].copy()
        if len(df_sim) == 0:
            print(f'[WARN] No data for sim period {sim_start} -> {sim_end}'); return None
        print(f'Sim period  : {sim_start} -> {sim_end}')
        print(f'Rows        : {len(df_sim):,}')
        print(f'PKs         : {sorted(df_sim["pk"].unique())}')
        for col in available_features:
            if col not in df_sim.columns: df_sim[col] = 0
        X_sim  = df_sim[available_features].apply(pd.to_numeric, errors='coerce').fillna(0)
        d_sim  = xgb.DMatrix(X_sim)
        s1_probs = np.zeros(len(X_sim))
        for m in base_models:
            best_iter = getattr(m, 'best_iteration', None)
            s1_probs += m.predict(d_sim, iteration_range=(0, best_iter)) if best_iter else m.predict(d_sim)
        s1_probs /= len(base_models)
        s1_flag  = s1_probs >= stage1_threshold
        s2_prob  = np.zeros(len(X_sim)); s2_bin = np.zeros(len(X_sim), dtype=int)
        if s1_flag.sum() > 0:
            X_flagged = X_sim[s1_flag].copy(); X_flagged['s1_score'] = s1_probs[s1_flag]
            s2_probs_flagged = stage2_model.predict(xgb.DMatrix(X_flagged))
            flagged_pos = np.where(s1_flag)[0]
            s2_prob[flagged_pos] = s2_probs_flagged
            s2_bin[flagged_pos]  = (s2_probs_flagged >= stage2_threshold).astype(int)
        results = df_sim[['via', 'sen', 'pk', 'anyo', 'mes', 'dia', 'diaSem', 'hor', 'min', 'dat']].copy()
        results['s1_probability']       = s1_probs
        results['s1_pred_binary']       = s1_flag.astype(int)
        results['s2_probability']       = s2_prob
        results['s2_pred_binary']       = s2_bin
        results['accident_probability'] = s2_prob
        results['accident_pred_binary'] = s2_bin
        for col, src in [('ACCIDENT_real', 'ACCIDENT'), ('mean_speed_real', 'mean_speed'),
                         ('intTot_real', 'intTot'), ('intP_real', 'intP')]:
            if src in df_sim.columns: results[col] = df_sim[src].values
        results['model_time'] = model_time; results['simulation_mode'] = 'two_stage_no_gnn'
        return results

    t0 = time.time()
    sim_results_2s = run_two_stage_simulation(
        df_full=df_full, base_models=base_models, stage2_model=stage2_model,
        available_features=available_features,
        sim_start=SIM_START_DATE, sim_end=SIM_END_DATE,
        pk_min=PK_MIN, pk_max=PK_MAX, model_time=model_time, output_dir=OUTPUT_DIR,
        stage1_threshold=S1_THR, stage2_threshold=S2_THR,
    )
    print(f'Two-stage simulation completed in {format_duration(time.time()-t0)}')

    if sim_results_2s is not None and 'ACCIDENT_real' in sim_results_2s.columns:
        df_e   = sim_results_2s.dropna(subset=['ACCIDENT_real', 's1_probability']).copy()
        y_t    = df_e['ACCIDENT_real'].astype(int).values
        s1p    = df_e['s1_probability'].values; s2p = df_e['s2_probability'].values
        s1b    = df_e['s1_pred_binary'].values; s2b = df_e['s2_pred_binary'].values
        cm_s1  = confusion_matrix(y_t, s1b); cm_s2 = confusion_matrix(y_t, s2b)
        tp1s, fp1s, fn1s, pr1s, rc1s, f1_1s = _metrics(cm_s1)
        tp2s, fp2s, fn2s, pr2s, rc2s, f1_2s = _metrics(cm_s2)
        auc1s  = average_precision_score(y_t, s1p); auc2s = average_precision_score(y_t, s2p)

        print(f'\n{"":22s}  {"Stage 1":>10s}  {"Two-Stage":>10s}')
        print('-' * 47)
        for label, v1, v2 in [('TP', tp1s, tp2s), ('FP', fp1s, fp2s), ('FN', fn1s, fn2s),
                               ('Precision', pr1s, pr2s), ('Recall', rc1s, rc2s),
                               ('F1', f1_1s, f1_2s), ('PR-AUC', auc1s, auc2s)]:
            if isinstance(v1, float): print(f'{label:22s}  {v1:>10.4f}  {v2:>10.4f}')
            else:                     print(f'{label:22s}  {v1:>10,}  {v2:>10,}')

        df_e['date'] = pd.to_datetime(df_e['dat']).dt.date
        daily_2s = df_e.groupby('date').agg(
            real_accidents=('ACCIDENT_real', 'sum'),
            s1_flags=('s1_pred_binary', 'sum'), s2_flags=('s2_pred_binary', 'sum'))
        print('\nPer-day comparison:'); print(daily_2s.to_string())

        sim_dir_2s  = os.path.join(OUTPUT_DIR, f'simulation_two_stage_{model_time}')
        os.makedirs(sim_dir_2s, exist_ok=True)
        sim_path_2s = os.path.join(sim_dir_2s, f'simulation_two_stage_{model_time}.csv')
        sim_results_2s.to_csv(sim_path_2s, sep=';', index=False)
        print(f'\n[SUCCESS] Two-stage simulation saved: {sim_path_2s}')

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        for ax, title, cm in zip(axes[:2], ['Stage 1 alone', 'Two-Stage (S1 + S2)'], [cm_s1, cm_s2]):
            sns.heatmap(cm, annot=True, fmt=',d', cmap='Blues', ax=ax,
                        xticklabels=['Pred NEG', 'Pred POS'],
                        yticklabels=['Real NEG', 'Real POS'], cbar=False)
            ax.set_title(title, fontsize=12, fontweight='bold')
        ax3 = axes[2]
        pc1s, rc1c, _ = precision_recall_curve(y_t, s1p)
        pc2s, rc2c, _ = precision_recall_curve(y_t, s2p)
        ax3.plot(rc1c, pc1s, label=f'Stage 1   (AUCPR={auc1s:.4f})', color='steelblue')
        ax3.plot(rc2c, pc2s, label=f'Two-Stage (AUCPR={auc2s:.4f})', color='crimson')
        ax3.scatter([rc1s], [pr1s], s=100, c='steelblue', zorder=5, marker='*')
        ax3.scatter([rc2s], [pr2s], s=100, c='crimson',   zorder=5, marker='*')
        ax3.axhline(y_t.mean(), color='gray', linestyle=':', alpha=0.6,
                    label=f'Baseline (prev={y_t.mean():.5f})')
        ax3.set_xlabel('Recall', fontsize=10); ax3.set_ylabel('Precision', fontsize=10)
        ax3.set_title('PR Curve — Simulation', fontsize=12, fontweight='bold')
        ax3.legend(fontsize=9); ax3.set_xlim(0, 1); ax3.set_ylim(0, 1)
        ax3.spines['top'].set_visible(False); ax3.spines['right'].set_visible(False)
        plt.suptitle(f'Two-Stage Simulation | {VERSION} | {SIM_START_DATE} -> {SIM_END_DATE}',
                     fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        _savefig(fig, "stage8_two_stage_simulation")

    return sim_results_2s


# =============================================================================
# STAGE 9: FINAL MERGED EVALUATION
# =============================================================================

def stage9_final_evaluation(sim_results, sim_results_2s):
    print('=' * 80)
    print('STAGE 9: Final Merged Evaluation (Two-Stage)')
    print('=' * 80)

    if sim_results_2s is None or len(sim_results_2s) == 0:
        print('[WARN] Two-stage simulation results not available.'); return

    df_s1 = sim_results.copy().reset_index(drop=True)
    df_s2 = sim_results_2s.copy().reset_index(drop=True)
    df_merged = df_s1[['via', 'sen', 'pk', 'anyo', 'mes', 'dia', 'diaSem',
                        'hor', 'min', 'dat', 'ACCIDENT_real']].copy()
    df_merged['s1_probability']       = df_s1['accident_probability'].values
    df_merged['s1_pred_binary']       = df_s1['accident_pred_binary'].values
    df_merged['s2_probability']       = df_s2['s2_probability'].values
    df_merged['s2_pred_binary']       = df_s2['s2_pred_binary'].values
    df_merged['accident_probability'] = df_merged['s2_probability']
    df_merged['accident_pred_binary'] = df_merged['s2_pred_binary']

    df = df_merged.dropna(subset=['ACCIDENT_real', 'accident_probability']).copy()
    df['ACCIDENT_real']        = (df['ACCIDENT_real'].astype(float) > 0).astype(int)
    df['accident_probability'] = df['accident_probability'].astype(float).clip(0, 1)
    df['s1_probability']       = df['s1_probability'].astype(float).clip(0, 1)

    yt     = df['ACCIDENT_real'].to_numpy(); yprob = df['accident_probability'].to_numpy()
    ypred  = df['accident_pred_binary'].to_numpy(); s1prob = df['s1_probability'].to_numpy()
    s1bin  = df['s1_pred_binary'].to_numpy()

    TP = int(((ypred == 1) & (yt == 1)).sum()); TN = int(((ypred == 0) & (yt == 0)).sum())
    FP = int(((ypred == 1) & (yt == 0)).sum()); FN = int(((ypred == 0) & (yt == 1)).sum())
    prec_val = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    rec_val  = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1_val   = 2 * prec_val * rec_val / (prec_val + rec_val) if (prec_val + rec_val) > 0 else 0.0
    beta     = 2
    f2_val   = ((1 + beta**2) * prec_val * rec_val / (beta**2 * prec_val + rec_val)
                if (beta**2 * prec_val + rec_val) > 0 else 0.0)
    aucpr    = average_precision_score(yt, yprob)
    aucpr_s1 = average_precision_score(yt, s1prob)

    s1_tp  = int(((s1bin == 1) & (yt == 1)).sum()); s1_fp = int(((s1bin == 1) & (yt == 0)).sum())
    s1_fn  = int(((s1bin == 0) & (yt == 1)).sum()); s1_tn = int(len(df) - s1_tp - s1_fp - s1_fn)
    s1_prec = s1_tp / (s1_tp + s1_fp) if (s1_tp + s1_fp) > 0 else 0.0
    s1_rec  = s1_tp / (s1_tp + s1_fn) if (s1_tp + s1_fn) > 0 else 0.0
    s1_f1   = 2*s1_prec*s1_rec/(s1_prec+s1_rec) if (s1_prec+s1_rec) > 0 else 0.0

    print(f'Rows: {len(df):,}  Prevalence: {df["ACCIDENT_real"].mean():.6f}')
    print(f'Sim period: {SIM_START_DATE} -> {SIM_END_DATE}')

    cm = np.array([[TP, FP], [FN, TN]])
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=False, cmap='Blues', ax=ax,
                xticklabels=['Real Positive', 'Real Negative'],
                yticklabels=['Predicted Positive', 'Predicted Negative'],
                linewidths=0.5, square=True, vmin=0, vmax=cm.max() * 3)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j+0.5, i+0.5, f'{cm[i,j]:,}', ha='center', va='center',
                    fontsize=14, fontweight='bold', color='black')
    ax.set_title(f'Confusion Matrix — Two-Stage | {VERSION} | {SIM_START_DATE} -> {SIM_END_DATE}',
                 fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel('REAL', fontsize=12); ax.set_ylabel('PREDICTED', fontsize=12)
    plt.tight_layout()
    _savefig(fig, "stage9_confusion_matrix_twostage")

    prec_s1_c, rec_s1_c, _ = precision_recall_curve(yt, s1prob)
    prec_s2_c, rec_s2_c, _ = precision_recall_curve(yt, yprob)
    prevalence = yt.mean()
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.axhline(prevalence, color='grey', linestyle=':', alpha=0.6,
               label=f'Random baseline (prevalence={prevalence:.5f})')
    ax.plot(rec_s1_c, prec_s1_c, label=f'Stage 1   (AUCPR={aucpr_s1:.4f})',
            color='steelblue', linewidth=1.5)
    ax.plot(rec_s2_c, prec_s2_c, label=f'Two-Stage (AUCPR={aucpr:.4f})',
            color='crimson', linewidth=1.5)
    ax.scatter([s1_rec], [s1_prec], s=120, c='steelblue', zorder=5, marker='*',
               edgecolors='black', linewidths=1,
               label=f't=0.5 S1: P={s1_prec:.4f}, R={s1_rec:.4f}')
    ax.scatter([rec_val], [prec_val], s=120, c='crimson', zorder=5, marker='*',
               edgecolors='black', linewidths=1,
               label=f't=0.5 S2: P={prec_val:.4f}, R={rec_val:.4f}')
    ax.set_xlabel('Recall (Sensitivity)', fontsize=12); ax.set_ylabel('Precision', fontsize=12)
    ax.set_title(f'Precision-Recall Curve | {VERSION} | {SIM_START_DATE} -> {SIM_END_DATE}',
                 fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.legend(fontsize=10, loc='upper right')
    plt.tight_layout()
    _savefig(fig, "stage9_precision_recall_twostage")

    summary = pd.DataFrame([
        {'model': 'Stage 1', 'sim_period': f'{SIM_START_DATE} -> {SIM_END_DATE}',
         'n_rows': len(df), 'TP': s1_tp, 'FP': s1_fp, 'FN': s1_fn, 'TN': s1_tn,
         'Precision': round(s1_prec, 4), 'Recall': round(s1_rec, 4),
         'F1': round(s1_f1, 4),
         'F2': round((1+4)*s1_prec*s1_rec/(4*s1_prec+s1_rec)
                     if (4*s1_prec+s1_rec) > 0 else 0, 4),
         'AUCPR': round(aucpr_s1, 4)},
        {'model': 'Two-Stage', 'sim_period': f'{SIM_START_DATE} -> {SIM_END_DATE}',
         'n_rows': len(df), 'TP': TP, 'FP': FP, 'FN': FN, 'TN': TN,
         'Precision': round(prec_val, 4), 'Recall': round(rec_val, 4),
         'F1': round(f1_val, 4), 'F2': round(f2_val, 4), 'AUCPR': round(aucpr, 4)},
    ]).set_index('model')
    print('\nSimulation evaluation metrics @ threshold=0.5:')
    print(summary.to_string())


# =============================================================================
# STAGE 10: EDA — FEATURE ENGINEERING ANALYSIS
# =============================================================================

def stage10_eda(X_train_xgb, X_test_xgb, y_train_acc, y_test_acc,
                available_features, ensemble_preds, STAGE1_THRESHOLD):
    print('=' * 80)
    print('STAGE 10: Feature Engineering EDA')
    print('=' * 80)

    # --- 10A: Feature Quality Audit ---
    print('\n--- 10A: Feature Quality Audit ---')
    df_audit = X_train_xgb.copy(); df_audit['ACCIDENT'] = y_train_acc.values
    feat_stats = []
    for col in available_features:
        s = df_audit[col]
        acc_mean    = df_audit.loc[df_audit['ACCIDENT']==1, col].mean()
        nonacc_mean = df_audit.loc[df_audit['ACCIDENT']==0, col].mean()
        feat_stats.append({'feature': col, 'std': s.std(), 'zero_rate': (s==0).mean(),
                           'nan_rate': s.isna().mean(), 'n_unique': s.nunique(),
                           'mean_acc': acc_mean, 'mean_nonacc': nonacc_mean,
                           'mean_diff': abs(acc_mean - nonacc_mean)})
    df_fstats = pd.DataFrame(feat_stats).set_index('feature').sort_values('std')
    print("Near-zero variance features (std < 0.01):")
    print(df_fstats[df_fstats['std'] < 0.01][['std','zero_rate','n_unique']].to_string())
    print("\nHigh zero-rate features (>80% zeros):")
    print(df_fstats[df_fstats['zero_rate'] > 0.80].sort_values('zero_rate', ascending=False)
          [['zero_rate','std','mean_acc','mean_nonacc']].to_string())

    top_shap = [f for f in ['pk_crash_rate_mob','mean_speed','speed_zscore','pk_crash_rate',
                             'vol_zscore','speed_std_4','curv_x_speed','dia','flow_regime',
                             'vol_lag1','delta_speed_2','wind_gust_excess','pend_x_speed',
                             'pk_crash_rate_log','speed_mean_4','mes','vol_cv_2','vol_grad_up',
                             '5min','speed_grad_dn'] if f in available_features]
    if top_shap:
        corr_mat = df_audit[top_shap].corr(method='spearman')
        fig, ax  = plt.subplots(figsize=(14, 12))
        mask     = np.triu(np.ones_like(corr_mat, dtype=bool))
        sns.heatmap(corr_mat, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                    center=0, vmin=-1, vmax=1, ax=ax, linewidths=0.3, annot_kws={'size': 7})
        ax.set_title('Spearman Correlation — Top SHAP Features', fontweight='bold')
        plt.tight_layout()
        _savefig(fig, "eda10a_correlation_heatmap")
        print("\nHighly correlated pairs (|r| > 0.7):")
        for i in range(len(corr_mat)):
            for j in range(i+1, len(corr_mat)):
                r = corr_mat.iloc[i, j]
                if abs(r) > 0.7:
                    print(f"  {corr_mat.index[i]:30s} <-> {corr_mat.columns[j]:30s}  r={r:.3f}")

    # --- 10B: Separability ---
    print('\n--- 10B: Per-Feature Separability ---')
    acc_mask_tr  = y_train_acc.values == 1
    nacc_mask_tr = y_train_acc.values == 0
    X_arr_tr     = X_train_xgb[available_features].values
    feat_sep = []
    for fi, col in enumerate(available_features):
        pos_vals = X_arr_tr[acc_mask_tr, fi]; neg_vals = X_arr_tr[nacc_mask_tr, fi]
        pooled_std = np.sqrt((pos_vals.std()**2 + neg_vals.std()**2) / 2)
        cohens_d   = abs(pos_vals.mean() - neg_vals.mean()) / (pooled_std + 1e-9)
        ks_stat, _ = ks_2samp(pos_vals, neg_vals, alternative='two-sided')
        feat_sep.append({'feature': col, 'cohens_d': cohens_d, 'ks_stat': ks_stat,
                         'mean_acc': pos_vals.mean(), 'mean_nonacc': neg_vals.mean()})
    df_sep = pd.DataFrame(feat_sep).set_index('feature').sort_values('cohens_d', ascending=False)
    print("Feature separability (sorted by Cohen's d):")
    print(df_sep[['cohens_d','ks_stat','mean_acc','mean_nonacc']].to_string())

    top25 = df_sep.head(25).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    axes[0].barh(top25['feature'][::-1], top25['cohens_d'][::-1], color='steelblue')
    for xv, lbl in [(0.2,'small'),(0.5,'medium'),(0.8,'large')]:
        axes[0].axvline(xv, linestyle='--', alpha=0.7, label=f'{lbl} ({xv})')
    axes[0].set_xlabel("Cohen's d"); axes[0].set_title("Separability: Cohen's d (top 25)", fontweight='bold')
    axes[0].legend(fontsize=8)
    axes[1].barh(top25['feature'][::-1], top25['ks_stat'][::-1], color='crimson')
    axes[1].axvline(0.1, linestyle='--', alpha=0.7, color='orange', label='noticeable (0.1)')
    axes[1].set_xlabel("KS statistic"); axes[1].set_title("Separability: KS statistic (top 25)", fontweight='bold')
    axes[1].legend(fontsize=8)
    plt.suptitle("Per-Feature Accident vs Non-Accident Separability (train set)", fontsize=13, fontweight='bold')
    plt.tight_layout()
    _savefig(fig, "eda10b_separability_bar")

    top8 = df_sep.head(8).index.tolist()
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    for ax, feat in zip(axes.flatten(), top8):
        fi = available_features.index(feat)
        lo, hi = np.percentile(X_arr_tr[:, fi], [1, 99])
        ax.hist(np.clip(X_arr_tr[nacc_mask_tr, fi], lo, hi), bins=40, alpha=0.5,
                density=True, color='steelblue', label='No accident')
        ax.hist(np.clip(X_arr_tr[acc_mask_tr,  fi], lo, hi), bins=40, alpha=0.7,
                density=True, color='crimson',   label='Accident')
        ax.set_title(f'{feat}\nd={df_sep.loc[feat,"cohens_d"]:.3f}', fontsize=9)
        ax.legend(fontsize=7); ax.set_yticks([])
    plt.suptitle("Feature Distributions: Accident vs Non-Accident (top 8 by Cohen's d)",
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    _savefig(fig, "eda10b_feature_distributions_top8")

    # --- 10C: Error Analysis ---
    print('\n--- 10C: Error Analysis (FP vs TN, FN vs TP) ---')
    s1_thresh = STAGE1_THRESHOLD
    y_te  = y_test_acc.reset_index(drop=True).values
    X_te  = X_test_xgb.reset_index(drop=True)[available_features].values
    s1_prob = ensemble_preds
    fp_mask = (s1_prob >= s1_thresh) & (y_te == 0)
    tn_mask = (s1_prob <  s1_thresh) & (y_te == 0)
    tp_mask = (s1_prob >= s1_thresh) & (y_te == 1)
    fn_mask = (s1_prob <  s1_thresh) & (y_te == 1)
    print(f"Test set: TP={tp_mask.sum()}  FP={fp_mask.sum()}  FN={fn_mask.sum()}  TN={tn_mask.sum()}")

    fp_tn_sep = []
    for fi, col in enumerate(available_features):
        fp_v = X_te[fp_mask, fi]; tn_v = X_te[tn_mask, fi]
        pooled_std = np.sqrt((fp_v.std()**2 + tn_v.std()**2) / 2)
        d  = abs(fp_v.mean() - tn_v.mean()) / (pooled_std + 1e-9)
        ks, _ = ks_2samp(fp_v, tn_v)
        fp_tn_sep.append({'feature': col, 'cohens_d': d, 'ks_stat': ks,
                          'mean_fp': fp_v.mean(), 'mean_tn': tn_v.mean()})
    df_fp_tn = pd.DataFrame(fp_tn_sep).set_index('feature').sort_values('cohens_d', ascending=False)
    print("\nFP vs TN (top 20 distinguishing features):")
    print(df_fp_tn.head(20)[['cohens_d','ks_stat','mean_fp','mean_tn']].to_string())

    top12_fptn = df_fp_tn.head(12).index.tolist()
    fig, axes = plt.subplots(3, 4, figsize=(20, 12))
    for ax, feat in zip(axes.flatten(), top12_fptn):
        fi = available_features.index(feat)
        lo, hi = np.percentile(X_te[:, fi], [1, 99])
        ax.hist(np.clip(X_te[tn_mask, fi], lo, hi), bins=30, alpha=0.5,
                density=True, color='steelblue', label=f'TN ({tn_mask.sum():,})')
        ax.hist(np.clip(X_te[fp_mask, fi], lo, hi), bins=30, alpha=0.7,
                density=True, color='orange',    label=f'FP ({fp_mask.sum():,})')
        ax.set_title(f'{feat}\nd={df_fp_tn.loc[feat,"cohens_d"]:.3f}', fontsize=9)
        ax.legend(fontsize=7); ax.set_yticks([])
    plt.suptitle("FP vs TN Feature Distributions (top 12)", fontsize=12, fontweight='bold')
    plt.tight_layout()
    _savefig(fig, "eda10c1_fp_tn_distributions")

    fn_tp_sep = []
    for fi, col in enumerate(available_features):
        fn_v = X_te[fn_mask, fi]; tp_v = X_te[tp_mask, fi]
        if len(fn_v) == 0 or len(tp_v) == 0: continue
        pooled_std = np.sqrt((fn_v.std()**2 + tp_v.std()**2) / 2)
        d  = abs(fn_v.mean() - tp_v.mean()) / (pooled_std + 1e-9)
        ks, _ = ks_2samp(fn_v, tp_v)
        fn_tp_sep.append({'feature': col, 'cohens_d': d, 'ks_stat': ks,
                          'mean_fn': fn_v.mean(), 'mean_tp': tp_v.mean()})
    df_fn_tp = pd.DataFrame(fn_tp_sep).set_index('feature').sort_values('cohens_d', ascending=False)
    print(f"\nFN: {fn_mask.sum()}  TP: {tp_mask.sum()}")
    print("FN vs TP (top 20 distinguishing features):")
    print(df_fn_tp.head(20)[['cohens_d','ks_stat','mean_fn','mean_tp']].to_string())

    top12_fntp = df_fn_tp.head(12).index.tolist()
    fig, axes = plt.subplots(3, 4, figsize=(20, 12))
    for ax, feat in zip(axes.flatten(), top12_fntp):
        fi = available_features.index(feat)
        all_v = np.concatenate([X_te[fn_mask, fi], X_te[tp_mask, fi]])
        lo, hi = np.percentile(all_v, [1, 99])
        ax.hist(np.clip(X_te[tp_mask, fi], lo, hi), bins=20, alpha=0.5,
                density=True, color='steelblue', label=f'TP detected ({tp_mask.sum()})')
        ax.hist(np.clip(X_te[fn_mask, fi], lo, hi), bins=20, alpha=0.7,
                density=True, color='crimson',   label=f'FN missed ({fn_mask.sum()})')
        ax.set_title(f'{feat}\nd={df_fn_tp.loc[feat,"cohens_d"]:.3f}', fontsize=9)
        ax.legend(fontsize=7); ax.set_yticks([])
    plt.suptitle("FN vs TP Feature Distributions", fontsize=12, fontweight='bold')
    plt.tight_layout()
    _savefig(fig, "eda10c2_fn_tp_distributions")

    # --- 10D: Temporal Patterns ---
    print('\n--- 10D: Temporal Pattern Analysis ---')
    df_te = X_test_xgb.reset_index(drop=True).copy()
    df_te['ACCIDENT'] = y_te; df_te['s1_prob'] = s1_prob; df_te['pred_type'] = 'TN'
    df_te.loc[tp_mask, 'pred_type'] = 'TP'; df_te.loc[fp_mask, 'pred_type'] = 'FP'
    df_te.loc[fn_mask, 'pred_type'] = 'FN'

    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    for label, mask, color in [('TP detected', tp_mask, 'steelblue'), ('FN missed', fn_mask, 'crimson')]:
        cnt = df_te.loc[mask].groupby('hor').size().reindex(range(24), fill_value=0)
        axes[0,0].plot(cnt.index, cnt.values, marker='o', label=label, color=color)
    axes[0,0].set_xlabel('Hour of day'); axes[0,0].set_ylabel('Count')
    axes[0,0].set_title('Accidents by Hour: Detected vs Missed', fontweight='bold')
    axes[0,0].legend()
    fp_hour = df_te.loc[fp_mask].groupby('hor').size().reindex(range(24), fill_value=0)
    axes[0,1].bar(fp_hour.index, fp_hour.values, color='orange')
    axes[0,1].set_xlabel('Hour of day'); axes[0,1].set_ylabel('FP count')
    axes[0,1].set_title('False Positives by Hour of Day', fontweight='bold')
    for label, mask, color, offset in [('TP', tp_mask, 'steelblue', 0.0), ('FN', fn_mask, 'crimson', 0.4)]:
        cnt = df_te.loc[mask].groupby('diaSem').size().reindex(range(7), fill_value=0)
        axes[0,2].bar([x+offset for x in cnt.index], cnt.values, width=0.4, label=label, color=color, alpha=0.7)
    axes[0,2].set_xlabel('Day of week (0=Mon)'); axes[0,2].set_ylabel('Count')
    axes[0,2].set_xticks(range(7)); axes[0,2].set_xticklabels(['Mon','Tue','Wed','Thu','Fri','Sat','Sun'])
    axes[0,2].set_title('Accidents by Day of Week', fontweight='bold'); axes[0,2].legend()
    axes[1,0].hist(s1_prob[tp_mask], bins=30, alpha=0.7, density=True, color='steelblue', label=f'TP (n={tp_mask.sum()})')
    axes[1,0].hist(s1_prob[fn_mask], bins=30, alpha=0.7, density=True, color='crimson',   label=f'FN (n={fn_mask.sum()})')
    axes[1,0].axvline(s1_thresh, color='black', linestyle='--', label=f'thr={s1_thresh}')
    axes[1,0].set_xlabel('Stage 1 probability'); axes[1,0].set_title('Stage 1 Score: TP vs FN', fontweight='bold')
    axes[1,0].legend()
    for feat_name, ax_idx in [('pk_crash_rate_mob', (1,1)), ('mean_speed', (1,2))]:
        if feat_name in available_features:
            fi = available_features.index(feat_name)
            for vals, label, color in [(X_te[tp_mask, fi],'TP','steelblue'), (X_te[fn_mask, fi],'FN','crimson')]:
                lo, hi = np.percentile(vals, [1,99])
                axes[ax_idx].hist(np.clip(vals,lo,hi), bins=25, alpha=0.7, density=True, color=color, label=label)
            axes[ax_idx].set_xlabel(feat_name)
            axes[ax_idx].set_title(f'{feat_name}: TP vs FN', fontweight='bold')
            axes[ax_idx].legend()
    plt.suptitle('Temporal and Operational Patterns — Detected vs Missed Accidents',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    _savefig(fig, "eda10d_temporal_patterns")

    print("\nFN temporal breakdown:")
    for col in ['hor', 'diaSem', 'mes']:
        if col in df_te.columns:
            print(f"\n{col}:")
            print(f"  TP: {df_te.loc[tp_mask, col].value_counts().sort_index().to_dict()}")
            print(f"  FN: {df_te.loc[fn_mask, col].value_counts().sort_index().to_dict()}")

    # --- 10E: Recommendations ---
    print('\n--- 10E: Feature Engineering Recommendations ---')
    print("=" * 80); print("FEATURE GAP ANALYSIS"); print("=" * 80)
    print("\nTop-15 features by Cohen's d:"); print(df_sep[['cohens_d','ks_stat']].head(15).to_string())
    low_var = df_fstats[(df_fstats['std'] < 0.01) | (df_fstats['zero_rate'] > 0.95)]
    print("\nProblematic features (near-zero variance or >95% zeros):")
    print(low_var[['std','zero_rate']].to_string() if len(low_var) else "None found.")
    print("\nModel blind spots (FN differ most from TP):")
    print(df_fn_tp.head(10)[['cohens_d','mean_fn','mean_tp']].to_string())
    print("\nFalse alarm triggers (FP differ most from TN):")
    print(df_fp_tn.head(10)[['cohens_d','mean_fp','mean_tn']].to_string())
    recommendations = [
        ("HIGH",   "Longer lag features (1h, 2h)",
         "speed_lag12/24, vol_lag12/24, speed_std_12/24, speed_trend_1h.",
         "Current lags stop at 20 min. Accidents follow gradual speed deterioration."),
        ("HIGH",   "Time-of-day x segment crash rate",
         "pk_hour_crash_rate, pk_weekday_crash_rate per (pk, hour_bin).",
         "pk_crash_rate is a lifetime average; accidents cluster at specific time slots."),
        ("MEDIUM", "Cross-segment upstream/downstream speed",
         "upstream_speed_delta, downstream_speed_delta, upstream_vol_ratio.",
         "Congestion propagates; upstream slowdown predicts current-segment risk."),
        ("MEDIUM", "Incident recency",
         "hours_since_last_accident_on_pk, recent_accident_within_1h.",
         "Secondary accident risk is elevated after an incident."),
        ("MEDIUM", "Weather data fix + continuous values",
         "Fix precip_change_1d_vs_3d (std=0). Add precipitation mm/h, temperature C.",
         "Binary weather flags discard magnitude; current merge may have a date-alignment bug."),
        ("LOW",    "Speed drop asymmetry",
         "speed_drop_max_4, n_speed_drops_4, speed_recover_flag.",
         "Symmetric std treats brake waves and smooth deceleration equally."),
    ]
    print("\n" + "=" * 80 + "\nRECOMMENDATIONS\n" + "=" * 80)
    for priority, name, candidates, rationale in recommendations:
        print(f"\n[{priority}] {name}")
        print(f"  Candidates : {candidates}")
        print(f"  Rationale  : {rationale}")
    print("=" * 80)


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_pipeline():
    t_start = time.time()
    print("=" * 80)
    print(f"V34 XGBoost Two-Stage OOF Simulation Pipeline")
    print(f"Training:   {TRAIN_START_DATE} -> {TRAIN_END_DATE}")
    print(f"Simulation: {SIM_START_DATE}   -> {SIM_END_DATE}")
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"Viz dir:    {VIZ_DIR}")
    print("=" * 80)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(VIZ_DIR, exist_ok=True)

    # Stage 0: Load data
    df_full, fe_only_cols = stage0_load_data()

    # Stage 1: Feature prep
    (df_train, X_train_xgb, X_test_xgb, y_train_acc, y_test_acc,
     available_features, model_time) = stage1_feature_prep(df_full, fe_only_cols)

    # df_train is a 20 GB copy that is no longer needed once X_train/X_test are built.
    # Free it now so ensemble training and OOF have ~20 GB more headroom.
    del df_train
    gc.collect()
    print(f'[MEM] df_train released. Proceeding to ensemble training.')

    # Stage 2: Train ensemble
    (base_models, ensemble_preds, y_pred_binary, roc_auc, pr_auc,
     s1_best_params) = stage2_train_ensemble(
        X_train_xgb, X_test_xgb, y_train_acc, y_test_acc, available_features, model_time)
    gc.collect()
    print(f'[MEM] Optuna study released (s1_best_params extracted).')

    # Stage 3: Stage 1 simulation
    sim_results = stage3_simulation(df_full, base_models, available_features, model_time)

    # Stage 4: Evaluate Stage 1 simulation
    stage4_evaluate_simulation(sim_results)

    # Stage 5: Feature importance + SHAP (Stage 1)
    stage5_feature_importance_shap(base_models, X_test_xgb, available_features)

    # Stage 6: Feature-space overlap
    stage6_feature_space_overlap(X_train_xgb, X_test_xgb, y_train_acc, y_test_acc,
                                 available_features, ensemble_preds)

    # Stage 7: Two-stage OOF training
    (stage2_model, X_s2_tr, y_s2_tr, s2_full_prob, s2_full_bin,
     y_test_np, flagged_idx, s2_test_probs,
     STAGE1_THRESHOLD, STAGE2_THRESHOLD,
     MIN_RECALL_CONSTRAINT, MAX_FAR_TARGET) = stage7_two_stage(
        X_train_xgb, X_test_xgb, y_train_acc, y_test_acc,
        available_features, ensemble_preds, base_models, model_time, s1_best_params)

    # Stage 7B: Stage 2 SHAP
    stage7b_shap_stage2(stage2_model, X_s2_tr, y_s2_tr)

    # Stage 7D: Threshold sweep
    S1_THR, S2_THR = stage7d_threshold_sweep(
        ensemble_preds, y_test_np, flagged_idx, s2_test_probs,
        MIN_RECALL_CONSTRAINT, MAX_FAR_TARGET)

    # Stage 8: Two-stage simulation
    sim_results_2s = stage8_two_stage_simulation(
        df_full, base_models, stage2_model, available_features, model_time, S1_THR, S2_THR)

    # Stage 9: Final merged evaluation
    stage9_final_evaluation(sim_results, sim_results_2s)

    # Stage 10: EDA
    stage10_eda(X_train_xgb, X_test_xgb, y_train_acc, y_test_acc,
                available_features, ensemble_preds, STAGE1_THRESHOLD)

    print("\n" + "=" * 80)
    print(f"Pipeline complete in {format_duration(time.time() - t_start)}")
    print(f"All plots saved to: {VIZ_DIR}")
    print("=" * 80)
    return True



# =============================================================================
# CONFIGURATION
# =============================================================================

PIPELINE_VERSION = "v50_xgboost_only"

# Output directory for the V50 XGBoost training artefacts.  Override via
# environment variable when launching from a SLURM script.  When invoked
# from `run_ablation_study_xgboost_only.sh` we instead write into the
# GNN experiment dir (PREVIOUS_EXPERIMENT_DIR) — this matches the v6+
# convention that the simulation runner expects.
OUTPUT_DIR = os.environ.get(
    "V50_XGBOOST_OUTPUT_DIR",
    f"{_AP7_EXPERIMENTS}/v50_xgb_with_gnn",
)

# GNN experiment dir is the only required CLI argument.  Bash runner sets
# this via `PREVIOUS_EXPERIMENT_DIR` (resolved from --gnn_v14 + --run-id=...).
GNN_EXPERIMENT_DIR = os.environ.get(
    "GNN_EXPERIMENT_DIR", os.environ.get("PREVIOUS_EXPERIMENT_DIR", "")
)

# ──────────────────────────────────────────────────────────────────────────────
# Date / window configuration — aligned with the V17 GNN run
# (`log_output/output_gnn_only_2538314.log`).  The V17 GNN was trained on
# two disjoint 1-month windows; we restrict XGBoost training to the union
# of those windows so the GNN predictions feeding XGBoost stay in-
# distribution.  Simulation covers the whole month of June 2025.
#
# These values OVERRIDE whatever `run_ablation_study_xgboost_only.sh` writes
# onto the module — `run_xgboost_only()` re-pins them after the bash runner
# has finished mutating the module attributes (see `_V17_ALIGNED_*` below).
# ──────────────────────────────────────────────────────────────────────────────
_V17_ALIGNED_TRAIN_START = "2024-06-01"
_V17_ALIGNED_TRAIN_END   = "2025-06-01"
_V17_ALIGNED_SIM_START   = "2025-06-01"
_V17_ALIGNED_SIM_END     = "2025-07-01"   # whole June 2025

# Disjoint training windows that match the V17 GNN training distribution.
# Rows outside these windows (and outside the simulation window) are dropped
# from `df_full` after feature engineering.
GNN_TRAIN_WINDOWS = [
    ("2024-06-01", "2024-07-01"),
    ("2025-05-01", "2025-06-01"),
]
# Env-overridable per job (data-quantity ablation, rolling-origin folds), e.g.
#   BENCH_TRAIN_WINDOWS="2024-05-01:2024-06-01;2025-04-01:2025-05-01"
# Windows are separated by ';' — NOT ',', because SLURM's --export splits its
# value on commas and would silently truncate the list to its first entry.
_env_tw = os.environ.get("BENCH_TRAIN_WINDOWS", "")
if _env_tw:
    _sep = ";" if ";" in _env_tw else ","
    GNN_TRAIN_WINDOWS = [tuple(w.split(":", 1)) for w in _env_tw.split(_sep) if ":" in w]
    print(f"[V51] BENCH_TRAIN_WINDOWS override active: {GNN_TRAIN_WINDOWS}")

TRAIN_START_DATE = _V17_ALIGNED_TRAIN_START
TRAIN_END_DATE   = _V17_ALIGNED_TRAIN_END
SIM_START_DATE   = _V17_ALIGNED_SIM_START
SIM_END_DATE     = _V17_ALIGNED_SIM_END

# NOTE: this MUST come after the V17-aligned assignments above, which would
# otherwise overwrite it. The base CSV is loaded over
# [min(TRAIN_START, SIM_START), max(TRAIN_END, SIM_END)]; if the overridden
# windows fall outside the V17-aligned dates they are filtered away before they
# can match, leaving an empty training split (ZeroDivisionError in
# stage1_feature_prep). Widen the train dates to span the requested windows.
if _env_tw and GNN_TRAIN_WINDOWS:
    TRAIN_START_DATE = min(w[0] for w in GNN_TRAIN_WINDOWS)
    TRAIN_END_DATE   = max(w[1] for w in GNN_TRAIN_WINDOWS)
    print(f"[V51]   train date span widened to {TRAIN_START_DATE} -> {TRAIN_END_DATE}")
PK_MIN           = PK_MIN
PK_MAX           = PK_MAX
TIME_RESOLUTION  = TIME_RESOLUTION

VIZ_DIR = os.path.join(OUTPUT_DIR, "visualizations")

# ── Module-level attrs that `run_ablation_study_xgboost_only.sh` mutates ─────
# The bash runner does e.g. `ablation.DATA_FILE = ...` and
# `base.ABLATION_SETTINGS['include_weather_1d'] = ...` (where `base` is v5).
# We share the underlying dicts with v34 so dict-level mutations propagate
# automatically; scalars are propagated explicitly inside run_xgboost_only().
TEST_MODE         = TEST_MODE
MEMORIZE_MODE     = MEMORIZE_MODE        # XGB train-on-everything overfit mode
ABLATION_SETTINGS = ABLATION_SETTINGS    # shared dict
FEATURE_GROUPS    = FEATURE_GROUPS       # shared dict
XGBOOST_CONFIG    = XGBOOST_CONFIG       # shared dict
DATA_FILE         = DATA_FILE
DATA_PATH         = str(DATA_PATH)


# Resolved by stage0 from GNN_TRAIN_WINDOWS (env-overridable via
# $BENCH_TRAIN_TEST_BOUNDARY); drives BOTH the chronological train/test split
# in stage1_feature_prep and the FE-baseline cutoff, so they cannot disagree.
TRAIN_TEST_BOUNDARY = None


# =============================================================================
# GNN-BYPASS FRAME ALIGNMENT
# =============================================================================
#
# Everything downstream of stage 0 slices the frame *positionally*:
#   * `build_engineered_features` estimates its (pk, hor, diaSem) baselines --
#     z-score mu/sigma, `pk_crash_rate`, free-flow `v_ff` -- from
#     `df.iloc[:int(0.70*n)]`,
#   * `stage1_feature_prep` splits train/test at `int(len(X) * 0.8)`.
#
# So the row LAYOUT, not just the row contents, decides what the model is
# trained on and tested against. The GNN branch reshapes that layout twice
# inside `generate_gnn_predictions_on_training_data` (v5):
#
#   1. it re-sorts by (via, sen, pk, dat)  [v5:668]. `via` is constant (AP-7
#      maps to 0), so the effective order is sen-major -- while stage 0A-bis
#      leaves the frame pk-major, sorted by (pk, sen, dat);
#   2. it produces no prediction for the first `seq_len` rows of each
#      (via, sen, pk) group, nor for groups no longer than `seq_len`
#      [v5:576-595]; the caller drops those with dropna(mean_speed_gnn).
#
# The bypass branches (MEMORIZE / OBSERVED / LAGGED) used to do neither, and
# the resulting pk-major layout silently wrecked the comparison: the baseline
# slice covered only PKs 120-188, so 29.8% of rows got z-score NaN (-> 0) and
# `pk_crash_rate` = 0.0, and the internal test split became PKs 199-219 with
# 20 of its 21 PKs completely absent from training -- an extrapolation set,
# not a held-out set. That is what drove the observed-traffic arm to a test
# PR-AUC of 0.006 while its June simulation still read 0.38.
#
# These two helpers reproduce the GNN layout for the bypass branches so the
# arms differ in the traffic values ALONE: same rows, same 70% baseline
# cutoff, same 80/20 boundary, same test set.
# =============================================================================

_GNN_LAYOUT_LOC_COLS = list(_v21_mod._LOCATION_KEYS)

# The canonical layout lives in the FE module (the one that requires it); this
# name is kept because the simulation script and older callers import it.
_align_to_gnn_layout = _v21_mod.sort_location_major


def _drop_gnn_warmup_rows(df, seq_len, label="V50"):
    """Drop exactly the rows the GNN branch has no prediction for.

    Mirrors `dropna(subset=["mean_speed_gnn"])` in the GNN branch: the first
    `seq_len` rows of every (via, sen, pk) group, plus any group not longer
    than `seq_len`. Rows whose aliased traffic is already NaN (the lagged
    mode's shift warm-up) go with them, which is the drop the lagged branch
    documented but never actually performed.

    `df` must already be in `_align_to_gnn_layout` order.
    """
    loc_cols = [c for c in _GNN_LAYOUT_LOC_COLS if c in df.columns]
    if not loc_cols or seq_len <= 0:
        return df
    grp = df.groupby(loc_cols, sort=False)
    # pos-from-start + pos-from-end + 1 == group size; derived from cumcount
    # alone so we never select a grouping key out of the groupby (which pandas
    # treats inconsistently across versions).
    pos = grp.cumcount()
    grp_size = pos + grp.cumcount(ascending=False) + 1
    keep = (pos >= seq_len) & (grp_size > seq_len)
    if "mean_speed_gnn" in df.columns:
        keep &= df["mean_speed_gnn"].notna()
    n_before = len(df)
    df = df[keep].reset_index(drop=True)
    print(f"[{label}] Dropped {n_before - len(df):,} warm-up rows (first {seq_len} "
          f"of each (via, sen, pk) group) — mirrors the GNN branch's dropna")
    return df


# =============================================================================
# STAGE 0 (v50): LOAD DATA + GNN INFERENCE + V21 FEATURE ENGINEERING
# =============================================================================

def stage0_load_data_with_gnn(gnn_experiment_dir):
    """Load base CSV, run V14 GNN inference, recompute V21 feature engineering.

    Returns (df_full, fe_only_cols) — same contract as v34.stage0_load_data
    so the downstream V34 stages are drop-in compatible.
    """
    print("=" * 80)
    print("V50 STAGE 0A: BASE CSV LOADING")
    print("=" * 80)

    # ── 0A. Load full base CSV (raw mean_speed/intTot/intP) ───────────────
    _date_min = min(pd.Timestamp(TRAIN_START_DATE), pd.Timestamp(SIM_START_DATE))
    _date_max = max(pd.Timestamp(TRAIN_END_DATE),   pd.Timestamp(SIM_END_DATE))

    t0 = time.time()
    df_full, _ = load_data(
        pk_min=PK_MIN, pk_max=PK_MAX,
        date_min=_date_min, date_max=_date_max,
    )
    print(f"[V50] Base CSV loaded in {time.time()-t0:.1f}s — {len(df_full):,} rows")

    # ── 0A-bis. Window filter (V17 GNN alignment) BEFORE GNN inference ────
    # Restricting to the V17 GNN training windows + June-2025 sim window now
    # (rather than after FE) keeps GNN sequence creation memory-bounded — going
    # from ~23M rows down to ~5M before the per-(pk, sen) sequence expansion.
    # Boundary effect: ~12 rows × n_pks at the May→Jun-2024 / Apr→May-2025 gap
    # transitions will get GNN sequences that span the gap — those predictions
    # are bad but get partly washed out by the dropna step in 0C and the V21
    # baseline math; this matches behaviour the V17 GNN already saw at training
    # time (windows trained disjointly).
    print("=" * 80)
    print("V50 STAGE 0A-bis: WINDOW FILTERING (V17 GNN ALIGNMENT)")
    print("=" * 80)
    df_full["dat"] = pd.to_datetime(df_full["dat"])
    keep_mask = pd.Series(False, index=df_full.index)
    for w_start, w_end in GNN_TRAIN_WINDOWS:
        m = (df_full["dat"] >= pd.Timestamp(w_start)) & (df_full["dat"] < pd.Timestamp(w_end))
        keep_mask |= m
        print(f"  + GNN train window {w_start} → {w_end}: {m.sum():,} rows")
    sim_mask = (
        (df_full["dat"] >= pd.Timestamp(SIM_START_DATE))
        & (df_full["dat"] < pd.Timestamp(SIM_END_DATE))
    )
    keep_mask |= sim_mask
    print(f"  + Sim window      {SIM_START_DATE} → {SIM_END_DATE}: {sim_mask.sum():,} rows")
    n_before = len(df_full)
    df_full = df_full[keep_mask].copy()
    # Canonical (via, sen, pk, dat) layout — location first, timestamp second —
    # so v5's GNN sequence creation produces sane windows and every downstream
    # lag sees consecutive time steps. Also resets the index to a contiguous
    # range, which the FE stage assumes. Deduplicates the ~5.3% double-mapped
    # segment-intervals the raw CSV carries (see `dedup_segment_intervals`);
    # returns in canonical order.
    df_full = _v21_mod.dedup_segment_intervals(df_full, label="V50")
    print(f"[V50] Window filter: {n_before:,} → {len(df_full):,} rows "
          f"({n_before - len(df_full):,} dropped)")
    sys.stdout.flush()

    # Make _v5_load_data / get_time_resolution_minutes see V34's data file
    # (also needed by `_v5_get_time_resolution_minutes` below in memorize mode).
    _v5_mod.DATA_FILE = DATA_FILE
    _v5_mod.DATA_PATH = str(DATA_PATH)
    _v21_mod.DATA_FILE = DATA_FILE
    _v21_mod.DATA_PATH = str(DATA_PATH)

    # GNN inference + traffic replacement are skipped under MEMORIZE_MODE so
    # the booster trains on the *real* mean_speed / intTot / intP columns.
    # The matching simulation script (`ablation_study_v51_simulation_only.py`)
    # applies the same skip, so train and sim see identical inputs.
    _LAG_BINS = int(os.environ.get("BENCH_LAGGED_TRAFFIC", "0"))  # >0 = lagged OBSERVED traffic, no forecast
    # BENCH_OBSERVED_TRAFFIC=1 substitutes the CONTEMPORANEOUS observed traffic
    # for the forecast, giving the real-time upper bound: identical classifier,
    # identical training windows, identical protocol, only the traffic source
    # differs. It deliberately does NOT reuse MEMORIZE_MODE, which additionally
    # collapses the windows to May-2025 and forces overfit hyper-parameters --
    # both wrong for this comparison. Must be passed to the simulation too.
    _OBS_TRAFFIC = os.environ.get("BENCH_OBSERVED_TRAFFIC", "0") == "1"
    # BENCH_UNION_TRAFFIC=1 keeps the forecast AND the observation side by side,
    # plus their difference, in one feature vector. Section 5.2 reports the two
    # as non-nested arms and reads the forecast's advantage as denoising; if
    # that reading is right, a model given both should dominate either and lean
    # little on the difference term. Unlike the observed-traffic flag this does
    # NOT bypass the forecaster: the GNN branch runs, and the observation is
    # carried alongside it.
    _UNION_TRAFFIC = os.environ.get("BENCH_UNION_TRAFFIC", "0") == "1"
    if MEMORIZE_MODE or _LAG_BINS > 0 or _OBS_TRAFFIC:
        _mode = (f"LAGGED-TRAFFIC (t-{_LAG_BINS} bins, no forecasting model)" if _LAG_BINS > 0
                 else ("OBSERVED-TRAFFIC (real-time upper bound)" if _OBS_TRAFFIC
                       else "MEMORIZE_MODE — real traffic"))
        print("=" * 80)
        print(f"V50 STAGE 0B/0C/0D: SKIPPED ({_mode}) — GNN deactivated, traffic aliased into *_gnn")
        print("=" * 80)
        gnn_model_time   = None
        gnn_metadata     = {}
        interval_minutes = _v5_get_time_resolution_minutes()
        seq_len          = SEQUENCE_LENGTH_BY_INTERVAL.get(interval_minutes, 8)
        # build_engineered_features gates lag/diff/std/zscore/gradient/flow/hv/
        # geometry-interaction/wet_speed groups on mean_speed_gnn / intTot_gnn /
        # intP_gnn. Alias real (MEMORIZE) or LAGGED-real (lag mode) traffic into
        # those names so the 48 FE cols are built identically to the GNN pipeline.
        #
        # Adopt the GNN branch's row layout FIRST: every positional consumer
        # downstream (the 70% FE baseline cutoff, the 80/20 split) keys off it,
        # so this is what makes the arms comparable. See the block comment on
        # `_align_to_gnn_layout` above.
        df_full = _align_to_gnn_layout(df_full)
        if _LAG_BINS > 0:
            # Inject lagged observed traffic (e.g. t-168h = same weekday/hour one
            # week prior) as the "forecast" — traffic without a forecasting model.
            # Per-(via, sen, pk) chronological shift; the first _LAG_BINS rows of
            # each group become NaN and are dropped just below, like the GNN
            # warm-up.
            _loc = [c for c in _GNN_LAYOUT_LOC_COLS if c in df_full.columns]
            for orig, gnn_col in (("mean_speed", "mean_speed_gnn"),
                                  ("intTot",     "intTot_gnn"),
                                  ("intP",       "intP_gnn")):
                if orig in df_full.columns:
                    df_full[gnn_col] = df_full.groupby(_loc)[orig].shift(_LAG_BINS)
                    print(f"  {gnn_col:<14s} ←  {orig}.shift({_LAG_BINS}) (lagged observed)")
        else:
            for orig, gnn_col in (("mean_speed", "mean_speed_gnn"),
                                  ("intTot",     "intTot_gnn"),
                                  ("intP",       "intP_gnn")):
                if orig in df_full.columns:
                    df_full[gnn_col] = df_full[orig]
                    print(f"  {gnn_col:<14s} ←  {orig} (alias)")
        # Same warm-up rows the GNN branch loses to its sequence window, so the
        # row count — and therefore both positional cutoffs — match exactly.
        df_full = _drop_gnn_warmup_rows(df_full, seq_len, label="V50")
        # The GNN branch recomputes `car` from the replaced traffic columns
        # (Stage 0D). Without this the skip branch leaves `car` at its raw-CSV
        # value, which is a constant lane count -- a dead feature -- so the two
        # paths would not be comparable. Mirror it exactly.
    else:
        # ── 0B. Load V14 GNN model from the supplied experiment dir ───────
        print("=" * 80)
        print("V50 STAGE 0B: GNN MODEL LOADING (V14)")
        print("=" * 80)

        if not gnn_experiment_dir or not os.path.isdir(gnn_experiment_dir):
            raise FileNotFoundError(
                f"GNN experiment dir not found: {gnn_experiment_dir!r}. "
                "Pass --gnn-experiment-dir or set $GNN_EXPERIMENT_DIR."
            )

        gnn_model_time = _v5_detect_model_time(gnn_experiment_dir)
        print(f"[V50-GNN] Experiment dir : {gnn_experiment_dir}")
        print(f"[V50-GNN] Detected model_time: {gnn_model_time}")

        (gnn_model, gnn_metadata, gnn_info,
         scaler_temporal, scaler_static, scaler_targets) = (
            _v5_load_pretrained_gnn_model(gnn_experiment_dir, gnn_model_time)
        )

        interval_minutes = _v5_get_time_resolution_minutes()
        seq_len = SEQUENCE_LENGTH_BY_INTERVAL.get(interval_minutes, 8)
        print(f"[V50-GNN] Interval {interval_minutes}min  →  sequence_length = {seq_len}")

        # ── 0C. Run GNN inference (mean_speed_gnn / intTot_gnn / intP_gnn) ─
        print("=" * 80)
        print("V50 STAGE 0C: GNN INFERENCE")
        print("=" * 80)

        df_full = _v5_generate_gnn_predictions(
            gnn_model, df_full,
            scaler_temporal, scaler_static, scaler_targets,
            gnn_metadata, sequence_length=seq_len,
            experiment_dir=gnn_experiment_dir,
        )
        n_before = len(df_full)
        df_full = df_full.dropna(subset=["mean_speed_gnn"]).copy()
        print(f"[V50-GNN] Dropped {n_before - len(df_full):,} rows without GNN predictions "
              f"(first {seq_len} rows of each (pk, sen) group)")

        # Free GPU memory used by the GNN — not needed for XGBoost training.
        try:
            import torch
            del gnn_model
            torch.cuda.empty_cache()
        except Exception:
            pass

        # ── 0D. Replace traffic features with GNN predictions ─────────────
        print("=" * 80)
        print("V50 STAGE 0D: TRAFFIC FEATURE REPLACEMENT")
        print("=" * 80)

        if _UNION_TRAFFIC:
            for _c in ("mean_speed", "intTot", "intP"):
                if _c in df_full.columns:
                    df_full[f"{_c}_obs"] = df_full[_c].astype("float32")
            print("  [union] observation snapshotted before replacement")

        replacements = {"mean_speed": "mean_speed_gnn",
                        "intTot":     "intTot_gnn",
                        "intP":       "intP_gnn"}
        for orig, gnn_col in replacements.items():
            if orig in df_full.columns and gnn_col in df_full.columns:
                df_full[orig] = df_full[gnn_col]
                print(f"  {orig:<10s}  ←  {gnn_col}")

    # ── 0E. V21 feature engineering on GNN-predicted traffic ──────────────
    print("=" * 80)
    print("V50 STAGE 0E: V21 FEATURE ENGINEERING")
    print("=" * 80)
    print("[V50-FE] Feature engineering groups:")
    for k, on in FEATURE_ENGINEERING_CONFIG.items():
        print(f"  [{'ON ' if on else 'off'}] {k}")
    sys.stdout.flush()

    # Per-(pk, hor, diaSem) baselines end at the TRAIN/TEST BOUNDARY — the same
    # timestamp stage 1 splits on — so the baselines never see the internal
    # test rows OR the sim window. (History: this was a positional 70% prefix
    # of a location-major frame, which leaked 901 of June-2025's 1,072 labels
    # into `pk_crash_rate` and made features depend on sort order; the first
    # fix cut baselines at SIM_START_DATE, which un-leaked the sim but left
    # the internal test rows inside the baseline slice.) A date cutoff is
    # order-invariant, so the forecast and GNN-bypass arms get byte-identical
    # baselines. The simulation resolves the SAME boundary from the same
    # window list — mismatched baselines would mean scoring the model on
    # features it was not trained on.
    global TRAIN_TEST_BOUNDARY
    TRAIN_TEST_BOUNDARY = _v21_mod.resolve_train_test_boundary(
        GNN_TRAIN_WINDOWS, override=os.environ.get("BENCH_TRAIN_TEST_BOUNDARY"))
    print(f"[V50-FE] train/test boundary (split + baseline cutoff): "
          f"{TRAIN_TEST_BOUNDARY}")
    cols_before = set(df_full.columns)
    df_full, new_features = build_engineered_features(
        df_full, 0, baseline_end_date=TRAIN_TEST_BOUNDARY)
    fe_only_cols = [c for c in df_full.columns if c not in cols_before]
    print(f"[V50-FE] Engineered {len(fe_only_cols)} new feature columns")

    # BENCH_COV_ENGINEERED gives the direct (covariates-only) arm the same
    # engineering budget the forecast channels get, so Section 5.3(a)'s gap can
    # be attributed to the decomposition rather than to feature engineering.
    if os.environ.get("BENCH_COV_ENGINEERED", "0") == "1":
        df_full, _cov_cols = _v21_mod.build_covariate_engineered_features(
            df_full, baseline_end_date=TRAIN_TEST_BOUNDARY)
        fe_only_cols = list(fe_only_cols) + [c for c in _cov_cols
                                             if c not in fe_only_cols]

    if _UNION_TRAFFIC:
        _union_cols = []
        for _c in ("mean_speed", "intTot", "intP"):
            o, g = f"{_c}_obs", f"{_c}_gnn"
            if o in df_full.columns and g in df_full.columns:
                df_full[f"d_{_c}"] = (df_full[g] - df_full[o]).astype("float32")
                _union_cols += [o, f"d_{_c}"]
        fe_only_cols = list(fe_only_cols) + [c for c in _union_cols
                                             if c not in fe_only_cols]
        print(f"[union] added {len(_union_cols)} observation + difference "
              f"columns -> {_union_cols}")


    # ── 0F. Final shape report (mirrors v34.stage0_load_data tail) ────────
    print(f"Shape: {df_full.shape}")
    print(f"Date range: {df_full['dat'].min()} to {df_full['dat'].max()}")
    print(f"PKs present: {sorted(df_full['pk'].unique())}")
    if "ACCIDENT" in df_full.columns:
        print(f"ACCIDENT distribution:\n{df_full['ACCIDENT'].value_counts()}")

    # Save GNN provenance alongside XGBoost artefacts so the simulation
    # script knows which GNN+FE setup the model was trained on.
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "v50_gnn_provenance.json"), "w") as f:
        json.dump({
            "gnn_used":           not bool(MEMORIZE_MODE),
            "memorize_mode":      bool(MEMORIZE_MODE),
            "gnn_experiment_dir": None if MEMORIZE_MODE else gnn_experiment_dir,
            "gnn_model_time":     gnn_model_time,
            "gnn_metadata":       {k: v for k, v in gnn_metadata.items()
                                    if isinstance(v, (str, int, float, bool, list, dict, type(None)))},
            "interval_minutes":   interval_minutes,
            "sequence_length":    seq_len,
            "fe_only_cols":       fe_only_cols,
            "feature_engineering_config": FEATURE_ENGINEERING_CONFIG,
            "train_cutoff_frac":  0.70,
            "timestamp":          datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }, f, indent=2, default=str)

    return df_full, fe_only_cols



# =============================================================================
# CONFIGURATION
# =============================================================================

PIPELINE_VERSION = "v51_xgboost_only"

# Output directory for the V51 XGBoost training artefacts.
OUTPUT_DIR = os.environ.get(
    "V51_XGBOOST_OUTPUT_DIR",
    f"{_AP7_EXPERIMENTS}/v51_xgb_with_gnn",
)

GNN_EXPERIMENT_DIR = os.environ.get(
    "GNN_EXPERIMENT_DIR", os.environ.get("PREVIOUS_EXPERIMENT_DIR", "")
)

# Date / window configuration — inherit V50's V17 GNN alignment so the GNN
# predictions feeding XGBoost stay in-distribution.
_V17_ALIGNED_TRAIN_START = _V17_ALIGNED_TRAIN_START
_V17_ALIGNED_TRAIN_END   = _V17_ALIGNED_TRAIN_END
_V17_ALIGNED_SIM_START   = _V17_ALIGNED_SIM_START
_V17_ALIGNED_SIM_END     = _V17_ALIGNED_SIM_END

GNN_TRAIN_WINDOWS = list(GNN_TRAIN_WINDOWS)

TRAIN_START_DATE = _V17_ALIGNED_TRAIN_START
TRAIN_END_DATE   = _V17_ALIGNED_TRAIN_END
SIM_START_DATE   = _V17_ALIGNED_SIM_START
SIM_END_DATE     = _V17_ALIGNED_SIM_END
PK_MIN           = PK_MIN
PK_MAX           = PK_MAX
TIME_RESOLUTION  = TIME_RESOLUTION

VIZ_DIR = os.path.join(OUTPUT_DIR, "visualizations")

# Module-level attrs the bash runner mutates — mirror v50.
TEST_MODE         = TEST_MODE
MEMORIZE_MODE     = MEMORIZE_MODE
ABLATION_SETTINGS = ABLATION_SETTINGS
FEATURE_GROUPS    = FEATURE_GROUPS
XGBOOST_CONFIG    = XGBOOST_CONFIG
DATA_FILE         = DATA_FILE
DATA_PATH         = str(DATA_PATH)


# =============================================================================
# WIDER TREE SEARCH RANGES
# =============================================================================
# Maps parameter name → (low, high). The patcher (below) intercepts
# `Trial.suggest_int` / `Trial.suggest_float` calls, replaces (low, high)
# for matching names, and forwards everything else unchanged. Param names
# not in the dict are unaffected.

_S1_INT_RANGES = {
    "max_depth":        (6, 12),
    "min_child_weight": (5, 30),
}
_S1_FLOAT_RANGES = {
    "gamma":            (0.0, 3.0),
    "learning_rate":    (0.01, 0.3),
    "reg_lambda":       (0.0, 30.0),
    "subsample":        (0.7, 1.0),
    "colsample_bytree": (0.7, 1.0),
}
_S2_INT_RANGES = {
    "max_depth":        (4, 12),
    "min_child_weight": (3, 25),
}
_S2_FLOAT_RANGES = {
    "gamma":            (0.0, 3.0),
    "reg_alpha":        (0.0, 10.0),
    "reg_lambda":       (0.0, 10.0),
    "subsample":        (0.7, 1.0),
    "colsample_bytree": (0.7, 1.0),
}


@contextmanager
def _wider_optuna_ranges(int_ranges: dict, float_ranges: dict):
    """Temporarily widen `Trial.suggest_int` / `suggest_float` ranges for
    specific parameter names. Calls for other names pass through unchanged.

    V34's trial objectives hard-code search ranges (including inside a
    closure in `stage7_two_stage`). Patching the Trial methods lets V51
    widen those ranges without forking ~150 lines of optimisation code.
    """
    orig_int   = Trial.suggest_int
    orig_float = Trial.suggest_float

    def _new_int(self, name, low, high, *args, **kwargs):
        if name in int_ranges:
            new_low, new_high = int_ranges[name]
            low, high = int(new_low), int(new_high)
        return orig_int(self, name, low, high, *args, **kwargs)

    def _new_float(self, name, low, high, *args, **kwargs):
        if name in float_ranges:
            new_low, new_high = float_ranges[name]
            low, high = float(new_low), float(new_high)
        return orig_float(self, name, low, high, *args, **kwargs)

    Trial.suggest_int   = _new_int
    Trial.suggest_float = _new_float
    try:
        yield
    finally:
        Trial.suggest_int   = orig_int
        Trial.suggest_float = orig_float


# =============================================================================
# RUN PIPELINE  (v50 stage 0  +  v34 stages 1, 2, 5, 6, 7, 7B, 7D — wider trees)
# =============================================================================

def run_pipeline(gnn_experiment_dir):
    t_start = time.time()
    print("=" * 80)
    print("V51 XGBoost-only Two-Stage OOF Training (GNN-fed, WIDER trees)")
    print(f"Training:        {TRAIN_START_DATE} -> {TRAIN_END_DATE}")
    print(f"Output dir:      {OUTPUT_DIR}")
    print(f"Viz dir:         {VIZ_DIR}")
    print(f"GNN experiment:  {gnn_experiment_dir}")
    print(f"MEMORIZE_MODE:   {bool(MEMORIZE_MODE)}")
    print("Stage 1 widened ranges:")
    for k, (lo, hi) in {**_S1_INT_RANGES, **_S1_FLOAT_RANGES}.items():
        print(f"    {k:<20s}{lo} - {hi}")
    print("Stage 2 widened ranges:")
    for k, (lo, hi) in {**_S2_INT_RANGES, **_S2_FLOAT_RANGES}.items():
        print(f"    {k:<20s}{lo} - {hi}")
    print("=" * 80)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(VIZ_DIR, exist_ok=True)

    # Stage 0 (v50): base data + GNN + FE — identical to v50.
    df_full, fe_only_cols = stage0_load_data_with_gnn(gnn_experiment_dir)

    # Stage 1 (v34): feature prep + train/test split.
    (df_train, X_train_xgb, X_test_xgb, y_train_acc, y_test_acc,
     available_features, model_time) = stage1_feature_prep(df_full, fe_only_cols)

    del df_train
    gc.collect()
    print("[MEM] df_train released. Proceeding to ensemble training (wider Stage 1 trees).")

    # Stage 2 (v34): Stage 1 ensemble — widen Optuna search space.
    with _wider_optuna_ranges(_S1_INT_RANGES, _S1_FLOAT_RANGES):
        (base_models, ensemble_preds, y_pred_binary, roc_auc, pr_auc,
         s1_best_params) = stage2_train_ensemble(
            X_train_xgb, X_test_xgb, y_train_acc, y_test_acc,
            available_features, model_time)
    gc.collect()

    # Stage 5 (v34): Stage 1 SHAP / feature importance.
    stage5_feature_importance_shap(base_models, X_test_xgb, available_features)

    # Stage 6 (v34): feature-space overlap diagnostics.
    stage6_feature_space_overlap(
        X_train_xgb, X_test_xgb, y_train_acc, y_test_acc,
        available_features, ensemble_preds)

    # Stage 7 (v34): Stage 2 OOF training (F2 Optuna) — widen Stage 2 search.
    # Under MEMORIZE_MODE the standard Stage-2 OOF/Optuna recipe is replaced
    # with a Stage-1-style training: Stage 1 uses balanced bagging with
    # n_base=1 and a single negative subsample, so the booster only ever sees
    # `len(pos) * (1 + neg_pos_ratio)` rows — many leaf paths over the
    # remaining negatives stay unreachable. Training Stage 2 the same way
    # but with a different seed gives the architecture a second balanced
    # subset (different negatives, different tree topology) and roughly
    # doubles the leaf-path budget at memorize time. The inference contract
    # is preserved: Stage 2 still consumes `s1_score` + `available_features`
    # on rows flagged by Stage 1.
    if MEMORIZE_MODE:
        print('=' * 80)
        print('STAGE 7 (MEMORIZE): Stage-2 trained Stage-1-style on full pool')
        print('=' * 80)

        s2_memorize_params = {
            **s1_best_params,
            'tree_method':  'hist',
            'objective':    'binary:logistic',
            'eval_metric':  'aucpr',
            'random_state': RANDOM_STATE + 1,
            'nthread':      N_THREADS,
        }
        neg_pos_ratio = ENSEMBLE_CONFIG['neg_pos_ratio']

        X_all = pd.concat([X_train_xgb, X_test_xgb])
        y_all = pd.concat([y_train_acc,  y_test_acc])

        s1_score_all = np.zeros(len(X_all))
        d_all = xgb.DMatrix(X_all)
        for m in base_models:
            best_iter = getattr(m, 'best_iteration', 500)
            s1_score_all += m.predict(d_all, iteration_range=(0, best_iter))
        s1_score_all /= max(len(base_models), 1)
        X_all = X_all.copy()
        X_all['s1_score'] = s1_score_all

        X_s2_tr, y_s2_tr = create_balanced_subset(
            X_all, y_all,
            neg_pos_ratio=neg_pos_ratio,
            random_state=RANDOM_STATE + 1,
        )
        n_tp_s2 = int(y_s2_tr.sum())
        n_fp_s2 = int((y_s2_tr == 0).sum())
        print(f'  Stage 2 (MEMORIZE) pool: {len(X_s2_tr):,} '
              f'(TP={n_tp_s2:,}, FP={n_fp_s2:,}, seed={RANDOM_STATE + 1})')

        dtrain_s2 = xgb.DMatrix(X_s2_tr, label=y_s2_tr)
        stage2_model = xgb.train(
            s2_memorize_params, dtrain_s2,
            num_boost_round=500, verbose_eval=False,
        )
        stage2_model.best_iteration = 500
        stage2_path = os.path.join(
            OUTPUT_DIR, f'xgboost-stage2-fp-filter_version={model_time}.json'
        )
        stage2_model.save_model(stage2_path)
        print(f'  Stage 2 (MEMORIZE) model saved: {stage2_path}')

        X_test_r     = X_test_xgb.reset_index(drop=True)
        y_test_np    = y_test_acc.reset_index(drop=True).values
        test_s1_flag = ensemble_preds >= 0.5
        flagged_idx  = np.where(test_s1_flag)[0]
        if test_s1_flag.any():
            X_te_s2 = X_test_r[test_s1_flag].copy()
            X_te_s2['s1_score'] = ensemble_preds[test_s1_flag]
            s2_test_probs = stage2_model.predict(xgb.DMatrix(X_te_s2))
        else:
            s2_test_probs = np.zeros(0)

        S1_THR, S2_THR = 0.5, 0.5
        MIN_RECALL_CONSTRAINT, MAX_FAR_TARGET = 0.0, 1.0

        print('STAGE 7B / 7D: SKIPPED (MEMORIZE_MODE — diagnostics only)')
    else:
        with _wider_optuna_ranges(_S2_INT_RANGES, _S2_FLOAT_RANGES):
            (stage2_model, X_s2_tr, y_s2_tr, s2_full_prob, s2_full_bin,
             y_test_np, flagged_idx, s2_test_probs,
             STAGE1_THRESHOLD, STAGE2_THRESHOLD,
             MIN_RECALL_CONSTRAINT, MAX_FAR_TARGET) = stage7_two_stage(
                X_train_xgb, X_test_xgb, y_train_acc, y_test_acc,
                available_features, ensemble_preds, base_models,
                model_time, s1_best_params)

        # Stage 7B (v34): Stage 2 SHAP.
        stage7b_shap_stage2(stage2_model, X_s2_tr, y_s2_tr)

        # Stage 7D (v34): threshold sweep.
        S1_THR, S2_THR = stage7d_threshold_sweep(
            ensemble_preds, y_test_np, flagged_idx, s2_test_probs,
            MIN_RECALL_CONSTRAINT, MAX_FAR_TARGET)

    # ── Persist V51 manifest ────────────────────────────────────────────────
    manifest = {
        "pipeline_version":       PIPELINE_VERSION,
        "model_time":             model_time,
        "output_dir":             OUTPUT_DIR,
        "gnn_experiment_dir":     gnn_experiment_dir,
        "available_features":     available_features,
        "stage1_threshold":       float(S1_THR),
        "stage2_threshold":       float(S2_THR),
        "min_recall_constraint":  float(MIN_RECALL_CONSTRAINT),
        "max_far_target":         float(MAX_FAR_TARGET),
        "test_metrics_stage1":    {"roc_auc": float(roc_auc), "pr_auc": float(pr_auc)},
        "train_dates":            {"start": TRAIN_START_DATE, "end": TRAIN_END_DATE},
        "sim_dates":              {"start": SIM_START_DATE,   "end": SIM_END_DATE},
        "pk_range":               {"min": PK_MIN, "max": PK_MAX},
        "time_resolution":        TIME_RESOLUTION,
        "ablation_settings":      ABLATION_SETTINGS,
        "memorize_mode":          bool(MEMORIZE_MODE),
        "stage2_disabled":        False,
        "wider_tree_ranges": {
            "stage1_int":   _S1_INT_RANGES,
            "stage1_float": _S1_FLOAT_RANGES,
            "stage2_int":   _S2_INT_RANGES,
            "stage2_float": _S2_FLOAT_RANGES,
        },
        "timestamp":              datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    manifest_path = os.path.join(OUTPUT_DIR, f"v51_xgboost_manifest_{model_time}.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[V51] Manifest saved: {manifest_path}")

    # ── experiment_config_xgb_v51.json — consumed by the sim runner ─────────
    exp_config = {
        "model_time":         model_time,
        "test_mode":          TEST_MODE,
        "memorize_mode":      bool(MEMORIZE_MODE),
        "ablation_settings":  ABLATION_SETTINGS,
        "pipeline_version":   PIPELINE_VERSION,
        "gnn_experiment_dir": gnn_experiment_dir,
        "stage1_threshold":   float(S1_THR),
        "stage2_threshold":   float(S2_THR),
        "available_features": available_features,
        "train_dates":        {"start": TRAIN_START_DATE, "end": TRAIN_END_DATE},
        "sim_dates":          {"start": SIM_START_DATE,   "end": SIM_END_DATE},
        "timestamp":          datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    exp_config_path = os.path.join(OUTPUT_DIR, "experiment_config_xgb_v51.json")
    with open(exp_config_path, "w") as f:
        json.dump(exp_config, f, indent=2)
    print(f"[V51] Sim-runner config saved: {exp_config_path}")

    print("\n" + "=" * 80)
    print(f"V51 XGBoost-only training complete in {format_duration(time.time() - t_start)}")
    print(f"All plots saved to: {VIZ_DIR}")
    print(f"All models saved to: {OUTPUT_DIR}")
    print(f"Model time: {model_time}")
    print("Next step: ablation_study_v51_simulation_only.py "
          f"--xgboost-experiment-dir {OUTPUT_DIR} "
          f"--gnn-experiment-dir {gnn_experiment_dir}")
    print("=" * 80)
    return True



def _propagate_to_v34():
    """Pull v5's bash-runner-mutated dicts (ABLATION_SETTINGS, FEATURE_GROUPS)
    into this module's own dicts. v51 is self-contained — it doesn't import
    v34 or v50 — so there's nothing else to mirror onto.
    """
    global TEST_MODE, MEMORIZE_MODE, DATA_PATH

    import src.training.ablation_study_v5_xgboost_only as v5

    ABLATION_SETTINGS.update(v5.ABLATION_SETTINGS)
    if "weather_3d" in v5.FEATURE_GROUPS:
        FEATURE_GROUPS["weather_3d"] = list(v5.FEATURE_GROUPS["weather_3d"])

    if TEST_MODE is None:
        TEST_MODE = v5.TEST_MODE
    MEMORIZE_MODE = bool(MEMORIZE_MODE)
    if not isinstance(DATA_PATH, Path):
        DATA_PATH = Path(DATA_PATH)



def run_xgboost_only():
    """Entry point invoked by `bash_files/run_ablation_study_xgboost_only.sh`.

    Reads the GNN experiment dir from PREVIOUS_EXPERIMENT_DIR (set by the
    runner) and saves XGBoost artefacts into a per-XGBoost-run subfolder
    `<GNN_dir>/xgboost_<XGB_RUN_ID>/` so multiple XGBoost runs co-exist
    against the same GNN.
    """
    global OUTPUT_DIR, VIZ_DIR
    global TRAIN_START_DATE, TRAIN_END_DATE, SIM_START_DATE, SIM_END_DATE
    global GNN_TRAIN_WINDOWS

    TRAIN_START_DATE = _V17_ALIGNED_TRAIN_START
    TRAIN_END_DATE   = _V17_ALIGNED_TRAIN_END
    SIM_START_DATE   = _V17_ALIGNED_SIM_START
    SIM_END_DATE     = _V17_ALIGNED_SIM_END

    # MEMORIZE_MODE: collapse train and simulation to May 2025 only, so the
    # model is trained and evaluated on the exact same window. This is the
    # "perfect-fit" sanity check (does the pipeline reach 0 FP / 0 FN when
    # train ≡ sim?) — it has no generalisation meaning.
    if MEMORIZE_MODE:
        TRAIN_START_DATE  = "2025-05-01"
        TRAIN_END_DATE    = "2025-06-01"
        SIM_START_DATE    = "2025-05-01"
        SIM_END_DATE      = "2025-06-01"
        GNN_TRAIN_WINDOWS = [("2025-05-01", "2025-06-01")]
        print(f"[V51][MEMORIZE] Collapsed windows to May 2025 only: "
              f"train={TRAIN_START_DATE}->{TRAIN_END_DATE}, "
              f"sim={SIM_START_DATE}->{SIM_END_DATE}")

    # CLI overrides win over both the V17-aligned defaults AND the MEMORIZE
    # collapse above. Empty env vars mean "keep the current value". When any
    # date flag is set we also collapse GNN_TRAIN_WINDOWS to a single
    # train-window so the 0A-bis filter doesn't keep extra rows.
    _cli_ini_train = os.environ.get("INI_TRAIN", "")
    _cli_end_train = os.environ.get("END_TRAIN", "")
    _cli_ini_test  = os.environ.get("INI_TEST",  "")
    _cli_end_test  = os.environ.get("END_TEST",  "")
    if _cli_ini_train: TRAIN_START_DATE = _cli_ini_train
    if _cli_end_train: TRAIN_END_DATE   = _cli_end_train
    if _cli_ini_test:  SIM_START_DATE   = _cli_ini_test
    if _cli_end_test:  SIM_END_DATE     = _cli_end_test
    if any([_cli_ini_train, _cli_end_train, _cli_ini_test, _cli_end_test]):
        GNN_TRAIN_WINDOWS = [(TRAIN_START_DATE, TRAIN_END_DATE)]
        print(f"[V51][CLI-OVERRIDE] train={TRAIN_START_DATE}->{TRAIN_END_DATE}, "
              f"sim={SIM_START_DATE}->{SIM_END_DATE}")

    gnn_experiment_dir = (
        os.environ.get("PREVIOUS_EXPERIMENT_DIR", "")
        or os.environ.get("GNN_EXPERIMENT_DIR", "")
        or GNN_EXPERIMENT_DIR
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
        print(f"[V51] XGB_RUN_ID not set; using timestamp fallback: {xgb_run_id}")

    OUTPUT_DIR = os.path.join(gnn_experiment_dir, f"xgboost_{xgb_run_id}")
    VIZ_DIR    = os.path.join(OUTPUT_DIR, "visualizations")
    print(f"[V51] XGBoost artefacts will be saved under: {OUTPUT_DIR}")

    _propagate_to_v34()
    return run_pipeline(gnn_experiment_dir)


def main():
    global OUTPUT_DIR, VIZ_DIR, MEMORIZE_MODE

    parser = argparse.ArgumentParser(
        description="V51 XGBoost-only training: GNN-fed two-stage OOF pipeline with WIDER trees."
    )
    parser.add_argument(
        "--gnn-experiment-dir",
        default=GNN_EXPERIMENT_DIR,
        help="V14 GNN experiment dir (must contain best-model + scalers + metadata).",
    )
    parser.add_argument(
        "--output-dir",
        default=OUTPUT_DIR,
        help="Where to save the V51 XGBoost artefacts.",
    )
    parser.add_argument(
        "--memorize-mode",
        action="store_true",
        default=bool(MEMORIZE_MODE),
        help="Enable v34 MEMORIZE_MODE (train on the full set with fixed hyperparams "
             "and no early stopping — for sanity-check overfitting).",
    )
    args = parser.parse_args()

    OUTPUT_DIR = args.output_dir
    VIZ_DIR    = os.path.join(OUTPUT_DIR, "visualizations")
    MEMORIZE_MODE = bool(args.memorize_mode)

    if not args.gnn_experiment_dir:
        print("[ERROR] --gnn-experiment-dir is required (or set $GNN_EXPERIMENT_DIR).")
        sys.exit(1)

    _propagate_to_v34()
    return run_pipeline(args.gnn_experiment_dir)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
