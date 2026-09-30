#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V51 Simulation-only: GNN-fed Two-Stage XGBoost Rolling Simulation
==================================================================
Companion to `ablation_study_v51_xgboost_only.py`.

Identical pipeline to V50's simulation companion — V14 GNN inference →
traffic feature replacement → V21 feature engineering → two-stage
XGBoost prediction — but loads V51 training artefacts (which differ from
V50 only in the Optuna search-space used for tree-shape and regularisation
hyperparameters; the saved booster format and metadata layout are the
same).

For the simulation window [SIM_START_DATE, SIM_END_DATE), the script runs
exactly the same pipeline that V51 training used:

  1. Load base CSV (raw mean_speed / intTot / intP).
  2. Run V14 GNN inference  →  mean_speed_gnn / intTot_gnn / intP_gnn.
  3. Replace ground-truth traffic columns with the GNN predictions.
  4. Recompute V21 feature engineering (groups 1-13) ON the GNN-predicted
     traffic — this is the point of the "with_GNN" rebuild: the engineered
     lags / z-scores / rolling stats that XGBoost sees at simulation time
     are derived from GNN outputs, never from ground truth.
  5. Run the two-stage XGBoost prediction (Stage 1 ensemble + Stage 2
     filter) loaded from the V51 XGBoost training run.
  6. Reuse V34's `stage4_evaluate_simulation`, `stage8_two_stage_simulation`
     and `stage9_final_evaluation` for reporting.

Usage
-----
  python -m src.training.ablation_study_v51_simulation_only \\
      --xgboost-experiment-dir $AP7_EXPERIMENTS_DIR/v51_xgb_with_gnn \\
      --gnn-experiment-dir     $AP7_EXPERIMENTS_DIR/v14_gnn_no-w1d_5min_<jobid>

Author: Gerard Franco
Date:   May 2026
"""
from src.paths import (  # portable paths -- see src/paths.py
    PROJECT_ROOT_STR as _AP7_ROOT,
    EXPERIMENTS_ROOT_STR as _AP7_EXPERIMENTS,
    DATA_DIR_STR as _AP7_DATA,
    TABLES_DIR as _AP7_TABLES,
    FIGURES_DIR as _AP7_FIGS,
)

import argparse
import gc
import glob
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# ── V34 (training-time pipeline; we reuse its evaluation stages) ───────────
import src.training.ablation_study_v34_simulation_oof_2024apr_2025sep as v34

# ── V5 (GNN inference infrastructure) ──────────────────────────────────────
import src.training.ablation_study_v5_xgboost_only as _v5_mod
from src.training.ablation_study_v5_xgboost_only import (
    detect_model_time as _v5_detect_model_time,
    load_pretrained_gnn_model as _v5_load_pretrained_gnn_model,
    generate_gnn_predictions_on_training_data as _v5_generate_gnn_predictions,
    get_time_resolution_minutes as _v5_get_time_resolution_minutes,
)

# ── V21 (feature-engineering pipeline) ─────────────────────────────────────
import src.training.ablation_study_v21_xgboost_only as _v21_mod
from src.training.ablation_study_v21_xgboost_only import (
    build_engineered_features,
    FEATURE_ENGINEERING_CONFIG,
    SEQUENCE_LENGTH_BY_INTERVAL,
)

# ── V51 training script (GNN-bypass frame alignment) ───────────────────────
# Shared with the trainer on purpose: the row layout these two helpers impose
# drives the positional 70% FE-baseline cutoff and the positional 80/20 split,
# so train and sim MUST derive it from the same code. Duplicating stage 0
# between the two scripts is what let the layouts drift apart in the first
# place.
from src.training.ablation_study_v51_xgboost_only import (
    _GNN_LAYOUT_LOC_COLS,
    _align_to_gnn_layout,
    _drop_gnn_warmup_rows,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

PIPELINE_VERSION = "v51_simulation_only"

# ──────────────────────────────────────────────────────────────────────────────
# Date / window configuration — must match the V17 GNN run used at training
# time (`log_output/output_gnn_only_2538314.log`) and the V51 XGBoost training
# script.  Training windows are the V17 GNN's two disjoint 1-month windows;
# simulation covers the whole month of June 2025.
# ──────────────────────────────────────────────────────────────────────────────
_V17_ALIGNED_TRAIN_START = "2024-06-01"
_V17_ALIGNED_TRAIN_END   = "2025-06-01"
_V17_ALIGNED_SIM_START   = "2025-06-01"
_V17_ALIGNED_SIM_END     = "2025-07-01"   # whole June 2025

# Mirrors GNN_TRAIN_WINDOWS in `ablation_study_v51_xgboost_only.py` so that
# FE rows kept at simulation time match the training distribution.
GNN_TRAIN_WINDOWS = [
    ("2024-06-01", "2024-07-01"),
    ("2025-05-01", "2025-06-01"),
]

TRAIN_START_DATE = _V17_ALIGNED_TRAIN_START
TRAIN_END_DATE   = _V17_ALIGNED_TRAIN_END
SIM_START_DATE   = _V17_ALIGNED_SIM_START
SIM_END_DATE     = _V17_ALIGNED_SIM_END
PK_MIN           = v34.PK_MIN
PK_MAX           = v34.PK_MAX
TIME_RESOLUTION  = v34.TIME_RESOLUTION

# ── Module-level attrs that `run_ablation_study_simulation_only.sh` mutates ──
# The bash runner sets `ablation.ABLATION_SETTINGS[...]`,
# `ablation.FEATURE_GROUPS[...]`, `ablation.TEST_MODE`,
# `ablation.SIM_START_DATE`, `ablation.SIM_END_DATE`,
# `ablation.SIMULATION_CONFIG[...]`, `ablation.DATA_FILE`,
# `ablation.DATA_PATH`, `ablation.XGBOOST_MODEL_TIME` directly on this module.
TEST_MODE         = v34.TEST_MODE
MEMORIZE_MODE     = v34.MEMORIZE_MODE        # propagated to v34 — no-op at sim time
ABLATION_SETTINGS = v34.ABLATION_SETTINGS    # shared dict — mutations propagate
FEATURE_GROUPS    = v34.FEATURE_GROUPS       # shared dict
DATA_FILE         = v34.DATA_FILE
DATA_PATH         = str(v34.DATA_PATH)

# v51 has no fine-tuning loop, so SIMULATION_CONFIG is just present for
# bash-runner compatibility (its dict-write is a no-op for us).
SIMULATION_CONFIG: dict = {
    "prediction_window_days":   7,
    "gnn_fine_tune_epochs":     0,
    "xgboost_fine_tune_rounds": 0,
}

XGBOOST_MODEL_TIME: str | None = None

# Tell the bash runner not to override our SIM_START_DATE / SIM_END_DATE.
# (The runner's defaults match v34's anyway, but be explicit.)
_CUSTOM_SIM_DATES = True


# =============================================================================
# ARTEFACT LOADING
# =============================================================================

def _detect_xgb_model_time(xgb_dir: str) -> str:
    """Find V51 XGBoost model_time from manifest or file pattern."""
    # Prefer the V51 manifest (written by the training script).
    manifests = sorted(glob.glob(os.path.join(xgb_dir, "v51_xgboost_manifest_*.json")))
    if manifests:
        with open(manifests[-1]) as f:
            return json.load(f)["model_time"]

    # Fallback: detect from the XGBoost metadata filename pattern used by V34.
    metas = glob.glob(os.path.join(xgb_dir, "xgboost-metadata_model=XGBoost-version=*.json"))
    if metas:
        return os.path.basename(metas[0]).split("version=")[1].rsplit(".json", 1)[0]

    raise FileNotFoundError(f"No V51 XGBoost manifest or metadata found in {xgb_dir}")


def load_v51_xgboost_artefacts(xgb_dir: str):
    """Load Stage 1 ensemble + Stage 2 filter + thresholds from a V51 dir."""
    print("=" * 80)
    print("V51-SIM STAGE A: LOADING XGBOOST ARTEFACTS")
    print("=" * 80)

    model_time = _detect_xgb_model_time(xgb_dir)
    print(f"[V51-SIM] XGBoost dir       : {xgb_dir}")
    print(f"[V51-SIM] XGBoost model_time: {model_time}")

    # Manifest (V51-specific): thresholds + feature list + GNN provenance.
    manifest_path = os.path.join(xgb_dir, f"v51_xgboost_manifest_{model_time}.json")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"V51 manifest missing: {manifest_path}")
    with open(manifest_path) as f:
        manifest = json.load(f)

    # Stage 1 ensemble metadata (paths to all base models).
    ens_meta_path = os.path.join(
        xgb_dir, f"xgboost-ensemble-metadata_target=accident_version={model_time}.json")
    if not os.path.exists(ens_meta_path):
        raise FileNotFoundError(f"Ensemble metadata missing: {ens_meta_path}")
    with open(ens_meta_path) as f:
        ens_meta = json.load(f)

    base_models = []
    for path in ens_meta["base_model_paths"]:
        m = xgb.Booster()
        m.load_model(path)
        base_models.append(m)
    print(f"[V51-SIM] Loaded {len(base_models)} Stage-1 base models")

    # Stage 2 filter — optional under memorize mode (training skipped it).
    stage2_disabled = bool(manifest.get("stage2_disabled", False))
    stage2_path = os.path.join(xgb_dir, f"xgboost-stage2-fp-filter_version={model_time}.json")
    if stage2_disabled or not os.path.exists(stage2_path):
        if stage2_disabled:
            print("[V51-SIM] Stage 2 disabled by training manifest (MEMORIZE_MODE)")
        else:
            print(f"[V51-SIM] Stage 2 filter not found, running Stage 1 only: {stage2_path}")
        stage2_model = None
    else:
        stage2_model = xgb.Booster()
        stage2_model.load_model(stage2_path)
        print(f"[V51-SIM] Loaded Stage-2 FP filter: {os.path.basename(stage2_path)}")

    available_features = manifest["available_features"]
    s1_thr = float(manifest.get("stage1_threshold", 0.5))
    s2_thr = float(manifest.get("stage2_threshold", 0.5))
    print(f"[V51-SIM] Thresholds: S1={s1_thr:.3f}  S2={s2_thr:.3f}")
    print(f"[V51-SIM] Available features: {len(available_features)}")

    return {
        "model_time":         model_time,
        "base_models":        base_models,
        "stage2_model":       stage2_model,
        "available_features": available_features,
        "stage1_threshold":   s1_thr,
        "stage2_threshold":   s2_thr,
        "manifest":           manifest,
    }


# =============================================================================
# DATA + GNN + FE PIPELINE  (mirrors v51_xgboost_only stage 0)
# =============================================================================

def prepare_simulation_dataframe(gnn_experiment_dir: str) -> pd.DataFrame:
    """Load base CSV, run GNN inference, recompute V21 FE.

    The output covers the full date range needed by V34's evaluation stages
    (training start through simulation end) so that:
      - V21 FE per-(pk, hor, diaSem) baselines stay leakage-safe
        (estimated on the first 70 % of rows just like in training).
      - V34's `stage8_two_stage_simulation` can slice by SIM_START / SIM_END
        the same way it did during training.
    """
    print("=" * 80)
    print("V51-SIM STAGE B: BASE CSV LOADING")
    print("=" * 80)

    _date_min = min(pd.Timestamp(TRAIN_START_DATE), pd.Timestamp(SIM_START_DATE))
    _date_max = max(pd.Timestamp(TRAIN_END_DATE),   pd.Timestamp(SIM_END_DATE))

    t0 = time.time()
    df_full, _ = v34.load_data(
        pk_min=PK_MIN, pk_max=PK_MAX,
        date_min=_date_min, date_max=_date_max,
    )
    print(f"[V51-SIM] Base CSV loaded in {time.time()-t0:.1f}s — {len(df_full):,} rows")

    # ── Window filter BEFORE GNN inference (memory-frugal) ───────────────
    # Matches the same pre-GNN window filter in `ablation_study_v51_xgboost_only.py`
    # so the simulation FE pipeline sees the same row distribution XGBoost
    # trained on. Drops ~75% of rows before sequence creation, keeping memory
    # well under the SLURM allocation.
    print("=" * 80)
    print("V51-SIM STAGE B-bis: WINDOW FILTERING (V17 GNN ALIGNMENT)")
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
    # plus removal of the raw CSV's double-mapped segment-intervals, mirroring
    # the trainer exactly (see `dedup_segment_intervals`).
    df_full = _v21_mod.dedup_segment_intervals(df_full, label="V51-SIM")
    print(f"[V51-SIM] Window filter: {n_before:,} → {len(df_full):,} rows "
          f"({n_before - len(df_full):,} dropped)")
    sys.stdout.flush()

    # ── Wire V5 / V21 to V34's data file (so they see the same CSV) ─────
    _v5_mod.DATA_FILE  = v34.DATA_FILE
    _v5_mod.DATA_PATH  = str(v34.DATA_PATH)
    _v21_mod.DATA_FILE = v34.DATA_FILE
    _v21_mod.DATA_PATH = str(v34.DATA_PATH)

    # ── GNN inference + traffic replacement (skipped under MEMORIZE_MODE) ─
    # In memorize mode we want XGBoost to see the *same* real traffic columns
    # the (memorize-mode) training run consumed, so the GNN is bypassed and
    # mean_speed / intTot / intP stay as the raw CSV values.
    _LAG_BINS = int(os.environ.get("BENCH_LAGGED_TRAFFIC", "0"))  # must match the training run
    _OBS_TRAFFIC = os.environ.get("BENCH_OBSERVED_TRAFFIC", "0") == "1"  # must match the training run
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
        print(f"V51-SIM STAGE C/D: SKIPPED ({_mode}) — GNN deactivated, traffic aliased into *_gnn")
        print("=" * 80)
        # Mirror the training-time alias (see v51 stage 0B/0C/0D branch) so
        # build_engineered_features produces the full 48 FE columns. Lag mode
        # injects lagged OBSERVED traffic (per-(via,sen,pk) chronological
        # shift), identical to the training path, so sim and train see the same
        # inputs.
        #
        # Adopt the GNN branch's row layout FIRST — `build_engineered_features`
        # below takes `int(len(df_full) * 0.70)` as its baseline cutoff, so a
        # pk-major frame would estimate the (pk, hor, diaSem) baselines from the
        # low PKs only and leave ~30% of June with z-score 0 / pk_crash_rate 0.
        # Same helpers as the trainer, imported, not re-implemented.
        _interval_minutes = _v5_get_time_resolution_minutes()
        _seq_len = SEQUENCE_LENGTH_BY_INTERVAL.get(_interval_minutes, 8)
        df_full = _align_to_gnn_layout(df_full)
        if _LAG_BINS > 0:
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
        df_full = _drop_gnn_warmup_rows(df_full, _seq_len, label="V51-SIM")
        # The GNN branch recomputes `car` from the replaced traffic columns
        # (Stage 0D). Without this the skip branch leaves `car` at its raw-CSV
        # value, which is a constant lane count -- a dead feature -- so the two
        # paths would not be comparable. Mirror it exactly.
    else:
        print("=" * 80)
        print("V51-SIM STAGE C: GNN INFERENCE")
        print("=" * 80)

        if not gnn_experiment_dir or not os.path.isdir(gnn_experiment_dir):
            raise FileNotFoundError(f"GNN experiment dir not found: {gnn_experiment_dir!r}")

        gnn_model_time = _v5_detect_model_time(gnn_experiment_dir)
        print(f"[V51-SIM-GNN] {gnn_experiment_dir}  →  model_time={gnn_model_time}")

        (gnn_model, gnn_metadata, gnn_info,
         scaler_temporal, scaler_static, scaler_targets) = (
            _v5_load_pretrained_gnn_model(gnn_experiment_dir, gnn_model_time)
        )

        interval_minutes = _v5_get_time_resolution_minutes()
        seq_len = SEQUENCE_LENGTH_BY_INTERVAL.get(interval_minutes, 8)
        print(f"[V51-SIM-GNN] interval={interval_minutes}min  sequence_length={seq_len}")

        df_full = _v5_generate_gnn_predictions(
            gnn_model, df_full,
            scaler_temporal, scaler_static, scaler_targets,
            gnn_metadata, sequence_length=seq_len,
            experiment_dir=gnn_experiment_dir,
        )
        n_before = len(df_full)
        df_full = df_full.dropna(subset=["mean_speed_gnn"]).copy()
        print(f"[V51-SIM-GNN] Dropped {n_before - len(df_full):,} rows without GNN predictions")

        try:
            import torch
            del gnn_model
            torch.cuda.empty_cache()
        except Exception:
            pass

        # ── Replace traffic with GNN predictions ─────────────────────────
        print("=" * 80)
        print("V51-SIM STAGE D: TRAFFIC FEATURE REPLACEMENT")
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

    # ── V21 feature engineering ──────────────────────────────────────────
    print("=" * 80)
    _fe_label = "REAL TRAFFIC" if MEMORIZE_MODE else "GNN-PREDICTED"
    print(f"V51-SIM STAGE E: V21 FEATURE ENGINEERING ({_fe_label})")
    print("=" * 80)
    print("[V51-SIM-FE] Active groups:")
    for k, on in FEATURE_ENGINEERING_CONFIG.items():
        print(f"  [{'ON ' if on else 'off'}] {k}")
    sys.stdout.flush()

    # Baselines end at the SAME train/test boundary the trainer used — the
    # model's features must be defined identically at training and scoring
    # time. Resolved from the same GNN_TRAIN_WINDOWS constants; for runs that
    # override the trainer's windows (rolling-origin folds), export
    # BENCH_TRAIN_TEST_BOUNDARY to BOTH jobs. (The old positional 70% prefix
    # leaked June labels into `pk_crash_rate` and made features depend on the
    # frame's sort order.)
    # The trainer honours BENCH_TRAIN_WINDOWS; this module did not, so a
    # rolling-origin fold scored its month with the *default* June windows.
    # For the May fold that put the baseline cutoff (2025-05-19) inside the
    # month being scored, leaking labels into `pk_crash_rate`; for July and
    # August it left the scoring features estimated from a different cutoff
    # than the model was trained with. Resolve from the same override the
    # trainer uses, so the two stages cannot disagree again.
    _env_tw = os.environ.get("BENCH_TRAIN_WINDOWS", "")
    _windows = GNN_TRAIN_WINDOWS
    if _env_tw:
        _sep = ";" if ";" in _env_tw else ","
        _windows = [tuple(w.split(":", 1)) for w in _env_tw.split(_sep) if ":" in w]
        print(f"[V51-SIM] BENCH_TRAIN_WINDOWS override active: {_windows}")
    _fe_boundary = _v21_mod.resolve_train_test_boundary(
        _windows, override=os.environ.get("BENCH_TRAIN_TEST_BOUNDARY"))
    print(f"[V51-SIM-FE] baseline cutoff (train/test boundary): {_fe_boundary}")
    df_full, new_features = build_engineered_features(
        df_full, 0, baseline_end_date=_fe_boundary)
    print(f"[V51-SIM-FE] Engineered {len(new_features)} new feature columns")

    if os.environ.get("BENCH_COV_ENGINEERED", "0") == "1":
        df_full, _cov_cols = _v21_mod.build_covariate_engineered_features(
            df_full, baseline_end_date=_fe_boundary)
        print(f"[COV-FE] rebuilt {len(_cov_cols)} engineered covariate columns")

    # Mirror of the trainer's union block. Without it the classifier would be
    # scored on columns _as_matrix() silently fills with zeros, which would look
    # like the observation adding nothing rather than like a missing column.
    if _UNION_TRAFFIC:
        _made = []
        for _c in ("mean_speed", "intTot", "intP"):
            o, g = f"{_c}_obs", f"{_c}_gnn"
            if o in df_full.columns and g in df_full.columns:
                df_full[f"d_{_c}"] = (df_full[g] - df_full[o]).astype("float32")
                _made += [o, f"d_{_c}"]
        print(f"[union] rebuilt {len(_made)} observation + difference columns")

    # ── Final shape report ───────────────────────────────────────────────
    print(f"Shape: {df_full.shape}")
    print(f"Date range: {df_full['dat'].min()} to {df_full['dat'].max()}")
    print(f"PKs present: {sorted(df_full['pk'].unique())}")
    if "ACCIDENT" in df_full.columns:
        print(f"ACCIDENT distribution:\n{df_full['ACCIDENT'].value_counts()}")

    return df_full


# =============================================================================
# STAGE Z: FEATURE-SPACE ANALYSIS (TP / FP / FN / TN)
# =============================================================================
#
# Goal: quantify how separable the TP / FP / TN partitions actually are in
# feature space.  If TP and FP have near-identical distributions on the
# features the model relies on, no threshold trick can reach 0 FP / 0 FN —
# the ceiling is set by feature-label collisions, not by the classifier.
#
# Outputs (under <output_dir>/feature_space_analysis_<model_time>/):
#   - feature_space_summary.csv         per-feature KS/Cohen-d for TP-FP and FP-TN
#   - feature_space_top_features.png    distribution overlays for top-K model feats
#   - feature_space_global_pca.png      2-D PCA of standardised features, colored
#                                       by TP/FP/FN/TN.
# =============================================================================

def stageZ_feature_space_analysis(df_full, sim_results, base_models,
                                  available_features, output_dir, viz_dir,
                                  model_time, top_k: int = 12):
    print("=" * 80)
    print("STAGE Z: FEATURE-SPACE ANALYSIS (TP vs FP vs TN)")
    print("=" * 80)

    if sim_results is None or len(sim_results) == 0:
        print("[STAGEZ] No sim_results — skipping.")
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import stats as _scs

    # ── 1. Re-attach feature columns to sim_results via (pk, dat) ────────
    sim = sim_results.copy()
    sim["dat"] = pd.to_datetime(sim["dat"])
    keep_cols = ["pk", "dat"] + [
        c for c in available_features
        if c in df_full.columns and c not in ("pk", "dat")
    ]
    feats_df = df_full[keep_cols].copy()
    feats_df["dat"] = pd.to_datetime(feats_df["dat"])
    sim = sim.merge(feats_df, on=["pk", "dat"], how="left")

    if "ACCIDENT_real" not in sim.columns:
        print("[STAGEZ] ACCIDENT_real missing — cannot compute TP/FP partitions.")
        return

    y_true = sim["ACCIDENT_real"].astype(int).to_numpy()
    y_pred = (sim["accident_probability"].astype(float) >= 0.5).astype(int).to_numpy()
    tp_mask = (y_true == 1) & (y_pred == 1)
    fp_mask = (y_true == 0) & (y_pred == 1)
    fn_mask = (y_true == 1) & (y_pred == 0)
    tn_mask = (y_true == 0) & (y_pred == 0)
    print(f"[STAGEZ] partitions  TP={tp_mask.sum()}  FP={fp_mask.sum()}  "
          f"FN={fn_mask.sum()}  TN={tn_mask.sum()}")

    if tp_mask.sum() == 0 or fp_mask.sum() == 0 or tn_mask.sum() == 0:
        print("[STAGEZ] Empty partition(s) — skipping distributional tests.")
        return

    out_dir = os.path.join(output_dir, f"feature_space_analysis_{model_time}")
    os.makedirs(out_dir, exist_ok=True)

    job_id = (
        os.environ.get("SLURM_JOB_ID", "")
        or os.environ.get("XGB_RUN_ID", "")
        or os.environ.get("JOB_ID", "")
    )
    suffix = f"_job{job_id}" if job_id else ""

    # ── 2. Model-importance ranking (Stage 1 ensemble, gain-weighted) ────
    feat_gain = {f: 0.0 for f in available_features}
    for m in base_models:
        try:
            score = m.get_score(importance_type="gain")
        except Exception:
            score = {}
        for k, v in score.items():
            if k in feat_gain:
                feat_gain[k] += float(v)
    n_models = max(1, len(base_models))
    feat_gain = {k: v / n_models for k, v in feat_gain.items()}
    ranked = sorted(feat_gain.items(), key=lambda kv: kv[1], reverse=True)
    top_feats = [f for f, _ in ranked[:top_k] if f in sim.columns]

    # ── 3. Per-feature KS + Cohen's d for TP-FP and FP-TN ────────────────
    def _cohen_d(a, b):
        a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
        a = a[~np.isnan(a)]; b = b[~np.isnan(b)]
        if len(a) < 2 or len(b) < 2:
            return np.nan
        s = np.sqrt(((len(a)-1) * a.var(ddof=1) + (len(b)-1) * b.var(ddof=1))
                    / max(1, len(a) + len(b) - 2))
        return (a.mean() - b.mean()) / s if s > 0 else np.nan

    rows = []
    for f in available_features:
        if f not in sim.columns:
            continue
        col = pd.to_numeric(sim[f], errors="coerce").to_numpy()
        tp_v, fp_v, tn_v = col[tp_mask], col[fp_mask], col[tn_mask]
        try:
            ks_tpfp = float(_scs.ks_2samp(tp_v[~np.isnan(tp_v)], fp_v[~np.isnan(fp_v)]).statistic)
        except Exception:
            ks_tpfp = np.nan
        try:
            ks_fptn = float(_scs.ks_2samp(fp_v[~np.isnan(fp_v)], tn_v[~np.isnan(tn_v)]).statistic)
        except Exception:
            ks_fptn = np.nan
        rows.append({
            "feature":        f,
            "model_gain":     feat_gain.get(f, 0.0),
            "ks_TP_vs_FP":    ks_tpfp,
            "ks_FP_vs_TN":    ks_fptn,
            "cohend_TP_vs_FP": _cohen_d(tp_v, fp_v),
            "cohend_FP_vs_TN": _cohen_d(fp_v, tn_v),
            "mean_TP":        float(np.nanmean(tp_v)) if np.isfinite(np.nanmean(tp_v)) else np.nan,
            "mean_FP":        float(np.nanmean(fp_v)) if np.isfinite(np.nanmean(fp_v)) else np.nan,
            "mean_TN":        float(np.nanmean(tn_v)) if np.isfinite(np.nanmean(tn_v)) else np.nan,
        })
    summary = pd.DataFrame(rows).sort_values("model_gain", ascending=False)
    summary_path = os.path.join(out_dir, "feature_space_summary.csv")
    summary.to_csv(summary_path, sep=";", index=False)
    print(f"[STAGEZ] Per-feature summary: {summary_path}")

    print("\n[STAGEZ] Top features by model gain (KS = distribution gap, 0=identical, 1=disjoint):")
    print(summary.head(top_k)[
        ["feature", "model_gain", "ks_TP_vs_FP", "ks_FP_vs_TN",
         "cohend_TP_vs_FP", "cohend_FP_vs_TN"]].to_string(index=False))

    print("\n[STAGEZ] Global mean KS over all features:")
    print(f"  TP-vs-FP : mean={summary['ks_TP_vs_FP'].mean():.3f}  "
          f"median={summary['ks_TP_vs_FP'].median():.3f}")
    print(f"  FP-vs-TN : mean={summary['ks_FP_vs_TN'].mean():.3f}  "
          f"median={summary['ks_FP_vs_TN'].median():.3f}")

    # ── 4. Plot: distribution overlays for top-K most-influential features ─
    n = len(top_feats)
    if n > 0:
        rows_p = (n + 2) // 3
        fig, axes = plt.subplots(rows_p, 3, figsize=(15, 3.4 * rows_p))
        axes = np.atleast_2d(axes).reshape(-1)
        for i, f in enumerate(top_feats):
            ax = axes[i]
            col = pd.to_numeric(sim[f], errors="coerce").to_numpy()
            for grp_name, mask, color in [
                ("TN", tn_mask, "#bbbbbb"),
                ("FP", fp_mask, "#d62728"),
                ("TP", tp_mask, "#2ca02c"),
            ]:
                v = col[mask]; v = v[~np.isnan(v)]
                if len(v) > 1:
                    ax.hist(v, bins=40, alpha=0.55, density=True,
                            color=color, label=f"{grp_name} (n={len(v)})")
            ks_tpfp = summary.loc[summary["feature"] == f, "ks_TP_vs_FP"].iloc[0]
            ks_fptn = summary.loc[summary["feature"] == f, "ks_FP_vs_TN"].iloc[0]
            ax.set_title(f"{f}\nKS(TP,FP)={ks_tpfp:.2f}  KS(FP,TN)={ks_fptn:.2f}",
                         fontsize=9)
            ax.legend(fontsize=7); ax.grid(alpha=0.3)
        for j in range(n, len(axes)):
            axes[j].axis("off")
        plt.suptitle("Feature distributions by partition — top model features",
                     fontsize=12, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        path = os.path.join(viz_dir, f"stageZ_top_features_distributions{suffix}.png")
        plt.savefig(path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"[STAGEZ] Saved top-features overlay: {path}")

    # ── 5. Global 2-D PCA over top-K features colored by partition ───────
    if n >= 2:
        try:
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler
            X = sim[top_feats].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy()
            X = StandardScaler().fit_transform(X)
            # Subsample TN for plot legibility (it dominates the canvas).
            tn_idx_all = np.where(tn_mask)[0]
            if len(tn_idx_all) > 5000:
                rng = np.random.RandomState(0)
                tn_idx = rng.choice(tn_idx_all, size=5000, replace=False)
            else:
                tn_idx = tn_idx_all
            keep = np.concatenate([
                np.where(tp_mask)[0], np.where(fp_mask)[0],
                np.where(fn_mask)[0], tn_idx,
            ])
            Xp = PCA(n_components=2, random_state=0).fit_transform(X[keep])
            labels = np.empty(len(keep), dtype=object)
            offs = 0
            for grp, mask in [("TP", tp_mask), ("FP", fp_mask),
                              ("FN", fn_mask), ("TN_sub", None)]:
                if grp == "TN_sub":
                    n_g = len(tn_idx)
                else:
                    n_g = mask.sum()
                labels[offs:offs + n_g] = grp
                offs += n_g
            fig, ax = plt.subplots(figsize=(8, 6))
            for grp, color, alpha, size in [
                ("TN_sub", "#bbbbbb", 0.35, 6),
                ("FP",     "#d62728", 0.7,  10),
                ("FN",     "#1f77b4", 0.9,  18),
                ("TP",     "#2ca02c", 0.9,  18),
            ]:
                m = labels == grp
                if m.any():
                    ax.scatter(Xp[m, 0], Xp[m, 1], s=size, c=color,
                               alpha=alpha, label=f"{grp} (n={m.sum()})",
                               edgecolors="none")
            ax.set_title("2-D PCA on top model features — partitions",
                         fontsize=11, fontweight="bold")
            ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
            ax.legend(loc="best"); ax.grid(alpha=0.3)
            path = os.path.join(viz_dir, f"stageZ_pca_partitions{suffix}.png")
            plt.tight_layout()
            plt.savefig(path, dpi=130, bbox_inches="tight")
            plt.close(fig)
            print(f"[STAGEZ] Saved PCA partition map: {path}")
        except Exception as e:
            print(f"[STAGEZ] PCA plot skipped: {e}")

    # ── 6. Standard reading guide (printed to the SLURM log) ─────────────
    print("\n" + "─" * 80)
    print("HOW TO READ THE STAGE-Z OUTPUTS")
    print("─" * 80)
    print(
        "feature_space_summary.csv (one row per feature)\n"
        "  • model_gain      Stage-1 ensemble gain — how much the classifier\n"
        "                    actually relies on this feature. High gain + low KS\n"
        "                    means the model leans on a feature that does NOT\n"
        "                    separate the partitions (a red flag).\n"
        "  • ks_TP_vs_FP     Kolmogorov-Smirnov statistic between TP and FP\n"
        "                    distributions. 0 = identical (FPs and TPs are\n"
        "                    indistinguishable on this feature → no model can\n"
        "                    fix them); 1 = fully disjoint (a sharper threshold\n"
        "                    on this feature alone could split them).\n"
        "                    Rule of thumb: <0.10 negligible, 0.10–0.25 small,\n"
        "                    0.25–0.50 moderate, >0.50 strong separation.\n"
        "  • ks_FP_vs_TN     Same statistic between FP and TN. High value means\n"
        "                    the FPs look unlike normal negatives — i.e. the\n"
        "                    model has a defensible reason to flag them; low\n"
        "                    value means the model fires on rows that look\n"
        "                    indistinguishable from the bulk of negatives.\n"
        "  • cohend_*        Cohen's d for the same pairs (signed effect size\n"
        "                    on the mean). |d|<0.2 negligible, 0.2–0.5 small,\n"
        "                    0.5–0.8 medium, >0.8 large. Sign tells which side\n"
        "                    of the comparison has the larger mean.\n"
        "  • mean_TP/FP/TN   Raw means per partition for sanity-checking the d.\n"
        "\n"
        "stageZ_top_features_distributions.png\n"
        "  Grid of histograms, one panel per top-gain feature. Three overlays:\n"
        "    grey = TN, red = FP, green = TP, all density-normalised so the\n"
        "    areas are comparable regardless of partition size.\n"
        "  Reading:\n"
        "    • TP and FP curves overlapping  → feature cannot tell them apart,\n"
        "      no threshold on it would help; the model's mistakes here are\n"
        "      structural (label noise / feature collisions).\n"
        "    • TP and FP curves shifted apart → feature DOES separate them but\n"
        "      the model failed to exploit the gap (room to improve via\n"
        "      capacity, depth, or interactions).\n"
        "    • FP and TN curves overlapping  → the model is firing on rows\n"
        "      that look just like normal negatives (over-confident).\n"
        "    • FP between TP and TN          → ambiguous middle ground; the\n"
        "      classic case where calibration / a second stage helps.\n"
        "  KS values for each pair are printed in the panel title for quick\n"
        "  triage.\n"
        "\n"
        "stageZ_pca_partitions.png\n"
        "  2-D PCA over the standardised top-K features. PC1/PC2 are the two\n"
        "  axes that capture the most variance — they have no intrinsic units;\n"
        "  only the *relative* layout of the colored points matters. TN is\n"
        "  subsampled to 5000 points so the plot stays readable.\n"
        "  Reading:\n"
        "    • Green TPs forming a tight cluster away from grey TNs → the\n"
        "      feature space already separates accidents from normal traffic.\n"
        "    • Red FPs sitting INSIDE the green TP cluster → FPs are feature-\n"
        "      space twins of true accidents (this is the structural ceiling;\n"
        "      no classifier can split them without new features).\n"
        "    • Red FPs sitting INSIDE the grey TN cloud → model is hallucinat-\n"
        "      ing; the input doesn't justify the positive call.\n"
        "    • Blue FNs near red FPs / green TPs → near-miss cases; a slightly\n"
        "      different threshold or calibration would catch them.\n"
        "    • Blue FNs scattered into the grey TN cloud → those accidents are\n"
        "      indistinguishable from normal traffic on these features —\n"
        "      another structural ceiling, addressable only with new signals.\n"
        "  Caveat: PCA is a *linear* projection, so two points that look close\n"
        "  in PC1/PC2 may still be far apart in the original feature space.\n"
        "  Use this plot for gestalt checks, the KS/Cohen-d table for numbers.\n"
    )
    print("[STAGEZ] Done.")


# =============================================================================
# RUN PIPELINE
# =============================================================================

def run_pipeline(xgb_experiment_dir: str, gnn_experiment_dir: str,
                 output_dir: str | None = None):
    global TRAIN_START_DATE, TRAIN_END_DATE, SIM_START_DATE, SIM_END_DATE
    global GNN_TRAIN_WINDOWS, MEMORIZE_MODE

    t_start = time.time()
    output_dir = output_dir or xgb_experiment_dir
    viz_dir    = os.path.join(output_dir, "visualizations")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(viz_dir,    exist_ok=True)

    # ── A. Load XGBoost artefacts (early — manifest tells us memorize_mode) ─
    art = load_v51_xgboost_artefacts(xgb_experiment_dir)
    base_models        = art["base_models"]
    stage2_model       = art["stage2_model"]
    available_features = art["available_features"]
    model_time         = art["model_time"]
    s1_thr             = art["stage1_threshold"]
    s2_thr             = art["stage2_threshold"]

    # MEMORIZE_MODE: collapse train and simulation to May 2025 only, matching
    # the training script. Triggered by the training manifest, the env-var
    # override the bash runner sets, or the module-level flag.
    _manifest_memorize = bool(art["manifest"].get("memorize_mode", False))
    _env_memorize      = os.environ.get("MEMORIZE_MODE_OVERRIDE", "") == "True"
    if _manifest_memorize or _env_memorize or MEMORIZE_MODE:
        MEMORIZE_MODE     = True
        TRAIN_START_DATE  = "2025-05-01"
        TRAIN_END_DATE    = "2025-06-01"
        SIM_START_DATE    = "2025-05-01"
        SIM_END_DATE      = "2025-06-01"
        GNN_TRAIN_WINDOWS = [("2025-05-01", "2025-06-01")]
        _propagate_to_v34()
        print(f"[V51-SIM][MEMORIZE] Collapsed windows to May 2025 only: "
              f"train={TRAIN_START_DATE}->{TRAIN_END_DATE}, "
              f"sim={SIM_START_DATE}->{SIM_END_DATE}")

    # CLI overrides win over both the V17-aligned defaults AND the MEMORIZE
    # collapse above. Empty env vars mean "keep the current value".
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
        _propagate_to_v34()
        print(f"[V51-SIM][CLI-OVERRIDE] train={TRAIN_START_DATE}->{TRAIN_END_DATE}, "
              f"sim={SIM_START_DATE}->{SIM_END_DATE}")

    print("=" * 80)
    print("V51 Simulation-only Two-Stage OOF (GNN-fed)")
    print(f"Simulation:        {SIM_START_DATE} -> {SIM_END_DATE}")
    print(f"XGBoost dir:       {xgb_experiment_dir}")
    print(f"GNN dir:           {gnn_experiment_dir}")
    print(f"Output dir:        {output_dir}")
    print("=" * 80)

    # Redirect V34 module-level paths so its evaluation stages write into
    # this run's directory.
    v34.OUTPUT_DIR = output_dir
    v34.VIZ_DIR    = viz_dir

    # Cross-check: the GNN supplied here should match the one used at
    # training time.  Warn (not error) so a fresh GNN run can be tried.
    expected_gnn = art["manifest"].get("gnn_experiment_dir")
    if expected_gnn and os.path.normpath(expected_gnn) != os.path.normpath(gnn_experiment_dir):
        print(f"[V51-SIM] WARNING: GNN dir differs from training "
              f"(training={expected_gnn}, sim={gnn_experiment_dir}). "
              "Proceeding — but predictions may drift if GNNs differ.")

    # ── B-E. Build df_full = base + GNN predictions + V21 FE ─────────────
    df_full = prepare_simulation_dataframe(gnn_experiment_dir)
    gc.collect()

    # ── F. Stage 1-only simulation (v34.stage3) ──────────────────────────
    sim_results = v34.stage3_simulation(
        df_full, base_models, available_features, model_time)

    # ── G. Stage 1 evaluation (v34.stage4) ───────────────────────────────
    v34.stage4_evaluate_simulation(sim_results)

    # ── H/I. Two-stage simulation (v34.stage8/stage9) — skipped if no S2 ──
    if stage2_model is not None:
        sim_results_2s = v34.stage8_two_stage_simulation(
            df_full, base_models, stage2_model, available_features,
            model_time, s1_thr, s2_thr)
        v34.stage9_final_evaluation(sim_results, sim_results_2s)
    else:
        print("=" * 80)
        print("STAGE 8 / 9: SKIPPED (no Stage-2 model — memorize-mode run)")
        print("=" * 80)

    # ── Z. Feature-space analysis: TP vs FP vs TN distributional gap ─────
    stageZ_feature_space_analysis(
        df_full, sim_results, base_models, available_features,
        output_dir, viz_dir, model_time)

    # ── J. Persist V51 simulation manifest ───────────────────────────────
    sim_manifest = {
        "pipeline_version":    PIPELINE_VERSION,
        "model_time":          model_time,
        "xgboost_experiment_dir": xgb_experiment_dir,
        "gnn_experiment_dir":  gnn_experiment_dir,
        "sim_dates":           {"start": SIM_START_DATE, "end": SIM_END_DATE},
        "pk_range":            {"min": PK_MIN, "max": PK_MAX},
        "time_resolution":     TIME_RESOLUTION,
        "stage1_threshold":    s1_thr,
        "stage2_threshold":    s2_thr,
        "feature_engineering_config": FEATURE_ENGINEERING_CONFIG,
        "ablation_settings":   v34.ABLATION_SETTINGS,
        "memorize_mode":       bool(v34.MEMORIZE_MODE),
        "timestamp":           datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(output_dir, f"v51_simulation_manifest_{model_time}.json"), "w") as f:
        json.dump(sim_manifest, f, indent=2)

    print("\n" + "=" * 80)
    print(f"V51 simulation complete in {v34.format_duration(time.time() - t_start)}")
    print(f"Results & plots in: {output_dir}")
    print("=" * 80)
    return True


def _propagate_to_v34():
    """Mirror this module's mutated attrs onto v34 so v34's stage functions
    see whatever the bash runner / caller wrote on this module.
    """
    v34.TEST_MODE        = TEST_MODE
    v34.MEMORIZE_MODE    = bool(MEMORIZE_MODE)
    v34.TRAIN_START_DATE = TRAIN_START_DATE
    v34.TRAIN_END_DATE   = TRAIN_END_DATE
    v34.SIM_START_DATE   = SIM_START_DATE
    v34.SIM_END_DATE     = SIM_END_DATE
    v34.DATA_FILE        = DATA_FILE
    v34.DATA_PATH        = Path(DATA_PATH) if not isinstance(DATA_PATH, Path) else DATA_PATH
    # ABLATION_SETTINGS / FEATURE_GROUPS share dicts with v34 already.


def run_simulation_only():
    """Entry point invoked by `bash_files/run_ablation_study_simulation_only.sh`.

    Reads XGBOOST_EXPERIMENT_DIR + GNN_EXPERIMENT_DIR from env (the runner
    resolves them from --xgb-run-id / --gnn-run-id) and runs the V51
    simulation pipeline.
    """
    global TRAIN_START_DATE, TRAIN_END_DATE, SIM_START_DATE, SIM_END_DATE

    # Re-pin to V17-aligned dates so neither the bash runner nor a stale
    # import-time value can drift the simulation away from the training
    # configuration.  (`_CUSTOM_SIM_DATES = True` already opts out of the
    # bash runner's SIM_* override; this also covers TRAIN_* for FE history.)
    TRAIN_START_DATE = _V17_ALIGNED_TRAIN_START
    TRAIN_END_DATE   = _V17_ALIGNED_TRAIN_END
    SIM_START_DATE   = _V17_ALIGNED_SIM_START
    SIM_END_DATE     = _V17_ALIGNED_SIM_END

    xgb_experiment_dir = os.environ.get("XGBOOST_EXPERIMENT_DIR", "")
    gnn_experiment_dir = os.environ.get("GNN_EXPERIMENT_DIR", "")

    if not xgb_experiment_dir or not os.path.isdir(xgb_experiment_dir):
        print(f"[ERROR] XGBOOST_EXPERIMENT_DIR not set or invalid: {xgb_experiment_dir!r}")
        sys.exit(1)
    if not gnn_experiment_dir or not os.path.isdir(gnn_experiment_dir):
        print(f"[ERROR] GNN_EXPERIMENT_DIR not set or invalid: {gnn_experiment_dir!r}")
        sys.exit(1)

    _propagate_to_v34()
    return run_pipeline(
        xgb_experiment_dir=xgb_experiment_dir,
        gnn_experiment_dir=gnn_experiment_dir,
        output_dir=xgb_experiment_dir,
    )


def main():
    global SIM_START_DATE, SIM_END_DATE, MEMORIZE_MODE

    parser = argparse.ArgumentParser(
        description="V51 Simulation-only: GNN-fed two-stage XGBoost rolling simulation."
    )
    parser.add_argument(
        "--xgboost-experiment-dir", required=True,
        help="V51 XGBoost training output dir (contains models + v51_xgboost_manifest_*.json).",
    )
    parser.add_argument(
        "--gnn-experiment-dir", required=True,
        help="V14 GNN experiment dir (must match the one used at V51 training time).",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Where to write the simulation reports/plots. Defaults to the XGBoost dir.",
    )
    parser.add_argument(
        "--sim-start-date", default=SIM_START_DATE,
        help="Override simulation start (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--sim-end-date", default=SIM_END_DATE,
        help="Override simulation end (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--memorize-mode",
        action="store_true",
        default=bool(MEMORIZE_MODE),
        help="Propagate MEMORIZE_MODE to v34 (no-op at sim time, kept for symmetry "
             "with the XGBoost runner so the flag chain stays consistent).",
    )
    args = parser.parse_args()

    SIM_START_DATE = args.sim_start_date
    SIM_END_DATE   = args.sim_end_date
    MEMORIZE_MODE  = bool(args.memorize_mode)

    _propagate_to_v34()

    return run_pipeline(
        xgb_experiment_dir=args.xgboost_experiment_dir,
        gnn_experiment_dir=args.gnn_experiment_dir,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
