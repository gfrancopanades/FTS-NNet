#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V52 Simulation-only: GNN-fed Two-Stage XGBoost Rolling Simulation
==================================================================
Companion to `ablation_study_v52_xgboost_only.py`.  Mirrors V51's simulation
pipeline but understands the V52 experiment block in the manifest:

  (a) Threshold lift — uses S1/S2 thresholds from the V52 manifest (the V52
      threshold sweep was run in BOTH memorize and non-memorize modes).
  (b) Stage-2 FP filter — V52 always trains it, so the simulation always
      runs the two-stage pipeline (V51's "stage 8/9 skipped" branch is gone).
  (c) PK × hour interaction features — the same `add_pk_hour_interactions`
      function used at training time is replayed on `df_full` so the
      Stage-1/Stage-2 boosters see the columns they were trained on.
  (f) Both memorize and non-memorize modes are first-class.

Manifest fall-back: if no `v52_xgboost_manifest_*.json` is found, falls back
to the V51 manifest and runs the V51 simulation pipeline unchanged.

Usage
-----
  python -m src.training.ablation_study_v52_simulation_only \\
      --xgboost-experiment-dir $AP7_EXPERIMENTS_DIR/v52_xgb_with_gnn \\
      --gnn-experiment-dir     $AP7_EXPERIMENTS_DIR/v17_gnn_full_5min_<jobid>

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

# ── V34 (we reuse its evaluation stages) ───────────────────────────────────
import src.training.ablation_study_v34_simulation_oof_2024apr_2025sep as v34

# ── V51 sim (we extend its simulation pipeline) ────────────────────────────
import src.training.ablation_study_v51_simulation_only as v51_sim

# ── V52 XGB (for the FE hooks used at training time) ───────────────────────
import src.training.ablation_study_v52_xgboost_only as v52_xgb
from src.training.ablation_study_v52_xgboost_only import (
    add_pk_hour_interactions,
    HOT_ZONE_PKS,
    EVENING_RUSH_HOURS,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

PIPELINE_VERSION = "v52_simulation_only"

# Inherit V51-aligned defaults; the V52 manifest will override them at runtime.
_V17_ALIGNED_TRAIN_START = v51_sim._V17_ALIGNED_TRAIN_START
_V17_ALIGNED_TRAIN_END   = v51_sim._V17_ALIGNED_TRAIN_END
_V17_ALIGNED_SIM_START   = v51_sim._V17_ALIGNED_SIM_START
_V17_ALIGNED_SIM_END     = v51_sim._V17_ALIGNED_SIM_END

GNN_TRAIN_WINDOWS = list(v51_sim.GNN_TRAIN_WINDOWS)

TRAIN_START_DATE = _V17_ALIGNED_TRAIN_START
TRAIN_END_DATE   = _V17_ALIGNED_TRAIN_END
SIM_START_DATE   = _V17_ALIGNED_SIM_START
SIM_END_DATE     = _V17_ALIGNED_SIM_END
PK_MIN           = v34.PK_MIN
PK_MAX           = v34.PK_MAX
TIME_RESOLUTION  = v34.TIME_RESOLUTION

TEST_MODE         = v34.TEST_MODE
MEMORIZE_MODE     = v34.MEMORIZE_MODE
ABLATION_SETTINGS = v34.ABLATION_SETTINGS
FEATURE_GROUPS    = v34.FEATURE_GROUPS
DATA_FILE         = v34.DATA_FILE
DATA_PATH         = str(v34.DATA_PATH)

SIMULATION_CONFIG: dict = dict(v51_sim.SIMULATION_CONFIG)

XGBOOST_MODEL_TIME: str | None = None

_CUSTOM_SIM_DATES = True


# =============================================================================
# V52 ARTEFACT LOADING
# =============================================================================

def _detect_xgb_model_time_v52(xgb_dir: str) -> tuple[str, str]:
    """Find V52 XGBoost model_time, falling back to V51.  Returns (model_time, schema)."""
    v52_manifests = sorted(glob.glob(os.path.join(xgb_dir, "v52_xgboost_manifest_*.json")))
    if v52_manifests:
        with open(v52_manifests[-1]) as f:
            return json.load(f)["model_time"], "v52"
    v51_manifests = sorted(glob.glob(os.path.join(xgb_dir, "v51_xgboost_manifest_*.json")))
    if v51_manifests:
        with open(v51_manifests[-1]) as f:
            return json.load(f)["model_time"], "v51"
    metas = glob.glob(os.path.join(xgb_dir, "xgboost-metadata_model=XGBoost-version=*.json"))
    if metas:
        return os.path.basename(metas[0]).split("version=")[1].rsplit(".json", 1)[0], "legacy"
    raise FileNotFoundError(f"No V52/V51 XGBoost manifest found in {xgb_dir}")


def load_v52_xgboost_artefacts(xgb_dir: str) -> dict:
    """Load Stage-1 ensemble + Stage-2 + thresholds + V52 experiment block."""
    print("=" * 80)
    print("V52-SIM STAGE A: LOADING XGBOOST ARTEFACTS")
    print("=" * 80)

    model_time, schema = _detect_xgb_model_time_v52(xgb_dir)
    print(f"[V52-SIM] XGBoost dir       : {xgb_dir}")
    print(f"[V52-SIM] XGBoost model_time: {model_time}")
    print(f"[V52-SIM] Manifest schema   : {schema}")

    if schema == "v52":
        manifest_path = os.path.join(xgb_dir, f"v52_xgboost_manifest_{model_time}.json")
    elif schema == "v51":
        manifest_path = os.path.join(xgb_dir, f"v51_xgboost_manifest_{model_time}.json")
    else:
        manifest_path = None

    manifest = {}
    if manifest_path and os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)

    # Stage-1 ensemble metadata.
    ens_meta_path = os.path.join(
        xgb_dir, f"xgboost-ensemble-metadata_target=accident_version={model_time}.json")
    if not os.path.exists(ens_meta_path):
        raise FileNotFoundError(f"Ensemble metadata missing: {ens_meta_path}")
    with open(ens_meta_path) as f:
        ens_meta = json.load(f)

    base_models = []
    for path in ens_meta["base_model_paths"]:
        # `base_model_paths` were written as absolute paths at training time
        # (relative to the XGBoost output dir).  If the XGBoost dir has been
        # moved/renamed since training (e.g. relocated into the per-run
        # `<gnn_dir>/xgboost_<XGB_RUN_ID>/` subfolder), fall back to looking
        # for the file by basename inside the *current* xgb_dir.
        if not os.path.exists(path):
            fallback = os.path.join(xgb_dir, os.path.basename(path))
            if os.path.exists(fallback):
                print(f"[V52-SIM] Base model path stale ({path!r}); "
                      f"using local copy: {fallback}")
                path = fallback
            else:
                raise FileNotFoundError(
                    f"Stage-1 base model not found at recorded path "
                    f"{path!r} or fallback {fallback!r}")
        m = xgb.Booster()
        m.load_model(path)
        base_models.append(m)
    print(f"[V52-SIM] Loaded {len(base_models)} Stage-1 base models")

    # Stage-2 FP filter — V52 always trains it (memorize too) so we expect it to exist.
    stage2_disabled = bool(manifest.get("stage2_disabled", False))
    stage2_path = os.path.join(xgb_dir, f"xgboost-stage2-fp-filter_version={model_time}.json")
    if stage2_disabled or not os.path.exists(stage2_path):
        if stage2_disabled:
            print("[V52-SIM] Stage 2 disabled by manifest")
        else:
            print(f"[V52-SIM] Stage 2 file not found, running Stage 1 only: {stage2_path}")
        stage2_model = None
    else:
        stage2_model = xgb.Booster()
        stage2_model.load_model(stage2_path)
        print(f"[V52-SIM] Loaded Stage-2 FP filter: {os.path.basename(stage2_path)}")

    available_features = manifest.get("available_features", [])
    s1_thr = float(manifest.get("stage1_threshold", 0.5))
    s2_thr = float(manifest.get("stage2_threshold", 0.5))
    print(f"[V52-SIM] Thresholds: S1={s1_thr:.3f}  S2={s2_thr:.3f}")
    print(f"[V52-SIM] Available features: {len(available_features)}")

    # V52 experiment block (only present in v52 manifests; default-safe for v51).
    experiments = manifest.get("experiments", {})
    print(f"[V52-SIM] Experiments enabled at training time:")
    print(f"           pk_hour_interactions = {experiments.get('pk_hour_interactions', False)}")
    print(f"           v52_extra_features   = {experiments.get('v52_extra_features', [])}")

    return {
        "model_time":         model_time,
        "schema":             schema,
        "base_models":        base_models,
        "stage2_model":       stage2_model,
        "available_features": available_features,
        "stage1_threshold":   s1_thr,
        "stage2_threshold":   s2_thr,
        "manifest":           manifest,
        "experiments":        experiments,
    }


# =============================================================================
# V52 PREPARE SIMULATION DATAFRAME
# =============================================================================

def prepare_simulation_dataframe_v52(gnn_experiment_dir: str,
                                     experiments: dict) -> pd.DataFrame:
    """V51's `prepare_simulation_dataframe` + V52 FE extensions (experiment c)."""
    # Mirror our module-level state onto V51 sim so its prepare_* sees the
    # same MEMORIZE_MODE / dates / windows we want.
    _propagate_to_v51_sim()

    df_full = v51_sim.prepare_simulation_dataframe(gnn_experiment_dir)

    if experiments.get("pk_hour_interactions", False):
        # Use the EXACT same hot-zone / evening-rush configuration as training.
        hot_pks = tuple(experiments.get("hot_zone_pks", HOT_ZONE_PKS) or HOT_ZONE_PKS)
        evening = tuple(experiments.get("evening_rush_hours", EVENING_RUSH_HOURS)
                        or EVENING_RUSH_HOURS)
        # Fallback: read from the manifest top-level if the experiments block
        # didn't carry them (older V52 manifest variants).
        df_full, added = add_pk_hour_interactions(
            df_full, hot_pks=hot_pks, evening_rush_hours=evening)
        print(f"[V52-SIM] Replayed pk × hour interactions: {added}")

    return df_full


# =============================================================================
# RUN PIPELINE
# =============================================================================

def run_pipeline_v52(xgb_experiment_dir: str, gnn_experiment_dir: str,
                     output_dir: str | None = None) -> bool:
    global TRAIN_START_DATE, TRAIN_END_DATE, SIM_START_DATE, SIM_END_DATE
    global GNN_TRAIN_WINDOWS, MEMORIZE_MODE

    t_start = time.time()
    output_dir = output_dir or xgb_experiment_dir
    viz_dir    = os.path.join(output_dir, "visualizations")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(viz_dir,    exist_ok=True)

    # ── A. Load XGBoost artefacts (early — manifest tells us memorize_mode) ─
    art = load_v52_xgboost_artefacts(xgb_experiment_dir)
    base_models        = art["base_models"]
    stage2_model       = art["stage2_model"]
    available_features = art["available_features"]
    model_time         = art["model_time"]
    s1_thr             = art["stage1_threshold"]
    s2_thr             = art["stage2_threshold"]
    experiments        = art["experiments"]

    # MEMORIZE_MODE / window collapse (mirror V51 sim's logic).
    _manifest_memorize = bool(art["manifest"].get("memorize_mode", False))
    _env_memorize      = os.environ.get("MEMORIZE_MODE_OVERRIDE", "") == "True"
    if _manifest_memorize or _env_memorize or MEMORIZE_MODE:
        MEMORIZE_MODE     = True
        TRAIN_START_DATE  = "2025-05-01"
        TRAIN_END_DATE    = "2025-06-01"
        SIM_START_DATE    = "2025-05-01"
        SIM_END_DATE      = "2025-06-01"
        GNN_TRAIN_WINDOWS = [("2025-05-01", "2025-06-01")]
        _propagate_to_v51_sim()
        print(f"[V52-SIM][MEMORIZE] Collapsed windows to May 2025 only: "
              f"train={TRAIN_START_DATE}->{TRAIN_END_DATE}, "
              f"sim={SIM_START_DATE}->{SIM_END_DATE}")

    # CLI / env overrides win.
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
        _propagate_to_v51_sim()
        print(f"[V52-SIM][CLI-OVERRIDE] train={TRAIN_START_DATE}->{TRAIN_END_DATE}, "
              f"sim={SIM_START_DATE}->{SIM_END_DATE}")

    print("=" * 80)
    print(f"V52 Simulation-only ({art['schema']} manifest)")
    print(f"Simulation:        {SIM_START_DATE} -> {SIM_END_DATE}")
    print(f"XGBoost dir:       {xgb_experiment_dir}")
    print(f"GNN dir:           {gnn_experiment_dir}")
    print(f"Output dir:        {output_dir}")
    print(f"S1 threshold:      {s1_thr:.4f}    S2 threshold: {s2_thr:.4f}")
    print(f"Stage-2 model:     {'loaded' if stage2_model is not None else 'NONE'}")
    print("=" * 80)

    # Redirect V34 module-level paths so its evaluation stages write here.
    v34.OUTPUT_DIR = output_dir
    v34.VIZ_DIR    = viz_dir

    # Cross-check: the GNN supplied here should match the one used at training.
    expected_gnn = art["manifest"].get("gnn_experiment_dir")
    if expected_gnn and os.path.normpath(expected_gnn) != os.path.normpath(gnn_experiment_dir):
        print(f"[V52-SIM] WARNING: GNN dir differs from training "
              f"(training={expected_gnn}, sim={gnn_experiment_dir}). "
              "Proceeding — predictions may drift if GNNs differ.")

    # ── B-E. Build df_full = base + GNN predictions + V21 FE + V52 FE ──────
    df_full = prepare_simulation_dataframe_v52(gnn_experiment_dir, experiments)
    gc.collect()

    # ── F. Stage-1-only simulation (v34.stage3) ────────────────────────────
    sim_results = v34.stage3_simulation(
        df_full, base_models, available_features, model_time)

    # ── G. Stage-1 evaluation (v34.stage4) ─────────────────────────────────
    v34.stage4_evaluate_simulation(sim_results)

    # ── H/I. Two-stage simulation — uses V52 thresholds from the manifest ──
    if stage2_model is not None:
        sim_results_2s = v34.stage8_two_stage_simulation(
            df_full, base_models, stage2_model, available_features,
            model_time, s1_thr, s2_thr)
        v34.stage9_final_evaluation(sim_results, sim_results_2s)
    else:
        print("=" * 80)
        print("STAGE 8 / 9: SKIPPED (no Stage-2 model)")
        print("=" * 80)

    # ── Z. Feature-space analysis: TP vs FP vs TN distributional gap ───────
    v51_sim.stageZ_feature_space_analysis(
        df_full, sim_results, base_models, available_features,
        output_dir, viz_dir, model_time)

    # ── J. Persist V52 simulation manifest ─────────────────────────────────
    sim_manifest = {
        "pipeline_version":       PIPELINE_VERSION,
        "model_time":             model_time,
        "xgboost_experiment_dir": xgb_experiment_dir,
        "gnn_experiment_dir":     gnn_experiment_dir,
        "sim_dates":              {"start": SIM_START_DATE, "end": SIM_END_DATE},
        "pk_range":               {"min": PK_MIN, "max": PK_MAX},
        "time_resolution":        TIME_RESOLUTION,
        "stage1_threshold":       s1_thr,
        "stage2_threshold":       s2_thr,
        "stage2_model_loaded":    bool(stage2_model is not None),
        "training_manifest_schema": art["schema"],
        "experiments_at_training": experiments,
        "ablation_settings":      v34.ABLATION_SETTINGS,
        "memorize_mode":          bool(v34.MEMORIZE_MODE),
        "timestamp":              datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    sim_manifest_path = os.path.join(
        output_dir, f"v52_simulation_manifest_{model_time}.json")
    with open(sim_manifest_path, "w") as f:
        json.dump(sim_manifest, f, indent=2, default=str)
    print(f"[V52-SIM] Sim manifest saved: {sim_manifest_path}")

    print("\n" + "=" * 80)
    print(f"V52 simulation complete in {v34.format_duration(time.time() - t_start)}")
    print(f"Results & plots in: {output_dir}")
    print("=" * 80)
    return True


# =============================================================================
# MODULE-ATTR PROPAGATION
# =============================================================================

def _propagate_to_v51_sim() -> None:
    """Mirror our module attrs onto V51 sim AND v34 so their stage functions
    see the same configuration.

    V51 sim already propagates to v34 internally, so we only need to set V51's
    attrs and call its `_propagate_to_v34` if available.
    """
    v51_sim.MEMORIZE_MODE     = bool(MEMORIZE_MODE)
    v51_sim.TRAIN_START_DATE  = TRAIN_START_DATE
    v51_sim.TRAIN_END_DATE    = TRAIN_END_DATE
    v51_sim.SIM_START_DATE    = SIM_START_DATE
    v51_sim.SIM_END_DATE      = SIM_END_DATE
    v51_sim.GNN_TRAIN_WINDOWS = list(GNN_TRAIN_WINDOWS)
    v51_sim.DATA_FILE         = DATA_FILE
    v51_sim.DATA_PATH         = str(DATA_PATH)
    v51_sim.TEST_MODE         = TEST_MODE

    if hasattr(v51_sim, "_propagate_to_v34"):
        try:
            v51_sim._propagate_to_v34()
        except Exception as e:
            print(f"[V52-SIM] v51_sim._propagate_to_v34() failed: {e}")


# =============================================================================
# ENTRY POINTS
# =============================================================================

def run_simulation_only() -> bool:
    """Entry point invoked by `bash_files/run_ablation_study_simulation_only.sh`."""
    global TRAIN_START_DATE, TRAIN_END_DATE, SIM_START_DATE, SIM_END_DATE

    # Re-pin to V17-aligned dates so neither the bash runner nor a stale
    # import-time value can drift the simulation away from the training config.
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

    _propagate_to_v51_sim()
    return run_pipeline_v52(
        xgb_experiment_dir=xgb_experiment_dir,
        gnn_experiment_dir=gnn_experiment_dir,
        output_dir=xgb_experiment_dir,
    )


def main() -> int:
    global SIM_START_DATE, SIM_END_DATE, MEMORIZE_MODE

    parser = argparse.ArgumentParser(
        description="V52 Simulation-only — V51 + (a)+(b)+(c)+(d)+(f) experiments."
    )
    parser.add_argument("--xgboost-experiment-dir", required=True,
                        help="V52 XGBoost training output dir.")
    parser.add_argument("--gnn-experiment-dir", required=True,
                        help="V14/V17 GNN experiment dir (must match training).")
    parser.add_argument("--output-dir", default=None,
                        help="Where to write reports/plots. Defaults to the XGBoost dir.")
    parser.add_argument("--sim-start-date", default=SIM_START_DATE,
                        help="Override simulation start (YYYY-MM-DD).")
    parser.add_argument("--sim-end-date", default=SIM_END_DATE,
                        help="Override simulation end (YYYY-MM-DD).")
    parser.add_argument("--memorize-mode", action="store_true",
                        default=bool(MEMORIZE_MODE),
                        help="Force MEMORIZE_MODE on (collapses windows to May 2025).")

    args = parser.parse_args()

    SIM_START_DATE = args.sim_start_date
    SIM_END_DATE   = args.sim_end_date
    MEMORIZE_MODE  = bool(args.memorize_mode)

    _propagate_to_v51_sim()

    return 0 if run_pipeline_v52(
        xgb_experiment_dir=args.xgboost_experiment_dir,
        gnn_experiment_dir=args.gnn_experiment_dir,
        output_dir=args.output_dir,
    ) else 1


if __name__ == "__main__":
    sys.exit(main())
