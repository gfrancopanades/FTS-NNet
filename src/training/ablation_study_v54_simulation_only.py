#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V54 Simulation-only
===================
Companion to `ablation_study_v54_xgboost_only.py`.

V54 simulation is identical to V53 simulation: it
  - replays pk×hour interactions at sim time when enabled in the manifest
  - runs Stage-1 simulation with the manifest's `stage1_features`
  - runs Two-Stage simulation slicing Stage-2 inputs to `stage2_features + s1_score`

The only differences are schema detection (`v54_xgboost_manifest_*.json`) and
the pipeline_version string written to the sim manifest.
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

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import src.training.ablation_study_v34_simulation_oof_2024apr_2025sep as v34
import src.training.ablation_study_v51_simulation_only as v51_sim
import src.training.ablation_study_v52_simulation_only as v52_sim
import src.training.ablation_study_v54_xgboost_only as v54_xgb  # noqa: F401
from src.training.ablation_study_v52_simulation_only import prepare_simulation_dataframe_v52


PIPELINE_VERSION = "v54_simulation_only"

_V17_ALIGNED_TRAIN_START = v51_sim._V17_ALIGNED_TRAIN_START
_V17_ALIGNED_TRAIN_END = v51_sim._V17_ALIGNED_TRAIN_END
_V17_ALIGNED_SIM_START = v51_sim._V17_ALIGNED_SIM_START
_V17_ALIGNED_SIM_END = v51_sim._V17_ALIGNED_SIM_END

GNN_TRAIN_WINDOWS = list(v51_sim.GNN_TRAIN_WINDOWS)
TRAIN_START_DATE = _V17_ALIGNED_TRAIN_START
TRAIN_END_DATE = _V17_ALIGNED_TRAIN_END
SIM_START_DATE = _V17_ALIGNED_SIM_START
SIM_END_DATE = _V17_ALIGNED_SIM_END
PK_MIN = v34.PK_MIN
PK_MAX = v34.PK_MAX
TIME_RESOLUTION = v34.TIME_RESOLUTION

TEST_MODE = v34.TEST_MODE
MEMORIZE_MODE = v34.MEMORIZE_MODE
ABLATION_SETTINGS = v34.ABLATION_SETTINGS
FEATURE_GROUPS = v34.FEATURE_GROUPS
DATA_FILE = v34.DATA_FILE
DATA_PATH = str(v34.DATA_PATH)

SIMULATION_CONFIG: dict = dict(v51_sim.SIMULATION_CONFIG)
XGBOOST_MODEL_TIME: str | None = None
_CUSTOM_SIM_DATES = True


def _detect_xgb_model_time_v54(xgb_dir: str) -> tuple[str, str]:
    """Locate XGB model_time, preferring v54 → v53 → v52 → v51 → legacy."""
    for schema in ("v54", "v53", "v52", "v51"):
        cands = sorted(glob.glob(os.path.join(xgb_dir, f"{schema}_xgboost_manifest_*.json")))
        if cands:
            with open(cands[-1]) as f:
                return json.load(f)["model_time"], schema
    metas = glob.glob(os.path.join(xgb_dir, "xgboost-metadata_model=XGBoost-version=*.json"))
    if metas:
        return os.path.basename(metas[0]).split("version=")[1].rsplit(".json", 1)[0], "legacy"
    raise FileNotFoundError(f"No V54/V53/V52/V51 XGBoost manifest found in {xgb_dir}")


def load_v54_xgboost_artefacts(xgb_dir: str) -> dict:
    print("=" * 80)
    print("V54-SIM STAGE A: LOADING XGBOOST ARTEFACTS")
    print("=" * 80)

    model_time, schema = _detect_xgb_model_time_v54(xgb_dir)
    print(f"[V54-SIM] XGBoost dir       : {xgb_dir}")
    print(f"[V54-SIM] XGBoost model_time: {model_time}")
    print(f"[V54-SIM] Manifest schema   : {schema}")

    manifest = {}
    if schema in ("v54", "v53", "v52", "v51"):
        manifest_path = os.path.join(xgb_dir, f"{schema}_xgboost_manifest_{model_time}.json")
        if os.path.exists(manifest_path):
            with open(manifest_path) as f:
                manifest = json.load(f)

    ens_meta_path = os.path.join(
        xgb_dir, f"xgboost-ensemble-metadata_target=accident_version={model_time}.json"
    )
    if not os.path.exists(ens_meta_path):
        raise FileNotFoundError(f"Ensemble metadata missing: {ens_meta_path}")
    with open(ens_meta_path) as f:
        ens_meta = json.load(f)

    base_models = []
    for path in ens_meta["base_model_paths"]:
        if not os.path.exists(path):
            fallback = os.path.join(xgb_dir, os.path.basename(path))
            if os.path.exists(fallback):
                print(f"[V54-SIM] Base model path stale ({path!r}); using local copy: {fallback}")
                path = fallback
            else:
                raise FileNotFoundError(f"Stage-1 base model not found: {path!r}")
        m = xgb.Booster()
        m.load_model(path)
        base_models.append(m)
    print(f"[V54-SIM] Loaded {len(base_models)} Stage-1 base models")

    stage2_disabled = bool(manifest.get("stage2_disabled", False))
    stage2_path = os.path.join(xgb_dir, f"xgboost-stage2-fp-filter_version={model_time}.json")
    if stage2_disabled or not os.path.exists(stage2_path):
        if stage2_disabled:
            print("[V54-SIM] Stage 2 disabled by manifest")
        else:
            print(f"[V54-SIM] Stage 2 file not found, running Stage 1 only: {stage2_path}")
        stage2_model = None
    else:
        stage2_model = xgb.Booster()
        stage2_model.load_model(stage2_path)
        print(f"[V54-SIM] Loaded Stage-2 FP filter: {os.path.basename(stage2_path)}")

    stage1_features = manifest.get("stage1_features") or manifest.get("available_features", [])
    stage2_features = manifest.get("stage2_features") or list(stage1_features)

    s1_thr = float(manifest.get("stage1_threshold", 0.5))
    s2_thr = float(manifest.get("stage2_threshold", 0.5))
    experiments = manifest.get("experiments", {})

    print(f"[V54-SIM] Thresholds: S1={s1_thr:.3f}  S2={s2_thr:.3f}")
    print(f"[V54-SIM] Stage-1 features: {len(stage1_features)}")
    print(f"[V54-SIM] Stage-2 features: {len(stage2_features)} (+ s1_score)")

    return {
        "model_time": model_time,
        "schema": schema,
        "base_models": base_models,
        "stage2_model": stage2_model,
        "stage1_features": stage1_features,
        "stage2_features": stage2_features,
        "stage1_threshold": s1_thr,
        "stage2_threshold": s2_thr,
        "manifest": manifest,
        "experiments": experiments,
    }


def _metrics_from_cm(cm: np.ndarray) -> tuple[int, int, int, float, float, float]:
    tn, fp, fn, tp = cm.ravel()
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    f1 = 2 * p * r / max(p + r, 1e-12)
    return int(tp), int(fp), int(fn), float(p), float(r), float(f1)


def stage8_two_stage_simulation_v54(df_full: pd.DataFrame,
                                    base_models: list,
                                    stage2_model: xgb.Booster,
                                    stage1_features: list[str],
                                    stage2_features: list[str],
                                    model_time: str,
                                    s1_thr: float,
                                    s2_thr: float,
                                    output_dir: str,
                                    viz_dir: str,
                                    sim_start: str,
                                    sim_end: str) -> pd.DataFrame | None:
    print("=" * 80)
    print("STAGE 8 (V54): Two-Stage Rolling Simulation — Stage-2 feature subset")
    print("=" * 80)

    start = pd.Timestamp(sim_start)
    end = pd.Timestamp(sim_end)
    df_sim = df_full[
        (df_full["pk"] >= PK_MIN) & (df_full["pk"] <= PK_MAX) &
        (df_full["dat"] >= start) & (df_full["dat"] < end)
    ].copy()
    if len(df_sim) == 0:
        print(f"[WARN] No data for sim period {sim_start} -> {sim_end}")
        return None

    for col in stage1_features:
        if col not in df_sim.columns:
            df_sim[col] = 0
    X_sim = df_sim[stage1_features].apply(pd.to_numeric, errors="coerce").fillna(0)
    d_sim = xgb.DMatrix(X_sim)

    s1_probs = np.zeros(len(X_sim))
    for m in base_models:
        best_iter = getattr(m, "best_iteration", None)
        s1_probs += (m.predict(d_sim, iteration_range=(0, best_iter)) if best_iter else m.predict(d_sim))
    s1_probs /= max(len(base_models), 1)

    s1_flag = s1_probs >= s1_thr
    s2_prob = np.zeros(len(X_sim))
    s2_bin = np.zeros(len(X_sim), dtype=int)
    if s1_flag.sum() > 0:
        X_flagged = X_sim[s1_flag].copy()
        X_flagged["s1_score"] = s1_probs[s1_flag]
        missing = [c for c in stage2_features if c not in X_flagged.columns]
        for c in missing:
            X_flagged[c] = 0
        X_flagged_s2 = X_flagged[list(stage2_features) + ["s1_score"]]
        s2_probs_flagged = stage2_model.predict(xgb.DMatrix(X_flagged_s2))
        flagged_pos = np.where(s1_flag)[0]
        s2_prob[flagged_pos] = s2_probs_flagged
        s2_bin[flagged_pos] = (s2_probs_flagged >= s2_thr).astype(int)

    results = df_sim[["via", "sen", "pk", "anyo", "mes", "dia", "diaSem", "hor", "min", "dat"]].copy()
    results["s1_probability"] = s1_probs
    results["s1_pred_binary"] = s1_flag.astype(int)
    results["s2_probability"] = s2_prob
    results["s2_pred_binary"] = s2_bin
    results["accident_probability"] = s2_prob
    results["accident_pred_binary"] = s2_bin
    if "ACCIDENT" in df_sim.columns:
        results["ACCIDENT_real"] = df_sim["ACCIDENT"].fillna(0).astype(int).values
    results["model_time"] = model_time
    results["simulation_mode"] = "two_stage_no_gnn_v54"

    if "ACCIDENT_real" in results.columns:
        df_e = results.dropna(subset=["ACCIDENT_real", "s1_probability"]).copy()
        y_t = df_e["ACCIDENT_real"].astype(int).values
        s1p = df_e["s1_probability"].values
        s2p = df_e["s2_probability"].values
        s1b = df_e["s1_pred_binary"].values
        s2b = df_e["s2_pred_binary"].values
        cm_s1 = confusion_matrix(y_t, s1b)
        cm_s2 = confusion_matrix(y_t, s2b)
        tp1, fp1, fn1, pr1, rc1, f1_1 = _metrics_from_cm(cm_s1)
        tp2, fp2, fn2, pr2, rc2, f1_2 = _metrics_from_cm(cm_s2)
        auc1 = average_precision_score(y_t, s1p)
        auc2 = average_precision_score(y_t, s2p)
        print(f"\n{'':22s}  {'Stage 1':>10s}  {'Two-Stage':>10s}")
        print("-" * 47)
        for label, v1, v2 in [("TP", tp1, tp2), ("FP", fp1, fp2), ("FN", fn1, fn2),
                              ("Precision", pr1, pr2), ("Recall", rc1, rc2),
                              ("F1", f1_1, f1_2), ("PR-AUC", auc1, auc2)]:
            if isinstance(v1, float):
                print(f"{label:22s}  {v1:>10.4f}  {v2:>10.4f}")
            else:
                print(f"{label:22s}  {v1:>10,}  {v2:>10,}")

        sim_dir = os.path.join(output_dir, f"simulation_two_stage_{model_time}")
        os.makedirs(sim_dir, exist_ok=True)
        sim_path = os.path.join(sim_dir, f"simulation_two_stage_{model_time}.csv")
        results.to_csv(sim_path, sep=";", index=False)
        print(f"\n[SUCCESS] Two-stage simulation saved: {sim_path}")

        fig, axes = plt.subplots(1, 1, figsize=(6, 5))
        pc2, rc2c, _ = precision_recall_curve(y_t, s2p)
        axes.plot(rc2c, pc2, label=f"Two-Stage (AUCPR={auc2:.4f})", color="crimson")
        axes.axhline(y_t.mean(), color="gray", linestyle=":", alpha=0.6, label=f"Baseline (prev={y_t.mean():.5f})")
        axes.set_xlabel("Recall"); axes.set_ylabel("Precision")
        axes.set_title("PR Curve — Two-Stage (V54)", fontsize=12, fontweight="bold")
        axes.legend(fontsize=9)
        os.makedirs(viz_dir, exist_ok=True)
        fig_path = os.path.join(viz_dir, "stage8_two_stage_simulation.png")
        plt.tight_layout()
        plt.savefig(fig_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"[PLOT] Saved: {fig_path}")

    return results


def _propagate_to_v52_sim() -> None:
    v52_sim.MEMORIZE_MODE = bool(MEMORIZE_MODE)
    v52_sim.TRAIN_START_DATE = TRAIN_START_DATE
    v52_sim.TRAIN_END_DATE = TRAIN_END_DATE
    v52_sim.SIM_START_DATE = SIM_START_DATE
    v52_sim.SIM_END_DATE = SIM_END_DATE
    v52_sim.GNN_TRAIN_WINDOWS = list(GNN_TRAIN_WINDOWS)
    v52_sim.DATA_FILE = DATA_FILE
    v52_sim.DATA_PATH = str(DATA_PATH)
    v52_sim.TEST_MODE = TEST_MODE
    if hasattr(v52_sim, "_propagate_to_v51_sim"):
        v52_sim._propagate_to_v51_sim()


def run_pipeline_v54(xgb_experiment_dir: str, gnn_experiment_dir: str, output_dir: str | None = None) -> bool:
    global TRAIN_START_DATE, TRAIN_END_DATE, SIM_START_DATE, SIM_END_DATE
    global GNN_TRAIN_WINDOWS, MEMORIZE_MODE

    t_start = time.time()
    output_dir = output_dir or xgb_experiment_dir
    viz_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)

    art = load_v54_xgboost_artefacts(xgb_experiment_dir)
    base_models = art["base_models"]
    stage2_model = art["stage2_model"]
    stage1_features = art["stage1_features"]
    stage2_features = art["stage2_features"]
    model_time = art["model_time"]
    s1_thr = art["stage1_threshold"]
    s2_thr = art["stage2_threshold"]
    experiments = art["experiments"]

    _manifest_memorize = bool(art["manifest"].get("memorize_mode", False))
    _env_memorize = os.environ.get("MEMORIZE_MODE_OVERRIDE", "") == "True"
    if _manifest_memorize or _env_memorize or MEMORIZE_MODE:
        MEMORIZE_MODE = True
        TRAIN_START_DATE = "2025-05-01"
        TRAIN_END_DATE = "2025-06-01"
        SIM_START_DATE = "2025-05-01"
        SIM_END_DATE = "2025-06-01"
        GNN_TRAIN_WINDOWS = [("2025-05-01", "2025-06-01")]
        _propagate_to_v52_sim()

    _cli_ini_test = os.environ.get("INI_TEST", "")
    _cli_end_test = os.environ.get("END_TEST", "")
    if _cli_ini_test:
        SIM_START_DATE = _cli_ini_test
    if _cli_end_test:
        SIM_END_DATE = _cli_end_test
    if any([_cli_ini_test, _cli_end_test]):
        _propagate_to_v52_sim()

    v34.OUTPUT_DIR = output_dir
    v34.VIZ_DIR = viz_dir

    df_full = prepare_simulation_dataframe_v52(gnn_experiment_dir, experiments)
    gc.collect()

    sim_results = v34.stage3_simulation(df_full, base_models, stage1_features, model_time)
    v34.stage4_evaluate_simulation(sim_results)

    if stage2_model is not None:
        sim_results_2s = stage8_two_stage_simulation_v54(
            df_full=df_full,
            base_models=base_models,
            stage2_model=stage2_model,
            stage1_features=stage1_features,
            stage2_features=stage2_features,
            model_time=model_time,
            s1_thr=s1_thr,
            s2_thr=s2_thr,
            output_dir=output_dir,
            viz_dir=viz_dir,
            sim_start=SIM_START_DATE,
            sim_end=SIM_END_DATE,
        )
        v34.stage9_final_evaluation(sim_results, sim_results_2s)

    v51_sim.stageZ_feature_space_analysis(
        df_full, sim_results, base_models, stage1_features, output_dir, viz_dir, model_time
    )

    sim_manifest = {
        "pipeline_version": PIPELINE_VERSION,
        "model_time": model_time,
        "xgboost_experiment_dir": xgb_experiment_dir,
        "gnn_experiment_dir": gnn_experiment_dir,
        "sim_dates": {"start": SIM_START_DATE, "end": SIM_END_DATE},
        "pk_range": {"min": PK_MIN, "max": PK_MAX},
        "time_resolution": TIME_RESOLUTION,
        "stage1_threshold": s1_thr,
        "stage2_threshold": s2_thr,
        "stage2_model_loaded": bool(stage2_model is not None),
        "training_manifest_schema": art["schema"],
        "stage1_features": stage1_features,
        "stage2_features": stage2_features,
        "experiments_at_training": experiments,
        "ablation_settings": v34.ABLATION_SETTINGS,
        "memorize_mode": bool(v34.MEMORIZE_MODE),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    sim_manifest_path = os.path.join(output_dir, f"v54_simulation_manifest_{model_time}.json")
    with open(sim_manifest_path, "w") as f:
        json.dump(sim_manifest, f, indent=2, default=str)
    print(f"[V54-SIM] Sim manifest saved: {sim_manifest_path}")

    print("\n" + "=" * 80)
    print(f"V54 simulation complete in {v34.format_duration(time.time() - t_start)}")
    print(f"Results & plots in: {output_dir}")
    print("=" * 80)
    return True


def run_simulation_only() -> bool:
    global TRAIN_START_DATE, TRAIN_END_DATE, SIM_START_DATE, SIM_END_DATE
    TRAIN_START_DATE = _V17_ALIGNED_TRAIN_START
    TRAIN_END_DATE = _V17_ALIGNED_TRAIN_END
    SIM_START_DATE = _V17_ALIGNED_SIM_START
    SIM_END_DATE = _V17_ALIGNED_SIM_END

    xgb_experiment_dir = os.environ.get("XGBOOST_EXPERIMENT_DIR", "")
    gnn_experiment_dir = os.environ.get("GNN_EXPERIMENT_DIR", "")
    if not xgb_experiment_dir or not os.path.isdir(xgb_experiment_dir):
        print(f"[ERROR] XGBOOST_EXPERIMENT_DIR not set or invalid: {xgb_experiment_dir!r}")
        sys.exit(1)
    if not gnn_experiment_dir or not os.path.isdir(gnn_experiment_dir):
        print(f"[ERROR] GNN_EXPERIMENT_DIR not set or invalid: {gnn_experiment_dir!r}")
        sys.exit(1)

    _propagate_to_v52_sim()
    return run_pipeline_v54(xgb_experiment_dir, gnn_experiment_dir, output_dir=xgb_experiment_dir)


def main() -> int:
    global SIM_START_DATE, SIM_END_DATE, MEMORIZE_MODE
    p = argparse.ArgumentParser(description="V54 Simulation-only.")
    p.add_argument("--xgboost-experiment-dir", required=True)
    p.add_argument("--gnn-experiment-dir", required=True)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--sim-start-date", default=SIM_START_DATE)
    p.add_argument("--sim-end-date", default=SIM_END_DATE)
    p.add_argument("--memorize-mode", action="store_true", default=bool(MEMORIZE_MODE))
    args = p.parse_args()

    SIM_START_DATE = args.sim_start_date
    SIM_END_DATE = args.sim_end_date
    MEMORIZE_MODE = bool(args.memorize_mode)
    _propagate_to_v52_sim()
    return 0 if run_pipeline_v54(args.xgboost_experiment_dir, args.gnn_experiment_dir, output_dir=args.output_dir) else 1


if __name__ == "__main__":
    sys.exit(main())

