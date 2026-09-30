#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V57 Simulation-only
===================
Companion to `ablation_study_v57_xgboost_only.py`.

V57 simulation is V55 simulation with the Stage-2 model swapped: the FP
filter is a tabular transformer loaded from
`transformer-stage2-fp-filter_version=<model_time>.pt` instead of an
XGBoost booster. The checkpoint is self-describing (architecture config,
feature names, standardisation stats), so the sim rebuilds the model from
the checkpoint alone and applies the thresholds the trainer's sweep wrote
into `v57_xgboost_manifest_*.json`.
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
from src.training.sim_dirname import sim_dir_name
import pandas as pd
import torch
import xgboost as xgb
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import src.training.ablation_study_v34_simulation_oof_2024apr_2025sep as v34
import src.training.ablation_study_v51_simulation_only as v51_sim
import src.training.ablation_study_v54_simulation_only as v54_sim
from src.training.ablation_study_v52_simulation_only import prepare_simulation_dataframe_v52
from src.training.ablation_study_v57_xgboost_only import (
    TabTransformerFPFilter,
    _predict_transformer,
)


PIPELINE_VERSION = "v57_simulation_only"

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


def _detect_xgb_model_time_v57(xgb_dir: str) -> tuple[str, str]:
    """Locate XGB model_time, preferring v59 → v58 → v57 → v55 → v54 → v53 → v52 → v51 → legacy.

    V58/V59 reuse the V57 transformer checkpoint format byte-for-byte, so the
    loader below reconstructs them unchanged; only the manifest name differs,
    hence "v59"/"v58" are added to the front of the preference list. (V59 = V58
    model with a precision-first operating-point selection in training.)"""
    for schema in ("v59", "v58", "v57", "v55", "v54", "v53", "v52", "v51"):
        cands = sorted(glob.glob(os.path.join(xgb_dir, f"{schema}_xgboost_manifest_*.json")))
        if cands:
            with open(cands[-1]) as f:
                return json.load(f)["model_time"], schema
    metas = glob.glob(os.path.join(xgb_dir, "xgboost-metadata_model=XGBoost-version=*.json"))
    if metas:
        return os.path.basename(metas[0]).split("version=")[1].rsplit(".json", 1)[0], "legacy"
    raise FileNotFoundError(f"No V57/V55/V54/V53/V52/V51 XGBoost manifest found in {xgb_dir}")


def _load_stage2_transformer(xgb_dir: str, model_time: str):
    """Rebuild the Stage-2 transformer from its self-describing checkpoint.

    Returns (model, feature_names, feat_mean, feat_scale) or None if the
    checkpoint is missing.
    """
    ckpt_path = os.path.join(
        xgb_dir, f"transformer-stage2-fp-filter_version={model_time}.pt"
    )
    if not os.path.exists(ckpt_path):
        return None
    ckpt = torch.load(ckpt_path, map_location="cpu")
    arch = ckpt["arch_config"]
    model = TabTransformerFPFilter(
        n_features=arch["n_features"],
        d_model=arch["d_model"],
        n_heads=arch["n_heads"],
        n_layers=arch["n_layers"],
        dropout=arch["dropout"],
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    feat_mean = np.asarray(ckpt["feat_mean"], dtype=np.float32)
    feat_scale = np.asarray(ckpt["feat_scale"], dtype=np.float32)
    print(f"[V57-SIM] Loaded Stage-2 transformer: {os.path.basename(ckpt_path)}")
    print(f"[V57-SIM]   arch: d_model={arch['d_model']}  heads={arch['n_heads']}  "
          f"layers={arch['n_layers']}  dropout={arch['dropout']:.3f}  "
          f"n_features={arch['n_features']}")
    return model, list(ckpt["feature_names"]), feat_mean, feat_scale


def load_v57_xgboost_artefacts(xgb_dir: str) -> dict:
    print("=" * 80)
    print("V57-SIM STAGE A: LOADING ARTEFACTS (Stage-1 XGB + Stage-2 transformer)")
    print("=" * 80)

    model_time, schema = _detect_xgb_model_time_v57(xgb_dir)
    print(f"[V57-SIM] XGBoost dir       : {xgb_dir}")
    print(f"[V57-SIM] XGBoost model_time: {model_time}")
    print(f"[V57-SIM] Manifest schema   : {schema}")

    manifest: dict = {}
    if schema != "legacy":
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
                print(f"[V57-SIM] Base model path stale ({path!r}); using local copy: {fallback}")
                path = fallback
            else:
                raise FileNotFoundError(f"Stage-1 base model not found: {path!r}")
        m = xgb.Booster()
        m.load_model(path)
        base_models.append(m)
    print(f"[V57-SIM] Loaded {len(base_models)} Stage-1 base models")

    stage2_disabled = bool(manifest.get("stage2_disabled", False))
    stage2_bundle = None
    if stage2_disabled:
        print("[V57-SIM] Stage 2 disabled by manifest")
    else:
        stage2_bundle = _load_stage2_transformer(xgb_dir, model_time)
        if stage2_bundle is None:
            print(f"[V57-SIM] Stage-2 transformer checkpoint not found, "
                  f"running Stage 1 only")

    stage1_features = manifest.get("stage1_features") or manifest.get("available_features", [])
    stage2_features = manifest.get("stage2_features") or list(stage1_features)

    s1_thr = float(manifest.get("stage1_threshold", 0.5))
    s2_thr = float(manifest.get("stage2_threshold", 0.5))
    experiments = manifest.get("experiments", {})

    print(f"[V57-SIM] Thresholds: S1={s1_thr:.4f}  S2={s2_thr:.4f}")
    print(f"[V57-SIM] Stage-1 features: {len(stage1_features)}")
    print(f"[V57-SIM] Stage-2 features: {len(stage2_features)} (+ s1_score)")

    return {
        "model_time": model_time,
        "schema": schema,
        "base_models": base_models,
        "stage2_bundle": stage2_bundle,
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


def stage8_two_stage_simulation_v57(df_full: pd.DataFrame,
                                    base_models: list,
                                    stage2_bundle: tuple,
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
    print("STAGE 8 (V57): Two-Stage Rolling Simulation — TRANSFORMER Stage 2")
    print("=" * 80)

    stage2_model, s2_ckpt_features, feat_mean, feat_scale = stage2_bundle

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
        # Reconstruct EXACTLY the column order the transformer was trained on
        # (stored in the checkpoint); missing columns are zero-filled.
        missing = [c for c in s2_ckpt_features if c not in X_flagged.columns]
        for c in missing:
            X_flagged[c] = 0
        if missing:
            print(f"[V57-SIM] WARNING: zero-filled {len(missing)} Stage-2 "
                  f"feature(s) missing from sim data: {missing}")
        X_flagged_s2 = X_flagged[s2_ckpt_features].values.astype(np.float32)
        X_flagged_std = (X_flagged_s2 - feat_mean) / feat_scale
        s2_probs_flagged = _predict_transformer(stage2_model, X_flagged_std, "cpu")
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
    results["simulation_mode"] = "two_stage_transformer_v57"

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

        # keyed on model + training window + simulation window (sim_dirname.py)
        sim_dir = os.path.join(output_dir, sim_dir_name(
            "simulation_two_stage", model_time,
            globals().get("TRAIN_START_DATE"), globals().get("TRAIN_END_DATE"),
            globals().get("SIM_START_DATE"), globals().get("SIM_END_DATE")))
        os.makedirs(sim_dir, exist_ok=True)
        sim_path = os.path.join(sim_dir, f"simulation_two_stage_{model_time}.csv")
        results.to_csv(sim_path, sep=";", index=False)
        print(f"\n[SUCCESS] Two-stage simulation saved: {sim_path}")

        fig, axes = plt.subplots(1, 1, figsize=(6, 5))
        pc2, rc2c, _ = precision_recall_curve(y_t, s2p)
        axes.plot(rc2c, pc2, label=f"Two-Stage transformer (AUCPR={auc2:.4f})", color="crimson")
        axes.axhline(y_t.mean(), color="gray", linestyle=":", alpha=0.6,
                     label=f"Baseline (prev={y_t.mean():.5f})")
        axes.set_xlabel("Recall"); axes.set_ylabel("Precision")
        axes.set_title("PR Curve — Two-Stage transformer (V57)", fontsize=12, fontweight="bold")
        axes.legend(fontsize=9)
        os.makedirs(viz_dir, exist_ok=True)
        fig_path = os.path.join(viz_dir, "stage8_two_stage_simulation.png")
        plt.tight_layout()
        plt.savefig(fig_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"[PLOT] Saved: {fig_path}")

    return results


def _propagate_to_v54_sim() -> None:
    v54_sim.MEMORIZE_MODE = bool(MEMORIZE_MODE)
    v54_sim.TRAIN_START_DATE = TRAIN_START_DATE
    v54_sim.TRAIN_END_DATE = TRAIN_END_DATE
    v54_sim.SIM_START_DATE = SIM_START_DATE
    v54_sim.SIM_END_DATE = SIM_END_DATE
    v54_sim.GNN_TRAIN_WINDOWS = list(GNN_TRAIN_WINDOWS)
    v54_sim.DATA_FILE = DATA_FILE
    v54_sim.DATA_PATH = str(DATA_PATH)
    v54_sim.TEST_MODE = TEST_MODE
    if hasattr(v54_sim, "_propagate_to_v52_sim"):
        v54_sim._propagate_to_v52_sim()


def run_pipeline_v57(xgb_experiment_dir: str, gnn_experiment_dir: str,
                     output_dir: str | None = None) -> bool:
    global TRAIN_START_DATE, TRAIN_END_DATE, SIM_START_DATE, SIM_END_DATE
    global GNN_TRAIN_WINDOWS, MEMORIZE_MODE

    t_start = time.time()
    output_dir = output_dir or xgb_experiment_dir
    viz_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)

    art = load_v57_xgboost_artefacts(xgb_experiment_dir)
    base_models = art["base_models"]
    stage2_bundle = art["stage2_bundle"]
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
        _propagate_to_v54_sim()

    _cli_ini_test = os.environ.get("INI_TEST", "")
    _cli_end_test = os.environ.get("END_TEST", "")
    if _cli_ini_test:
        SIM_START_DATE = _cli_ini_test
    if _cli_end_test:
        SIM_END_DATE = _cli_end_test
    if any([_cli_ini_test, _cli_end_test]):
        _propagate_to_v54_sim()

    v34.OUTPUT_DIR = output_dir
    v34.VIZ_DIR = viz_dir

    df_full = prepare_simulation_dataframe_v52(gnn_experiment_dir, experiments)
    gc.collect()

    sim_results = v34.stage3_simulation(df_full, base_models, stage1_features, model_time)
    v34.stage4_evaluate_simulation(sim_results)

    if stage2_bundle is not None:
        sim_results_2s = stage8_two_stage_simulation_v57(
            df_full=df_full,
            base_models=base_models,
            stage2_bundle=stage2_bundle,
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
        "stage2_model_loaded": bool(stage2_bundle is not None),
        "stage2_model_type": "transformer",
        "training_manifest_schema": art["schema"],
        "stage1_features": stage1_features,
        "stage2_features": stage2_features,
        "experiments_at_training": experiments,
        "ablation_settings": v34.ABLATION_SETTINGS,
        "memorize_mode": bool(v34.MEMORIZE_MODE),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    sim_manifest_path = os.path.join(output_dir, f"v57_simulation_manifest_{model_time}.json")
    with open(sim_manifest_path, "w") as f:
        json.dump(sim_manifest, f, indent=2, default=str)
    print(f"[V57-SIM] Sim manifest saved: {sim_manifest_path}")

    print("\n" + "=" * 80)
    print(f"V57 simulation complete in {v34.format_duration(time.time() - t_start)}")
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

    _propagate_to_v54_sim()
    return run_pipeline_v57(xgb_experiment_dir, gnn_experiment_dir,
                            output_dir=xgb_experiment_dir)


def main() -> int:
    global SIM_START_DATE, SIM_END_DATE, MEMORIZE_MODE
    p = argparse.ArgumentParser(description="V57 Simulation-only (transformer Stage 2).")
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
    _propagate_to_v54_sim()
    return 0 if run_pipeline_v57(args.xgboost_experiment_dir, args.gnn_experiment_dir,
                                 output_dir=args.output_dir) else 1


if __name__ == "__main__":
    sys.exit(main())
