#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Version-agnostic rolling simulation for the Layer-2 classifier benchmarks
=========================================================================

A SINGLE simulation module serves every BL-C* benchmark (v68..v73): it reads
the classifier token + features + threshold + replayed-FE config straight from
the `benchmark_classifier_manifest_*.json` written by the trainer, so it never
needs to know which benchmark it is running. Each
`ablation_study_v6X_simulation_only.py` just aliases itself to this module via
`sys.modules` (the same trick V58-sim uses to reuse V57-sim), so the generic
`run_ablation_study_simulation_only.sh` resolves `--v68 .. --v73` here.

It reuses V52's `prepare_simulation_dataframe_v52` to rebuild the *identical*
GNN-forecasted feature frame the classifier was trained on (same FE replay),
scores it with the saved classifier, applies the trainer's operating-point
threshold, and reports the frozen-protocol confusion-matrix metrics + PR curve.

Author: Gerard Franco
Date:   June 2026
"""

from __future__ import annotations
from src.paths import (  # portable paths -- see src/paths.py
    PROJECT_ROOT_STR as _AP7_ROOT,
    EXPERIMENTS_ROOT_STR as _AP7_EXPERIMENTS,
    DATA_DIR_STR as _AP7_DATA,
    TABLES_DIR as _AP7_TABLES,
    FIGURES_DIR as _AP7_FIGS,
)

import gc
import glob
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             precision_recall_curve, roc_auc_score)

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import src.training.ablation_study_v34_simulation_oof_2024apr_2025sep as v34
import src.training.ablation_study_v51_simulation_only as v51_sim
import src.training.ablation_study_v52_simulation_only as v52_sim
from src.training.ablation_study_v52_simulation_only import prepare_simulation_dataframe_v52
from src.models.benchmark_classifiers import load_benchmark_classifier
from src.training.sim_dirname import sim_dir_name


PIPELINE_VERSION = "benchmark_classifier_sim"

# ── Module attributes the generic sim bash runner pokes (mirror V57-sim) ──────
TEST_MODE = v34.TEST_MODE
MEMORIZE_MODE = v34.MEMORIZE_MODE
ABLATION_SETTINGS = v34.ABLATION_SETTINGS
FEATURE_GROUPS = v34.FEATURE_GROUPS
DATA_FILE = v34.DATA_FILE
DATA_PATH = str(v34.DATA_PATH)
SIMULATION_CONFIG = dict(v51_sim.SIMULATION_CONFIG)
TRAIN_START_DATE = v51_sim._V17_ALIGNED_TRAIN_START
TRAIN_END_DATE = v51_sim._V17_ALIGNED_TRAIN_END
# Env-overridable so validation runs can extend the sim window (e.g.
# BENCH_SIM_END=2025-10-01 for Jun->Sep multi-month temporal validation) with
# the classifier + thresholds kept frozen. Default = the June benchmark month.
SIM_START_DATE = os.environ.get("BENCH_SIM_START", v51_sim._V17_ALIGNED_SIM_START)
SIM_END_DATE = os.environ.get("BENCH_SIM_END", v51_sim._V17_ALIGNED_SIM_END)
PK_MIN = v34.PK_MIN
PK_MAX = v34.PK_MAX
TIME_RESOLUTION = v34.TIME_RESOLUTION
XGBOOST_MODEL_TIME: str | None = None
_CUSTOM_SIM_DATES = True   # tell the runner not to clobber our SIM dates


def _detect_model_time(xgb_dir: str) -> str:
    if XGBOOST_MODEL_TIME:
        return XGBOOST_MODEL_TIME
    cands = sorted(glob.glob(os.path.join(xgb_dir, "benchmark_classifier_manifest_*.json")))
    if not cands:
        raise FileNotFoundError(f"No benchmark_classifier_manifest_*.json in {xgb_dir}")
    return json.load(open(cands[-1]))["model_time"]


def _metrics_from_cm(cm):
    tn, fp, fn, tp = cm.ravel()
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    f1 = 2 * p * r / max(p + r, 1e-12)
    f2 = 5 * p * r / max(4 * p + r, 1e-12)
    far = fp / max(tp + fp, 1)
    return dict(tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn),
                precision=float(p), recall=float(r), f1=float(f1), f2=float(f2),
                far=float(far))


def _propagate_sim_dates():
    for mod in (v52_sim, v51_sim, v34):
        for attr, val in (("SIM_START_DATE", SIM_START_DATE),
                          ("SIM_END_DATE", SIM_END_DATE),
                          ("TRAIN_START_DATE", TRAIN_START_DATE),
                          ("TRAIN_END_DATE", TRAIN_END_DATE)):
            if hasattr(mod, attr):
                setattr(mod, attr, val)


def run_pipeline(xgb_experiment_dir: str, gnn_experiment_dir: str,
                 output_dir: str | None = None) -> bool:
    t0 = time.time()
    output_dir = output_dir or xgb_experiment_dir
    viz_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)

    model_time = _detect_model_time(xgb_experiment_dir)
    manifest_path = os.path.join(
        xgb_experiment_dir, f"benchmark_classifier_manifest_{model_time}.json")
    manifest = json.load(open(manifest_path))
    token = manifest.get("classifier_token", "?")
    features = manifest["stage1_features"]
    threshold = float(manifest.get("stage1_threshold", 0.5))
    experiments = manifest.get("experiments", {})

    print("=" * 80)
    print(f"LAYER-2 BENCHMARK SIMULATION — token={token}  model_time={model_time}")
    print(f"  XGB dir: {xgb_experiment_dir}")
    print(f"  GNN dir: {gnn_experiment_dir}")
    print(f"  threshold={threshold:.4f}  features={len(features)}  "
          f"sim={SIM_START_DATE}->{SIM_END_DATE}")
    print("=" * 80)

    _propagate_sim_dates()
    clf = load_benchmark_classifier(xgb_experiment_dir, model_time)
    df_full = prepare_simulation_dataframe_v52(gnn_experiment_dir, experiments)
    gc.collect()

    start, end = pd.Timestamp(SIM_START_DATE), pd.Timestamp(SIM_END_DATE)
    df_full["dat"] = pd.to_datetime(df_full["dat"])
    df_sim = df_full[(df_full["pk"] >= PK_MIN) & (df_full["pk"] <= PK_MAX)
                     & (df_full["dat"] >= start) & (df_full["dat"] < end)].copy()
    if len(df_sim) == 0:
        print(f"[WARN] No sim data for {SIM_START_DATE}->{SIM_END_DATE}")
        return False

    for col in features:
        if col not in df_sim.columns:
            df_sim[col] = 0.0
    X_sim = df_sim[features].apply(pd.to_numeric, errors="coerce").fillna(0)
    proba = clf.predict_proba(X_sim)
    pred = (proba >= threshold).astype(int)

    results = df_sim[["via", "sen", "pk", "anyo", "mes", "dia", "diaSem",
                      "hor", "min", "dat"]].copy()
    results["accident_probability"] = proba
    results["accident_pred_binary"] = pred
    results["classifier_token"] = token
    results["model_time"] = model_time
    if "ACCIDENT" in df_sim.columns:
        results["ACCIDENT_real"] = df_sim["ACCIDENT"].fillna(0).astype(int).values

    metrics = {}
    if "ACCIDENT_real" in results.columns:
        y = results["ACCIDENT_real"].values
        cm = confusion_matrix(y, pred, labels=[0, 1])
        metrics = _metrics_from_cm(cm)
        metrics["pr_auc"] = float(average_precision_score(y, proba)) if y.sum() else 0.0
        try:
            metrics["roc_auc"] = float(roc_auc_score(y, proba)) if y.sum() else 0.5
        except Exception:
            metrics["roc_auc"] = 0.5
        print(f"\n[{token}] frozen-protocol metrics ({SIM_START_DATE}->{SIM_END_DATE}):")
        for k in ("tp", "fp", "fn", "precision", "recall", "f1", "f2", "far",
                  "pr_auc", "roc_auc"):
            v = metrics[k]
            print(f"   {k:10s} {v:,.4f}" if isinstance(v, float) else f"   {k:10s} {v:,}")

        fig, ax = plt.subplots(figsize=(6, 5))
        pc, rc, _ = precision_recall_curve(y, proba)
        ax.plot(rc, pc, color="navy", label=f"{token} (AUCPR={metrics['pr_auc']:.4f})")
        ax.axhline(y.mean(), color="gray", ls=":", label=f"prevalence={y.mean():.5f}")
        ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
        ax.set_title(f"PR — Layer-2 benchmark ({token})", fontweight="bold")
        ax.legend(fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, f"benchmark_pr_curve_{token}.png"),
                    dpi=120, bbox_inches="tight")
        plt.close(fig)

    # A simulation is identified by the model, its TRAINING window and its
    # SIMULATION window: same model + different sim window gives different
    # predictions for the same rows (the FE baselines are estimated from the
    # assembled frame), and same sim window + different training window is a
    # different model. Folders keyed on model_time alone silently overwrote
    # each other; see src/training/sim_dirname.py for the full history.
    _tr = manifest.get("train_dates") or {}
    sim_dir = os.path.join(output_dir, sim_dir_name(
        "simulation_benchmark", model_time,
        _tr.get("start", TRAIN_START_DATE), _tr.get("end", TRAIN_END_DATE),
        SIM_START_DATE, SIM_END_DATE))
    os.makedirs(sim_dir, exist_ok=True)
    results.to_csv(os.path.join(sim_dir, f"simulation_benchmark_{token}_{model_time}.csv"),
                   sep=";", index=False)

    sim_manifest = {
        "pipeline_version": PIPELINE_VERSION,
        "benchmark_layer": "L2_classifier",
        "classifier_token": token,
        "model_time": model_time,
        "xgboost_experiment_dir": xgb_experiment_dir,
        "gnn_experiment_dir": gnn_experiment_dir,
        "sim_dates": {"start": SIM_START_DATE, "end": SIM_END_DATE},
        "threshold": threshold,
        "metrics": metrics,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(output_dir,
                           f"benchmark_classifier_sim_manifest_{model_time}.json"), "w") as f:
        json.dump(sim_manifest, f, indent=2, default=str)

    print(f"\n[DONE] Layer-2 benchmark sim ({token}) in "
          f"{v34.format_duration(time.time() - t0)} -> {sim_dir}")
    return True


def run_simulation_only() -> bool:
    xgb_experiment_dir = os.environ.get("XGBOOST_EXPERIMENT_DIR", "")
    gnn_experiment_dir = os.environ.get("GNN_EXPERIMENT_DIR", "")
    if not xgb_experiment_dir or not os.path.isdir(xgb_experiment_dir):
        print(f"[ERROR] XGBOOST_EXPERIMENT_DIR invalid: {xgb_experiment_dir!r}")
        sys.exit(1)
    if not gnn_experiment_dir or not os.path.isdir(gnn_experiment_dir):
        print(f"[ERROR] GNN_EXPERIMENT_DIR invalid: {gnn_experiment_dir!r}")
        sys.exit(1)
    return run_pipeline(xgb_experiment_dir, gnn_experiment_dir,
                        output_dir=xgb_experiment_dir)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Layer-2 benchmark simulation.")
    p.add_argument("--xgboost-experiment-dir", required=True)
    p.add_argument("--gnn-experiment-dir", required=True)
    args = p.parse_args()
    os.environ["XGBOOST_EXPERIMENT_DIR"] = args.xgboost_experiment_dir
    os.environ["GNN_EXPERIMENT_DIR"] = args.gnn_experiment_dir
    sys.exit(0 if run_simulation_only() else 1)
