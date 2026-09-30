#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V52 XGBoost-only: V51 + four FP-reduction experiments
======================================================
Builds on V51 (`ablation_study_v51_xgboost_only.py`) with the four
highest-priority experiments identified in the V51 TP/FP/FN analysis notebook
(`notebooks/v51_v17_TPFPFN_analysis.ipynb`):

  (a) Threshold lift — wider sweep grid (S1 0.50 → 0.99 step 0.025; S2
      0.10 → 0.95 step 0.05) and lower min-recall constraint (0.85 vs
      V51's 0.95) so the optimal cut can move toward higher precision.
      Run in BOTH memorize and non-memorize modes (V51 skips the sweep
      under MEMORIZE_MODE).

  (b) Stage-2 FP filter is already trained in memorize mode by V51 itself
      (its memorize Stage-2 recipe is balanced-bagging Stage-1-style with
      a different seed).  V52 ADDITIONALLY runs the threshold sweep on top
      so the manifest carries non-trivial S1/S2 thresholds in memorize too.

  (c) Explicit pk × hour interaction features targeting the precision-deficit
      hotspots from cell 16.3 (PK ∈ {124-126, 149-150} × hour ∈ {17, 18}).
      Adds: pk_x_hor, pk_x_hor_sin, pk_x_hor_cos, is_pk_hot_zone,
            pk_hot_x_evening_rush.

  (f) Both memorize and non-memorize modes are first-class: every experiment
      flag works the same in either mode, and the threshold sweep runs in
      both.

V52 is otherwise identical to V51 — same Optuna search spaces, same V17 GNN
window alignment, same V21 feature engineering, same V14 GNN inference.

Usage
-----
Drop-in replacement for V51's runner:
    python -m src.training.ablation_study_v52_xgboost_only

Module attributes the bash runner / caller may set BEFORE main():
    v52.MEMORIZE_MODE             (bool)
    v52.ADD_PK_HOUR_INTERACTIONS  (bool, default True)
    v52.MIN_RECALL_CONSTRAINT     (float, default 0.85)
    v52.THRESHOLD_GRID_S1         (np.ndarray, default 0.50→0.99 step 0.025)
    v52.THRESHOLD_GRID_S2         (np.ndarray, default 0.10→0.95 step 0.05)

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

import gc
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import xgboost as xgb

# ── Project root on sys.path ────────────────────────────────────────────────
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# ── V21 (canonical frame layout, shared by every stage) ────────────────────
import src.training.ablation_study_v21_xgboost_only as _v21_mod

# ── V51 (base XGBoost pipeline V52 extends) ────────────────────────────────
import src.training.ablation_study_v51_xgboost_only as v51
from src.training.ablation_study_v51_xgboost_only import (
    stage0_load_data_with_gnn,
    stage1_feature_prep,
    stage2_train_ensemble,
    stage5_feature_importance_shap,
    stage6_feature_space_overlap,
    stage7_two_stage,
    stage7b_shap_stage2,
    create_balanced_subset,
    format_duration,
    _wider_optuna_ranges,
    _S1_INT_RANGES, _S1_FLOAT_RANGES,
    _S2_INT_RANGES, _S2_FLOAT_RANGES,
    RANDOM_STATE, N_THREADS, ENSEMBLE_CONFIG,
    OUTPUT_DIR, VIZ_DIR,
    TRAIN_START_DATE, TRAIN_END_DATE, SIM_START_DATE, SIM_END_DATE,
    PK_MIN, PK_MAX, TIME_RESOLUTION,
    ABLATION_SETTINGS, TEST_MODE, MEMORIZE_MODE,
)


# =============================================================================
# V52 CONFIGURATION
# =============================================================================

PIPELINE_VERSION = "v52_xgboost_only"

# (c) — pk × hour interaction features.
ADD_PK_HOUR_INTERACTIONS: bool = True
HOT_ZONE_PKS = (124, 125, 126, 149, 150)
EVENING_RUSH_HOURS = (17, 18)

# (a) — threshold sweep configuration (also used by V52 sim).
MIN_RECALL_CONSTRAINT: float = 0.85
MAX_FAR_TARGET:        float = 1.0
THRESHOLD_GRID_S1 = np.arange(0.50, 1.00, 0.025)
THRESHOLD_GRID_S2 = np.arange(0.10, 0.95, 0.05)

# Mirror V51's MEMORIZE_MODE so external callers can mutate either module attr.
MEMORIZE_MODE = bool(v51.MEMORIZE_MODE)


# =============================================================================
# EXPERIMENT (c): PK × HOUR INTERACTION FEATURES
# =============================================================================

def add_pk_hour_interactions(df: pd.DataFrame,
                             hot_pks=HOT_ZONE_PKS,
                             evening_rush_hours=EVENING_RUSH_HOURS) -> tuple[pd.DataFrame, list[str]]:
    """Add explicit pk × hour interaction columns.  Returns (df, new_feature_names)."""
    print("=" * 80)
    print("V52 EXPERIMENT (c): PK × HOUR INTERACTION FEATURES")
    print("=" * 80)

    df = df.copy()
    new_feats: list[str] = []

    if "pk" in df.columns and "hor" in df.columns:
        df["pk_x_hor"] = df["pk"].astype(float) * df["hor"].astype(float)
        new_feats.append("pk_x_hor")

    if "pk" in df.columns and "hor_sin" in df.columns and "hor_cos" in df.columns:
        df["pk_x_hor_sin"] = df["pk"].astype(float) * df["hor_sin"].astype(float)
        df["pk_x_hor_cos"] = df["pk"].astype(float) * df["hor_cos"].astype(float)
        new_feats.extend(["pk_x_hor_sin", "pk_x_hor_cos"])

    if "pk" in df.columns:
        df["is_pk_hot_zone"] = df["pk"].isin(hot_pks).astype(int)
        new_feats.append("is_pk_hot_zone")

    if "pk" in df.columns and "hor" in df.columns:
        df["pk_hot_x_evening_rush"] = (
            df["pk"].isin(hot_pks) & df["hor"].isin(evening_rush_hours)
        ).astype(int)
        new_feats.append("pk_hot_x_evening_rush")

    print(f"[V52-INTERACTIONS] hot zone PKs={list(hot_pks)}  "
          f"evening rush hours={list(evening_rush_hours)}")
    print(f"[V52-INTERACTIONS] Added {len(new_feats)} features: {new_feats}")
    return df, new_feats


# =============================================================================
# EXPERIMENT (a): WIDER THRESHOLD SWEEP (runs in memorize too)
# =============================================================================

def sweep_thresholds_v52(ensemble_preds: np.ndarray,
                         y_test_np: np.ndarray,
                         flagged_idx: np.ndarray,
                         s2_test_probs: np.ndarray,
                         s1_grid: np.ndarray | None = None,
                         s2_grid: np.ndarray | None = None,
                         min_recall: float | None = None,
                         max_far: float | None = None,
                         optimise_for: str = "f2") -> tuple[float, float, dict]:
    """V52 threshold sweep — wider S1 grid, lower default min-recall.

    Differences vs V51's `stage7d_threshold_sweep`:
      - S1 grid extends to 0.995 (V51 stopped at 0.50).
      - Defaults to F2 maximisation (recall-weighted) under a min-recall
        constraint, instead of pure precision maximisation.
      - Runs unconditionally — V51 skips this in MEMORIZE_MODE.

    Returns (S1_THR, S2_THR, sweep_summary_dict).
    """
    s1_grid    = s1_grid if s1_grid is not None else THRESHOLD_GRID_S1
    s2_grid    = s2_grid if s2_grid is not None else THRESHOLD_GRID_S2
    min_recall = min_recall if min_recall is not None else MIN_RECALL_CONSTRAINT
    max_far    = max_far    if max_far    is not None else MAX_FAR_TARGET

    print("=" * 80)
    print("V52 EXPERIMENT (a): WIDER THRESHOLD SWEEP")
    print("=" * 80)
    print(f"  S1 grid: {len(s1_grid)} values from {s1_grid[0]:.3f} to {s1_grid[-1]:.3f}")
    print(f"  S2 grid: {len(s2_grid)} values from {s2_grid[0]:.3f} to {s2_grid[-1]:.3f}")
    print(f"  min_recall = {min_recall:.3f}   max_far = {max_far:.3f}   "
          f"optimise_for = {optimise_for}")

    s2_test_probs_full = np.zeros(len(y_test_np))
    if len(flagged_idx) > 0 and len(s2_test_probs) == len(flagged_idx):
        s2_test_probs_full[flagged_idx] = s2_test_probs

    best = {"s1": 0.5, "s2": 0.5, "score": -np.inf,
            "precision": 0.0, "recall": 0.0, "far": 1.0,
            "f1": 0.0, "f2": 0.0, "tp": 0, "fp": 0, "fn": 0}

    rows = []
    for s1_t in s1_grid:
        s1_flag_mask = ensemble_preds >= s1_t
        if s1_flag_mask.sum() == 0:
            continue
        s1_flag_idx = np.where(s1_flag_mask)[0]

        for s2_t in s2_grid:
            final_pred = np.zeros(len(y_test_np), dtype=int)
            final_pred[s1_flag_idx[s2_test_probs_full[s1_flag_idx] >= s2_t]] = 1

            tp = int(((final_pred == 1) & (y_test_np == 1)).sum())
            fp = int(((final_pred == 1) & (y_test_np == 0)).sum())
            fn = int(((final_pred == 0) & (y_test_np == 1)).sum())
            recall    = tp / max(tp + fn, 1)
            precision = tp / max(tp + fp, 1)
            far       = fp / max(tp + fp, 1)
            f1        = 2 * precision * recall / max(precision + recall, 1e-12)
            f2        = 5 * precision * recall / max(4 * precision + recall, 1e-12)

            score = {"f2": f2, "f1": f1, "precision": precision, "recall": recall}.get(optimise_for, f2)

            rows.append({"s1": float(s1_t), "s2": float(s2_t),
                         "tp": tp, "fp": fp, "fn": fn,
                         "precision": precision, "recall": recall,
                         "far": far, "f1": f1, "f2": f2})

            if recall >= min_recall and far <= max_far and score > best["score"]:
                best.update({"s1": float(s1_t), "s2": float(s2_t), "score": score,
                             "precision": precision, "recall": recall,
                             "far": far, "f1": f1, "f2": f2,
                             "tp": tp, "fp": fp, "fn": fn})

    if best["score"] == -np.inf:
        # Graceful fallback: keep the user's optimise_for and pick the (S1,S2)
        # that maximises it, dropping the recall/far floors. This is more
        # useful than reverting to F2 (which loses the user's stated goal).
        print(f"[V52-SWEEP] WARNING: no (S1, S2) achieves recall ≥ {min_recall} AND "
              f"FAR ≤ {max_far}.  Falling back to max-{optimise_for} unconstrained.")
        if rows:
            def _row_score(r):
                return r.get(optimise_for, r["f2"])
            best_unconstrained = max(rows, key=_row_score)
            best.update({"s1": best_unconstrained["s1"], "s2": best_unconstrained["s2"],
                         "score": float(_row_score(best_unconstrained)),
                         **{k: best_unconstrained[k]
                            for k in ["precision", "recall", "far", "f1", "f2",
                                      "tp", "fp", "fn"]}})

    print(f"[V52-SWEEP] best ({optimise_for}): "
          f"S1={best['s1']:.3f}  S2={best['s2']:.3f}  "
          f"P={best['precision']:.4f}  R={best['recall']:.4f}  "
          f"FAR={best['far']:.4f}  F2={best['f2']:.4f}  "
          f"(TP={best['tp']}  FP={best['fp']}  FN={best['fn']})")

    return best["s1"], best["s2"], {"best": best, "all": rows}


# =============================================================================
# V52 RUN PIPELINE
# =============================================================================

def run_pipeline_v52(gnn_experiment_dir: str) -> bool:
    """V52 main pipeline — wraps V51's stage functions with V52 hooks.

    Flow:
        Stage 0 (V51)              load CSV + GNN + V21 FE
        ─ V52 (c) pk × hour feats  if ADD_PK_HOUR_INTERACTIONS
        Stage 1 (V51)              feature prep + train/test split
        Stage 2 (V51)              Stage-1 ensemble (widened Optuna ranges)
        Stage 5 (V51)              Stage-1 SHAP / importance
        Stage 6 (V51)              feature-space overlap diagnostics
        Stage 7 (V51)              Stage-2 trainer (memorize OR OOF)
        ─ V52 (a) threshold sweep  unconditionally (V51 skipped in memorize)
        Save V52 manifest          stamps experiment flags + sweep result
    """
    # Honour any external overrides written onto either v51 or v52.
    _propagate_to_v51()

    t_start = time.time()
    print("=" * 80)
    print("V52 XGBoost-only — V51 + (a)+(b)+(c)+(d)+(f) experiments")
    print(f"Training:        {v51.TRAIN_START_DATE} -> {v51.TRAIN_END_DATE}")
    print(f"Output dir:      {v51.OUTPUT_DIR}")
    print(f"GNN experiment:  {gnn_experiment_dir}")
    print(f"MEMORIZE_MODE:   {bool(v51.MEMORIZE_MODE)}")
    print("V52 experiment flags:")
    print(f"  (a) threshold sweep        : enabled "
          f"(min_recall={MIN_RECALL_CONSTRAINT}, "
          f"S1 grid {THRESHOLD_GRID_S1[0]:.3f}-{THRESHOLD_GRID_S1[-1]:.3f}, "
          f"S2 grid {THRESHOLD_GRID_S2[0]:.3f}-{THRESHOLD_GRID_S2[-1]:.3f})")
    print(f"  (b) Stage-2 in memorize    : delegated to V51 (already trained)")
    print(f"  (c) pk × hour interactions : {ADD_PK_HOUR_INTERACTIONS}")
    print(f"  (f) memorize + holdout     : both modes supported")
    print("=" * 80)

    os.makedirs(v51.OUTPUT_DIR, exist_ok=True)
    os.makedirs(v51.VIZ_DIR, exist_ok=True)

    # ── Stage 0: Base CSV + GNN inference + V21 FE (delegate to V51) ────────
    df_full, fe_only_cols = stage0_load_data_with_gnn(gnn_experiment_dir)

    # ── V52 EXPERIMENT (c): Add pk × hour interaction features ──────────────
    v52_extra_features: list[str] = []
    if ADD_PK_HOUR_INTERACTIONS:
        df_full, v52_extra_features = add_pk_hour_interactions(df_full)
        # Tag them as FE-only so V51's stage1_feature_prep treats them like
        # the V21 engineered columns (they end up in `available_features`).
        fe_only_cols = list(fe_only_cols) + [
            f for f in v52_extra_features if f not in fe_only_cols
        ]

    # ── Stage 1: feature prep + train/test split (delegate to V51) ──────────
    (df_train, X_train_xgb, X_test_xgb, y_train_acc, y_test_acc,
     available_features, model_time) = stage1_feature_prep(df_full, fe_only_cols)

    # Sanity-check that V52 features made it through.
    if ADD_PK_HOUR_INTERACTIONS:
        missing_v52 = [f for f in v52_extra_features
                       if f not in available_features and f in X_train_xgb.columns]
        if missing_v52:
            print(f"[V52] Adding {len(missing_v52)} V52 interaction feats to available_features: "
                  f"{missing_v52}")
            available_features = list(available_features) + missing_v52

    del df_train
    gc.collect()
    print("[MEM] df_train released. Proceeding to ensemble training.")

    # ── Stage 2: Stage-1 ensemble training (V51 widened ranges) ─────────────
    with _wider_optuna_ranges(_S1_INT_RANGES, _S1_FLOAT_RANGES):
        (base_models, ensemble_preds, y_pred_binary, roc_auc, pr_auc,
         s1_best_params) = stage2_train_ensemble(
            X_train_xgb, X_test_xgb, y_train_acc, y_test_acc,
            available_features, model_time)
    gc.collect()

    # ── Stage 5: Stage-1 SHAP / importance ──────────────────────────────────
    stage5_feature_importance_shap(base_models, X_test_xgb, available_features)

    # ── Stage 6: Feature-space overlap diagnostics ──────────────────────────
    stage6_feature_space_overlap(
        X_train_xgb, X_test_xgb, y_train_acc, y_test_acc,
        available_features, ensemble_preds)

    # ── Stage 7: Stage-2 trainer ────────────────────────────────────────────
    # V51 already handles MEMORIZE_MODE inside stage7_two_stage / its inline
    # memorize branch.  V52 keeps that exact behaviour but ALSO runs the
    # threshold sweep on whichever Stage-2 model was produced.
    s2_test_probs = np.zeros(0)
    flagged_idx   = np.array([], dtype=int)
    y_test_np     = y_test_acc.reset_index(drop=True).values
    stage2_model  = None

    if v51.MEMORIZE_MODE:
        # Replicate V51's memorize-mode Stage-2 recipe inline so V52 owns the
        # references it needs for the threshold sweep below (s2_test_probs,
        # flagged_idx).  Logic mirrors V51 lines 2462-2497.
        print("=" * 80)
        print("STAGE 7 (MEMORIZE): Stage-2 trained Stage-1-style on full pool")
        print("=" * 80)

        s2_memorize_params = {
            **s1_best_params,
            "tree_method":  "hist",
            "objective":    "binary:logistic",
            "eval_metric":  "aucpr",
            "random_state": RANDOM_STATE + 1,
            "nthread":      N_THREADS,
        }
        neg_pos_ratio = ENSEMBLE_CONFIG["neg_pos_ratio"]

        X_all = pd.concat([X_train_xgb, X_test_xgb])
        y_all = pd.concat([y_train_acc, y_test_acc])

        s1_score_all = np.zeros(len(X_all))
        d_all = xgb.DMatrix(X_all)
        for m in base_models:
            best_iter = getattr(m, "best_iteration", 500)
            s1_score_all += m.predict(d_all, iteration_range=(0, best_iter))
        s1_score_all /= max(len(base_models), 1)
        X_all = X_all.copy()
        X_all["s1_score"] = s1_score_all

        X_s2_tr, y_s2_tr = create_balanced_subset(
            X_all, y_all,
            neg_pos_ratio=neg_pos_ratio,
            random_state=RANDOM_STATE + 1,
        )
        n_tp_s2 = int(y_s2_tr.sum())
        n_fp_s2 = int((y_s2_tr == 0).sum())
        print(f"  Stage 2 (MEMORIZE) pool: {len(X_s2_tr):,} "
              f"(TP={n_tp_s2:,}, FP={n_fp_s2:,})")

        dtrain_s2 = xgb.DMatrix(X_s2_tr, label=y_s2_tr)
        stage2_model = xgb.train(
            s2_memorize_params, dtrain_s2,
            num_boost_round=500, verbose_eval=False,
        )
        stage2_model.best_iteration = 500
        stage2_path = os.path.join(
            v51.OUTPUT_DIR,
            f"xgboost-stage2-fp-filter_version={model_time}.json",
        )
        stage2_model.save_model(stage2_path)
        print(f"  Stage 2 (MEMORIZE) model saved: {stage2_path}")

        X_test_r     = X_test_xgb.reset_index(drop=True)
        test_s1_flag = ensemble_preds >= 0.5
        flagged_idx  = np.where(test_s1_flag)[0]
        if test_s1_flag.any():
            X_te_s2 = X_test_r[test_s1_flag].copy()
            X_te_s2["s1_score"] = ensemble_preds[test_s1_flag]
            s2_test_probs = stage2_model.predict(xgb.DMatrix(X_te_s2))

    else:
        with _wider_optuna_ranges(_S2_INT_RANGES, _S2_FLOAT_RANGES):
            (stage2_model, X_s2_tr, y_s2_tr, s2_full_prob, s2_full_bin,
             y_test_np, flagged_idx, s2_test_probs,
             _S1_THR_v34, _S2_THR_v34,
             _MIN_REC_v34, _MAX_FAR_v34) = stage7_two_stage(
                X_train_xgb, X_test_xgb, y_train_acc, y_test_acc,
                available_features, ensemble_preds, base_models,
                model_time, s1_best_params)
        stage7b_shap_stage2(stage2_model, X_s2_tr, y_s2_tr)

    # ── V52 EXPERIMENT (a): Threshold sweep — runs in BOTH modes ────────────
    S1_THR, S2_THR, sweep_summary = sweep_thresholds_v52(
        ensemble_preds=ensemble_preds,
        y_test_np=y_test_np,
        flagged_idx=flagged_idx,
        s2_test_probs=s2_test_probs,
        s1_grid=THRESHOLD_GRID_S1,
        s2_grid=THRESHOLD_GRID_S2,
        min_recall=MIN_RECALL_CONSTRAINT,
        max_far=MAX_FAR_TARGET,
        optimise_for="f2",
    )

    # ── Persist V52 manifests ───────────────────────────────────────────────
    write_v52_manifests(
        gnn_experiment_dir=gnn_experiment_dir,
        model_time=model_time,
        available_features=available_features,
        s1_thr=S1_THR, s2_thr=S2_THR,
        roc_auc=roc_auc, pr_auc=pr_auc,
        sweep_summary=sweep_summary,
        v52_extra_features=v52_extra_features,
    )

    print("\n" + "=" * 80)
    print(f"V52 XGBoost training complete in {format_duration(time.time() - t_start)}")
    print(f"All plots saved to:  {v51.VIZ_DIR}")
    print(f"All models saved to: {v51.OUTPUT_DIR}")
    print(f"Model time: {model_time}")
    print("Next step: ablation_study_v52_simulation_only.py "
          f"--xgboost-experiment-dir {v51.OUTPUT_DIR} "
          f"--gnn-experiment-dir {gnn_experiment_dir}")
    print("=" * 80)
    return True


def write_v52_manifests(*, gnn_experiment_dir: str, model_time: str,
                        available_features: list[str],
                        s1_thr: float, s2_thr: float,
                        roc_auc: float, pr_auc: float,
                        sweep_summary: dict,
                        v52_extra_features: list[str]) -> None:
    """Write `v52_xgboost_manifest_*.json` and `experiment_config_xgb_v52.json`.

    Schema is a strict superset of V51's so the V52 sim can fall back to V51
    sim with only the experiment-block additions.
    """
    experiments = {
        "pk_hour_interactions":   bool(ADD_PK_HOUR_INTERACTIONS),
        "v52_extra_features":     list(v52_extra_features),
        "stage2_in_memorize":     True,                     # always (V51 already does this)
        "threshold_sweep": {
            "min_recall_constraint": float(MIN_RECALL_CONSTRAINT),
            "max_far_target":        float(MAX_FAR_TARGET),
            "s1_grid":               [float(x) for x in THRESHOLD_GRID_S1],
            "s2_grid":               [float(x) for x in THRESHOLD_GRID_S2],
            "optimise_for":          "f2",
            "best":                  {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer))
                                          else v)
                                      for k, v in sweep_summary["best"].items()},
        },
        "hot_zone_pks":         list(HOT_ZONE_PKS),
        "evening_rush_hours":   list(EVENING_RUSH_HOURS),
    }

    manifest = {
        "pipeline_version":       PIPELINE_VERSION,
        "model_time":             model_time,
        "output_dir":             v51.OUTPUT_DIR,
        "gnn_experiment_dir":     gnn_experiment_dir,
        "available_features":     available_features,
        "stage1_threshold":       float(s1_thr),
        "stage2_threshold":       float(s2_thr),
        "min_recall_constraint":  float(MIN_RECALL_CONSTRAINT),
        "max_far_target":         float(MAX_FAR_TARGET),
        "test_metrics_stage1":    {"roc_auc": float(roc_auc), "pr_auc": float(pr_auc)},
        "train_dates":            {"start": v51.TRAIN_START_DATE, "end": v51.TRAIN_END_DATE},
        "sim_dates":              {"start": v51.SIM_START_DATE,   "end": v51.SIM_END_DATE},
        "pk_range":               {"min": v51.PK_MIN, "max": v51.PK_MAX},
        "time_resolution":        v51.TIME_RESOLUTION,
        "ablation_settings":      v51.ABLATION_SETTINGS,
        "memorize_mode":          bool(v51.MEMORIZE_MODE),
        "stage2_disabled":        False,
        "experiments":            experiments,
        "based_on":               "ablation_study_v51_xgboost_only",
        "wider_tree_ranges": {
            "stage1_int":   _S1_INT_RANGES,
            "stage1_float": _S1_FLOAT_RANGES,
            "stage2_int":   _S2_INT_RANGES,
            "stage2_float": _S2_FLOAT_RANGES,
        },
        "timestamp":              datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    manifest_path = os.path.join(
        v51.OUTPUT_DIR, f"v52_xgboost_manifest_{model_time}.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"[V52] Manifest saved: {manifest_path}")

    exp_config = {
        "model_time":         model_time,
        "test_mode":          v51.TEST_MODE,
        "memorize_mode":      bool(v51.MEMORIZE_MODE),
        "ablation_settings":  v51.ABLATION_SETTINGS,
        "pipeline_version":   PIPELINE_VERSION,
        "gnn_experiment_dir": gnn_experiment_dir,
        "stage1_threshold":   float(s1_thr),
        "stage2_threshold":   float(s2_thr),
        "available_features": available_features,
        "train_dates":        {"start": v51.TRAIN_START_DATE, "end": v51.TRAIN_END_DATE},
        "sim_dates":          {"start": v51.SIM_START_DATE,   "end": v51.SIM_END_DATE},
        "experiments":        experiments,
        "timestamp":          datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    exp_config_path = os.path.join(v51.OUTPUT_DIR, "experiment_config_xgb_v52.json")
    with open(exp_config_path, "w") as f:
        json.dump(exp_config, f, indent=2, default=str)
    print(f"[V52] Sim-runner config saved: {exp_config_path}")


# =============================================================================
# MODULE-ATTR PROPAGATION
# =============================================================================

def _propagate_to_v51() -> None:
    """Mirror V52's mutable module attrs onto V51 so V51's stage functions see
    whatever the bash runner / caller wrote on this module.

    The bash runner targets V51 attrs by default; allowing both means callers
    can write to either module and the configuration stays consistent.
    """
    # MEMORIZE_MODE flow:  v52 → v51 → v34   (v51 already propagates to v34)
    v51.MEMORIZE_MODE = bool(MEMORIZE_MODE) or bool(v51.MEMORIZE_MODE)
    if hasattr(v51, "_propagate_to_v34"):
        try:
            v51._propagate_to_v34()
        except Exception as e:
            print(f"[V52] v51._propagate_to_v34() failed: {e}")


# =============================================================================
# ENTRY POINTS
# =============================================================================

def run_xgboost_only() -> bool:
    """Entry point invoked by `bash_files/run_ablation_study_xgboost_only.sh`.

    Mirrors V51's signature: reads the GNN experiment dir from env (the runner
    resolves it from --gnn-run-id) and runs the V52 pipeline. The bash runner
    exports `PREVIOUS_EXPERIMENT_DIR`; accept `GNN_EXPERIMENT_DIR` and the
    v51 module-level constant as fallbacks so both names work.

    Also re-points `v51.OUTPUT_DIR` (and `v51.VIZ_DIR`) to the per-XGBoost-run
    subfolder `<gnn_dir>/xgboost_<XGB_RUN_ID>/` so multiple V52 training runs
    against the same GNN co-exist instead of clobbering the static v51 dir,
    and the V52 simulation runner can locate the manifest through its
    `xgboost_<XGB_RUN_ID>/` resolution path.
    """
    gnn_experiment_dir = (
        os.environ.get("PREVIOUS_EXPERIMENT_DIR", "")
        or os.environ.get("GNN_EXPERIMENT_DIR", "")
        or getattr(v51, "GNN_EXPERIMENT_DIR", "")
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
        print(f"[V52] XGB_RUN_ID not set; using timestamp fallback: {xgb_run_id}")

    v51.OUTPUT_DIR = os.path.join(gnn_experiment_dir, f"xgboost_{xgb_run_id}")
    v51.VIZ_DIR    = os.path.join(v51.OUTPUT_DIR, "visualizations")
    print(f"[V52] XGBoost artefacts will be saved under: {v51.OUTPUT_DIR}")

    return run_pipeline_v52(gnn_experiment_dir)


def main() -> int:
    global MEMORIZE_MODE, ADD_PK_HOUR_INTERACTIONS, MIN_RECALL_CONSTRAINT

    import argparse
    parser = argparse.ArgumentParser(
        description="V52 XGBoost-only — V51 + (a)+(b)+(c)+(d)+(f) experiments."
    )
    parser.add_argument("--gnn-experiment-dir", required=True,
                        help="V14/V17 GNN experiment dir (for GNN inference at training time).")
    parser.add_argument("--memorize-mode", action="store_true", default=False,
                        help="Force MEMORIZE_MODE on (defaults to V51's value).")
    parser.add_argument("--no-pk-hour", action="store_true", default=False,
                        help="Disable experiment (c) — pk × hour interactions.")
    parser.add_argument("--min-recall", type=float, default=MIN_RECALL_CONSTRAINT,
                        help=f"Experiment (a) — min recall constraint (default {MIN_RECALL_CONSTRAINT}).")

    args = parser.parse_args()

    if args.memorize_mode:
        MEMORIZE_MODE = True
        v51.MEMORIZE_MODE = True
    if args.no_pk_hour:
        ADD_PK_HOUR_INTERACTIONS = False
    MIN_RECALL_CONSTRAINT  = float(args.min_recall)

    return 0 if run_pipeline_v52(args.gnn_experiment_dir) else 1


if __name__ == "__main__":
    sys.exit(main())
