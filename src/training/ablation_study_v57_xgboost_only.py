#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V57 XGBoost-only: V55 with a TRANSFORMER as the Stage-2 FP detector
====================================================================

Identical recipe to V55 except step 4 — the Stage-2 false-positive filter
is an FT-Transformer-style tabular transformer instead of an XGBoost model:

  1. Stage 1 trained with AUCPR Optuna (same as V54/V55).
  2. Stage-1 OOF threshold sweep: pick the S1 threshold that maximises
     RECALL under min_recall ≥ 0.9 (same as V55).
  3. Use that picked threshold to filter Stage-2's training pool.
  4. NEW — Stage 2 is a tabular transformer (per-feature token embedding +
     [CLS] token + TransformerEncoder + linear head), trained with
     BCE-with-logits (pos-weighted) and a small Optuna search over
     lr / d_model / depth / dropout, early-stopped on validation AUCPR.
  5. Final test-time sweep: S1 LOCKED at the training-time pick; only the
     transformer's S2 threshold is swept, PRECISION maximisation under
     min_precision ≥ 0.9.

Artefacts:
  - `transformer-stage2-fp-filter_version=<model_time>.pt`
    (torch checkpoint bundling state_dict + architecture config +
     feature names + standardisation stats — fully self-describing).
  - `v57_xgboost_manifest_<model_time>.json` with
    experiments.stage2_model_type = "transformer".

Based on: ablation_study_v55_xgboost_only (Stage-2 swapped for transformer).

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
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
import xgboost as xgb
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold, train_test_split

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import src.training.ablation_study_v51_xgboost_only as v51
import src.training.ablation_study_v54_xgboost_only as v54
import src.training.ablation_study_v55_xgboost_only as v55
from src.training.ablation_study_v51_xgboost_only import (
    format_duration,
    stage0_load_data_with_gnn,
    stage1_feature_prep,
    stage5_feature_importance_shap,
    stage6_feature_space_overlap,
    _S1_FLOAT_RANGES,
    _S1_INT_RANGES,
    _wider_optuna_ranges,
    ENSEMBLE_CONFIG,
    N_THREADS,
    RANDOM_STATE,
)
from src.training.ablation_study_v52_xgboost_only import (
    add_pk_hour_interactions,
    sweep_thresholds_v52,
    EVENING_RUSH_HOURS,
    HOT_ZONE_PKS,
    THRESHOLD_GRID_S2,
)
from src.training.ablation_study_v54_xgboost_only import (
    stage2_train_ensemble_v54,
    _resolve_stage1_features,
    _resolve_stage2_features,
    _slice_for_stage2,
    _ALLOWED_OPTUNA_METRICS,
)
from src.training.ablation_study_v55_xgboost_only import sweep_stage1_threshold


# =============================================================================
# V57 CONFIGURATION
# =============================================================================

PIPELINE_VERSION = "v57_xgboost_only"

# Inherit V53/V54 feature pruning + experimental knobs (same as V55).
STAGE1_DROP_FEATURES: list[str] = list(v54.STAGE1_DROP_FEATURES)
STAGE2_FEATURES: list[str] | None = (
    list(v54.STAGE2_FEATURES) if v54.STAGE2_FEATURES is not None else None
)
ADD_PK_HOUR_INTERACTIONS: bool = v54.ADD_PK_HOUR_INTERACTIONS
MEMORIZE_MODE: bool = bool(v51.MEMORIZE_MODE)

# V57 recipe defaults — AUCPR for the Stage-1 Optuna search (Stage 2 is a
# transformer, scored on validation AUCPR directly).
OPTUNA_METRIC_STAGE1: str = "aucpr"
OPTUNA_MIN_RECALL_STAGE1: float = 0.30  # unused when metric=aucpr, kept for compat
OPTUNA_METRIC_STAGE2: str = "aucpr"     # informational; transformer always uses AUCPR
OPTUNA_MIN_RECALL_STAGE2: float = 0.30

# Stage-1 OOF threshold sweep (same as V55).
STAGE1_OPTIMISE_FOR: str = "recall"
STAGE1_MIN_RECALL: float = 0.90
STAGE1_MIN_PRECISION: float = 0.0

# Stage-2 (final) threshold sweep — S1 is LOCKED to the training-time pick.
OPTIMISE_FOR: str = "precision"
MIN_RECALL_CONSTRAINT: float = 0.0
MIN_PRECISION_CONSTRAINT: float = 0.90
MAX_FAR_TARGET: float = 1.0 - MIN_PRECISION_CONSTRAINT  # = 0.10

# Threshold grids (same shapes as V55).
STAGE1_OOF_GRID = np.unique(np.round(np.concatenate([
    np.arange(0.005, 0.05 + 1e-9, 0.005),
    np.arange(0.05, 0.50 + 1e-9, 0.025),
    np.arange(0.50, 1.00 + 1e-9, 0.025),
]), 4))
THRESHOLD_GRID_S2 = THRESHOLD_GRID_S2  # inherited

# Transformer training knobs.
TRANSFORMER_OPTUNA_TRIALS: int = 10
TRANSFORMER_MAX_EPOCHS: int = 100
TRANSFORMER_PATIENCE: int = 10
TRANSFORMER_BATCH_SIZE: int = 256
TRANSFORMER_DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================================
# Stage-2 transformer model (FT-Transformer-lite for tabular FP filtering)
# =============================================================================

class TabTransformerFPFilter(nn.Module):
    """Per-feature token embedding + [CLS] + TransformerEncoder + linear head.

    Each scalar feature x_j is mapped to a d_model-dim token via its own
    affine map (w_j * x_j + b_j). A learned [CLS] token is prepended; the
    encoder output at [CLS] feeds the classification head. Inputs are
    expected pre-standardised (the checkpoint stores mean/scale).
    """

    def __init__(self, n_features: int, d_model: int = 64, n_heads: int = 4,
                 n_layers: int = 2, dropout: float = 0.1, ff_mult: int = 2):
        super().__init__()
        self.n_features = n_features
        self.feature_weight = nn.Parameter(torch.empty(n_features, d_model))
        self.feature_bias = nn.Parameter(torch.zeros(n_features, d_model))
        nn.init.normal_(self.feature_weight, std=0.02)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_token, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=ff_mult * d_model,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, F) -> tokens (B, F, d_model)
        tokens = x.unsqueeze(-1) * self.feature_weight.unsqueeze(0) + self.feature_bias.unsqueeze(0)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        h = self.encoder(torch.cat([cls, tokens], dim=1))
        return self.head(h[:, 0, :]).squeeze(-1)  # logits (B,)


def _predict_transformer(model: nn.Module, X: np.ndarray,
                         device: str, batch_size: int = 4096) -> np.ndarray:
    model.eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.as_tensor(X[i:i + batch_size], dtype=torch.float32, device=device)
            probs.append(torch.sigmoid(model(xb)).cpu().numpy())
    return np.concatenate(probs) if probs else np.zeros(0)


def _fit_transformer(X_tr: np.ndarray, y_tr: np.ndarray,
                     X_va: np.ndarray, y_va: np.ndarray,
                     *, d_model: int, n_heads: int, n_layers: int,
                     dropout: float, lr: float, weight_decay: float,
                     max_epochs: int, patience: int, batch_size: int,
                     device: str, seed: int = RANDOM_STATE,
                     verbose: bool = False) -> tuple[nn.Module, float, int]:
    """Train one transformer; early-stop on validation AUCPR.

    Returns (best_model_on_cpu, best_val_aucpr, best_epoch).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = TabTransformerFPFilter(
        n_features=X_tr.shape[1], d_model=d_model, n_heads=n_heads,
        n_layers=n_layers, dropout=dropout,
    ).to(device)

    n_pos = max(int(y_tr.sum()), 1)
    n_neg = max(int((y_tr == 0).sum()), 1)
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    X_tr_t = torch.as_tensor(X_tr, dtype=torch.float32)
    y_tr_t = torch.as_tensor(y_tr, dtype=torch.float32)
    ds = torch.utils.data.TensorDataset(X_tr_t, y_tr_t)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True,
                                     generator=torch.Generator().manual_seed(seed))

    best_aucpr = -np.inf
    best_state = None
    best_epoch = 0
    epochs_no_improve = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss) * len(xb)

        va_probs = _predict_transformer(model, X_va, device)
        va_aucpr = average_precision_score(y_va, va_probs) if y_va.sum() > 0 else 0.0

        if verbose and (epoch % 10 == 0 or epoch == 1):
            print(f"    epoch {epoch:3d}  loss={total_loss / max(len(ds), 1):.5f}  "
                  f"val_aucpr={va_aucpr:.4f}")

        if va_aucpr > best_aucpr:
            best_aucpr = va_aucpr
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                if verbose:
                    print(f"    early stop at epoch {epoch} (best={best_epoch}, "
                          f"val_aucpr={best_aucpr:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to("cpu")
    return model, float(best_aucpr), int(best_epoch)


# =============================================================================
# Stage-2 training (V57) — OOF + Stage-1 sweep + filter, then TRANSFORMER fit
# =============================================================================

def _train_stage2_transformer_v57(*, X_train_xgb, X_test_xgb, y_train_acc, y_test_acc,
                                  base_models, ensemble_preds, stage2_features,
                                  s1_best_params, model_time, output_dir,
                                  stage1_optimise_for: str,
                                  stage1_min_recall: float,
                                  stage1_min_precision: float):
    """V57 Stage-2 trainer.

    Steps 1-3 are copied verbatim from V55's `_train_stage2_holdout_v55`
    (OOF Stage-1 probs → Stage-1 OOF sweep → pool filtering). Step 4 swaps
    the XGBoost Stage-2 Optuna+fit for a tabular transformer with its own
    small Optuna search, early-stopped on validation AUCPR.
    """
    X_tr = X_train_xgb.reset_index(drop=True)
    y_tr = y_train_acc.reset_index(drop=True)
    stage1_features = list(X_tr.columns)
    neg_pos_ratio = ENSEMBLE_CONFIG["neg_pos_ratio"]

    # 1. OOF Stage-1 probabilities on the training set (unchanged from V55)
    oof_n_folds = 5
    oof_n_models = 5
    mean_iter = int(np.mean([getattr(m, "best_iteration", 100) for m in base_models]))
    oof_rounds = max(50, mean_iter)
    oof_params = {**s1_best_params, "tree_method": "hist",
                  "objective": "binary:logistic", "eval_metric": "aucpr",
                  "nthread": N_THREADS, "seed": RANDOM_STATE}

    oof_kf = StratifiedKFold(n_splits=oof_n_folds, shuffle=True, random_state=RANDOM_STATE)
    s1_tr_probs = np.zeros(len(X_tr))
    rng = np.random.RandomState(RANDOM_STATE)
    for fold_i, (tr_idx, va_idx) in enumerate(oof_kf.split(X_tr.values, y_tr.values), 1):
        X_fold = X_tr.iloc[tr_idx]
        y_fold = y_tr.iloc[tr_idx]
        pos_ix = np.where(y_fold.values == 1)[0]
        neg_ix = np.where(y_fold.values == 0)[0]
        fold_probs = np.zeros(len(va_idx))
        for _ in range(oof_n_models):
            samp_neg = rng.choice(neg_ix, size=len(pos_ix) * neg_pos_ratio, replace=False)
            bag_idx = np.concatenate([pos_ix, samp_neg])
            rng.shuffle(bag_idx)
            Xb = X_fold.iloc[bag_idx][stage1_features]
            yb = y_fold.iloc[bag_idx]
            m = xgb.train(oof_params, xgb.DMatrix(Xb, label=yb),
                          num_boost_round=oof_rounds, verbose_eval=False)
            fold_probs += m.predict(xgb.DMatrix(X_tr.iloc[va_idx][stage1_features]))
        s1_tr_probs[va_idx] = fold_probs / oof_n_models

    # 2. Stage-1 OOF threshold sweep (same as V55)
    print("=" * 80)
    print("V57 STAGE 7A: STAGE-1 OOF THRESHOLD SWEEP")
    print("=" * 80)
    print(f"optimise_for={stage1_optimise_for}  "
          f"min_recall={stage1_min_recall}  min_precision={stage1_min_precision}")
    stage1_threshold, s1_sweep_summary = sweep_stage1_threshold(
        y_true=y_tr.values, y_prob=s1_tr_probs,
        optimise_for=stage1_optimise_for,
        min_recall=stage1_min_recall,
        min_precision=stage1_min_precision,
        grid=STAGE1_OOF_GRID,
    )

    # 3. Filter both train and test pools at the picked Stage-1 threshold
    s2_mask = s1_tr_probs >= stage1_threshold
    X_s2_full_tr = X_tr[s2_mask].copy()
    X_s2_full_tr["s1_score"] = s1_tr_probs[s2_mask]
    y_s2_tr = y_tr[s2_mask].copy()

    X_test_r = X_test_xgb.reset_index(drop=True)
    y_test_r = y_test_acc.reset_index(drop=True)
    test_s1_flag = ensemble_preds >= stage1_threshold
    X_s2_full_te = X_test_r[test_s1_flag].copy()
    X_s2_full_te["s1_score"] = ensemble_preds[test_s1_flag]
    y_s2_te = y_test_r[test_s1_flag].copy()

    X_s2_tr = _slice_for_stage2(X_s2_full_tr, stage2_features)
    X_s2_te = _slice_for_stage2(X_s2_full_te, stage2_features)
    s2_feature_names = list(X_s2_tr.columns)

    n_tp_s2 = int(y_s2_tr.sum())
    n_fp_s2 = int((y_s2_tr == 0).sum())
    print(f"[V57] Stage-2 training pool @ S1≥{stage1_threshold:.4f}: "
          f"{len(y_s2_tr):,} rows  (TP={n_tp_s2}, FP={n_fp_s2})")
    print(f"[V57] Stage-2 test pool     @ S1≥{stage1_threshold:.4f}: "
          f"{len(y_s2_te):,} rows  (TP={int(y_s2_te.sum())}, "
          f"FP={int((y_s2_te == 0).sum())})")

    # 4. NEW IN V57 — Stage-2 TRANSFORMER (standardise → Optuna → final fit)
    print("=" * 80)
    print("V57 STAGE 7B: Stage-2 TRANSFORMER FP filter (Optuna on val AUCPR)")
    print("=" * 80)
    print(f"Device: {TRANSFORMER_DEVICE}  trials={TRANSFORMER_OPTUNA_TRIALS}  "
          f"max_epochs={TRANSFORMER_MAX_EPOCHS}  patience={TRANSFORMER_PATIENCE}")

    torch.set_num_threads(N_THREADS)

    X_s2_tr_np = X_s2_tr.values.astype(np.float32)
    X_s2_te_np = X_s2_te.values.astype(np.float32)
    y_s2_tr_np = y_s2_tr.values.astype(np.float32)
    y_s2_te_np = y_s2_te.values.astype(np.float32)

    # Standardise on the Stage-2 training pool; stats go into the checkpoint.
    feat_mean = X_s2_tr_np.mean(axis=0)
    feat_scale = X_s2_tr_np.std(axis=0)
    feat_scale[feat_scale < 1e-8] = 1.0
    X_s2_tr_std = (X_s2_tr_np - feat_mean) / feat_scale
    X_s2_te_std = (X_s2_te_np - feat_mean) / feat_scale

    # Inner split for Optuna + early stopping (the test pool is reserved for
    # the final threshold sweep, mirroring V55 where Optuna used CV on train).
    stratify = y_s2_tr_np if 0 < y_s2_tr_np.sum() < len(y_s2_tr_np) else None
    X_in_tr, X_in_va, y_in_tr, y_in_va = train_test_split(
        X_s2_tr_std, y_s2_tr_np, test_size=0.2,
        random_state=RANDOM_STATE, stratify=stratify,
    )

    def _t_objective(trial: optuna.trial.Trial) -> float:
        d_model = trial.suggest_categorical("d_model", [32, 64, 128])
        n_heads = trial.suggest_categorical("n_heads", [2, 4])
        n_layers = trial.suggest_int("n_layers", 1, 3)
        dropout = trial.suggest_float("dropout", 0.0, 0.3)
        lr = trial.suggest_float("lr", 1e-4, 3e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        _, va_aucpr, _ = _fit_transformer(
            X_in_tr, y_in_tr, X_in_va, y_in_va,
            d_model=d_model, n_heads=n_heads, n_layers=n_layers,
            dropout=dropout, lr=lr, weight_decay=weight_decay,
            max_epochs=min(TRANSFORMER_MAX_EPOCHS, 40),
            patience=max(TRANSFORMER_PATIENCE // 2, 5),
            batch_size=TRANSFORMER_BATCH_SIZE,
            device=TRANSFORMER_DEVICE,
        )
        return va_aucpr

    t_study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    t_study.optimize(_t_objective, n_trials=TRANSFORMER_OPTUNA_TRIALS,
                     show_progress_bar=True)
    print(f"Stage 2 transformer best val AUCPR: {t_study.best_value:.4f}")
    for k, v in t_study.best_params.items():
        print(f"  {k}: {v}")

    bp = dict(t_study.best_params)
    stage2_model, final_va_aucpr, best_epoch = _fit_transformer(
        X_in_tr, y_in_tr, X_in_va, y_in_va,
        d_model=bp["d_model"], n_heads=bp["n_heads"], n_layers=bp["n_layers"],
        dropout=bp["dropout"], lr=bp["lr"], weight_decay=bp["weight_decay"],
        max_epochs=TRANSFORMER_MAX_EPOCHS,
        patience=TRANSFORMER_PATIENCE,
        batch_size=TRANSFORMER_BATCH_SIZE,
        device=TRANSFORMER_DEVICE,
        verbose=True,
    )
    print(f"[V57] Final transformer: val AUCPR={final_va_aucpr:.4f} (epoch {best_epoch})")

    arch_config = {
        "n_features": int(X_s2_tr_std.shape[1]),
        "d_model": int(bp["d_model"]),
        "n_heads": int(bp["n_heads"]),
        "n_layers": int(bp["n_layers"]),
        "dropout": float(bp["dropout"]),
    }
    checkpoint = {
        "state_dict": stage2_model.state_dict(),
        "arch_config": arch_config,
        "train_config": {
            "lr": float(bp["lr"]),
            "weight_decay": float(bp["weight_decay"]),
            "batch_size": int(TRANSFORMER_BATCH_SIZE),
            "best_epoch": int(best_epoch),
            "val_aucpr": float(final_va_aucpr),
        },
        "feature_names": s2_feature_names,
        "feat_mean": feat_mean.tolist(),
        "feat_scale": feat_scale.tolist(),
        "model_time": model_time,
        "pipeline_version": PIPELINE_VERSION,
    }
    stage2_path = os.path.join(
        output_dir, f"transformer-stage2-fp-filter_version={model_time}.pt"
    )
    torch.save(checkpoint, stage2_path)
    print(f"\nStage 2 transformer saved: {stage2_path}")

    s2_test_probs = _predict_transformer(stage2_model, X_s2_te_std, "cpu")
    y_test_np = y_test_r.values
    flagged_idx = np.where(test_s1_flag)[0]
    s2_best_params = {**bp, "best_epoch": int(best_epoch)}
    return (stage2_model, flagged_idx, s2_test_probs, y_test_np,
            s2_best_params, float(stage1_threshold), s1_sweep_summary)


# =============================================================================
# Manifests
# =============================================================================

def write_v57_manifests(*, gnn_experiment_dir: str, model_time: str,
                        stage1_features: list[str], stage1_dropped: list[str],
                        stage2_features: list[str],
                        s1_thr: float, s2_thr: float,
                        roc_auc: float, pr_auc: float,
                        sweep_summary: dict,
                        s1_sweep_summary: dict,
                        v52_extra_features: list[str],
                        s1_best_params: dict,
                        s2_best_params: dict) -> None:
    def _coerce(v):
        if isinstance(v, (np.floating, np.integer)):
            return float(v) if isinstance(v, np.floating) else int(v)
        return v

    experiments = {
        "pk_hour_interactions": bool(ADD_PK_HOUR_INTERACTIONS),
        "v52_extra_features": list(v52_extra_features),
        "stage1_dropped_features": list(stage1_dropped),
        "stage1_drop_request": list(STAGE1_DROP_FEATURES),
        "stage2_features": list(stage2_features),
        "stage2_inherits_stage1": STAGE2_FEATURES is None,
        "stage2_model_type": "transformer",
        "optuna_objective_stage1": {
            "metric": OPTUNA_METRIC_STAGE1,
            "min_recall": float(OPTUNA_MIN_RECALL_STAGE1),
        },
        "optuna_objective_stage2": {
            "metric": "aucpr_transformer_val",
            "trials": int(TRANSFORMER_OPTUNA_TRIALS),
        },
        "stage1_sweep": {
            "optimise_for": STAGE1_OPTIMISE_FOR,
            "min_recall_constraint": float(STAGE1_MIN_RECALL),
            "min_precision_constraint": float(STAGE1_MIN_PRECISION),
            "picked_threshold": float(s1_thr),
            "feasible": bool(s1_sweep_summary.get("feasible", False)),
            "summary": {k: _coerce(v) for k, v in s1_sweep_summary.items()
                        if k != "feasible"},
        },
        "threshold_sweep": {
            "optimise_for": OPTIMISE_FOR,
            "min_recall_constraint": float(MIN_RECALL_CONSTRAINT),
            "min_precision_constraint": float(MIN_PRECISION_CONSTRAINT),
            "max_far_target": float(MAX_FAR_TARGET),
            "s1_locked_to": float(s1_thr),
            "s2_grid": [float(x) for x in THRESHOLD_GRID_S2],
            "best": {k: _coerce(v) for k, v in sweep_summary["best"].items()},
        },
        "hot_zone_pks": list(HOT_ZONE_PKS),
        "evening_rush_hours": list(EVENING_RUSH_HOURS),
        "stage1_best_params": dict(s1_best_params),
        "stage2_best_params": dict(s2_best_params),
    }

    manifest = {
        "pipeline_version": PIPELINE_VERSION,
        "model_time": model_time,
        "output_dir": v51.OUTPUT_DIR,
        "gnn_experiment_dir": gnn_experiment_dir,
        "available_features": stage1_features,
        "stage1_features": stage1_features,
        "stage2_features": stage2_features,
        "stage1_threshold": float(s1_thr),
        "stage2_threshold": float(s2_thr),
        "min_recall_constraint": float(MIN_RECALL_CONSTRAINT),
        "min_precision_constraint": float(MIN_PRECISION_CONSTRAINT),
        "max_far_target": float(MAX_FAR_TARGET),
        "test_metrics_stage1": {"roc_auc": float(roc_auc), "pr_auc": float(pr_auc)},
        "train_dates": {"start": v51.TRAIN_START_DATE, "end": v51.TRAIN_END_DATE},
        "sim_dates": {"start": v51.SIM_START_DATE, "end": v51.SIM_END_DATE},
        "pk_range": {"min": v51.PK_MIN, "max": v51.PK_MAX},
        "time_resolution": v51.TIME_RESOLUTION,
        "ablation_settings": v51.ABLATION_SETTINGS,
        "memorize_mode": bool(v51.MEMORIZE_MODE),
        "stage2_disabled": False,
        "stage2_model_type": "transformer",
        "experiments": experiments,
        "based_on": "ablation_study_v55_xgboost_only",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    manifest_path = os.path.join(v51.OUTPUT_DIR, f"v57_xgboost_manifest_{model_time}.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"[V57] Manifest saved: {manifest_path}")

    exp_config = {
        "model_time": model_time,
        "test_mode": v51.TEST_MODE,
        "memorize_mode": bool(v51.MEMORIZE_MODE),
        "ablation_settings": v51.ABLATION_SETTINGS,
        "pipeline_version": PIPELINE_VERSION,
        "gnn_experiment_dir": gnn_experiment_dir,
        "stage1_threshold": float(s1_thr),
        "stage2_threshold": float(s2_thr),
        "stage2_model_type": "transformer",
        "available_features": stage1_features,
        "stage1_features": stage1_features,
        "stage2_features": stage2_features,
        "train_dates": {"start": v51.TRAIN_START_DATE, "end": v51.TRAIN_END_DATE},
        "sim_dates": {"start": v51.SIM_START_DATE, "end": v51.SIM_END_DATE},
        "experiments": experiments,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    exp_config_path = os.path.join(v51.OUTPUT_DIR, "experiment_config_xgb_v57.json")
    with open(exp_config_path, "w") as f:
        json.dump(exp_config, f, indent=2, default=str)
    print(f"[V57] Sim-runner config saved: {exp_config_path}")


# =============================================================================
# Pipeline
# =============================================================================

def _propagate_to_v54() -> None:
    v54.ADD_PK_HOUR_INTERACTIONS = bool(ADD_PK_HOUR_INTERACTIONS)
    v54.MEMORIZE_MODE = bool(MEMORIZE_MODE) or bool(v54.MEMORIZE_MODE)
    v51.MEMORIZE_MODE = bool(MEMORIZE_MODE) or bool(v51.MEMORIZE_MODE)
    if hasattr(v54, "_propagate_to_v52"):
        try:
            v54._propagate_to_v52()
        except Exception as e:
            print(f"[V57] v54._propagate_to_v52() failed: {e}")


def run_pipeline_v57(gnn_experiment_dir: str) -> bool:
    _propagate_to_v54()
    t_start = time.time()

    print("=" * 80)
    print("V57 XGBoost-only — V55 with TRANSFORMER Stage-2 FP detector")
    print(f"Output dir:      {v51.OUTPUT_DIR}")
    print(f"GNN experiment:  {gnn_experiment_dir}")
    print(f"MEMORIZE_MODE:   {bool(v51.MEMORIZE_MODE)}")
    print(f"Device:          {TRANSFORMER_DEVICE}")
    print("Optuna objective config:")
    print(f"  Stage 1: metric={OPTUNA_METRIC_STAGE1}  min_recall={OPTUNA_MIN_RECALL_STAGE1}")
    print(f"  Stage 2: transformer, val AUCPR, trials={TRANSFORMER_OPTUNA_TRIALS}")
    print("Stage-1 OOF threshold sweep config:")
    print(f"  optimise_for={STAGE1_OPTIMISE_FOR}  "
          f"min_recall={STAGE1_MIN_RECALL}  min_precision={STAGE1_MIN_PRECISION}")
    print("Stage-2 (final) threshold sweep config:")
    print(f"  optimise_for={OPTIMISE_FOR}  "
          f"min_recall={MIN_RECALL_CONSTRAINT}  "
          f"min_precision={MIN_PRECISION_CONSTRAINT} "
          f"(max_far={MAX_FAR_TARGET:.3f})")
    print("  S1 LOCKED at the training-time Stage-1 pick")
    print("=" * 80)

    if bool(v51.MEMORIZE_MODE):
        raise NotImplementedError("V57 does not yet implement MEMORIZE_MODE (no holdout pool).")

    os.makedirs(v51.OUTPUT_DIR, exist_ok=True)
    os.makedirs(v51.VIZ_DIR, exist_ok=True)

    df_full, fe_only_cols = stage0_load_data_with_gnn(gnn_experiment_dir)

    v52_extra_features: list[str] = []
    if ADD_PK_HOUR_INTERACTIONS:
        df_full, v52_extra_features = add_pk_hour_interactions(df_full)
        fe_only_cols = list(fe_only_cols) + [f for f in v52_extra_features if f not in fe_only_cols]

    time_res_min = v51._get_time_resolution_minutes()

    (df_train, X_train_xgb, X_test_xgb, y_train_acc, y_test_acc,
     available_features, model_time) = stage1_feature_prep(df_full, fe_only_cols)
    del df_train
    gc.collect()

    stage1_features, stage1_dropped = _resolve_stage1_features(available_features)
    if stage1_dropped:
        X_train_xgb = X_train_xgb[stage1_features]
        X_test_xgb = X_test_xgb[stage1_features]

    stage2_features = _resolve_stage2_features(stage1_features)

    # Stage-1 training (reused from V54 — AUCPR Optuna by default)
    with _wider_optuna_ranges(_S1_INT_RANGES, _S1_FLOAT_RANGES):
        (base_models, ensemble_preds, _y_pred_binary, roc_auc, pr_auc,
         s1_best_params) = stage2_train_ensemble_v54(
            X_train_xgb, X_test_xgb, y_train_acc, y_test_acc,
            stage1_features, model_time,
            optuna_metric=OPTUNA_METRIC_STAGE1,
            optuna_min_recall=float(OPTUNA_MIN_RECALL_STAGE1),
        )

    stage5_feature_importance_shap(base_models, X_test_xgb, stage1_features)
    stage6_feature_space_overlap(
        X_train_xgb, X_test_xgb, y_train_acc, y_test_acc,
        stage1_features, ensemble_preds,
    )

    # Stage-1 threshold sweep + Stage-2 TRANSFORMER training (V57-specific)
    (stage2_model, flagged_idx, s2_test_probs, y_test_np, s2_best_params,
     stage1_threshold_picked, s1_sweep_summary) = _train_stage2_transformer_v57(
        X_train_xgb=X_train_xgb, X_test_xgb=X_test_xgb,
        y_train_acc=y_train_acc, y_test_acc=y_test_acc,
        base_models=base_models, ensemble_preds=ensemble_preds,
        stage2_features=stage2_features,
        s1_best_params=s1_best_params,
        model_time=model_time, output_dir=v51.OUTPUT_DIR,
        stage1_optimise_for=STAGE1_OPTIMISE_FOR,
        stage1_min_recall=float(STAGE1_MIN_RECALL),
        stage1_min_precision=float(STAGE1_MIN_PRECISION),
    )

    # Final test-time sweep — S1 LOCKED, S2 swept (same as V55)
    print("=" * 80)
    print(f"V57 FINAL SWEEP — S1 locked at {stage1_threshold_picked:.4f}, S2 swept")
    print("=" * 80)
    S1_THR, S2_THR, sweep_summary = sweep_thresholds_v52(
        ensemble_preds=ensemble_preds,
        y_test_np=y_test_np,
        flagged_idx=flagged_idx,
        s2_test_probs=s2_test_probs,
        s1_grid=np.array([float(stage1_threshold_picked)]),
        s2_grid=THRESHOLD_GRID_S2,
        min_recall=float(MIN_RECALL_CONSTRAINT),
        max_far=float(MAX_FAR_TARGET),
        optimise_for=OPTIMISE_FOR,
    )

    write_v57_manifests(
        gnn_experiment_dir=gnn_experiment_dir,
        model_time=model_time,
        stage1_features=stage1_features,
        stage1_dropped=stage1_dropped,
        stage2_features=stage2_features,
        s1_thr=S1_THR, s2_thr=S2_THR,
        roc_auc=roc_auc, pr_auc=pr_auc,
        sweep_summary=sweep_summary,
        s1_sweep_summary=s1_sweep_summary,
        v52_extra_features=v52_extra_features,
        s1_best_params=s1_best_params,
        s2_best_params=s2_best_params,
    )

    print("\n" + "=" * 80)
    print(f"V57 XGBoost training complete in {format_duration(time.time() - t_start)}")
    print(f"All plots saved to:  {v51.VIZ_DIR}")
    print(f"All models saved to: {v51.OUTPUT_DIR}")
    print(f"Model time: {model_time}")
    print(f"S1 LOCKED at: {S1_THR:.4f}  |  S2 (transformer) picked: {S2_THR:.4f}")
    print("Next step: ablation_study_v57_simulation_only.py "
          f"--xgboost-experiment-dir {v51.OUTPUT_DIR} "
          f"--gnn-experiment-dir {gnn_experiment_dir}")
    print("=" * 80)
    return True


def run_xgboost_only() -> bool:
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
        print(f"[V57] XGB_RUN_ID not set; using timestamp fallback: {xgb_run_id}")

    v51.OUTPUT_DIR = os.path.join(gnn_experiment_dir, f"xgboost_{xgb_run_id}")
    v51.VIZ_DIR = os.path.join(v51.OUTPUT_DIR, "visualizations")
    print(f"[V57] XGBoost artefacts will be saved under: {v51.OUTPUT_DIR}")
    return run_pipeline_v57(gnn_experiment_dir)


def main() -> int:
    global MEMORIZE_MODE, ADD_PK_HOUR_INTERACTIONS
    global OPTUNA_METRIC_STAGE1, OPTUNA_MIN_RECALL_STAGE1
    global STAGE1_OPTIMISE_FOR, STAGE1_MIN_RECALL, STAGE1_MIN_PRECISION
    global MIN_RECALL_CONSTRAINT, MIN_PRECISION_CONSTRAINT, MAX_FAR_TARGET, OPTIMISE_FOR
    global TRANSFORMER_OPTUNA_TRIALS

    import argparse
    p = argparse.ArgumentParser(
        description="V57 XGBoost-only — V55 with transformer Stage-2 FP detector.")
    p.add_argument("--gnn-experiment-dir", required=True)
    p.add_argument("--memorize-mode", action="store_true", default=False)
    p.add_argument("--no-pk-hour", action="store_true", default=False)

    # Optuna metric / floors (Stage 1 only — Stage 2 transformer uses AUCPR)
    p.add_argument("--optuna-metric-s1", type=str, default=OPTUNA_METRIC_STAGE1,
                   choices=sorted(_ALLOWED_OPTUNA_METRICS))
    p.add_argument("--optuna-min-recall-s1", type=float, default=OPTUNA_MIN_RECALL_STAGE1)
    p.add_argument("--transformer-trials", type=int, default=TRANSFORMER_OPTUNA_TRIALS)

    # Stage-1 OOF threshold sweep
    p.add_argument("--stage1-optimise-for", type=str, default=STAGE1_OPTIMISE_FOR,
                   choices=["precision", "recall", "f1", "f2"],
                   help="Stage-1 OOF sweep objective.")
    p.add_argument("--stage1-min-recall", type=float, default=STAGE1_MIN_RECALL)
    p.add_argument("--stage1-min-precision", type=float, default=STAGE1_MIN_PRECISION)

    # Stage-2 (final) threshold sweep
    p.add_argument("--min-recall", type=float, default=MIN_RECALL_CONSTRAINT)
    p.add_argument("--min-precision", type=float, default=MIN_PRECISION_CONSTRAINT)
    p.add_argument("--optimise-for", type=str, default=OPTIMISE_FOR,
                   choices=["precision", "recall", "f1", "f2"])

    args = p.parse_args()

    if args.memorize_mode:
        MEMORIZE_MODE = True
        v51.MEMORIZE_MODE = True
    if args.no_pk_hour:
        ADD_PK_HOUR_INTERACTIONS = False

    OPTUNA_METRIC_STAGE1 = str(args.optuna_metric_s1)
    OPTUNA_MIN_RECALL_STAGE1 = float(args.optuna_min_recall_s1)
    TRANSFORMER_OPTUNA_TRIALS = int(args.transformer_trials)

    STAGE1_OPTIMISE_FOR = str(args.stage1_optimise_for)
    STAGE1_MIN_RECALL = float(args.stage1_min_recall)
    STAGE1_MIN_PRECISION = float(args.stage1_min_precision)

    MIN_RECALL_CONSTRAINT = float(args.min_recall)
    MIN_PRECISION_CONSTRAINT = float(args.min_precision)
    MAX_FAR_TARGET = float(1.0 - MIN_PRECISION_CONSTRAINT)
    OPTIMISE_FOR = str(args.optimise_for)

    return 0 if run_pipeline_v57(args.gnn_experiment_dir) else 1


if __name__ == "__main__":
    sys.exit(main())
