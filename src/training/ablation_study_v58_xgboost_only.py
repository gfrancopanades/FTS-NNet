#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V58 XGBoost-only: V57 with a HARDENED Stage-2 transformer training recipe
=========================================================================

V57 simulation analysis (job 2613091) showed the Stage-2 transformer working
as an FP filter (FP 117,687 → 1,154) but at a steep recall cost: it dropped
959 true positives that Stage-1 had already caught (Recall 0.846 → 0.572) and
its two-stage PR-AUC (0.5669) was *below* Stage-1's (0.5769) — i.e. as a
ranker it was not adding separation, only buying precision at a threshold.

The root causes in V57's transformer trainer were optimization-side, not
architectural:

  * Validation was a RANDOM stratified split of temporally-ordered rows, so
    early-stopping / Optuna selected models that leaked time structure and
    generalised poorly to the out-of-time June-2025 simulation.
  * Loss was plain pos-weighted BCE — with the extreme FP:TP imbalance in the
    Stage-2 pool the loss is dominated by easy negatives; no hard-example
    focusing.
  * Constant LR, only 10 Optuna trials, patience 10.

V58 changes ONLY the transformer training recipe (steps 1-3 — the Stage-1
ensemble, OOF Stage-1 sweep and pool filtering — are byte-identical to V57).
The saved checkpoint keeps the exact same self-describing format
(`arch_config` = n_features/d_model/n_heads/n_layers/dropout + feat stats +
feature_names), so the existing `ablation_study_v57_simulation_only.py`
loader reconstructs it unchanged.

What is new in V58 step 4:

  1. CHRONOLOGICAL validation split (last `VAL_FRAC` of the time-ordered
     Stage-2 pool is held out), with a stratified-random fallback only if the
     temporal tail is too positive-poor to score AUCPR reliably. This makes
     early-stopping / Optuna honest about out-of-time generalisation.
  2. SELECTABLE loss, chosen by Optuna per trial:
       - "focal"        : focal loss (α fixed, γ tuned) — focuses on the hard
                          FP/TP boundary instead of easy negatives.
       - "bce_posweight": V57's pos-weighted BCE (kept as a baseline arm).
       - "bce_sampler"  : plain BCE + class-balanced WeightedRandomSampler.
     (The three modes never double-correct the imbalance.)
  3. Cosine LR schedule with linear warmup (per-epoch).
  4. More Optuna trials (20) and slightly higher patience (12).

Artefacts:
  - `transformer-stage2-fp-filter_version=<model_time>.pt`  (same format as V57)
  - `v58_xgboost_manifest_<model_time>.json`  (experiments.stage2_training_recipe
     records the V58 knobs; experiments.stage2_model_type = "transformer").

Based on: ablation_study_v57_xgboost_only (transformer trainer hardened).

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
import math
import os
import sys
import time
from datetime import datetime

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import xgboost as xgb
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import src.training.ablation_study_v51_xgboost_only as v51
import src.training.ablation_study_v54_xgboost_only as v54
import src.training.ablation_study_v57_xgboost_only as v57
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
# The transformer module + inference helper are unchanged from V57 (keeps the
# checkpoint architecture identical so the V57 simulation loader still works).
from src.training.ablation_study_v57_xgboost_only import (
    TabTransformerFPFilter,
    _predict_transformer,
    STAGE1_OOF_GRID,
)


# =============================================================================
# V58 CONFIGURATION
# =============================================================================

PIPELINE_VERSION = "v58_xgboost_only"

# Inherit V54 feature pruning + experimental knobs (same chain as V57).
STAGE1_DROP_FEATURES: list[str] = list(v54.STAGE1_DROP_FEATURES)
STAGE2_FEATURES: list[str] | None = (
    list(v54.STAGE2_FEATURES) if v54.STAGE2_FEATURES is not None else None
)
ADD_PK_HOUR_INTERACTIONS: bool = v54.ADD_PK_HOUR_INTERACTIONS
MEMORIZE_MODE: bool = bool(v51.MEMORIZE_MODE)

# Stage-1 Optuna (unchanged from V57).
OPTUNA_METRIC_STAGE1: str = "aucpr"
OPTUNA_MIN_RECALL_STAGE1: float = 0.30
OPTUNA_METRIC_STAGE2: str = "aucpr"
OPTUNA_MIN_RECALL_STAGE2: float = 0.30

# Stage-1 OOF threshold sweep (same as V57).
STAGE1_OPTIMISE_FOR: str = "recall"
STAGE1_MIN_RECALL: float = 0.90
STAGE1_MIN_PRECISION: float = 0.0

# Stage-2 (final) threshold sweep — S1 LOCKED to the training-time pick.
OPTIMISE_FOR: str = "precision"
MIN_RECALL_CONSTRAINT: float = 0.0
MIN_PRECISION_CONSTRAINT: float = 0.90
MAX_FAR_TARGET: float = 1.0 - MIN_PRECISION_CONSTRAINT  # = 0.10

THRESHOLD_GRID_S2 = THRESHOLD_GRID_S2  # inherited

# -----------------------------------------------------------------------------
# Transformer training knobs (V58 — hardened).
# -----------------------------------------------------------------------------
TRANSFORMER_OPTUNA_TRIALS: int = 20        # was 10 in V57
TRANSFORMER_MAX_EPOCHS: int = 100
TRANSFORMER_PATIENCE: int = 12             # was 10 in V57
TRANSFORMER_BATCH_SIZE: int = 256
TRANSFORMER_DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

# Chronological validation split for honest out-of-time early stopping.
VAL_FRAC: float = 0.2
# If the temporal-tail validation slice has fewer than this many positives,
# fall back to a stratified random split so AUCPR stays meaningful.
MIN_VAL_POSITIVES: int = 10

# Focal-loss positive-class weight (γ is tuned by Optuna). 0.75 leans toward
# recall of the rare positive class without ignoring negatives entirely.
FOCAL_ALPHA: float = 0.75
# Linear-warmup fraction of the cosine LR schedule.
WARMUP_FRAC: float = 0.1


# =============================================================================
# Loss + schedule helpers (V58)
# =============================================================================

class _FocalLoss(nn.Module):
    """Binary focal loss on logits (Lin et al. 2017), mean-reduced.

    alpha weights the positive class; gamma down-weights easy examples so the
    optimiser concentrates on the hard FP/TP boundary that dominates the
    Stage-2 pool.
    """

    def __init__(self, alpha: float, gamma: float):
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        pos = targets >= 0.5
        pt = torch.where(pos, p, 1.0 - p).clamp(min=1e-6, max=1.0)
        alpha_t = torch.where(pos, torch.as_tensor(self.alpha, device=logits.device),
                              torch.as_tensor(1.0 - self.alpha, device=logits.device))
        return (alpha_t * (1.0 - pt) ** self.gamma * bce).mean()


def _cosine_warmup_lambda(max_epochs: int, warmup_frac: float):
    warmup = max(1, int(round(warmup_frac * max_epochs)))

    def _fn(epoch: int) -> float:  # epoch is 0-indexed by LambdaLR
        if epoch < warmup:
            return float(epoch + 1) / float(warmup)
        progress = (epoch - warmup) / max(1, max_epochs - warmup)
        # floor the cosine at 0.05·lr so a long tail of epochs still learns.
        return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return _fn


def _temporal_or_stratified_split(X: np.ndarray, y: np.ndarray, *,
                                  val_frac: float, min_val_pos: int, seed: int):
    """Chronological tail split (rows are time-ordered), with a stratified
    fallback when the tail is too positive-poor to score AUCPR.

    Returns (X_tr, X_va, y_tr, y_va, split_kind).
    """
    n = len(X)
    cut = int(round(n * (1.0 - val_frac)))
    cut = min(max(cut, 1), n - 1)
    X_tr, X_va = X[:cut], X[cut:]
    y_tr, y_va = y[:cut], y[cut:]

    if y_va.sum() >= min_val_pos and y_tr.sum() >= min_val_pos:
        return X_tr, X_va, y_tr, y_va, "temporal"

    # Fallback: stratified random (V57 behaviour) so val always has positives.
    stratify = y if 0 < y.sum() < len(y) else None
    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=val_frac, random_state=seed, stratify=stratify,
    )
    return X_tr, X_va, y_tr, y_va, "stratified_fallback"


# =============================================================================
# Hardened transformer fit (V58)
# =============================================================================

def _fit_transformer_v58(X_tr: np.ndarray, y_tr: np.ndarray,
                         X_va: np.ndarray, y_va: np.ndarray,
                         *, d_model: int, n_heads: int, n_layers: int,
                         dropout: float, lr: float, weight_decay: float,
                         loss_type: str, focal_gamma: float,
                         max_epochs: int, patience: int, batch_size: int,
                         device: str, seed: int = RANDOM_STATE,
                         verbose: bool = False) -> tuple[nn.Module, float, int]:
    """Train one transformer; early-stop on validation AUCPR.

    V58 additions vs V57's `_fit_transformer`:
      * `loss_type` in {"focal", "bce_posweight", "bce_sampler"}.
      * cosine LR schedule with linear warmup.
      * class-balanced WeightedRandomSampler for the "bce_sampler" arm.

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

    # --- loss + sampler selection (no double-correction of the imbalance) ----
    sampler = None
    if loss_type == "focal":
        criterion = _FocalLoss(alpha=FOCAL_ALPHA, gamma=focal_gamma)
    elif loss_type == "bce_sampler":
        criterion = nn.BCEWithLogitsLoss()
        # balance classes by inverse frequency, drawn with replacement.
        class_w = np.where(y_tr >= 0.5, 1.0 / n_pos, 1.0 / n_neg).astype(np.float64)
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(class_w, dtype=torch.double),
            num_samples=len(y_tr), replacement=True,
            generator=torch.Generator().manual_seed(seed),
        )
    else:  # "bce_posweight" — V57 baseline arm
        pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32, device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, _cosine_warmup_lambda(max_epochs, WARMUP_FRAC))

    X_tr_t = torch.as_tensor(X_tr, dtype=torch.float32)
    y_tr_t = torch.as_tensor(y_tr, dtype=torch.float32)
    ds = TensorDataset(X_tr_t, y_tr_t)
    if sampler is not None:
        dl = DataLoader(ds, batch_size=batch_size, sampler=sampler)
    else:
        dl = DataLoader(ds, batch_size=batch_size, shuffle=True,
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
        scheduler.step()

        va_probs = _predict_transformer(model, X_va, device)
        va_aucpr = average_precision_score(y_va, va_probs) if y_va.sum() > 0 else 0.0

        if verbose and (epoch % 10 == 0 or epoch == 1):
            print(f"    epoch {epoch:3d}  loss={total_loss / max(len(ds), 1):.5f}  "
                  f"lr={optimizer.param_groups[0]['lr']:.2e}  val_aucpr={va_aucpr:.4f}")

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
# Stage-2 training (V58) — OOF + Stage-1 sweep + filter, then HARDENED fit
# =============================================================================

def _train_stage2_transformer_v58(*, X_train_xgb, X_test_xgb, y_train_acc, y_test_acc,
                                  base_models, ensemble_preds, stage2_features,
                                  s1_best_params, model_time, output_dir,
                                  stage1_optimise_for: str,
                                  stage1_min_recall: float,
                                  stage1_min_precision: float):
    """V58 Stage-2 trainer.

    Steps 1-3 (OOF Stage-1 probs → Stage-1 OOF sweep → pool filtering) are
    identical to V57/V55. Step 4 uses the hardened `_fit_transformer_v58` with
    a chronological validation split and a richer Optuna search (loss type +
    focal γ in addition to the V57 arch/optim knobs).
    """
    X_tr = X_train_xgb.reset_index(drop=True)
    y_tr = y_train_acc.reset_index(drop=True)
    stage1_features = list(X_tr.columns)
    neg_pos_ratio = ENSEMBLE_CONFIG["neg_pos_ratio"]

    # 1. OOF Stage-1 probabilities on the training set (unchanged from V57)
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

    # 2. Stage-1 OOF threshold sweep (same as V57)
    print("=" * 80)
    print("V58 STAGE 7A: STAGE-1 OOF THRESHOLD SWEEP")
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

    # 3. Filter both train and test pools at the picked Stage-1 threshold.
    #    NOTE: rows stay in their original (chronological) order through the
    #    mask, which V58's temporal validation split relies on.
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
    print(f"[V58] Stage-2 training pool @ S1≥{stage1_threshold:.4f}: "
          f"{len(y_s2_tr):,} rows  (TP={n_tp_s2}, FP={n_fp_s2})")
    print(f"[V58] Stage-2 test pool     @ S1≥{stage1_threshold:.4f}: "
          f"{len(y_s2_te):,} rows  (TP={int(y_s2_te.sum())}, "
          f"FP={int((y_s2_te == 0).sum())})")

    # 4. HARDENED V58 transformer (standardise → temporal split → Optuna → fit)
    print("=" * 80)
    print("V58 STAGE 7B: Stage-2 TRANSFORMER FP filter — hardened recipe")
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

    # Chronological inner split (with stratified fallback) for Optuna + early
    # stopping. The test pool is reserved for the final threshold sweep.
    X_in_tr, X_in_va, y_in_tr, y_in_va, split_kind = _temporal_or_stratified_split(
        X_s2_tr_std, y_s2_tr_np, val_frac=VAL_FRAC,
        min_val_pos=MIN_VAL_POSITIVES, seed=RANDOM_STATE,
    )
    print(f"[V58] Inner validation split: {split_kind}  "
          f"(train={len(y_in_tr):,} [pos={int(y_in_tr.sum())}], "
          f"val={len(y_in_va):,} [pos={int(y_in_va.sum())}])")

    def _t_objective(trial: optuna.trial.Trial) -> float:
        d_model = trial.suggest_categorical("d_model", [32, 64, 128])
        n_heads = trial.suggest_categorical("n_heads", [2, 4])
        n_layers = trial.suggest_int("n_layers", 1, 3)
        dropout = trial.suggest_float("dropout", 0.0, 0.3)
        lr = trial.suggest_float("lr", 1e-4, 3e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        loss_type = trial.suggest_categorical(
            "loss_type", ["focal", "bce_posweight", "bce_sampler"])
        # focal_gamma only matters for the focal arm; suggested unconditionally
        # so Optuna's search space stays well-defined across trials.
        focal_gamma = trial.suggest_float("focal_gamma", 0.5, 3.0)
        _, va_aucpr, _ = _fit_transformer_v58(
            X_in_tr, y_in_tr, X_in_va, y_in_va,
            d_model=d_model, n_heads=n_heads, n_layers=n_layers,
            dropout=dropout, lr=lr, weight_decay=weight_decay,
            loss_type=loss_type, focal_gamma=focal_gamma,
            max_epochs=min(TRANSFORMER_MAX_EPOCHS, 40),
            patience=max(TRANSFORMER_PATIENCE // 2, 6),
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
    stage2_model, final_va_aucpr, best_epoch = _fit_transformer_v58(
        X_in_tr, y_in_tr, X_in_va, y_in_va,
        d_model=bp["d_model"], n_heads=bp["n_heads"], n_layers=bp["n_layers"],
        dropout=bp["dropout"], lr=bp["lr"], weight_decay=bp["weight_decay"],
        loss_type=bp["loss_type"], focal_gamma=bp["focal_gamma"],
        max_epochs=TRANSFORMER_MAX_EPOCHS,
        patience=TRANSFORMER_PATIENCE,
        batch_size=TRANSFORMER_BATCH_SIZE,
        device=TRANSFORMER_DEVICE,
        verbose=True,
    )
    print(f"[V58] Final transformer: val AUCPR={final_va_aucpr:.4f} (epoch {best_epoch})  "
          f"loss={bp['loss_type']}  gamma={bp['focal_gamma']:.2f}")

    # arch_config keys MUST match what the V57 sim loader reconstructs.
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
            "loss_type": str(bp["loss_type"]),
            "focal_gamma": float(bp["focal_gamma"]),
            "focal_alpha": float(FOCAL_ALPHA),
            "val_split": split_kind,
            "lr_schedule": "cosine_warmup",
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
    s2_best_params = {**bp, "best_epoch": int(best_epoch), "val_split": split_kind}
    return (stage2_model, flagged_idx, s2_test_probs, y_test_np,
            s2_best_params, float(stage1_threshold), s1_sweep_summary)


# =============================================================================
# Manifests
# =============================================================================

def write_v58_manifests(*, gnn_experiment_dir: str, model_time: str,
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
        # V58-specific record of the hardened transformer recipe.
        "stage2_training_recipe": {
            "val_split": "temporal_tail_with_stratified_fallback",
            "val_frac": float(VAL_FRAC),
            "min_val_positives": int(MIN_VAL_POSITIVES),
            "loss_choices": ["focal", "bce_posweight", "bce_sampler"],
            "focal_alpha": float(FOCAL_ALPHA),
            "lr_schedule": "cosine_warmup",
            "warmup_frac": float(WARMUP_FRAC),
            "optuna_trials": int(TRANSFORMER_OPTUNA_TRIALS),
            "patience": int(TRANSFORMER_PATIENCE),
            "max_epochs": int(TRANSFORMER_MAX_EPOCHS),
            "chosen_loss_type": str(s2_best_params.get("loss_type", "")),
            "chosen_focal_gamma": float(s2_best_params.get("focal_gamma", 0.0)),
            "chosen_val_split": str(s2_best_params.get("val_split", "")),
        },
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
        "based_on": "ablation_study_v57_xgboost_only",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    manifest_path = os.path.join(v51.OUTPUT_DIR, f"v58_xgboost_manifest_{model_time}.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"[V58] Manifest saved: {manifest_path}")

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
    exp_config_path = os.path.join(v51.OUTPUT_DIR, "experiment_config_xgb_v58.json")
    with open(exp_config_path, "w") as f:
        json.dump(exp_config, f, indent=2, default=str)
    print(f"[V58] Sim-runner config saved: {exp_config_path}")


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
            print(f"[V58] v54._propagate_to_v52() failed: {e}")


def run_pipeline_v58(gnn_experiment_dir: str) -> bool:
    _propagate_to_v54()
    t_start = time.time()

    print("=" * 80)
    print("V58 XGBoost-only — V57 with a HARDENED transformer Stage-2 trainer")
    print(f"Output dir:      {v51.OUTPUT_DIR}")
    print(f"GNN experiment:  {gnn_experiment_dir}")
    print(f"MEMORIZE_MODE:   {bool(v51.MEMORIZE_MODE)}")
    print(f"Device:          {TRANSFORMER_DEVICE}")
    print("Optuna objective config:")
    print(f"  Stage 1: metric={OPTUNA_METRIC_STAGE1}  min_recall={OPTUNA_MIN_RECALL_STAGE1}")
    print(f"  Stage 2: transformer, val AUCPR, trials={TRANSFORMER_OPTUNA_TRIALS}")
    print("  Stage 2 recipe: temporal val split, focal/bce loss search, cosine LR")
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
        raise NotImplementedError("V58 does not yet implement MEMORIZE_MODE (no holdout pool).")

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

    # Stage-1 threshold sweep + Stage-2 TRANSFORMER training (V58-specific)
    (stage2_model, flagged_idx, s2_test_probs, y_test_np, s2_best_params,
     stage1_threshold_picked, s1_sweep_summary) = _train_stage2_transformer_v58(
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

    # Final test-time sweep — S1 LOCKED, S2 swept (same as V57)
    print("=" * 80)
    print(f"V58 FINAL SWEEP — S1 locked at {stage1_threshold_picked:.4f}, S2 swept")
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

    write_v58_manifests(
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
    print(f"V58 XGBoost training complete in {format_duration(time.time() - t_start)}")
    print(f"All plots saved to:  {v51.VIZ_DIR}")
    print(f"All models saved to: {v51.OUTPUT_DIR}")
    print(f"Model time: {model_time}")
    print(f"S1 LOCKED at: {S1_THR:.4f}  |  S2 (transformer) picked: {S2_THR:.4f}")
    print("Next step: ablation_study_v57_simulation_only.py "
          f"--xgboost-experiment-dir {v51.OUTPUT_DIR} "
          f"--gnn-experiment-dir {gnn_experiment_dir}")
    print("  (the V57 simulation loader reconstructs the V58 checkpoint unchanged)")
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
        print(f"[V58] XGB_RUN_ID not set; using timestamp fallback: {xgb_run_id}")

    v51.OUTPUT_DIR = os.path.join(gnn_experiment_dir, f"xgboost_{xgb_run_id}")
    v51.VIZ_DIR = os.path.join(v51.OUTPUT_DIR, "visualizations")
    print(f"[V58] XGBoost artefacts will be saved under: {v51.OUTPUT_DIR}")
    return run_pipeline_v58(gnn_experiment_dir)


def main() -> int:
    global MEMORIZE_MODE, ADD_PK_HOUR_INTERACTIONS
    global OPTUNA_METRIC_STAGE1, OPTUNA_MIN_RECALL_STAGE1
    global STAGE1_OPTIMISE_FOR, STAGE1_MIN_RECALL, STAGE1_MIN_PRECISION
    global MIN_RECALL_CONSTRAINT, MIN_PRECISION_CONSTRAINT, MAX_FAR_TARGET, OPTIMISE_FOR
    global TRANSFORMER_OPTUNA_TRIALS, TRANSFORMER_PATIENCE, VAL_FRAC, FOCAL_ALPHA

    import argparse
    p = argparse.ArgumentParser(
        description="V58 XGBoost-only — V57 with a hardened transformer Stage-2 trainer.")
    p.add_argument("--gnn-experiment-dir", required=True)
    p.add_argument("--memorize-mode", action="store_true", default=False)
    p.add_argument("--no-pk-hour", action="store_true", default=False)

    # Optuna metric / floors (Stage 1 only — Stage 2 transformer uses AUCPR)
    p.add_argument("--optuna-metric-s1", type=str, default=OPTUNA_METRIC_STAGE1,
                   choices=sorted(_ALLOWED_OPTUNA_METRICS))
    p.add_argument("--optuna-min-recall-s1", type=float, default=OPTUNA_MIN_RECALL_STAGE1)

    # V58 transformer recipe knobs
    p.add_argument("--transformer-trials", type=int, default=TRANSFORMER_OPTUNA_TRIALS)
    p.add_argument("--transformer-patience", type=int, default=TRANSFORMER_PATIENCE)
    p.add_argument("--val-frac", type=float, default=VAL_FRAC,
                   help="Chronological tail fraction held out for Stage-2 early stopping.")
    p.add_argument("--focal-alpha", type=float, default=FOCAL_ALPHA,
                   help="Positive-class weight for the focal-loss arm.")

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
    TRANSFORMER_PATIENCE = int(args.transformer_patience)
    VAL_FRAC = float(args.val_frac)
    FOCAL_ALPHA = float(args.focal_alpha)

    STAGE1_OPTIMISE_FOR = str(args.stage1_optimise_for)
    STAGE1_MIN_RECALL = float(args.stage1_min_recall)
    STAGE1_MIN_PRECISION = float(args.stage1_min_precision)

    MIN_RECALL_CONSTRAINT = float(args.min_recall)
    MIN_PRECISION_CONSTRAINT = float(args.min_precision)
    MAX_FAR_TARGET = float(1.0 - MIN_PRECISION_CONSTRAINT)
    OPTIMISE_FOR = str(args.optimise_for)

    return 0 if run_pipeline_v58(args.gnn_experiment_dir) else 1


if __name__ == "__main__":
    sys.exit(main())
