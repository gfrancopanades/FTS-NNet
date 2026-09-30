#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Self-contained pipeline for the Layer-3 end-to-end crash benchmarks (BL-E0/E1)
==============================================================================

BL-E0 (Transformer) and BL-E1 (MSGNN) bypass the proposed forecast->classify
decomposition: they are trained end-to-end on crash labels directly from
sequences of the raw reconstructed corridor state. There is no separate
forecaster and no XGBoost head, so they don't fit the gnn/xgboost/sim split —
this one module trains AND simulates them.

Pipeline (one SLURM job):
  1. Load the base CSV (`v5.load_data`), use V17's two training windows + the
     June->Oct 2025 sim window. State features = temporal features + raw
     mean_speed/intTot/intP (the Phase-1 state). Standardised on the training
     rows (no leakage).
  2. Build per-(via, sen, pk) sliding windows (length L) -> label = ACCIDENT at
     the next step. Chronological train/val split by target timestamp.
  3. Train the end-to-end model with class-weighted BCE, early-stopped on
     validation AUCPR.
  4. Constrained threshold sweep on validation (maximise precision s.t.
     recall >= 0.40 — the proposed method's operating-point contract).
  5. Frozen rolling simulation on the sim window: score, threshold, report
     confusion-matrix metrics + PR curve.

Artefacts under `<OUTPUT_BASE>/<version>_e2e_<prefix>_<res>_<jobid>/`.

Author: Gerard Franco | Date: June 2026
"""

from __future__ import annotations
from src.paths import (  # portable paths -- see src/paths.py
    PROJECT_ROOT_STR as _AP7_ROOT,
    EXPERIMENTS_ROOT_STR as _AP7_EXPERIMENTS,
    DATA_DIR_STR as _AP7_DATA,
    TABLES_DIR as _AP7_TABLES,
    FIGURES_DIR as _AP7_FIGS,
)

import json
import logging
import os
import time
import warnings
from datetime import datetime

import joblib
import numpy as np
from src.training.sim_dirname import sim_dir_name
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             precision_recall_curve, roc_auc_score)
from sklearn.preprocessing import StandardScaler

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from torch.utils.data import DataLoader, TensorDataset

from src.training import ablation_study_v5_gnn_only as v5
from src.training import ablation_study_v17_gnn_only as v17
# Canonical (via, sen, pk, dat) frame layout + duplicate-interval removal +
# train/test-boundary resolver — shared with the cascade so the two never
# drift apart again.
from src.training.ablation_study_v21_xgboost_only import (
    LOCATION_SORT_KEYS, sort_location_major, dedup_segment_intervals,
    resolve_train_test_boundary, _LOCATION_KEYS)
from src.training.ablation_study_v55_xgboost_only import sweep_stage1_threshold
from src.training.ablation_study_v56_xgboost_only import THRESHOLD_GRID
from src.models.benchmark_end2end import build_end2end, END2END_REGISTRY

TRAIN_WINDOWS = list(v17.TRAIN_WINDOWS)
# Rolling-origin support. SIM_START/SIM_END below already honour their env
# overrides, but the training window did not: every end-to-end arm inherited
# v17's hardcoded pair regardless of BENCH_TRAIN_WINDOWS, so a fold asked to
# test on July-2025 trained on June-2024+May-2025 anyway -- and a fold asked to
# test on May-2025 evaluated *inside* its own training window. Parsed exactly as
# in ablation_study_v51 (';' separated, because SLURM --export splits on ',').
_env_tw = os.environ.get("BENCH_TRAIN_WINDOWS", "")
if _env_tw:
    _sep = ";" if ";" in _env_tw else ","
    TRAIN_WINDOWS = [tuple(w.split(":", 1)) for w in _env_tw.split(_sep) if ":" in w]
    print(f"[E2E] BENCH_TRAIN_WINDOWS override active: {TRAIN_WINDOWS}")
SIM_START_DATE = os.environ.get("BENCH_SIM_START", "2025-06-01")
# Default Jun->Oct (native e2e roll). Overridable to shrink the sim tensor —
# the benchmark only scores June, so BENCH_SIM_END=2025-07-01 cuts the sim
# memory/time ~4x (needed for the feature-rich variants, 72 feats x 48 steps).
SIM_END_DATE = os.environ.get("BENCH_SIM_END", "2025-10-01")
PK_MIN, PK_MAX = 120, 220   # AP-7 corridor in the 5-min dataset (matches v34/v51)

STATE_SEQ_LEN = 48                  # 4h window at 5-min resolution (doc: L=48)
STATIC_FEATURES = ["car", "segment", "ang_curv", "ang_pend_pos",
                   "ang_pend_neg", "via", "sen", "pk"]
TARGET_STATE = ["mean_speed", "intTot", "intP"]   # Phase-1 corridor state

MAX_EPOCHS = 40
PATIENCE = 8
BATCH_SIZE = 1024
OPTIMISE_FOR = "precision"
MIN_RECALL_CONSTRAINT = 0.40


# ---------------------------------------------------------------------------
def _state_columns(df, selected_features):
    """State (temporal) input columns = selected temporal features + raw state,
    minus ids/bookkeeping/label."""
    exclude = set(STATIC_FEATURES) | {"dat", "min", "ACCIDENT", "window_id"}
    cols = [c for c in selected_features if c in df.columns and c not in exclude]
    for t in TARGET_STATE:
        if t in df.columns and t not in cols:
            cols.append(t)
    # numeric only
    cols = [c for c in cols if np.issubdtype(df[c].dtype, np.number)]
    return cols


def _filter_windows(df, windows):
    parts = []
    for s, e in windows:
        parts.append(df[(df["dat"] >= pd.Timestamp(s)) & (df["dat"] < pd.Timestamp(e))])
    return pd.concat(parts, axis=0) if parts else df.iloc[0:0]


def _build_sequences(df, state_cols, static_cols, pk_to_id, sen_to_id, L):
    """Per-(via, sen, pk) sliding windows -> (X, static, y, pk_ids, sen_ids, dat)."""
    df = df.sort_values(["via", "sen", "pk", "dat"]).reset_index(drop=True)
    state = df[state_cols].fillna(0).values.astype(np.float32)
    stat = df[static_cols].fillna(0).values.astype(np.float32)
    y_all = df["ACCIDENT"].fillna(0).astype(np.int8).values
    dat_all = df["dat"].values
    pk_all = df["pk"].values
    sen_all = df["sen"].values

    groups = df.groupby(["via", "sen", "pk"], sort=False).size()
    Xs, Ss, Ys, PK, SEN, DAT = [], [], [], [], [], []
    nf = state.shape[1]
    idx = 0
    for (via, sen, pk), n in groups.items():
        if n <= L:
            idx += n
            continue
        gst = state[idx:idx + n]
        wins = sliding_window_view(gst, (L, nf)).squeeze(axis=1)[: n - L]  # (n-L, L, nf)
        Xs.append(wins.copy())
        Ys.append(y_all[idx + L: idx + n])
        Ss.append(stat[idx + L: idx + n])
        DAT.append(dat_all[idx + L: idx + n])
        PK.append(np.full(n - L, pk_to_id.get(pk, 0), dtype=np.int64))
        SEN.append(np.full(n - L, sen_to_id.get(sen, 0), dtype=np.int64))
        idx += n
    if not Xs:
        raise ValueError("No sequences built — check window length vs data.")
    return (np.concatenate(Xs), np.concatenate(Ss), np.concatenate(Ys),
            np.concatenate(PK), np.concatenate(SEN), np.concatenate(DAT))


def _loader(X, S, y, pk, sen, batch_size, shuffle):
    ds = TensorDataset(torch.as_tensor(X), torch.as_tensor(y, dtype=torch.float32),
                       torch.as_tensor(pk), torch.as_tensor(S), torch.as_tensor(sen))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def build_experiment_name(version):
    return (f"{version}_e2e_{v5.get_experiment_prefix()}_"
            f"{v5.get_time_resolution_token()}_{v5.get_job_id()}")


# ---------------------------------------------------------------------------
def run_end2end(version: str, architecture: str, title: str) -> bool:
    if architecture not in END2END_REGISTRY:
        raise KeyError(f"Unknown end-to-end arch {architecture!r}")
    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore")

    model_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(v5.OUTPUT_BASE_DIR, build_experiment_name(version))
    viz_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    max_epochs = 3 if getattr(v5, "TEST_MODE", False) else MAX_EPOCHS
    patience = 2 if getattr(v5, "TEST_MODE", False) else PATIENCE

    print("=" * 80)
    print(f"{version.upper()} END-TO-END BENCHMARK — {title}")
    print(f"Architecture: {architecture}  | Output: {output_dir}")
    print(f"Train windows: {TRAIN_WINDOWS} | Sim: {SIM_START_DATE}->{SIM_END_DATE}")
    print(f"State seq len L={STATE_SEQ_LEN} | device={device}")
    print("=" * 80)

    t0 = time.time()
    df_full, selected_features = v5.load_data()
    df_full["dat"] = pd.to_datetime(df_full["dat"])
    df_full = df_full[(df_full["pk"] >= PK_MIN) & (df_full["pk"] <= PK_MAX)].copy()
    # Remove the raw CSV's double-mapped segment-intervals BEFORE lagging /
    # sequence building — duplicates insert fake time steps into every
    # (via, sen, pk) group, corrupting rolling windows and inflating evaluation
    # row counts. Same shared helper (and policy) as the cascade.
    df_full = dedup_segment_intervals(df_full, label="E2E")

    # ── LAGGED-TRAFFIC MODE (E2E_LAG_TRAFFIC) ─────────────────────────────
    # 72-h-legal deep sequence model USING real traffic without forecasting it:
    # lag the traffic channels by N bins (t-72h = 864) so the window carries the
    # observed traffic available at issuance, while covariates stay current for
    # the target interval. Parallels the flat lagged-traffic classifier.
    _e2e_lag = int(os.environ.get("E2E_LAG_TRAFFIC", "0"))
    if _e2e_lag > 0:
        df_full = sort_location_major(df_full)
        _loc = [c for c in _LOCATION_KEYS if c in df_full.columns]
        for _c in ("mean_speed", "intTot", "intP", "car"):
            if _c in df_full.columns:
                df_full[_c] = df_full.groupby(_loc)[_c].shift(_e2e_lag)
        df_full = df_full.dropna(subset=["mean_speed"]).reset_index(drop=True)
        print(f"[E2E-LAG-TRAFFIC] traffic lagged by {_e2e_lag} bins (t-72h, no forecast); "
              f"covariates kept current -> {len(df_full):,} rows")

    static_cols = [c for c in STATIC_FEATURES if c in df_full.columns]
    state_cols = _state_columns(df_full, selected_features)

    # ── 72-h-LEGAL COVARIATE-ONLY MODE (E2E_NO_TRAFFIC) ────────────────────
    # The honest "end-to-end deep learning without decomposition" baseline for
    # a 72-h horizon: no traffic is observable that far ahead, so a legitimate
    # no-forecast model may use ONLY exogenous covariates (calendar, 3-day
    # weather forecast, geometry, mobility). Drop every traffic channel —
    # including `car` (= intTot - intP), which is traffic mislabelled as static.
    # The deep model then learns crashes directly from the covariate window.
    if os.environ.get("E2E_NO_TRAFFIC", "").lower() in ("1", "true", "yes"):
        traffic = {"mean_speed", "intTot", "intP", "car",
                   "mean_speed_gnn", "intTot_gnn", "intP_gnn"}
        state_cols = [c for c in state_cols if c not in traffic]
        static_cols = [c for c in static_cols if c not in traffic]
        print(f"[E2E-NO-TRAFFIC] 72h-legal covariate-only: dropped traffic (incl. car) "
              f"-> state={len(state_cols)} static={len(static_cols)}")
        print(f"[E2E-NO-TRAFFIC] covariate state cols: {state_cols}")

    print(f"[E2E] state features: {len(state_cols)} | static: {len(static_cols)}")

    df_tr_all = _filter_windows(df_full, TRAIN_WINDOWS)
    df_sim = df_full[(df_full["dat"] >= pd.Timestamp(SIM_START_DATE))
                     & (df_full["dat"] < pd.Timestamp(SIM_END_DATE))].copy()

    # pk / sen id maps over the union (so sim ids are known to the embedding).
    pks = sorted(pd.unique(pd.concat([df_tr_all["pk"], df_sim["pk"]])))
    pk_to_id = {pk: i for i, pk in enumerate(pks)}
    sens = sorted(pd.unique(pd.concat([df_tr_all["sen"], df_sim["sen"]])))
    sen_to_id = {s: (0 if i == 0 else 1) for i, s in enumerate(sens)}  # dec->0, cre->1

    # Standardise state features on training rows (no leakage), apply to all.
    scaler = StandardScaler().fit(df_tr_all[state_cols].fillna(0).values)
    for d in (df_tr_all, df_sim):
        d[state_cols] = scaler.transform(d[state_cols].fillna(0).values)

    Xtr, Str, ytr, pktr, sentr, dattr = _build_sequences(
        df_tr_all, state_cols, static_cols, pk_to_id, sen_to_id, STATE_SEQ_LEN)
    # chronological train/val split by target timestamp (80/20)
    cut = np.quantile(dattr.astype("datetime64[ns]").astype(np.int64), 0.8)
    tr_mask = dattr.astype("datetime64[ns]").astype(np.int64) <= cut
    va_mask = ~tr_mask
    n_pos = max(int(ytr[tr_mask].sum()), 1); n_neg = max(int((ytr[tr_mask] == 0).sum()), 1)
    pos_weight = n_neg / n_pos
    print(f"[E2E] train seqs={tr_mask.sum():,} (pos={int(ytr[tr_mask].sum())}) "
          f"val seqs={va_mask.sum():,} (pos={int(ytr[va_mask].sum())}) pos_weight={pos_weight:.1f}")

    model = build_end2end(
        architecture, input_size=Xtr.shape[2], num_pks=len(pk_to_id),
        static_feature_dim=len(static_cols), pos_weight=pos_weight)

    tr_loader = _loader(Xtr[tr_mask], Str[tr_mask], ytr[tr_mask], pktr[tr_mask],
                        sentr[tr_mask], BATCH_SIZE, True)
    va_loader = _loader(Xtr[va_mask], Str[va_mask], ytr[va_mask], pktr[va_mask],
                        sentr[va_mask], BATCH_SIZE, False)
    trainer = pl.Trainer(
        max_epochs=max_epochs, accelerator="gpu" if device == "cuda" else "cpu",
        devices=1, logger=False, enable_progress_bar=False, enable_model_summary=False,
        callbacks=[EarlyStopping(monitor="val_aucpr", mode="max", patience=patience)],
        gradient_clip_val=1.0, precision="16-mixed" if device == "cuda" else 32)
    trainer.fit(model, tr_loader, va_loader)

    # threshold sweep on validation
    model.eval().to(device)
    va_prob = _predict_loader(model, va_loader, device)
    threshold, sweep = sweep_stage1_threshold(
        y_true=ytr[va_mask], y_prob=va_prob, optimise_for=OPTIMISE_FOR,
        min_recall=MIN_RECALL_CONSTRAINT, min_precision=0.0, grid=THRESHOLD_GRID)

    # save artefacts
    arch_config = {"input_size": int(Xtr.shape[2]), "num_pks": int(len(pk_to_id)),
                   "static_feature_dim": int(len(static_cols))}
    torch.save({"state_dict": model.to("cpu").state_dict(), "architecture": architecture,
                "arch_config": arch_config},
               os.path.join(output_dir, f"end2end-model_version={model_time}.pt"))
    joblib.dump(scaler, os.path.join(output_dir, f"state-scaler_version={model_time}.pkl"))
    joblib.dump({"pk_to_id": pk_to_id, "sen_to_id": sen_to_id},
                os.path.join(output_dir, f"id-maps_version={model_time}.pkl"))

    manifest = {
        "pipeline_version": f"{version}_end2end",
        "benchmark_version": version, "benchmark_layer": "L3_end2end",
        "architecture": architecture, "model_time": model_time,
        "state_cols": state_cols, "static_cols": static_cols,
        "seq_len": STATE_SEQ_LEN, "threshold": float(threshold),
        "threshold_sweep": {k: (float(v) if isinstance(v, (np.floating, float)) else v)
                            for k, v in sweep.items()},
        "train_windows": [{"start": s, "end": e} for s, e in TRAIN_WINDOWS],
        "sim_dates": {"start": SIM_START_DATE, "end": SIM_END_DATE},
        "pk_range": {"min": PK_MIN, "max": PK_MAX},
        "arch_config": arch_config, "pos_weight": float(pos_weight),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(output_dir, f"end2end_manifest_{model_time}.json"), "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    print(f"[E2E] training complete in {v5.format_duration(time.time() - t0)} | "
          f"threshold={threshold:.4f}")

    # ── frozen rolling simulation ────────────────────────────────────────────
    _simulate(model, df_sim, state_cols, static_cols, pk_to_id, sen_to_id,
              STATE_SEQ_LEN, threshold, architecture, version, model_time,
              output_dir, viz_dir, device)

    print("=" * 80)
    print(f"{version.upper()} ({architecture}) END-TO-END COMPLETE -> {output_dir}")
    print("=" * 80)
    return True


def _predict_loader(model, loader, device):
    probs = []
    model.eval()
    with torch.no_grad():
        for x, _y, pk, st, sen in loader:
            logits = model(x.to(device), pk.to(device), st.to(device), sen.to(device))
            probs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probs) if probs else np.array([])


def _simulate(model, df_sim, state_cols, static_cols, pk_to_id, sen_to_id, L,
              threshold, architecture, version, model_time, output_dir, viz_dir, device):
    if len(df_sim) == 0:
        print("[E2E-SIM] No sim data."); return
    X, S, y, pk, sen, dat = _build_sequences(
        df_sim, state_cols, static_cols, pk_to_id, sen_to_id, L)
    loader = _loader(X, S, y, pk, sen, BATCH_SIZE, False)
    proba = _predict_loader(model.to(device), loader, device)
    pred = (proba >= threshold).astype(int)

    cm = confusion_matrix(y, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
    f1 = 2 * p * r / max(p + r, 1e-12); f2 = 5 * p * r / max(4 * p + r, 1e-12)
    metrics = dict(tp=int(tp), fp=int(fp), fn=int(fn), precision=float(p), recall=float(r),
                   f1=float(f1), f2=float(f2), far=float(fp / max(tp + fp, 1)),
                   pr_auc=float(average_precision_score(y, proba)) if y.sum() else 0.0,
                   roc_auc=float(roc_auc_score(y, proba)) if (y.sum() and (y == 0).any()) else 0.5)
    print(f"\n[E2E-SIM {architecture}] frozen-protocol metrics "
          f"({SIM_START_DATE}->{SIM_END_DATE}):")
    for k, v in metrics.items():
        print(f"   {k:10s} {v:,.4f}" if isinstance(v, float) else f"   {k:10s} {v:,}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 5))
    pc, rc, _ = precision_recall_curve(y, proba)
    ax.plot(rc, pc, color="darkgreen", label=f"{architecture} (AUCPR={metrics['pr_auc']:.4f})")
    ax.axhline(y.mean(), color="gray", ls=":", label=f"prevalence={y.mean():.5f}")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title(f"PR — Layer-3 end-to-end ({architecture})", fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, f"e2e_pr_curve_{architecture}.png"),
                dpi=120, bbox_inches="tight")
    plt.close(fig)

    # keyed on model + training window + simulation window (see sim_dirname.py)
    _tw = TRAIN_WINDOWS if isinstance(TRAIN_WINDOWS, (list, tuple)) and TRAIN_WINDOWS else []
    _trs = min(w[0] for w in _tw) if _tw else None
    _tre = max(w[1] for w in _tw) if _tw else None
    sim_dir = os.path.join(output_dir, sim_dir_name(
        "simulation_e2e", model_time, _trs, _tre, SIM_START_DATE, SIM_END_DATE))
    os.makedirs(sim_dir, exist_ok=True)
    pd.DataFrame({"dat": dat, "pk_id": pk, "sen_id": sen,
                  "accident_probability": proba, "accident_pred_binary": pred,
                  "ACCIDENT_real": y}).to_csv(
        os.path.join(sim_dir, f"simulation_e2e_{architecture}_{model_time}.csv"),
        sep=";", index=False)
    with open(os.path.join(output_dir, f"end2end_sim_manifest_{model_time}.json"), "w") as f:
        json.dump({"architecture": architecture, "version": version,
                   "model_time": model_time, "sim_dates": {"start": SIM_START_DATE,
                   "end": SIM_END_DATE}, "threshold": float(threshold),
                   "metrics": metrics}, f, indent=2, default=str)
    print(f"[E2E-SIM] saved -> {sim_dir}")
