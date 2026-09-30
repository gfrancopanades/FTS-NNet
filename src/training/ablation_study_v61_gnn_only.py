#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Ablation Study V61 — BL-F0 Historical Average (Layer-1 forecasting benchmark)
=============================================================================

Non-parametric seasonal-naive baseline. The forecast for a corridor state is
the **average daily profile from the same weekdays of the prior-year window**,
with the **volume corrected by the recent weeks**:

  * PROFILE (shape of mean_speed, intTot, intP) is averaged over the prior-year
    window (V17's first training window, 2024-06-01 → 2024-07-01), keyed by
    (via, sen, pk, diaSem, hor, min).
  * VOLUME CORRECTION is a per-(via, sen, pk) factor
    mean(intTot | recent window) / mean(intTot | prior-year window) computed
    from V17's second training window (2025-05-01 → 2025-06-01) and applied to
    the two intensity channels.

There is no learned model, so V61 produces a GNN-experiment dir whose
"forecaster" is the saved HistoricalAverageForecaster lookup. The arch-aware
`load_pretrained_gnn_model` + `generate_gnn_predictions_on_training_data`
detect `architecture == 'historical_average'` and fill the *_gnn forecast
columns by calendar join — so the frozen Phase-3 cascade and the rolling
simulation consume HA's forecasts with no further changes.

Distinguishes genuine learned dynamics from calendar periodicity: any
forecaster that does not clearly beat HA provides no evidence of learning.

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

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime

import joblib
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler

from src.training import ablation_study_v5_gnn_only as v5
from src.training import ablation_study_v17_gnn_only as v17
from src.models.benchmark_forecasters import (
    HistoricalAverageForecaster,
    HISTORICAL_AVERAGE_ARCH,
)

PIPELINE_VERSION = "v61"
ARCHITECTURE = HISTORICAL_AVERAGE_ARCH
TITLE = "BL-F0 Historical Average (prior-year weekday profile, volume-corrected)"

# V17's two disjoint windows: [0] prior-year profile, [1] recent volume.
PROFILE_WINDOW = v17.TRAIN_WINDOWS[0]
VOLUME_WINDOW = v17.TRAIN_WINDOWS[1]

_STATIC_FEATURES = ["car", "segment", "ang_curv", "ang_pend_pos",
                    "ang_pend_neg", "via", "sen", "pk"]
_TARGETS = ["mean_speed", "intTot", "intP"]


def build_experiment_name() -> str:
    return (f"{PIPELINE_VERSION}_gnn_{v5.get_experiment_prefix()}_"
            f"{v5.get_time_resolution_token()}_{v5.get_job_id()}")


def _filter_window(df: pd.DataFrame, window) -> pd.DataFrame:
    start, end = window
    mask = (df["dat"] >= pd.Timestamp(start)) & (df["dat"] < pd.Timestamp(end))
    return df[mask].copy()


def _fit_scaler(df: pd.DataFrame, cols):
    cols = [c for c in cols if c in df.columns]
    sc = MinMaxScaler()
    sc.fit(df[cols].fillna(0))
    return sc


def main() -> bool:
    model_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = build_experiment_name()
    output_dir = os.path.join(v5.OUTPUT_BASE_DIR, experiment_name)
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 80)
    print(f"ABLATION STUDY {PIPELINE_VERSION.upper()} — {TITLE}")
    print("=" * 80)
    print(f"Experiment:    {experiment_name}")
    print(f"Output:        {output_dir}")
    print(f"Profile window (prior year): {PROFILE_WINDOW[0]} -> {PROFILE_WINDOW[1]}")
    print(f"Volume window  (recent):     {VOLUME_WINDOW[0]} -> {VOLUME_WINDOW[1]}")
    print(f"Test Mode:     {v5.TEST_MODE}")
    print("=" * 80)
    sys.stdout.flush()

    t0 = time.time()
    print("\n[STAGE 0] Loading data...")
    df_full, _selected_features = v5.load_data()
    df_full["dat"] = pd.to_datetime(df_full["dat"])

    df_profile = _filter_window(df_full, PROFILE_WINDOW)
    df_recent = _filter_window(df_full, VOLUME_WINDOW)
    print(f"[INFO] Profile window rows: {len(df_profile):,}  | "
          f"Recent window rows: {len(df_recent):,}")
    if len(df_profile) == 0 or len(df_recent) == 0:
        print("[ERROR] One of the HA windows is empty — check date coverage.")
        return False

    print("\n[STAGE 1] Building Historical-Average profile + volume correction...")
    ha = HistoricalAverageForecaster().fit(df_profile, df_recent)
    print(f"[HA] {ha.meta}")

    # ── Persist artefacts in the GNN-experiment-dir layout ────────────────────
    ha_path = os.path.join(
        output_dir, f"historical-average_model=GNN-version={model_time}.pkl")
    ha.save(ha_path)
    print(f"[SAVE] HA artifact: {ha_path}")

    # Dummy checkpoint so the bash validity check (best-model + metadata + 3
    # scalers) passes; the HA loader branch never reads it.
    torch.save({}, os.path.join(
        output_dir, f"best-model_model=GNN-version={model_time}.pt"))

    # Valid (if unused) scalers, fitted on the profile window.
    joblib.dump(_fit_scaler(df_profile, ["hor", "diaSem", "dia", "mes"]),
                os.path.join(output_dir, f"scaler-temporal_model=GNN-version={model_time}.pkl"))
    joblib.dump(_fit_scaler(df_profile, _STATIC_FEATURES),
                os.path.join(output_dir, f"scaler-static_model=GNN-version={model_time}.pkl"))
    joblib.dump(_fit_scaler(df_profile, _TARGETS),
                os.path.join(output_dir, f"scaler-targets_model=GNN-version={model_time}.pkl"))

    num_pks = int(df_full["pk"].nunique())
    static_size = len([c for c in _STATIC_FEATURES if c in df_full.columns])
    metadata = {
        "model_name": "HistoricalAverageForecaster",
        "architecture": ARCHITECTURE,
        "benchmark_version": PIPELINE_VERSION,
        "benchmark_paper": "2nd article (Layer-1 forecasting)",
        "model_time": model_time,
        "input_size": None,
        "static_size": static_size,
        "output_size": len(_TARGETS),
        "num_pks": num_pks,
        "best_val_loss": "N/A",
        "hyperparameters": {},
        "ha_meta": ha.meta,
        "profile_window": {"start": PROFILE_WINDOW[0], "end": PROFILE_WINDOW[1]},
        "volume_window": {"start": VOLUME_WINDOW[0], "end": VOLUME_WINDOW[1]},
        "ablation_settings": v5.ABLATION_SETTINGS,
        "test_mode": v5.TEST_MODE,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(
            output_dir, f"best-model-metadata_model=GNN-version={model_time}.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    config = {
        "experiment_name": experiment_name,
        "model_time": model_time,
        "architecture": ARCHITECTURE,
        "benchmark_version": PIPELINE_VERSION,
        "pipeline_version": f"{PIPELINE_VERSION}_gnn_benchmark_historical_average",
        "profile_window": {"start": PROFILE_WINDOW[0], "end": PROFILE_WINDOW[1]},
        "volume_window": {"start": VOLUME_WINDOW[0], "end": VOLUME_WINDOW[1]},
        "test_mode": v5.TEST_MODE,
        "ablation_settings": v5.ABLATION_SETTINGS,
        "next_step": "Run an xgboost-only version with "
                     "PREVIOUS_EXPERIMENT_DIR pointing to this output",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    for name in ("experiment_config.json", f"experiment_config_{PIPELINE_VERSION}.json"):
        with open(os.path.join(output_dir, name), "w") as f:
            json.dump(config, f, indent=2)

    summary = {
        "experiment_name": experiment_name,
        "architecture": ARCHITECTURE,
        "benchmark_version": PIPELINE_VERSION,
        "total_duration_minutes": (time.time() - t0) / 60,
        "ha_meta": ha.meta,
        "completed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "next_step": f"Run xgboost-only with PREVIOUS_EXPERIMENT_DIR={output_dir}",
    }
    with open(os.path.join(output_dir, "experiment_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print(f"V61 (Historical Average) BUILD COMPLETE in "
          f"{v5.format_duration(time.time() - t0)}")
    print(f"Results saved to: {output_dir}")
    print("=" * 80)
    sys.stdout.flush()
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=TITLE)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--time-resolution", choices=v5.TIME_RESOLUTION_CHOICES,
                       help="Time resolution token to use")
    for token in v5.TIME_RESOLUTION_CHOICES:
        minutes = token.replace("min", "")
        group.add_argument(f"--{minutes}min", dest="time_resolution",
                           action="store_const", const=token,
                           help=argparse.SUPPRESS)
    args = parser.parse_args()
    if getattr(args, "time_resolution", None):
        chosen = args.time_resolution
        v5.DATA_FILE = re.sub(r"\d+min", chosen, v5.DATA_FILE, count=1)
        v5.DATA_PATH = os.path.join(_AP7_DATA, v5.DATA_FILE)
        print(f"[CONFIG] Using DATA_FILE={v5.DATA_FILE} for resolution {chosen}")
    sys.exit(0 if main() else 1)
