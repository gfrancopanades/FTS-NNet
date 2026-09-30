#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Ablation Study v101 — rolling-origin fold, test month Jul-2025
==========================================================

Stage-1 GeoLSTM for the rolling-origin validation of H3. Identical architecture
and objective to the reference two-month model; the only change is the training
window, which is rolled so that it lies strictly BEFORE the fold's test month:

    train [('2024-07-01', '2024-08-01'), ('2025-06-01', '2025-07-01')]   ->   test Jul-2025

The Optuna budget is reduced to 12 trials (against 30 for the reference
Jun-2025 fold) to keep five folds affordable; this is reported in the paper.

Author: Gerard Franco | Date: August 2026
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
import os
import re
import sys

from src.training import ablation_study_v5_gnn_only as v5
from src.training import ablation_study_v17_gnn_only as v17
from src.training import benchmark_forecaster_gnn_common as common
from src.training.benchmark_forecaster_gnn_common import run_benchmark_gnn

PIPELINE_VERSION = "v101"
ARCHITECTURE = "geolstm"
TITLE = "v101 GeoLSTM — rolling-origin fold, test month Jul-2025"

FOLD_WINDOWS = [('2024-07-01', '2024-08-01'), ('2025-06-01', '2025-07-01')]
common.TRAIN_WINDOWS = list(FOLD_WINDOWS)
v17.TRAIN_WINDOWS = list(FOLD_WINDOWS)
if hasattr(v5, "GNN_CONFIG"):
    v5.GNN_CONFIG["trials"] = 12


def main():
    print("=" * 70)
    print(TITLE)
    print(f"  training windows: {FOLD_WINDOWS}")
    print(f"  Optuna trials   : {v5.GNN_CONFIG.get('trials')}")
    print("=" * 70)
    return run_benchmark_gnn(PIPELINE_VERSION, ARCHITECTURE, TITLE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=TITLE)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--time-resolution", choices=v5.TIME_RESOLUTION_CHOICES)
    for token in v5.TIME_RESOLUTION_CHOICES:
        group.add_argument(f"--{token.replace('min','')}min", dest="time_resolution",
                           action="store_const", const=token, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if getattr(args, "time_resolution", None):
        chosen = args.time_resolution
        v5.DATA_FILE = re.sub(r"\d+min", chosen, v5.DATA_FILE, count=1)
        v5.DATA_PATH = os.path.join(_AP7_DATA, v5.DATA_FILE)
    sys.exit(0 if main() else 1)
