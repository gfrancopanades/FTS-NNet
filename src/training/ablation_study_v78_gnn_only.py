#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Ablation Study V78 — GeoLSTM trained on a FULL YEAR (data-quantity ablation)
============================================================================

Identical to V62 (BL-F1 GeoLSTM: vanilla stacked LSTM + static geometry, no
graph) in architecture, Optuna search, objective and evaluation — the ONLY
change is the training window: one contiguous year 2024-06-01 -> 2025-06-01
(June 2024 through May 2025, both included) instead of V17's two disjoint
months (June 2024 + May 2025).

Purpose (article): show that the tight two-month protocol reaches results
equivalent to training on the whole year, justifying the reduced windows.
Note the chronological 80/20 split then places validation on roughly the last
~2.4 months of the year (mid-March -> May 2025) instead of late May only.

Author: Gerard Franco | Date: July 2026
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

PIPELINE_VERSION = "v78"
ARCHITECTURE = "geolstm"
TITLE = ("GeoLSTM full-year ablation (train 2024-06-01 -> 2025-06-01, "
         "otherwise identical to V62/BL-F1)")

# ---------------------------------------------------------------------------
# The one deviation from V62: a single contiguous 12-month training window.
# Patch BOTH the benchmark common's copy and v17's original, since v17's
# window-aware patches to v5 (prepare_gnn_data / sequence masking) read
# v17.TRAIN_WINDOWS. End date is exclusive: May 2025 is fully included.
# ---------------------------------------------------------------------------
FULL_YEAR_WINDOWS = [("2024-06-01", "2025-06-01")]
common.TRAIN_WINDOWS = list(FULL_YEAR_WINDOWS)
v17.TRAIN_WINDOWS = list(FULL_YEAR_WINDOWS)


def main():
    return run_benchmark_gnn(PIPELINE_VERSION, ARCHITECTURE, TITLE)


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
