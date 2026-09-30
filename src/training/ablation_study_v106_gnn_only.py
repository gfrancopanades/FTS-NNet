#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Ablation Study v106 — Graph WaveNet WITH graph message passing.

Identical to v65 in architecture, Optuna objective, training windows and
evaluation. One change: sequences are reordered time-major before batching, so
each minibatch holds co-temporal samples across kilometre posts and
`build_highway_graph` returns real p<->p+-1 edges instead of self-loops.

v65 — like every other graph baseline in Table 2 panel (a) — was trained
under location-major batching, which left its spatial operator inert in
97-99.7 % of batches. Its published number therefore measures its temporal
core, not the architecture. This run measures the architecture.

Trial budget: this architecture's per-trial cost is an outlier -- the
location-major v65 run took 6 d 21 h for 30 trials on two months of data,
against 17 h - 1 d 11 h for every other arm. It is therefore run with
AP7_GNN_TRIALS=10 and footnoted, in the same way STGCN's truncated search is.

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
from src.training.benchmark_forecaster_gnn_common import run_benchmark_gnn
from src.training.time_major_patch import activate as _activate_time_major

PIPELINE_VERSION = "v106"
ARCHITECTURE = "graphwavenet"
TITLE = "BL-F4b Graph WaveNet + message passing (dilated conv + adaptive adj, time-major)"

_activate_time_major(v5, tag="v106-TIME-MAJOR")


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
