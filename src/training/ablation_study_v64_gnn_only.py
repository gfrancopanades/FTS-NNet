#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Ablation Study V64 — BL-F3 STGCN (Layer-1 forecasting benchmark)
================================================================

Spatio-Temporal Graph Convolutional Network (Yu et al., 2018,
https://doi.org/10.48550/arXiv.1709.04875). Replaces recurrence with stacked
ST-Conv blocks: gated temporal convolution (causal 1D conv + GLU) -> Chebyshev
spatial graph convolution -> gated temporal convolution. No hidden state is
propagated; predictions come from a single forward pass (single-step decoder).

Represents the pure-convolutional branch of the STGNN family; STGCN vs. DCRNN
(V63) disentangles whether the temporal inductive bias of recurrence is needed
for long-horizon motorway forecasting.

Trained with the identical V17 windows / V14 RMSE-objective Optuna search as
the main model. Feeds the frozen Phase-3 cascade via the arch-aware GNN loader.

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
import os
import re
import sys

from src.training import ablation_study_v5_gnn_only as v5
from src.training.benchmark_forecaster_gnn_common import run_benchmark_gnn

PIPELINE_VERSION = "v64"
ARCHITECTURE = "stgcn"
TITLE = "BL-F3 STGCN (gated temporal conv + Chebyshev graph conv, no recurrence)"


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
