#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Ablation Study v111 — BL-C14 MLP + soft-target focal, proximity-weighted labels (Layer-2 crash-classifier benchmark)

The +/-15 min labelling turns each crash into a RUN of positive intervals and\nasserts that the interval three steps before the crash is exactly as\ncrash-like as the one containing it. The window exists because the operator's\ntimestamps carry uncertainty, and that uncertainty is greatest at the run's\nedges. Targets therefore decay from each run's centre and the loss\ninterpolates rather than thresholding at 0.5. The EVALUATION label is\nuntouched: this changes what the model fits, never what it is scored on.

Receives the identical u_{i,t} features as every other Layer-2 arm (same v51
stage0 + stage1 feature prep + V52 FE replay) and the same operating-point
contract, so any difference is attributable to the classifier alone. Trainer,
optimiser, early stopping and focal parameters are BL-C5's, unchanged.

Author: Gerard Franco | Date: August 2026
"""
from __future__ import annotations

import sys

from src.training.benchmark_classifier_common import run_xgboost_only_for

PIPELINE_VERSION = "v111"
CLASSIFIER_TOKEN = "mlp_prox"


def run_xgboost_only():
    return run_xgboost_only_for(PIPELINE_VERSION, CLASSIFIER_TOKEN)


def main():
    return 0 if run_xgboost_only() else 1


if __name__ == "__main__":
    sys.exit(main())
