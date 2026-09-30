#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Ablation Study v109 — BL-C12 MLP + focal, context-gated feature blocks (Layer-2 crash-classifier benchmark)

The 52 features are not one homogeneous vector: they form four semantic\nblocks whose usefulness is context-dependent. Section 6 shows geometry\ncontributes only through its interaction with deceleration, never alone. A\ncontext network emits one gate per block so the model can weight traffic\ndynamics differently in a congested rush hour than at 3 a.m., instead of\nleaving the first linear layer to discover that interaction unaided.

Receives the identical u_{i,t} features as every other Layer-2 arm (same v51
stage0 + stage1 feature prep + V52 FE replay) and the same operating-point
contract, so any difference is attributable to the classifier alone. Trainer,
optimiser, early stopping and focal parameters are BL-C5's, unchanged.

Author: Gerard Franco | Date: August 2026
"""
from __future__ import annotations

import sys

from src.training.benchmark_classifier_common import run_xgboost_only_for

PIPELINE_VERSION = "v109"
CLASSIFIER_TOKEN = "mlp_gated"


def run_xgboost_only():
    return run_xgboost_only_for(PIPELINE_VERSION, CLASSIFIER_TOKEN)


def main():
    return 0 if run_xgboost_only() else 1


if __name__ == "__main__":
    sys.exit(main())
