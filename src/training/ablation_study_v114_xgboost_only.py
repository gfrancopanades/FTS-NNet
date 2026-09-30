#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Ablation Study v114 — BL-T1 GBDT, proximity-weighted crash runs (Layer-2 crash-classifier benchmark)

`scale_pos_weight` compresses every positive into one number, but the\n+/-15 min labelling makes a crash a RUN whose centre is far more certainly a\ncrash than its edges. XGBoost accepts per-row `sample_weight`, so the insight\nthat won BL-C14 reaches the split criterion directly.

Receives the identical u_{i,t} features as every other Layer-2 arm and the
same operating-point contract, and inherits the searched XGBoost
hyperparameters, so a difference is attributable to the stated change.

Author: Gerard Franco | Date: August 2026
"""
from __future__ import annotations

import sys

from src.training.benchmark_classifier_common import run_xgboost_only_for

PIPELINE_VERSION = "v114"
CLASSIFIER_TOKEN = "gbdt_prox"


def run_xgboost_only():
    return run_xgboost_only_for(PIPELINE_VERSION, CLASSIFIER_TOKEN)


def main():
    return 0 if run_xgboost_only() else 1


if __name__ == "__main__":
    sys.exit(main())
