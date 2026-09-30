#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Ablation Study v115 — BL-T2 GBDT, local-residual features (Layer-2 crash-classifier benchmark)

A split on `mean_speed < 85` conflates a congested rush hour at one post with\nfree flow at another. The leading traffic channels are replaced by residuals\nagainst per-(pk, hour-of-week) baselines fitted strictly before the train/test\ncutoff, so every split is already a statement about departure from local\nnormality.

Receives the identical u_{i,t} features as every other Layer-2 arm and the
same operating-point contract, and inherits the searched XGBoost
hyperparameters, so a difference is attributable to the stated change.

Author: Gerard Franco | Date: August 2026
"""
from __future__ import annotations

import sys

from src.training.benchmark_classifier_common import run_xgboost_only_for

PIPELINE_VERSION = "v115"
CLASSIFIER_TOKEN = "gbdt_resid"


def run_xgboost_only():
    return run_xgboost_only_for(PIPELINE_VERSION, CLASSIFIER_TOKEN)


def main():
    return 0 if run_xgboost_only() else 1


if __name__ == "__main__":
    sys.exit(main())
