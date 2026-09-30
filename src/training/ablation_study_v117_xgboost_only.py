#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Ablation Study v117 — BL-T4 GBDT, day-block bootstrap ensemble (Layer-2 crash-classifier benchmark)

`subsample` draws rows independently, splitting crash runs across the in-bag\nboundary and letting a tree fit one half and be scored on the other. Sampling\nwhole (pk, day) blocks is the unit of independence the day-block bootstrap of\nSection 3.5.4 already assumes, so training and inference stop contradicting\neach other.

Receives the identical u_{i,t} features as every other Layer-2 arm and the
same operating-point contract, and inherits the searched XGBoost
hyperparameters, so a difference is attributable to the stated change.

Author: Gerard Franco | Date: August 2026
"""
from __future__ import annotations

import sys

from src.training.benchmark_classifier_common import run_xgboost_only_for

PIPELINE_VERSION = "v117"
CLASSIFIER_TOKEN = "gbdt_block"


def run_xgboost_only():
    return run_xgboost_only_for(PIPELINE_VERSION, CLASSIFIER_TOKEN)


def main():
    return 0 if run_xgboost_only() else 1


if __name__ == "__main__":
    sys.exit(main())
