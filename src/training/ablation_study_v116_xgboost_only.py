#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Ablation Study v116 — BL-T3 GBDT, monotone-constrained (Layer-2 crash-classifier benchmark)

With 6,092 positives, asserting known physics beats spending capacity\nrediscovering it. Every leading feature in Table 7 carries a positive signed\nSHAP value, and each direction is an established freeway-safety finding; those\ndirections become monotone constraints. A constrained model also cannot output\n'more speed variance lowered the risk', which is what an authority needs it\nnever to say.

Receives the identical u_{i,t} features as every other Layer-2 arm and the
same operating-point contract, and inherits the searched XGBoost
hyperparameters, so a difference is attributable to the stated change.

Author: Gerard Franco | Date: August 2026
"""
from __future__ import annotations

import sys

from src.training.benchmark_classifier_common import run_xgboost_only_for

PIPELINE_VERSION = "v116"
CLASSIFIER_TOKEN = "gbdt_mono"


def run_xgboost_only():
    return run_xgboost_only_for(PIPELINE_VERSION, CLASSIFIER_TOKEN)


def main():
    return 0 if run_xgboost_only() else 1


if __name__ == "__main__":
    sys.exit(main())
