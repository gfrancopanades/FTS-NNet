#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ablation Study V84 — H2 covariates-only ablation (classifier: mlp_focal)
=====================================================================

Trains the 'mlp_focal' benchmark classifier WITHOUT any forecast-derived
features (the 4 substituted traffic channels + the V21 FE columns engineered
from them; covariate-based pk-hour interactions are kept). Everything else —
data, windows, threshold contract, simulation — is identical to the standard
benchmark run. The gap to the with-forecast twin measures the information the
traffic-forecast stage contributes beyond its own covariate inputs (H2).

Author: Gerard Franco | Date: July 2026
"""

from __future__ import annotations

import os
import sys

os.environ["BENCH_COVARIATES_ONLY"] = "1"

from src.training.benchmark_classifier_common import run_xgboost_only_for

PIPELINE_VERSION = "v84"
CLASSIFIER_TOKEN = "mlp_focal"


def run_xgboost_only():
    return run_xgboost_only_for(PIPELINE_VERSION, CLASSIFIER_TOKEN)


def main():
    return 0 if run_xgboost_only() else 1


if __name__ == "__main__":
    sys.exit(main())
