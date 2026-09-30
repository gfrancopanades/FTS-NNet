#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Ablation Study v110 — BL-C13 MLP + focal, spatial-neighbour context (Layer-2 crash-classifier benchmark)

Crash risk is a field over the corridor, not a set of independent cells: a\nqueue reaching kilometre post p was at p-1 five minutes earlier. The\nengineered features carry each location's own history but nothing about the\nroad immediately upstream, so a plain MLP cannot see a shockwave arriving.\nThis appends the neighbours' traffic channels at the same instant and the\nspatial gradient between them, which is what a shockwave physically is.

Receives the identical u_{i,t} features as every other Layer-2 arm (same v51
stage0 + stage1 feature prep + V52 FE replay) and the same operating-point
contract, so any difference is attributable to the classifier alone. Trainer,
optimiser, early stopping and focal parameters are BL-C5's, unchanged.

Author: Gerard Franco | Date: August 2026
"""
from __future__ import annotations

import sys

from src.training.benchmark_classifier_common import run_xgboost_only_for

PIPELINE_VERSION = "v110"
CLASSIFIER_TOKEN = "mlp_ctx"


def run_xgboost_only():
    return run_xgboost_only_for(PIPELINE_VERSION, CLASSIFIER_TOKEN)


def main():
    return 0 if run_xgboost_only() else 1


if __name__ == "__main__":
    sys.exit(main())
