#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ablation Study V120 -- BL-T5 Focal-objective GBDT (Layer-2 benchmark)
=====================================================================

Section 5.3(e) attributes the neural/tree separation to the training objective
rather than to the model family: the neural arms reshape the loss for a 0.19 %
positive rate, while the tree arms could only be told about imbalance through
row weights. That attribution is a conjecture in the manuscript. This arm tests
it directly by giving a gradient-boosted tree the same focal objective, with
`scale_pos_weight` removed so the objective is the only imbalance channel, and
the same 25-trial budget as every other Layer-2 arm.

Author: Gerard Franco | Date: August 2026
"""

from __future__ import annotations

import sys

from src.training.benchmark_classifier_common import run_xgboost_only_for

PIPELINE_VERSION = "v120"
CLASSIFIER_TOKEN = "gbdt_focal"


def run_xgboost_only():
    return run_xgboost_only_for(PIPELINE_VERSION, CLASSIFIER_TOKEN)


def main():
    return 0 if run_xgboost_only() else 1


if __name__ == "__main__":
    sys.exit(main())
