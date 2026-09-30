#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ablation Study V73 — BL-C5 MLP + focal loss (Layer-2 crash-classifier benchmark)
=====================================================================

Replaces the proposed Phase-3 cascade with: BL-C5 MLP + focal loss. Receives the identical
u_{i,t} features from the proposed GNN-LSTM (same v51 stage0 + stage1 feature
prep + V52 FE replay) and the same operating-point contract (maximise precision
s.t. recall >= 0.40). See benchmark_classifier_common for the shared pipeline.

Author: Gerard Franco | Date: June 2026
"""

from __future__ import annotations

import sys

from src.training.benchmark_classifier_common import run_xgboost_only_for

PIPELINE_VERSION = "v73"
CLASSIFIER_TOKEN = "mlp_focal"


def run_xgboost_only():
    return run_xgboost_only_for(PIPELINE_VERSION, CLASSIFIER_TOKEN)


def main():
    return 0 if run_xgboost_only() else 1


if __name__ == "__main__":
    sys.exit(main())
