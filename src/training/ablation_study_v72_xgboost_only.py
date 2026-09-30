#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ablation Study V72 — BL-C4 Two-stage DL cascade (Jin et al. 2023) (Layer-2 crash-classifier benchmark)
=====================================================================

Replaces the proposed Phase-3 cascade with: BL-C4 Two-stage DL cascade (Jin et al. 2023). Receives the identical
u_{i,t} features from the proposed GNN-LSTM (same v51 stage0 + stage1 feature
prep + V52 FE replay) and the same operating-point contract (maximise precision
s.t. recall >= 0.40). See benchmark_classifier_common for the shared pipeline.

Author: Gerard Franco | Date: June 2026
"""

from __future__ import annotations

import sys

from src.training.benchmark_classifier_common import run_xgboost_only_for

PIPELINE_VERSION = "v72"
CLASSIFIER_TOKEN = "two_stage_mlp"


def run_xgboost_only():
    return run_xgboost_only_for(PIPELINE_VERSION, CLASSIFIER_TOKEN)


def main():
    return 0 if run_xgboost_only() else 1


if __name__ == "__main__":
    sys.exit(main())
