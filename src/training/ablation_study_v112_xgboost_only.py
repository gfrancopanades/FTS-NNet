#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Ablation Study v112 — BL-C15 MLP deviation-residual + stratum-anchored focal (Layer-2 crash-classifier benchmark)

Two branches whose sum is the logit: a learned local baseline b(place, time,\ngeometry, historical rate) and a departure d(traffic dynamics) that can only\npush risk away from it. The separation states architecturally what Section 6\nfinds empirically -- risk is carried by departure from local normality, not by\nwhere crashes are common -- and stops the network re-deriving a hotspot map\nthat Table 2 shows scores at chance on its own. Minibatches are drawn from a\nsingle (5 km band x 3 h band) stratum, so the gradient contrasts an interval\nagainst its own place and hour rather than against a quiet rural post at 3 a.m.

Receives the identical u_{i,t} features as every other Layer-2 arm (same v51
stage0 + stage1 feature prep + V52 FE replay) and the same operating-point
contract, so any difference is attributable to the classifier alone. Trainer,
optimiser, early stopping and focal parameters are BL-C5's, unchanged.

Author: Gerard Franco | Date: August 2026
"""
from __future__ import annotations

import sys

from src.training.benchmark_classifier_common import run_xgboost_only_for

PIPELINE_VERSION = "v112"
CLASSIFIER_TOKEN = "mlp_devres"


def run_xgboost_only():
    return run_xgboost_only_for(PIPELINE_VERSION, CLASSIFIER_TOKEN)


def main():
    return 0 if run_xgboost_only() else 1


if __name__ == "__main__":
    sys.exit(main())
