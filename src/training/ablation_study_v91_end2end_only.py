#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Ablation Study V91 — BL-E0 Transformer end-to-end crash predictor (Layer 3)
===========================================================================

Abdel-Aty et al. (2024, https://doi.org/10.1038/s41598-024-75350-z). A
transformer encoder ingests a window of reconstructed corridor states and
directly outputs crash probability — no forecasting/classification
decomposition. Trained end-to-end on crash labels, evaluated under the frozen
rolling protocol. Demonstrates the cost of lacking the proposed multi-phase
decomposition under long-horizon frozen conditions.

Self-contained (train + frozen simulation in one job); see
`benchmark_end2end_common.run_end2end`.

Author: Gerard Franco | Date: June 2026
"""

from __future__ import annotations

import os
import sys

# This arm's input mode (traffic lagged by 864 five-minute bins (72 h)). The reported run was launched with
# E2E_LAG_TRAFFIC=864 in its environment; default it here so the module cannot
# silently run as the observed-traffic arm when the variable is omitted.
os.environ.setdefault("E2E_LAG_TRAFFIC", "864")

from src.training.benchmark_end2end_common import run_end2end

PIPELINE_VERSION = "v91"
ARCHITECTURE = "e2e_lstm"
TITLE = "BL-E6 LSTM e2e, LAGGED real traffic t-72h (72h-legal, no forecast)"


def main():
    return 0 if run_end2end(PIPELINE_VERSION, ARCHITECTURE, TITLE) else 1


if __name__ == "__main__":
    sys.exit(main())
