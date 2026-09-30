#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Ablation Study V83 — BL-E0 Transformer end-to-end crash predictor (Layer 3)
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

import sys

from src.training.benchmark_end2end_common import run_end2end

PIPELINE_VERSION = "v83"
ARCHITECTURE = "e2e_lstm"
TITLE = "BL-E3 LSTM end-to-end crash predictor (literature-standard RTCP baseline)"


def main():
    return 0 if run_end2end(PIPELINE_VERSION, ARCHITECTURE, TITLE) else 1


if __name__ == "__main__":
    sys.exit(main())
