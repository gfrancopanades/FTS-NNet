#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Ablation Study V75 — BL-E1 MSGNN end-to-end crash predictor (Layer 3)
=====================================================================

Tran et al. (2023, https://doi.org/10.1016/j.trc.2023.104354). A multi-
structured GNN fusing two relational structures over the corridor — (1) spatial
adjacency along each carriageway and (2) a data-driven similarity graph (cosine
similarity of encoded state windows) — into a single crash-incident predictor.
Trained end-to-end on crash labels (no forecasting stage), evaluated under the
frozen rolling protocol. Exposes the value of the proposed explicit 72-hour rollout
vs. a GNN that bypasses the forecast->classify decomposition.

Self-contained (train + frozen simulation in one job); see
`benchmark_end2end_common.run_end2end`.

Author: Gerard Franco | Date: June 2026
"""

from __future__ import annotations

import sys

from src.training.benchmark_end2end_common import run_end2end

PIPELINE_VERSION = "v75"
ARCHITECTURE = "e2e_msgnn"
TITLE = "BL-E1 MSGNN end-to-end crash predictor (Tran et al., 2023)"


def main():
    return 0 if run_end2end(PIPELINE_VERSION, ARCHITECTURE, TITLE) else 1


if __name__ == "__main__":
    sys.exit(main())
