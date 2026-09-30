#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V70 simulation — alias to the version-agnostic Layer-2 benchmark simulation.

The simulation auto-detects the classifier (xgb_smote) + features + threshold from
the `benchmark_classifier_manifest_*.json` in the XGBoost experiment dir, so
every BL-C* sim shares one implementation. We alias via sys.modules (same trick
as v58-sim -> v57-sim) so the generic runner's attribute pokes and the final
`run_simulation_only()` call land on the real module.
"""

from __future__ import annotations

import sys

import src.training.benchmark_classifier_sim as _bench_sim

sys.modules[__name__] = _bench_sim
