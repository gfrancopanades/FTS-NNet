#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""V112 simulation — alias to the version-agnostic Layer-2 benchmark simulation.

The simulation auto-detects the classifier token, its feature list and its
frozen threshold from `benchmark_classifier_manifest_*.json` in the training
output dir, so every BL-C* arm shares one implementation. Aliasing via
`sys.modules` (the same trick v73 uses) makes the generic runner's attribute
pokes and its final `run_simulation_only()` call land on the real module.

Without this file the runner silently falls back to `v6_simulation_only`, which
looks for XGBoost metadata that a neural benchmark classifier never writes.
"""
from __future__ import annotations

import sys

import src.training.benchmark_classifier_sim as _bench_sim

sys.modules[__name__] = _bench_sim
