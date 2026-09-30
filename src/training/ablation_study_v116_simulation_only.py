#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""V116 simulation — alias to the version-agnostic Layer-2 benchmark simulation.

Without this file the runner silently falls back to `v6_simulation_only`, which
looks for XGBoost metadata a benchmark classifier never writes.
"""
from __future__ import annotations

import sys

import src.training.benchmark_classifier_sim as _bench_sim

sys.modules[__name__] = _bench_sim
