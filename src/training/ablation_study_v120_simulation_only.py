#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V120 simulation -- alias to the version-agnostic Layer-2 benchmark sim."""

from __future__ import annotations

import sys

import src.training.benchmark_classifier_sim as _bench_sim

sys.modules[__name__] = _bench_sim
