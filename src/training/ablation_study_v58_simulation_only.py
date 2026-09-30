#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V58 Simulation-only — thin alias to the V57 simulation pipeline
===============================================================

V58 changed ONLY the Stage-2 transformer *training* recipe
(`ablation_study_v58_xgboost_only.py`). The artefacts it writes are
byte-compatible with the V57 simulation loader:

  * the transformer checkpoint keeps the identical self-describing format
    (`arch_config` + `feature_names` + `feat_mean`/`feat_scale`), and
  * `ablation_study_v57_simulation_only._detect_xgb_model_time_v57` already
    lists "v58" first in its manifest-schema preference, so it picks up
    `v58_xgboost_manifest_*.json` and `experiment_config_xgb_v58.json`.

So there is no V58-specific simulation logic to write. This module exists
purely so the generic runner
(`bash_files/run_ablation_study_simulation_only.sh`), which dispatches via
`importlib.import_module(f"...ablation_study_{VERSION}_simulation_only")`,
resolves `--v58` to the real V57 implementation instead of silently falling
back to `v6_simulation_only` (which has no transformer Stage 2).

We alias through `sys.modules` rather than re-exporting names: the runner
*reassigns* scalar module attributes (`ablation.TEST_MODE = ...`,
`ablation.DATA_FILE = ...`, ...) and then calls `ablation.run_simulation_only()`.
A `from ... import *` would let those scalar reassignments land on this shim
instead of the module whose globals `run_simulation_only` actually reads.
Making this module *be* the V57 module guarantees every poke and the final
call hit the real implementation — i.e. `--v58` behaves exactly like `--v57`,
only the manifest/config it loads differ (which is what we want).
"""

from __future__ import annotations

import sys

import src.training.ablation_study_v57_simulation_only as _v57_sim

# Replace this module object with the V57 simulation module so that
# `importlib.import_module("...ablation_study_v58_simulation_only")` returns
# the real V57 module, with all its functions and mutable/scalar globals.
sys.modules[__name__] = _v57_sim
