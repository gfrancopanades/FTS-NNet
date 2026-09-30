#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Canonical naming for simulation output folders.

A simulation is identified by three things, and all three must appear in the
folder name:

  * ``model_time``  - which trained model produced it;
  * the **training** window - two models trained on different windows are
    different models even at the same ``model_time`` prefix;
  * the **simulation** window - the engineered-feature baselines are estimated
    from the assembled frame, so the *same* model scored over a different
    window yields different predictions for the very same rows.

Historically folders were keyed on ``model_time`` alone. Two consequences bit
us hard enough to justify this module:

  1. Re-simulating a model over a new window silently **overwrote** the previous
     run. A 13-month rerun destroyed the June-only simulations of six models,
     including the proposed configuration.
  2. Because the overwrite was invisible, a later harvest clipped a 13-month
     frame down to June and compared it against baselines scored in a one-month
     frame - understating the proposed model's headline AUPRC by 0.028.

The canonical form is::

    <prefix>_<model_time>_tr<YYYYMMDD>-<YYYYMMDD>_sim<YYYYMMDD>-<YYYYMMDD>

Both ranges are start-inclusive / end-exclusive, matching the simulation code.
``parse_sim_dirname`` returns None for legacy folders so callers can fall back.
"""

from __future__ import annotations

import re

__all__ = ["sim_dir_name", "parse_sim_dirname", "SIM_DIR_RE"]

SIM_DIR_RE = re.compile(
    r"_tr(?P<tr_start>\d{8})-(?P<tr_end>\d{8})"
    r"_sim(?P<sim_start>\d{8})-(?P<sim_end>\d{8})$"
)

# Tolerated intermediate form: only the simulation window was tagged. Two
# spellings were emitted before this module existed - ``_win<start><end>`` and
# ``_win<start>_<end>`` - so the separator is optional.
_LEGACY_WIN_RE = re.compile(r"_win(?P<sim_start>\d{8})_?(?P<sim_end>\d{8})$")


def _compact(d) -> str:
    """'2025-06-01' | datetime | None -> '20250601' ('00000000' when unknown)."""
    if d is None:
        return "00000000"
    s = str(d)[:10].replace("-", "").replace("/", "")
    return s if len(s) == 8 and s.isdigit() else "00000000"


def sim_dir_name(prefix: str, model_time: str,
                 train_start, train_end, sim_start, sim_end) -> str:
    """Build the canonical simulation folder name.

    Unknown dates degrade to ``00000000`` rather than raising, so a simulation
    never fails to write its results just because provenance is incomplete.
    """
    return (f"{prefix}_{model_time}"
            f"_tr{_compact(train_start)}-{_compact(train_end)}"
            f"_sim{_compact(sim_start)}-{_compact(sim_end)}")


def parse_sim_dirname(name: str):
    """Return dict(tr_start, tr_end, sim_start, sim_end) or None if untagged.

    Legacy ``_win<start><end>`` folders resolve with the training window left
    as None, which is enough for window-aware discovery.
    """
    m = SIM_DIR_RE.search(name)
    if m:
        return {k: v for k, v in m.groupdict().items()}
    m = _LEGACY_WIN_RE.search(name)
    if m:
        return dict(tr_start=None, tr_end=None,
                    sim_start=m.group("sim_start"), sim_end=m.group("sim_end"))
    return None
