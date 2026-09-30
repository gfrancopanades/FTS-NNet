#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Ablation Study V86 — GNN-LSTM (with message passing)
====================================================

Identical to the proposed V17 GNN-LSTM in architecture, Optuna search (same
trials/objective), windows and evaluation — with ONE change: sequences are
reordered TIME-MAJOR before batching, so each (shuffle=False) batch contains
co-temporal samples across locations and `build_highway_graph` produces real
pk-neighbor edges. This activates the GCN message passing that the standard
location-major stream leaves at self-loops ~98-99.7% of the time (see
`src/evaluation/check_graph_edges.py`, job 2663343).

All changes are contained in this module via a monkeypatch of
`v5.create_gnn_sequences` — no shared code is modified; every existing
pipeline remains byte-identical.

NOTE: the downstream cascade for this model needs the same time-major
treatment in the inference path (`generate_gnn_predictions`) — see the v86
simulation wrapper before running xgb/sim stages.

Author: Gerard Franco | Date: July 2026
"""

from __future__ import annotations
from src.paths import (  # portable paths -- see src/paths.py
    PROJECT_ROOT_STR as _AP7_ROOT,
    EXPERIMENTS_ROOT_STR as _AP7_EXPERIMENTS,
    DATA_DIR_STR as _AP7_DATA,
    TABLES_DIR as _AP7_TABLES,
    FIGURES_DIR as _AP7_FIGS,
)

import argparse
import os
import re
import sys

import numpy as np

from src.training import ablation_study_v5_gnn_only as v5
from src.training import ablation_study_v17_gnn_only as v17

PIPELINE_VERSION = "v86"
TITLE = "V86 — GNN-LSTM (with message passing): time-major batches activate GCN edges"


def _time_major_order(pk_ids: np.ndarray, sen: np.ndarray) -> np.ndarray:
    """Order sequences so consecutive samples are co-temporal across locations.

    The incoming stream is per-(pk,sen) chronological runs concatenated. Key =
    position-within-run; sorting primarily by that position interleaves all
    locations at (approximately) the same timestamp into adjacent samples.
    """
    n = len(pk_ids)
    key = pk_ids.astype(np.int64) * 10 + sen.astype(np.int64)
    change = np.r_[True, key[1:] != key[:-1]]
    run_id = np.cumsum(change) - 1
    run_start = np.maximum.accumulate(np.where(change, np.arange(n), 0))
    pos = np.arange(n) - run_start
    return np.lexsort((run_id, pos))  # primary: pos, secondary: run


_orig_create = v5.create_gnn_sequences


def _create_gnn_sequences_time_major(*args, **kwargs):
    seq, static, tgt, pk, sen, num_pks = _orig_create(*args, **kwargs)
    order = _time_major_order(np.asarray(pk), np.asarray(sen))
    n = len(order)
    # quick edge sanity on the first batch-sized chunk
    chunk = np.asarray(pk)[order[:512]]
    vals, cnts = np.unique(chunk, return_counts=True)
    c = dict(zip(vals.tolist(), cnts.tolist()))
    edges = 2 * sum(cn * c.get(v + 1, 0) for v, cn in c.items())
    print(f"[V86-TIME-MAJOR] reordered {n:,} sequences; first-512 chunk: "
          f"{len(vals)} distinct pks, ~{edges} graph edges (was ~0 location-major)")
    return (seq[order], static[order], tgt[order], pk[order], sen[order], num_pks)


# Contained monkeypatches: sequence order + experiment naming/version.
v5.create_gnn_sequences = _create_gnn_sequences_time_major
v17.PIPELINE_VERSION = PIPELINE_VERSION


def _build_experiment_name() -> str:
    prefix = v5.get_experiment_prefix()
    resolution = v5.get_time_resolution_token()
    job_id = v5.get_job_id()
    return f"v86_gnn_{prefix}_{resolution}_{job_id}"


v17.build_experiment_name = _build_experiment_name


def main():
    print("=" * 70)
    print(TITLE)
    print("=" * 70)
    return v17.main()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=TITLE)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--time-resolution", choices=v5.TIME_RESOLUTION_CHOICES,
                       help="Time resolution token to use")
    for token in v5.TIME_RESOLUTION_CHOICES:
        minutes = token.replace("min", "")
        group.add_argument(f"--{minutes}min", dest="time_resolution",
                           action="store_const", const=token,
                           help=argparse.SUPPRESS)
    args = parser.parse_args()
    if getattr(args, "time_resolution", None):
        chosen = args.time_resolution
        v5.DATA_FILE = re.sub(r"\d+min", chosen, v5.DATA_FILE, count=1)
        v5.DATA_PATH = os.path.join(_AP7_DATA, v5.DATA_FILE)
        print(f"[CONFIG] Using DATA_FILE={v5.DATA_FILE} for resolution {chosen}")
    sys.exit(0 if main() else 1)
