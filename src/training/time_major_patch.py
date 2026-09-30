#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Activate graph message passing by reordering sequences time-major.

`build_highway_graph` builds the corridor graph *per minibatch*: it links a
sample at kilometre post p to a sample at p+-1 only if both are in the same
batch. Under the default location-major stream a batch holds consecutive
timesteps of ONE post, so no neighbour is ever present and the graph collapses
to self-loops in 97-99.7 % of batches (`src/evaluation/check_graph_edges.py`).
Every graph forecaster then runs as its temporal core with location embeddings,
and its spatial operator is inert -- silently, with no error raised.

Reordering the sequences time-major puts co-temporal samples across locations
into the same batch, so real p<->p+-1 edges exist and message passing fires.
V86 established the effect on the GNN-LSTM: downstream crash AUPRC went from
0.150 to 0.403 on the identical architecture. This module factors that patch
out of V86 so every graph baseline can be retrained under the same condition.

Usage -- import and call before the trainer builds its sequences::

    from src.training.time_major_patch import activate
    activate(v5)
"""
from __future__ import annotations

import numpy as np


def _time_major_order(pk_ids: np.ndarray, sen: np.ndarray) -> np.ndarray:
    """Indices that order samples by timestep first, location second.

    `pk_ids`/`sen` arrive in location-major order, so position within a
    (pk, sen) run is a proxy for the timestep. Sorting on that position, with
    the run identity as the tiebreak, interleaves the locations.
    """
    key = pk_ids.astype(np.int64) * 4 + sen.astype(np.int64)
    order_by_run = np.lexsort((np.arange(len(key)), key))
    run_id = np.empty(len(key), dtype=np.int64)
    pos = np.empty(len(key), dtype=np.int64)
    start = 0
    rid = 0
    sorted_key = key[order_by_run]
    for i in range(1, len(sorted_key) + 1):
        if i == len(sorted_key) or sorted_key[i] != sorted_key[start]:
            idx = order_by_run[start:i]
            run_id[idx] = rid
            pos[idx] = np.arange(i - start)
            rid += 1
            start = i
    return np.lexsort((run_id, pos))          # primary: timestep, secondary: run


def activate(v5_module, tag: str = "TIME-MAJOR") -> None:
    """Monkeypatch `v5_module.create_gnn_sequences` to emit time-major order.

    Contained: no shared source is edited, so any run without this call keeps
    the original behaviour exactly.
    """
    original = v5_module.create_gnn_sequences

    def _time_major(*args, **kwargs):
        seq, static, tgt, pk, sen, num_pks = original(*args, **kwargs)
        order = _time_major_order(np.asarray(pk), np.asarray(sen))
        chunk = np.asarray(pk)[order[:512]]
        vals, cnts = np.unique(chunk, return_counts=True)
        c = dict(zip(vals.tolist(), cnts.tolist()))
        edges = 2 * sum(cn * c.get(v + 1, 0) for v, cn in c.items())
        print(f"[{tag}] reordered {len(order):,} sequences; first-512 chunk: "
              f"{len(vals)} distinct pks, ~{edges} graph edges "
              f"(location-major gives ~0)")
        return (seq[order], static[order], tgt[order],
                pk[order], sen[order], num_pks)

    v5_module.create_gnn_sequences = _time_major


# ---------------------------------------------------------------------------
# Inference-side companion to `activate()`.
#
# `activate()` reorders sequences during TRAINING only. Inference builds its
# sequences through a different function (`_create_gnn_sequences_for_inference`)
# and sorts location-major, so a model trained with real corridor edges was
# being evaluated with a self-loop graph. These helpers let the inference path
# ask whether a given run needs the same treatment.
#
# Membership is derived by scanning the training modules rather than from a
# hand-kept list, because a hand-kept list is exactly what drifts.
# ---------------------------------------------------------------------------

def time_major_versions():
    """Versions whose training module performs time-major reordering."""
    import glob as _g, os as _o, re as _r
    here = _o.path.dirname(_o.path.abspath(__file__))
    out = set()
    for f in _g.glob(_o.path.join(here, "ablation_study_v*_gnn_only.py")):
        m = _r.search(r"_v(\d+)_gnn_only", f)
        if not m:
            continue
        try:
            src = open(f, errors="ignore").read()
        except OSError:
            continue
        if _r.search(r"time[_-]major", src, _r.I):
            out.add("v" + m.group(1))
    return out


def is_time_major_run(experiment_dir):
    """True when the model in `experiment_dir` was trained time-major.

    A `TIME_MAJOR` marker file in the run directory wins, so a run can declare
    its own regime; otherwise the version prefix of the directory name is
    matched against the modules that reorder.
    """
    import os as _o
    if not experiment_dir:
        return False
    if _o.path.exists(_o.path.join(experiment_dir, "TIME_MAJOR")):
        return True
    base = _o.path.basename(_o.path.normpath(experiment_dir))
    ver = base.split("_")[0]
    return ver in time_major_versions()
