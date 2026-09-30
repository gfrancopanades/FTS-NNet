#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Optuna search over Layer-2 classifiers, with an identical budget per arm.

The one property that matters more than the tuning itself: **selection never
sees the test month**. The search splits the TRAINING window chronologically,
fits on the earlier part and scores AUPRC on the later part, so the frozen
selection-then-validation contract of Section 3.5.1 is preserved. The winning
hyperparameters are then refit on the full training window, exactly as an
untuned arm is, and only then is the test month scored -- once.

Every searchable arm receives the same number of trials, which is the point:
the previous comparison gave logistic regression a 5-value grid, gave the tree
family a single hardcoded dict, and gave the neural family one hand-picked
recipe, then described the result as an identical tuning budget.
"""
from __future__ import annotations

import os
import time

import numpy as np
from sklearn.metrics import average_precision_score

from src.models.benchmark_classifiers import build_classifier, _chrono_split, day_index
from src.models.classifier_search_spaces import suggest, is_searchable, incumbent

TRIALS = int(os.environ.get("AP7_CLF_TRIALS", "0"))
TIMEOUT_S = int(os.environ.get("AP7_CLF_SEARCH_TIMEOUT", "0")) or None


def search_and_build(token, X_train, y_train, *, trials=None, seed=42):
    """Return (classifier, best_params, n_trials_done) — unsearched if disabled."""
    n = TRIALS if trials is None else trials
    if n <= 0 or not is_searchable(token):
        if n > 0:
            print(f"[CLF-SEARCH] {token}: no tunable surface, using defaults")
        return build_classifier(token), {}, 0

    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Split on whole days, not row position: the frame is location-major, so a
    # positional tail held out the northern 40 km of one carriageway rather than
    # the latest dates (see `_chrono_split`).
    tr, va = _chrono_split(len(X_train), days=day_index(X_train))
    Xtr, ytr = X_train.iloc[tr], np.asarray(y_train)[tr]
    Xva, yva = X_train.iloc[va], np.asarray(y_train)[va]
    print(f"[CLF-SEARCH] {token}: {n} trials | fit {len(tr):,} rows "
          f"({int(ytr.sum()):,} pos) -> score {len(va):,} rows "
          f"({int(yva.sum()):,} pos). Test month untouched.", flush=True)
    if yva.sum() == 0:
        print(f"[CLF-SEARCH] {token}: validation tail has no positives; defaults")
        return build_classifier(token), {}, 0

    t0 = time.time()

    def objective(trial):
        kw = suggest(token, trial)
        try:
            clf = build_classifier(token, **kw).fit(Xtr, ytr)
            return float(average_precision_score(yva, clf.predict_proba(Xva)))
        except Exception as exc:                       # a bad corner of the space
            print(f"  [trial {trial.number}] failed: {type(exc).__name__}: {exc}")
            raise optuna.TrialPruned()

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.NopPruner())

    # Seed trial 0 with the incumbent -- the library defaults the published
    # untuned arm used. The parameters were always *reachable*, but one point in
    # a six-dimensional space is never *reached* by 25 TPE draws, so the first
    # pass could and did return a configuration worse than the one it replaced
    # (mlp_focal: search best val 0.4601 -> test 0.4174, against an untuned test
    # score of 0.4722). Evaluating it explicitly makes the study's best at least
    # as good as the incumbent on validation, and exposes the selection gap.
    inc = incumbent(token)
    if inc:
        try:
            study.enqueue_trial(inc, skip_if_exists=True)
            print(f"[CLF-SEARCH] {token}: incumbent enqueued as trial 0 -> {inc}")
        except Exception as exc:
            print(f"[CLF-SEARCH] {token}: could not enqueue incumbent ({exc})")

    study.optimize(objective, n_trials=n, timeout=TIMEOUT_S,
                   catch=(Exception,), show_progress_bar=False)

    done = [t for t in study.trials if t.value is not None]
    if not done:
        print(f"[CLF-SEARCH] {token}: every trial failed, falling back to defaults")
        return build_classifier(token), {}, 0

    best = study.best_trial
    inc_val = next((t.value for t in done if inc and
                    all(t.params.get(k) == v for k, v in inc.items())), None)
    print(f"[CLF-SEARCH] {token}: best val AUPRC {best.value:.4f} "
          f"from {len(done)}/{n} trials in {time.time()-t0:.0f}s")
    if inc_val is not None:
        print(f"[CLF-SEARCH] {token}: incumbent scored {inc_val:.4f} on the same "
              f"split -> search gained {best.value - inc_val:+.4f} on VALIDATION")
    print(f"[CLF-SEARCH] {token}: params {best.params}", flush=True)
    # refit the winner on the FULL training window, as an untuned arm would be
    return build_classifier(token, **suggest(token, _Fixed(best.params))), best.params, len(done)


class _Fixed:
    """Replay a chosen trial's parameters through the same `suggest` function,
    so the refit is built by exactly the code path the search used."""
    def __init__(self, params):
        self.params = params

    def suggest_float(self, name, *a, **k):        return self.params.get(name, a[0] if a else 1.0)
    def suggest_int(self, name, *a, **k):          return self.params.get(name, a[0] if a else 1)
    def suggest_categorical(self, name, choices):  return self.params.get(name, choices[0])
