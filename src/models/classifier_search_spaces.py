#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Bayesian search spaces for every Layer-2 crash classifier.

Until now the Layer-2 arms were compared on library defaults: the tree family
used a single hardcoded dict (`max_depth=6, lr=0.05, n_estimators=400`), every
neural arm shared one hand-picked recipe, and only logistic regression received
any search at all -- a 5-value C grid. Section 3.5.3 nonetheless described the
benchmark as running under an "identical tuning budget", which is true only in
the sense that eleven of twelve budgets were zero.

That asymmetry matters in a specific direction. XGBoost's defaults are sensible
for a balanced problem and were never chosen for a 1:532 one, whereas focal loss
is designed for exactly this regime -- so part of the neural-vs-tree gap could
be a gap in how well each family's defaults happen to suit the task rather than
in the family itself.

These spaces let every arm receive the SAME number of Optuna trials against the
SAME objective (validation AUPRC on the chronological tail of the training
window). The test month is never touched: selection happens strictly inside
training data, so the frozen selection-then-validation contract is preserved.
"""
from __future__ import annotations


def suggest(token: str, trial):
    """Return a kwargs dict for `build_classifier(token, **kwargs)`."""
    t = (token or "").lower()

    # ---------------- neural family ---------------------------------------
    if t in ("mlp_focal", "mlp_gated", "mlp_ctx", "mlp_prox", "mlp_devres",
             "two_stage_mlp", "mlp_focal_two_stage"):
        kw = dict(
            alpha=trial.suggest_float("alpha", 0.05, 0.75),
            gamma=trial.suggest_float("gamma", 0.5, 5.0),
            hidden=trial.suggest_categorical(
                "hidden", ["128,64", "256,128,64", "512,256,128",
                           "512,512,256,128", "1024,512,256"]),
            dropout=trial.suggest_float("dropout", 0.05, 0.5),
            lr=trial.suggest_float("lr", 3e-5, 3e-3, log=True),
            batch_size=trial.suggest_categorical("batch_size", [1024, 2048, 4096, 8192]),
        )
        if t == "mlp_prox":
            # how sharply the soft target decays from the centre of a crash run
            kw["edge_weight"] = trial.suggest_float("edge_weight", 0.2, 0.95)
        return kw

    # BL-T5: same tree surface, but the imbalance handle is the objective's
    # (alpha, gamma) rather than a weight multiplier, which is the comparison
    # Section 5.3(e) needs.
    if t == "gbdt_focal":
        return dict(
            alpha=trial.suggest_float("alpha", 0.05, 0.95),
            gamma=trial.suggest_float("gamma", 0.5, 5.0),
            xgb_params=dict(
                max_depth=trial.suggest_int("max_depth", 3, 12),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                n_estimators=trial.suggest_int("n_estimators", 200, 1500, step=100),
                subsample=trial.suggest_float("subsample", 0.5, 1.0),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
                min_child_weight=trial.suggest_float("min_child_weight", 1.0, 50.0, log=True),
                reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 50.0, log=True),
                reg_alpha=trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
                tree_method="hist", n_jobs=-1))

    # ---------------- gradient-boosted trees ------------------------------
    if t in ("gbdt_one_stage", "gbdt_two_stage", "gbdt_mlp_supervisor", "xgb_smote",
             "gbdt_prox", "gbdt_resid", "gbdt_mono", "gbdt_block"):
        kw = dict(xgb_params=dict(
            max_depth=trial.suggest_int("max_depth", 3, 12),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            n_estimators=trial.suggest_int("n_estimators", 200, 1500, step=100),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
            min_child_weight=trial.suggest_float("min_child_weight", 1.0, 50.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 50.0, log=True),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            eval_metric="aucpr", tree_method="hist", n_jobs=-1))
        # the imbalance handle: a multiplier on the neg/pos ratio, so the search
        # can both under- and over-correct rather than being fixed at exactly 1x
        # scale_pos_weight is set inside each class from the observed neg/pos
        # ratio; expose it here so the search can under- or over-correct.
        kw["xgb_params"]["scale_pos_weight_mult"] = trial.suggest_float(
            "pos_weight_mult", 0.1, 3.0, log=True)
        if t == "gbdt_prox":
            # how sharply the row weight decays to the edge of a crash run
            kw["edge_weight"] = trial.suggest_float("edge_weight", 0.1, 0.95)
        if t == "gbdt_block":
            kw["n_members"] = trial.suggest_int("n_members", 4, 16)
            kw["block_frac"] = trial.suggest_float("block_frac", 0.5, 0.95)
        return kw

    # ---------------- bagging / forests ------------------------------------
    if t == "random_forest":
        # The class builds its forest as
        #   RandomForestClassifier(n_estimators=200, class_weight=..., **params)
        # so `params` must not contain n_estimators, class_weight, n_jobs or
        # random_state -- passing any of them raises "got multiple values for
        # keyword argument". Restrict the space to the keys it can actually take.
        return dict(grid=[dict(
            max_depth=trial.suggest_int("max_depth", 4, 30),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 50, log=True),
            max_features=trial.suggest_categorical(
                "max_features", ["sqrt", "log2", 0.3, 0.6]),
            criterion=trial.suggest_categorical("criterion", ["gini", "entropy"]))])
    if t == "balanced_bagging_xgb":
        return dict(
            n_members=trial.suggest_int("n_members", 5, 40),
            neg_pos_ratio=trial.suggest_float("neg_pos_ratio", 1.0, 50.0, log=True),
            num_boost_round=trial.suggest_int("num_boost_round", 100, 800, step=50),
            xgb_params=dict(
                max_depth=trial.suggest_int("bb_max_depth", 3, 10),
                eta=trial.suggest_float("bb_eta", 0.02, 0.3, log=True),
                subsample=trial.suggest_float("bb_subsample", 0.5, 1.0),
                colsample_bytree=trial.suggest_float("bb_colsample", 0.4, 1.0)))

    # ---------------- linear ----------------------------------------------
    if t == "logreg":
        return dict(C_grid=(trial.suggest_float("C", 1e-4, 1e2, log=True),))
    if t == "logreg_pk_effects":
        return {}          # the class exposes no kwargs surface

    # ---------------- no tunable surface ----------------------------------
    return {}


def is_searchable(token: str) -> bool:
    """`historical_risk` is an empirical-Bayes rate estimator with no free
    hyperparameters; searching it would spend the budget on noise."""
    return (token or "").lower() != "historical_risk" and bool(
        suggest(token, _NullTrial()))


class _NullTrial:
    """Probe the space without an Optuna study attached."""
    def suggest_float(self, *a, **k):        return 1.0
    def suggest_int(self, *a, **k):          return 1
    def suggest_categorical(self, n, c):     return c[0]


# ---------------------------------------------------------------------------
# The incumbent configuration of each arm, i.e. the library defaults the
# published Table 2 was produced with. A search that never evaluates the
# incumbent can return something worse than it -- which is exactly what happened
# to `mlp_focal` on the first pass (search best val 0.4601, test 0.4174, against
# an untuned test score of 0.4722). Seeding trial 0 with these values guarantees
# the study's best is at least as good as the incumbent ON VALIDATION, and makes
# the selection gap measurable rather than invisible.
# ---------------------------------------------------------------------------
INCUMBENT = {
    "mlp_focal":           dict(alpha=0.25, gamma=2.0, hidden="256,128,64",
                                dropout=0.3, lr=3e-4, batch_size=4096),
    "mlp_gated":           dict(alpha=0.25, gamma=2.0, hidden="256,128,64",
                                dropout=0.3, lr=3e-4, batch_size=4096),
    "mlp_ctx":             dict(alpha=0.25, gamma=2.0, hidden="256,128,64",
                                dropout=0.3, lr=3e-4, batch_size=4096),
    "mlp_prox":            dict(alpha=0.25, gamma=2.0, hidden="256,128,64",
                                dropout=0.3, lr=3e-4, batch_size=4096,
                                edge_weight=0.6),
    "mlp_devres":          dict(alpha=0.25, gamma=2.0, hidden="256,128,64",
                                dropout=0.3, lr=3e-4, batch_size=4096),
    "two_stage_mlp":       dict(alpha=0.25, gamma=2.0, hidden="256,128,64",
                                dropout=0.3, lr=3e-4, batch_size=4096),
    "mlp_focal_two_stage": dict(alpha=0.25, gamma=2.0, hidden="256,128,64",
                                dropout=0.3, lr=3e-4, batch_size=4096),
}
_XGB_INCUMBENT = dict(max_depth=6, learning_rate=0.05, n_estimators=400,
                      subsample=0.8, colsample_bytree=0.8, min_child_weight=1.0,
                      reg_lambda=1.0, reg_alpha=1e-4, pos_weight_mult=1.0)
for _t in ("gbdt_one_stage", "gbdt_two_stage", "gbdt_mlp_supervisor", "xgb_smote",
           "gbdt_prox", "gbdt_resid", "gbdt_mono", "gbdt_block"):
    INCUMBENT[_t] = dict(_XGB_INCUMBENT)
INCUMBENT["gbdt_focal"] = dict(
    {k: v for k, v in _XGB_INCUMBENT.items() if k != "pos_weight_mult"},
    alpha=0.25, gamma=2.0)
INCUMBENT["gbdt_prox"]["edge_weight"] = 0.4
INCUMBENT["gbdt_block"].update(n_members=8, block_frac=0.8)
INCUMBENT["random_forest"] = dict(max_depth=12, min_samples_leaf=5,
                                  max_features="sqrt", criterion="gini")
INCUMBENT["logreg"] = dict(C=1.0)
INCUMBENT["balanced_bagging_xgb"] = dict(n_members=10, neg_pos_ratio=10.0,
                                         num_boost_round=300, bb_max_depth=6,
                                         bb_eta=0.05, bb_subsample=0.8,
                                         bb_colsample=0.8)


def incumbent(token: str):
    """Parameters reproducing the published untuned arm, or None."""
    return INCUMBENT.get((token or "").lower())
