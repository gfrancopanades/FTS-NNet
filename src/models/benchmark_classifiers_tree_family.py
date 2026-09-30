#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Four gradient-boosted variants designed for THIS problem.

The tree arms in Table 2 are stock XGBoost on a binary target with a single
`scale_pos_weight`. Four assumptions in that setup are wrong here in specific,
separable ways, and each model below repairs exactly one:

  BL-T1  GBDT-Prox   `scale_pos_weight` compresses every positive into one
                     number, but the +/-15 min labelling makes a crash a RUN of
                     intervals whose centre is far more certainly a crash than
                     its edges. XGBoost takes per-row `sample_weight`, so the
                     insight that won BL-C14 applies directly.

  BL-T2  GBDT-Resid  a split on `mean_speed < 85` means something different at
                     km 141 in rush hour than at km 200 at midnight. The leading
                     traffic channels are replaced by residuals against
                     per-(pk, hour-of-week) baselines fitted strictly before the
                     train/test cutoff.

  BL-T3  GBDT-Mono   with 6,092 positives, asserting known physics beats
                     spending capacity rediscovering it. Section 6 found every
                     leading feature has a positive signed SHAP value; those
                     directions become monotone constraints.

  BL-T4  GBDT-Block  `subsample` draws rows independently, splitting crash runs
                     across the in-bag boundary and letting the model memorise
                     within-run correlation. Sampling whole (pk, day) blocks
                     matches the day-block bootstrap already used for the
                     confidence intervals, so training and inference share one
                     assumption about what is independent.

All four inherit the searched XGBoost hyperparameters, so a difference is
attributable to the stated change rather than to tuning.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
import xgboost as xgb

from src.models.benchmark_classifiers import (
    _BaseBenchmarkClassifier, RANDOM_STATE)

# reuse the family helpers written for the MLP variants
from src.models.benchmark_classifiers_mlp_family import _timestamp_key


def _base_params(kw, n_pos, n_neg):
    p = dict(kw.get("xgb_params", {}) or {})
    p.setdefault("max_depth", 6)
    p.setdefault("learning_rate", 0.05)
    p.setdefault("n_estimators", 400)
    p.setdefault("subsample", 0.8)
    p.setdefault("colsample_bytree", 0.8)
    p.setdefault("eval_metric", "aucpr")
    p.setdefault("tree_method", "hist")
    p.setdefault("n_jobs", -1)
    p.setdefault("random_state", RANDOM_STATE)
    mult = float(p.pop("scale_pos_weight_mult", 1.0))
    p.setdefault("scale_pos_weight", (n_neg / max(n_pos, 1)) * mult)
    return p


def _run_weights(X, y, edge=0.4):
    """Weight decaying from the centre of each crash run to its edges.

    Mirrors BL-C14's soft targets. A tree cannot take a fractional label, but it
    can take a fractional row weight, which reaches the split criterion the same
    way. Negatives keep weight 1.
    """
    yv = np.asarray(y, dtype=int)
    w = np.ones(len(yv), dtype=np.float32)
    ts = _timestamp_key(X)
    if ts is None or "pk" not in X.columns:
        print("[gbdt_prox] calendar columns absent -> uniform weights (= BL-C6)")
        return w
    d = pd.DataFrame({"pk": pd.to_numeric(X["pk"], errors="coerce").to_numpy(),
                      "ts": ts.to_numpy(), "y": yv, "i": np.arange(len(yv))})
    d = d.sort_values(["pk", "ts"], kind="mergesort")
    pos = d[d.y > 0]
    if len(pos):
        brk = (pos["pk"].to_numpy()[1:] != pos["pk"].to_numpy()[:-1])
        runs = np.concatenate([[0], np.cumsum(brk)])
        for r in np.unique(runs):
            idx = pos["i"].to_numpy()[runs == r]
            n = len(idx)
            if n <= 1:
                continue
            centre = (n - 1) / 2.0
            k = np.arange(n)
            w[idx] = (1.0 - (1.0 - edge) * np.abs(k - centre) / max(centre, 1e-9)).astype(np.float32)
    print(f"[gbdt_prox] {int((w < 1).sum()):,} positive rows down-weighted "
          f"(edge {edge})")
    return w


class GBDTProximity(_BaseBenchmarkClassifier):
    token = "gbdt_prox"

    def fit(self, X, y):
        self.features = list(X.columns)
        Xv = self._as_matrix(X); yv = np.asarray(y, dtype=int)
        w = _run_weights(X, yv, float(self.kwargs.get("edge_weight", 0.4)))
        p = _base_params(self.kwargs, int(yv.sum()), int((yv == 0).sum()))
        self.model = xgb.XGBClassifier(**p)
        self.model.fit(Xv, yv, sample_weight=w)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(self._as_matrix(X))[:, 1]


# =============================================================================
# BL-T2 — GBDT-Resid: features as departures from local normality
# =============================================================================

_RESID_CHANNELS = ("mean_speed", "intTot", "intP", "car", "speed_std_2",
                   "speed_std_4", "delta_speed_2", "hv_fraction")


class GBDTLocalResidual(_BaseBenchmarkClassifier):
    """Replace the leading traffic channels by residuals against a per-(pk,
    hour-of-week) baseline.

    A tree partitions on absolute values, so `mean_speed < 85` conflates a
    congested rush hour at one post with free flow at another. The engineered
    features do include z-scores, but the tree must spend depth discovering
    which of the 52 columns encode locality. Here the baseline is subtracted
    directly, so every split is already a statement about departure.

    The baselines are fitted on the TRAINING rows only and stored, then applied
    unchanged at scoring time -- the same discipline the engineered-feature
    baselines follow in Section 3.5.2b, and for the same reason.
    """

    token = "gbdt_resid"

    def _keys(self, X):
        pk = pd.to_numeric(X["pk"], errors="coerce").fillna(-1).astype(int)
        hor = (pd.to_numeric(X["hor"], errors="coerce").fillna(0).astype(int)
               if "hor" in X.columns else pd.Series(0, index=X.index))
        dow = (pd.to_numeric(X["diaSem"], errors="coerce").fillna(0).astype(int)
               if "diaSem" in X.columns else pd.Series(0, index=X.index))
        return pk.to_numpy(), (dow * 24 + hor).to_numpy()

    def _residualise(self, X, fit=False):
        cols = [c for c in _RESID_CHANNELS if c in X.columns]
        if not cols or "pk" not in X.columns:
            if fit:
                self._baselines = None
                print("[gbdt_resid] required columns absent -> raw features (= BL-C6)")
            return self._as_matrix(X)
        pk, how = self._keys(X)
        d = pd.DataFrame({c: pd.to_numeric(X[c], errors="coerce") for c in cols})
        d["_pk"], d["_how"] = pk, how
        if fit:
            self._baselines = d.groupby(["_pk", "_how"])[cols].median()
            self._global = d[cols].median()
            print(f"[gbdt_resid] fitted {len(self._baselines):,} (pk, hour-of-week) "
                  f"baselines over {len(cols)} channels")
        idx = pd.MultiIndex.from_arrays([pk, how])
        base = self._baselines.reindex(idx)
        out = self._as_matrix(X).copy()
        for j, c in enumerate(self.features):
            if c in cols:
                b = base[c].to_numpy()
                b = np.where(np.isnan(b), float(self._global[c]), b)
                out[:, j] = out[:, j] - b
        return out

    def fit(self, X, y):
        self.features = list(X.columns)
        yv = np.asarray(y, dtype=int)
        Xv = self._residualise(X, fit=True)
        p = _base_params(self.kwargs, int(yv.sum()), int((yv == 0).sum()))
        self.model = xgb.XGBClassifier(**p)
        self.model.fit(Xv, yv)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(self._residualise(X))[:, 1]


# =============================================================================
# BL-T3 — GBDT-Mono: assert the directions Section 6 established
# =============================================================================

# name pattern -> required direction. Every leading feature in Table 7 carries a
# POSITIVE signed SHAP value on crash intervals, and each direction below is
# also an established freeway-safety finding, not merely an observed
# correlation: dispersion and abrupt change raise risk; being unusually slow
# for the place and hour raises risk.
_MONOTONE = [
    (r"speed_std",              +1),   # speed variability
    (r"vol_std",                +1),   # volume variability
    (r"^accel_speed$",          +1),
    (r"speed_pct_below_normal", +1),
    (r"crash_rate",             +1),   # historical site risk
    (r"curv_x_delta_speed",     +1),   # decelerating on a curve
]


class GBDTMonotone(_BaseBenchmarkClassifier):
    """Constrain the response to be monotone in features whose direction is
    known, rather than spending 6,092 positives on rediscovering it.

    Two payoffs. Statistically, shrinking the hypothesis space is usually worth
    more than searching it when positives are this scarce. Operationally, a
    monotone model cannot produce the sentence "more speed variance lowered the
    predicted risk", which is what a road authority would need it never to say.
    """

    token = "gbdt_mono"

    def _constraints(self, features):
        out = []
        for f in features:
            d = 0
            for pat, sign in _MONOTONE:
                if re.search(pat, f):
                    d = sign
                    break
            out.append(d)
        n = sum(1 for v in out if v)
        print(f"[gbdt_mono] {n} of {len(features)} features constrained monotone "
              f"(+1): {[f for f, v in zip(features, out) if v][:8]}")
        return "(" + ",".join(str(v) for v in out) + ")"

    def fit(self, X, y):
        self.features = list(X.columns)
        Xv = self._as_matrix(X); yv = np.asarray(y, dtype=int)
        p = _base_params(self.kwargs, int(yv.sum()), int((yv == 0).sum()))
        p["monotone_constraints"] = self._constraints(self.features)
        self.model = xgb.XGBClassifier(**p)
        self.model.fit(Xv, yv)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(self._as_matrix(X))[:, 1]


# =============================================================================
# BL-T4 — GBDT-Block: subsample whole (pk, day) blocks, not independent rows
# =============================================================================

class GBDTDayBlock(_BaseBenchmarkClassifier):
    """Boost on day-block bootstrap samples instead of row-wise subsampling.

    XGBoost's `subsample` draws rows independently. Here rows are not
    independent: a crash occupies a run of consecutive intervals at one post,
    and traffic is autocorrelated for hours either side. Row-wise sampling
    splits a run across the in-bag/out-of-bag boundary, so a tree can fit one
    half and be scored on the other -- an optimistic signal that does not exist
    at deployment, where whole days are unseen.

    Sampling whole (pk, day) blocks is the same unit of independence the
    day-block bootstrap of Section 3.5.4 already assumes when it computes the
    confidence intervals. Training and inference then share one assumption
    rather than contradicting each other.

    Implemented as an explicit ensemble over block-bootstrap resamples, because
    XGBoost has no grouped-subsample hook; `subsample` is set to 1.0 inside each
    member so the only sampling is the block draw.
    """

    token = "gbdt_block"

    def fit(self, X, y):
        self.features = list(X.columns)
        Xv = self._as_matrix(X); yv = np.asarray(y, dtype=int)
        n_members = int(self.kwargs.get("n_members", 8))
        frac = float(self.kwargs.get("block_frac", 0.8))
        rng = np.random.default_rng(RANDOM_STATE)

        ts = _timestamp_key(X)
        if ts is None or "pk" not in X.columns:
            print("[gbdt_block] calendar columns absent -> single fit (= BL-C6)")
            blocks = np.zeros(len(yv), dtype=np.int64)
        else:
            day = (ts.to_numpy() // 10_000)          # yyyymmdd
            pk = pd.to_numeric(X["pk"], errors="coerce").fillna(-1).astype(np.int64).to_numpy()
            blocks = pk * 10**8 + day
        uniq = np.unique(blocks)
        print(f"[gbdt_block] {len(uniq):,} (pk, day) blocks; {n_members} members "
              f"at {frac:.0%} of blocks each")

        p = _base_params(self.kwargs, int(yv.sum()), int((yv == 0).sum()))
        p["subsample"] = 1.0            # the block draw IS the subsampling
        p["n_estimators"] = max(int(p["n_estimators"] // n_members), 50)
        self.models = []
        for m in range(n_members):
            take = rng.choice(uniq, size=max(int(len(uniq) * frac), 1), replace=False)
            sel = np.isin(blocks, take)
            if yv[sel].sum() == 0:
                continue
            mdl = xgb.XGBClassifier(**{**p, "random_state": RANDOM_STATE + m})
            mdl.fit(Xv[sel], yv[sel])
            self.models.append(mdl)
        print(f"[gbdt_block] fitted {len(self.models)} members")
        return self

    def predict_proba(self, X):
        Xv = self._as_matrix(X)
        return np.mean([m.predict_proba(Xv)[:, 1] for m in self.models], axis=0)


TREE_FAMILY = {c.token: c for c in
               (GBDTProximity, GBDTLocalResidual, GBDTMonotone, GBDTDayBlock)}
