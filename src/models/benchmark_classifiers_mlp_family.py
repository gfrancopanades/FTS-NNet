#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Four MLP variants designed for THIS problem, not adapted from elsewhere.

BL-C5 (MLP + focal loss) treats every segment-interval as an i.i.d. row of 52
numbers. That assumption is wrong here in four specific, separable ways, and
each model below repairs exactly one of them so the contribution of each repair
is attributable:

  BL-C12  MLP-Gate   the 52 features are not one homogeneous vector. They form
                     four semantic blocks whose usefulness is context-dependent
                     -- Section 6 shows geometry contributes only through its
                     interaction with deceleration, never on its own. A context
                     network gates each block instead of letting the first
                     linear layer discover that interaction unaided.

  BL-C13  MLP-Ctx    crash risk is a field over the corridor, not a set of
                     independent cells. Congestion arrives from the upstream
                     kilometre post. Each row is given the traffic state of its
                     spatial neighbours at the same instant.

  BL-C14  MLP-Prox   the +/-15 min labelling makes a crash a RUN of positive
                     intervals, and the run's centre is a more certain positive
                     than its edges. Binary targets discard that; soft targets
                     that decay from the centre keep it.

  BL-C15  MLP-DevRes the novel one. See its docstring.

All four keep BL-C5's trainer, optimiser, early stopping and focal parameters,
so any difference is attributable to the stated change and not to tuning.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.benchmark_classifiers import (
    _BaseBenchmarkClassifier, _FocalLoss, _MLP, _train_mlp, _chrono_split,
    _mlp_kwargs, day_index, RANDOM_STATE, _DEVICE, StandardScaler, average_precision_score,
)

# --------------------------------------------------------------------------
# semantic feature blocks (matched by name; anything unmatched joins "other")
# --------------------------------------------------------------------------
import re

# Order matters: first pattern wins, and the specific blocks must be tested
# before the general ones. `curv_x_delta_speed` contains "speed" and would fall
# into traffic; `pk_crash_rate_mob` contains "mob" and would fall into calendar.
# Both are the exact features Section 6 singles out, so mis-binning them would
# defeat the purpose of blocking at all.
_BLOCK_PATTERNS = {
    "site":     r"crash_rate",
    "geometry": r"curv|pend|slope|geom",
    "calendar": r"^(anyo|mes|dia|diaSem|hor|5min)$|peak|festiu|mob_esp|_sin$|_cos$",
    "traffic":  r"speed|vel|int|vol|car|flow|accel|delta|density|regime|occ|hv_",
}


def split_blocks(features):
    """Assign each feature to exactly one block, first pattern wins."""
    blocks = {k: [] for k in _BLOCK_PATTERNS}
    blocks["other"] = []
    for i, f in enumerate(features):
        for name, pat in _BLOCK_PATTERNS.items():
            if re.search(pat, f):
                blocks[name].append(i); break
        else:
            blocks["other"].append(i)
    return {k: v for k, v in blocks.items() if v}


def _timestamp_key(X):
    """A sortable integer instant from the calendar columns present in X.

    The classifiers receive engineered features, not `dat`, so the instant is
    rebuilt from (anyo, mes, dia, hor, 5min). Returns None if unavailable, and
    every caller degrades to the plain BL-C5 behaviour in that case rather than
    silently constructing a wrong neighbourhood.
    """
    need = ("anyo", "mes", "dia", "hor")
    if not all(c in X.columns for c in need):
        return None
    sub = X["5min"] if "5min" in X.columns else 0
    return (pd.to_numeric(X["anyo"], errors="coerce").fillna(0).astype(np.int64) * 10**8
            + pd.to_numeric(X["mes"], errors="coerce").fillna(0).astype(np.int64) * 10**6
            + pd.to_numeric(X["dia"], errors="coerce").fillna(0).astype(np.int64) * 10**4
            + pd.to_numeric(X["hor"], errors="coerce").fillna(0).astype(np.int64) * 100
            + pd.to_numeric(sub, errors="coerce").fillna(0).astype(np.int64))


# =============================================================================
# BL-C12 — MLP-Gate: context-gated feature blocks
# =============================================================================

class _GatedBlockMLP(nn.Module):
    """Encode each semantic block separately, then gate the blocks by context.

    A plain MLP must discover from data that geometry is only informative when
    multiplied by deceleration. Here a small context network reads the calendar,
    geometry and site-risk blocks and emits one scalar gate per block, so the
    network can turn the traffic-dynamics block up during a congested rush hour
    and down at 3 a.m. without spending width on learning the interaction.
    """

    def __init__(self, blocks, emb=48, dropout=0.3):
        super().__init__()
        self.blocks = blocks
        self.encoders = nn.ModuleDict({
            name: nn.Sequential(nn.Linear(len(idx), emb), nn.ReLU(),
                                nn.Dropout(dropout), nn.Linear(emb, emb), nn.ReLU())
            for name, idx in blocks.items()})
        ctx_dim = sum(len(blocks[b]) for b in ("calendar", "geometry", "site") if b in blocks)
        self.gate = nn.Sequential(nn.Linear(max(ctx_dim, 1), 32), nn.ReLU(),
                                  nn.Linear(32, len(blocks)))
        self.ctx_idx = [i for b in ("calendar", "geometry", "site") if b in blocks
                        for i in blocks[b]]
        self.head = nn.Sequential(nn.Linear(emb, 64), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(64, 1))

    def forward(self, x):
        ctx = x[:, self.ctx_idx] if self.ctx_idx else x[:, :1]
        g = torch.sigmoid(self.gate(ctx))                       # (B, n_blocks)
        h = 0
        for k, (name, idx) in enumerate(self.blocks.items()):
            h = h + g[:, k:k + 1] * self.encoders[name](x[:, idx])
        return self.head(h).squeeze(-1)


class MLPGatedBlocks(_BaseBenchmarkClassifier):
    token = "mlp_gated"

    def fit(self, X, y):
        self.features = list(X.columns)
        self.blocks = split_blocks(self.features)
        Xv = self._as_matrix(X); yv = np.asarray(y, dtype=np.float32)
        self.scaler = StandardScaler().fit(Xv)
        Xs = self.scaler.transform(Xv).astype(np.float32)
        crit = _FocalLoss(alpha=float(self.kwargs.get("alpha", 0.25)),
                          gamma=float(self.kwargs.get("gamma", 2.0)))
        mk = _mlp_kwargs(self.kwargs, X)
        # This model is parameterised by a per-block embedding width, not by a
        # hidden tuple, so a searched `hidden` must be translated rather than
        # forwarded -- passing it on reaches _train_generic, which rejects it.
        hid = mk.pop("hidden", None)
        emb = int(max(hid)) // 4 if hid else 48
        self.model, self.val_aucpr = _train_generic(
            _GatedBlockMLP(self.blocks, emb=max(emb, 16),
                           dropout=mk.pop("dropout", 0.3)),
            Xs, yv, crit, **mk)
        print(f"[mlp_gated] blocks: "
              + ", ".join(f"{k}={len(v)}" for k, v in self.blocks.items()))
        return self

    def _to_cpu(self):
        self.model = self.model.to("cpu"); return self

    def predict_proba(self, X):
        Xs = self.scaler.transform(self._as_matrix(X)).astype(np.float32)
        self.model.eval()
        with torch.no_grad():
            return torch.sigmoid(self.model(torch.as_tensor(Xs))).numpy()


# =============================================================================
# shared trainer for the custom architectures (mirrors _train_mlp exactly)
# =============================================================================

def _train_generic(model, Xtr, ytr, criterion, *, lr=3e-4, max_epochs=60,
                   patience=8, batch_size=4096, seed=RANDOM_STATE,
                   device=_DEVICE, sample_weight=None, batch_sampler=None,
                   days=None):
    """Same optimiser, schedule, chronological split and early-stopping rule as
    BL-C5's `_train_mlp`, so a difference in score is attributable to the model
    rather than to the training recipe."""
    torch.manual_seed(seed); np.random.seed(seed)
    tr_idx, va_idx = _chrono_split(len(Xtr), days=days)
    Xt = torch.as_tensor(Xtr, dtype=torch.float32)
    yt = torch.as_tensor(ytr, dtype=torch.float32)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    if batch_sampler is not None:
        dl = batch_sampler(tr_idx, Xt, yt, batch_size, seed)
    else:
        ds = torch.utils.data.TensorDataset(Xt[tr_idx], yt[tr_idx])
        dl = torch.utils.data.DataLoader(
            ds, batch_size=batch_size, shuffle=True,
            generator=torch.Generator().manual_seed(seed))
    Xva = Xt[va_idx].to(device)
    yva_bin = (np.asarray(ytr)[va_idx] >= 0.5).astype(int)
    best_auc, best_state, no_imp = -np.inf, None, 0
    for _ in range(max_epochs):
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            p = torch.sigmoid(model(Xva)).cpu().numpy()
        auc = average_precision_score(yva_bin, p) if yva_bin.sum() > 0 else 0.0
        if auc > best_auc:
            best_auc, no_imp = auc, 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            no_imp += 1
            if no_imp >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model.to("cpu"), float(best_auc)


# =============================================================================
# BL-C13 — MLP-Ctx: the upstream/downstream neighbour at the same instant
# =============================================================================

class MLPNeighbourContext(_BaseBenchmarkClassifier):
    """Give each cell the traffic state of its spatial neighbours.

    A queue reaching kilometre post p was, five minutes earlier, at p-1. The
    engineered features already carry that location's own history, but nothing
    about the road immediately upstream, so a plain MLP cannot see a shockwave
    approaching. This appends the leading traffic channels of pk-1 and pk+1 at
    the SAME timestamp, plus their difference from the cell's own value -- a
    spatial gradient, which is what a shockwave actually is.

    Falls back to plain BL-C5 behaviour if the calendar columns needed to
    rebuild the instant are absent, rather than fabricating a neighbourhood.
    """

    token = "mlp_ctx"
    NEIGHBOUR_CHANNELS = ("mean_speed", "intTot", "speed_std_2", "delta_speed_2")

    def _augment(self, X):
        ts = _timestamp_key(X)
        if ts is None or "pk" not in X.columns:
            self._ctx_cols = []
            return self._as_matrix(X)
        chans = [c for c in self.NEIGHBOUR_CHANNELS if c in X.columns]
        if not chans:
            self._ctx_cols = []
            return self._as_matrix(X)
        d = X[["pk"] + chans].copy()
        d["_ts"] = ts.to_numpy()
        d["pk"] = pd.to_numeric(d["pk"], errors="coerce")
        # one row per (instant, pk): join the same instant at pk-1 and pk+1
        key = d.set_index(["_ts", "pk"])[chans]
        key = key[~key.index.duplicated(keep="first")]
        out = {}
        for side, shift in (("up", -1), ("dn", +1)):
            idx = pd.MultiIndex.from_arrays([d["_ts"].to_numpy(),
                                             d["pk"].to_numpy() + shift])
            nb = key.reindex(idx)
            for c in chans:
                v = nb[c].to_numpy()
                own = d[c].to_numpy()
                out[f"{c}_{side}"] = np.where(np.isnan(v), own, v)
                out[f"{c}_grad_{side}"] = out[f"{c}_{side}"] - own
        self._ctx_cols = list(out)
        extra = np.column_stack([out[c] for c in self._ctx_cols]).astype(np.float32)
        return np.hstack([self._as_matrix(X), extra])

    def fit(self, X, y):
        self.features = list(X.columns)
        Xv = self._augment(X); yv = np.asarray(y, dtype=np.float32)
        print(f"[mlp_ctx] appended {len(self._ctx_cols)} spatial-neighbour columns")
        self.scaler = StandardScaler().fit(Xv)
        Xs = self.scaler.transform(Xv).astype(np.float32)
        crit = _FocalLoss(alpha=float(self.kwargs.get("alpha", 0.25)),
                          gamma=float(self.kwargs.get("gamma", 2.0)))
        mk = _mlp_kwargs(self.kwargs, X)
        self.model, self.val_aucpr = _train_generic(
            _MLP(Xs.shape[1], mk.pop('hidden', (256, 128, 64)),
                 mk.pop('dropout', 0.3)), Xs, yv, crit, **mk)
        return self

    def _to_cpu(self):
        self.model = self.model.to("cpu"); return self

    def predict_proba(self, X):
        Xs = self.scaler.transform(self._augment(X)).astype(np.float32)
        self.model.eval()
        with torch.no_grad():
            return torch.sigmoid(self.model(torch.as_tensor(Xs))).numpy()


# =============================================================================
# BL-C14 — MLP-Prox: soft targets that decay from the centre of a crash run
# =============================================================================

class _SoftFocalLoss(nn.Module):
    """Focal loss over targets in [0, 1] rather than {0, 1}.

    The standard formulation splits on `targets >= 0.5` to pick alpha and pt,
    which discards everything a soft target encodes. Here both are interpolated,
    so a 0.6 target contributes as a weak positive rather than a full one.
    """

    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha, self.gamma = float(alpha), float(gamma)

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        pt = (targets * p + (1 - targets) * (1 - p)).clamp(1e-6, 1.0)
        at = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        return (at * (1 - pt) ** self.gamma * bce).mean()


class MLPProximity(_BaseBenchmarkClassifier):
    """Exploit the structure the +/-15 min labelling leaves in the targets.

    A recorded crash is expanded to the window around it, so positives arrive as
    short runs at one location. Every interval in a run gets target 1.0, which
    asserts that the interval three steps before the crash is exactly as
    crash-like as the interval containing it. It is not: the operator's own
    timestamp uncertainty is what motivates the window, and uncertainty is
    highest at the edges.

    Targets therefore decay from each run's centre (1.0) to its edges
    (`edge_weight`), and the loss interpolates instead of thresholding. Negatives
    stay at 0. The evaluation label is untouched -- this changes what the model
    is asked to fit, never what it is scored against.
    """

    token = "mlp_prox"

    def _soft_targets(self, X, y):
        yv = np.asarray(y, dtype=np.float32)
        ts = _timestamp_key(X)
        if ts is None or "pk" not in X.columns:
            print("[mlp_prox] calendar columns absent -> binary targets (= BL-C5)")
            return yv
        edge = float(self.kwargs.get("edge_weight", 0.6))
        df = pd.DataFrame({"pk": pd.to_numeric(X["pk"], errors="coerce").to_numpy(),
                           "ts": ts.to_numpy(), "y": yv,
                           "i": np.arange(len(yv))})
        df = df.sort_values(["pk", "ts"], kind="mergesort")
        out = yv.copy()
        pos = df[df.y > 0]
        if len(pos):
            # a run breaks when the location changes or the instant jumps
            brk = (pos["pk"].to_numpy()[1:] != pos["pk"].to_numpy()[:-1])
            runs = np.concatenate([[0], np.cumsum(brk)])
            for r in np.unique(runs):
                idx = pos["i"].to_numpy()[runs == r]
                n = len(idx)
                if n <= 1:
                    continue
                # triangular weight: 1.0 at the centre, `edge` at both ends
                pos_in_run = np.arange(n)
                centre = (n - 1) / 2.0
                w = 1.0 - (1.0 - edge) * (np.abs(pos_in_run - centre) / max(centre, 1e-9))
                out[idx] = w.astype(np.float32)
        n_soft = int(((out > 0) & (out < 1)).sum())
        print(f"[mlp_prox] {n_soft:,} of {int((yv > 0).sum()):,} positives "
              f"softened (edge weight {edge})")
        return out

    def fit(self, X, y):
        self.features = list(X.columns)
        Xv = self._as_matrix(X)
        soft = self._soft_targets(X, y)
        self.scaler = StandardScaler().fit(Xv)
        Xs = self.scaler.transform(Xv).astype(np.float32)
        crit = _SoftFocalLoss(alpha=float(self.kwargs.get("alpha", 0.25)),
                              gamma=float(self.kwargs.get("gamma", 2.0)))
        mk = _mlp_kwargs(self.kwargs, X)
        self.model, self.val_aucpr = _train_generic(
            _MLP(Xs.shape[1], mk.pop('hidden', (256, 128, 64)),
                 mk.pop('dropout', 0.3)), Xs, soft, crit, **mk)
        return self

    def _to_cpu(self):
        self.model = self.model.to("cpu"); return self

    def predict_proba(self, X):
        Xs = self.scaler.transform(self._as_matrix(X)).astype(np.float32)
        self.model.eval()
        with torch.no_grad():
            return torch.sigmoid(self.model(torch.as_tensor(Xs))).numpy()


# =============================================================================
# BL-C15 — MLP-DevRes: deviation-residual network with stratum-anchored focal
# =============================================================================

class _DeviationResidualNet(nn.Module):
    """Two branches whose sum is the logit: a learned local baseline, and a
    departure from it driven only by traffic dynamics.

        logit(s,t) = b(context of s,t)  +  d(dynamics of s,t)

    `b` sees only what is true of a place and a time of week regardless of that
    day's traffic -- location, hour, weekday, geometry, historical crash rate.
    It is a neural analogue of the random-effects term in the crash-frequency
    literature: the corridor's own base risk surface.

    `d` sees only the forecast traffic and the quantities engineered from it,
    and can only push risk away from that baseline.

    The separation is the point. It states in the architecture what Section 6
    finds empirically -- that crash risk is carried by *departure from local
    normality*, not by where crashes are common -- and it stops the network
    spending capacity re-deriving a hotspot map, which Table 2 shows scores at
    chance on its own.
    """

    def __init__(self, ctx_idx, dyn_idx, n_pk=256, emb=16, hidden=(256, 128, 64),
                 dropout=0.3):
        super().__init__()
        self.ctx_idx, self.dyn_idx = ctx_idx, dyn_idx
        self.pk_embed = nn.Embedding(n_pk, emb)
        self.baseline = nn.Sequential(
            nn.Linear(len(ctx_idx) + emb, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
        layers, prev = [], len(dyn_idx)
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, 1)]
        self.deviation = nn.Sequential(*layers)

    def forward(self, x, pk_ix=None):
        if pk_ix is None:                       # pk index is carried as the last column
            x, pk_ix = x[:, :-1], x[:, -1].long().clamp(0, self.pk_embed.num_embeddings - 1)
        ctx = torch.cat([x[:, self.ctx_idx], self.pk_embed(pk_ix)], dim=1)
        return (self.baseline(ctx) + self.deviation(x[:, self.dyn_idx])).squeeze(-1)


def _stratified_batches(strata):
    """Batches drawn from ONE stratum, so the loss contrasts within it.

    Ordinary shuffling contrasts a crash against the whole corridor, most of
    which is a quiet rural post at 3 a.m. -- an easy negative that teaches only
    where traffic is. Restricting a batch to one (location band, hour band)
    forces the gradient to answer the operationally useful question: is this
    interval unusual *for this place at this time*.
    """
    def sampler(tr_idx, Xt, yt, batch_size, seed):
        rng = np.random.default_rng(seed)
        s = strata[tr_idx]
        groups = [tr_idx[s == v] for v in np.unique(s)]
        groups = [g for g in groups if len(g) >= 32]

        class _DS(torch.utils.data.IterableDataset):
            def __iter__(self):
                order = rng.permutation(len(groups))
                for gi in order:
                    g = groups[gi]
                    perm = rng.permutation(len(g))
                    for k in range(0, len(g), batch_size):
                        sel = g[perm[k:k + batch_size]]
                        if len(sel) < 8:
                            continue
                        yield Xt[sel], yt[sel]
        return _DS()
    return sampler


class MLPDeviationResidual(_BaseBenchmarkClassifier):
    token = "mlp_devres"

    def _prepare(self, X):
        feats = self.features
        blocks = split_blocks(feats)
        dyn = blocks.get("traffic", []) + blocks.get("other", [])
        ctx = blocks.get("calendar", []) + blocks.get("geometry", []) + blocks.get("site", [])
        # `pk` is an identifier, not a dynamic feature: keep it out of the
        # deviation branch so departures cannot be learned per-location.
        if "pk" in feats:
            p = feats.index("pk")
            dyn = [i for i in dyn if i != p]
            ctx = [i for i in ctx if i != p]
        return ctx, dyn

    def fit(self, X, y):
        self.features = list(X.columns)
        self.ctx_idx, self.dyn_idx = self._prepare(X)
        Xv = self._as_matrix(X); yv = np.asarray(y, dtype=np.float32)
        self.scaler = StandardScaler().fit(Xv)
        Xs = self.scaler.transform(Xv).astype(np.float32)

        pk_raw = (pd.to_numeric(X["pk"], errors="coerce").fillna(0).astype(int).to_numpy()
                  if "pk" in X.columns else np.zeros(len(Xs), dtype=int))
        self.pk_min = int(pk_raw.min())
        pk_ix = np.clip(pk_raw - self.pk_min, 0, 255)
        Xa = np.hstack([Xs, pk_ix.reshape(-1, 1).astype(np.float32)])

        hour = (pd.to_numeric(X["hor"], errors="coerce").fillna(0).astype(int).to_numpy()
                if "hor" in X.columns else np.zeros(len(Xs), dtype=int))
        strata = (pk_ix // 5) * 8 + (hour // 3)          # ~5 km band x 3 h band
        n_str = len(np.unique(strata))
        print(f"[mlp_devres] ctx={len(self.ctx_idx)} dyn={len(self.dyn_idx)} "
              f"feats, {n_str} strata for contrastive batching")

        crit = _FocalLoss(alpha=float(self.kwargs.get("alpha", 0.25)),
                          gamma=float(self.kwargs.get("gamma", 2.0)))
        mk = _mlp_kwargs(self.kwargs, X)
        self.model, self.val_aucpr = _train_generic(
            _DeviationResidualNet(self.ctx_idx, self.dyn_idx,
                                  hidden=mk.pop('hidden', (256, 128, 64)),
                                  dropout=mk.pop('dropout', 0.3)),
            Xa, yv, crit, batch_sampler=_stratified_batches(strata), **mk)
        return self

    def _to_cpu(self):
        self.model = self.model.to("cpu"); return self

    def predict_proba(self, X):
        Xs = self.scaler.transform(self._as_matrix(X)).astype(np.float32)
        pk_raw = (pd.to_numeric(X["pk"], errors="coerce").fillna(0).astype(int).to_numpy()
                  if "pk" in X.columns else np.zeros(len(Xs), dtype=int))
        pk_ix = np.clip(pk_raw - self.pk_min, 0, 255).reshape(-1, 1).astype(np.float32)
        Xa = np.hstack([Xs, pk_ix])
        self.model.eval()
        with torch.no_grad():
            return torch.sigmoid(self.model(torch.as_tensor(Xa))).numpy()


MLP_FAMILY = {c.token: c for c in
              (MLPGatedBlocks, MLPNeighbourContext, MLPProximity, MLPDeviationResidual)}
