"""
Crash-classifier benchmarks for the 2nd-article comparison (Layer 2)
=========================================================================

Each model here replaces the proposed Phase-3 predictor-supervisor cascade. They all
receive the IDENTICAL feature vectors u_{i,t} produced by the proposed GNN-LSTM
(Phase 2) — the trainer reuses v51's `stage0_load_data_with_gnn` +
`stage1_feature_prep`, so differences in precision/recall are attributable
purely to classifier design + imbalance strategy.

Uniform interface (so one trainer and one simulation handle all six):
    clf.fit(X_train_df, y_train_series) -> self
    clf.predict_proba(X_df) -> np.ndarray            # P(crash), shape (N,)
    clf.save(output_dir, model_time)
    BenchmarkClassifier.load(output_dir, model_time) -> clf   (token-agnostic)

Models:
    BL-C0  LogRegClassifier      — L2 logistic regression, class_weight balanced.
    BL-C1  RandomForestBench     — 200-tree RF, balanced_subsample.
    BL-C2  XGBSmoteClassifier    — single XGBoost on a SMOTE-oversampled fold.
    BL-C3  BalancedBaggingXGB    — M XGBoost members on balanced subsets
                                   (= Stage-1 alone, no Supervisor).
    BL-C4  TwoStageMLP           — Jin et al. 2023 two-stage DL cascade
                                   (no OOF / no threshold alignment).
    BL-C5  MLPFocal              — MLP trained end-to-end with focal loss.

The whole object is joblib-pickled on save (torch members are moved to CPU
first), so load needs no architecture metadata.

Author: Gerard Franco
Date:   June 2026
"""

from __future__ import annotations

import os

import joblib
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

import xgboost as xgb

# Seed for every fitted classifier. AP7_CLF_SEED lets the replication study
# of Section 6.2 refit an identical configuration under a different draw;
# unset, it is the 42 every reported run used.
RANDOM_STATE = int(os.environ.get("AP7_CLF_SEED", "42"))
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================================
# Shared base + persistence
# =============================================================================


def _mlp_kwargs(kw, X=None):
    """Translate searched hyperparameters into `_train_mlp` arguments.

    The neural arms previously hardcoded hidden=(256,128,64), dropout=0.3 and
    lr=3e-4, so a Bayesian search over those had nothing to act on. Anything not
    supplied falls back to exactly the previous default, so an unsearched run is
    byte-for-byte the old behaviour.
    """
    out = {}
    if X is not None:
        d = day_index(X)
        if d is not None:
            out["days"] = d
    h = kw.get("hidden")
    if h is not None:
        out["hidden"] = tuple(int(x) for x in str(h).split(",")) if isinstance(h, str) else tuple(h)
    for src, dst in (("dropout", "dropout"), ("lr", "lr"),
                     ("batch_size", "batch_size"), ("max_epochs", "max_epochs")):
        if kw.get(src) is not None:
            out[dst] = kw[src]
    return out


class _BaseBenchmarkClassifier:
    token = "base"

    def __init__(self, **kwargs):
        self.features: list[str] | None = None
        self.kwargs = kwargs

    def _as_matrix(self, X) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            cols = self.features if self.features is not None else list(X.columns)
            for c in cols:
                if c not in X.columns:
                    X[c] = 0.0
            return X[cols].apply(pd.to_numeric, errors="coerce").fillna(0).values.astype(np.float32)
        return np.asarray(X, dtype=np.float32)

    # torch members (if any) -> CPU so the pickle is portable.
    def _to_cpu(self):
        return self

    def fit(self, X, y):  # pragma: no cover - overridden
        raise NotImplementedError

    def predict_proba(self, X) -> np.ndarray:  # pragma: no cover - overridden
        raise NotImplementedError

    def save(self, output_dir: str, model_time: str) -> str:
        self._to_cpu()
        path = os.path.join(output_dir, f"benchmark-classifier_version={model_time}.joblib")
        joblib.dump(self, path)
        return path


def load_benchmark_classifier(output_dir: str, model_time: str):
    """Token-agnostic loader (the pickle carries its own class)."""
    return joblib.load(os.path.join(
        output_dir, f"benchmark-classifier_version={model_time}.joblib"))


# Back-compat alias used by the simulation loader.
class BenchmarkClassifier:
    load = staticmethod(load_benchmark_classifier)


# =============================================================================
# Torch helpers (focal loss + MLP + a compact training loop)
# =============================================================================

class _FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha, self.gamma = float(alpha), float(gamma)

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        pt = torch.where(targets >= 0.5, p, 1 - p).clamp(1e-6, 1.0)
        at = torch.where(targets >= 0.5, torch.as_tensor(self.alpha, device=logits.device),
                         torch.as_tensor(1 - self.alpha, device=logits.device))
        return (at * (1 - pt) ** self.gamma * bce).mean()


class _MLP(nn.Module):
    """3 hidden layers (256-128-64), BatchNorm + ReLU + dropout."""

    def __init__(self, n_features, hidden=(256, 128, 64), dropout=0.3):
        super().__init__()
        layers, prev = [], n_features
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def day_index(X):
    """Integer yyyymmdd per row, rebuilt from the calendar columns.

    The classifiers receive engineered features rather than `dat`, so the day is
    reconstructed. Returns None when the columns are absent, and every caller
    then degrades to the positional split rather than inventing an ordering.
    """
    need = ("anyo", "mes", "dia")
    cols = getattr(X, "columns", None)
    if cols is None or not all(c in cols for c in need):
        return None
    import pandas as _pd
    return (_pd.to_numeric(X["anyo"], errors="coerce").fillna(0).astype(np.int64) * 10000
            + _pd.to_numeric(X["mes"], errors="coerce").fillna(0).astype(np.int64) * 100
            + _pd.to_numeric(X["dia"], errors="coerce").fillna(0).astype(np.int64)).to_numpy()


def _chrono_split(n, val_frac=0.2, days=None):
    """Train/validation split for early stopping and hyperparameter selection.

    `days` is an array of yyyymmdd integers, one per row. When supplied, the
    split holds out the LATEST whole days -- which is what "chronological" has
    always meant in this codebase's docstrings and in Section 3.5.1.

    Without it the split is positional, and that was the bug this argument
    exists to fix. The assembled frame is sorted location-major
    (`via, sen, pk, ...` -- see `ablation_study_v51_xgboost_only`), so slicing
    the last 20 % of ROWS held out the northern 40 kilometre posts in one
    carriageway: 40 of 200 directional segments, zero segment overlap with
    training, an identical date range, and a 0.026 % crash rate against the
    corridor's 0.19 %. Models were therefore early-stopped, and hyperparameters
    selected, on a spatial holdout roughly nine times rarer than the task being
    scored -- with ~159 positives, an AUPRC standard error near +/-0.04, wider
    than most differences the search was choosing between.
    """
    if days is None:
        cut = max(1, min(n - 1, int(round(n * (1 - val_frac)))))
        return np.arange(cut), np.arange(cut, n)
    days = np.asarray(days)
    if days.shape[0] != n:
        raise ValueError(
            "_chrono_split: days has %d entries but X has %d rows -- the caller "
            "must subset the day array whenever it subsets X (e.g. stage-2 "
            "training on a stage-1 alarm mask)." % (days.shape[0], n))
    uniq = np.unique(days)
    if uniq.size < 3:                       # too few days to hold any out
        cut = max(1, min(n - 1, int(round(n * (1 - val_frac)))))
        return np.arange(cut), np.arange(cut, n)
    # take whole days from the end until val_frac of ROWS is reached
    counts = {d: int((days == d).sum()) for d in uniq}
    target, acc, held = n * val_frac, 0, []
    for d in uniq[::-1]:
        held.append(d); acc += counts[d]
        if acc >= target:
            break
    mask = np.isin(days, held)
    tr, va = np.flatnonzero(~mask), np.flatnonzero(mask)
    if va.size == 0 or tr.size == 0:
        cut = max(1, min(n - 1, int(round(n * (1 - val_frac)))))
        return np.arange(cut), np.arange(cut, n)
    return tr, va


def _train_mlp(Xtr, ytr, *, criterion, hidden=(256, 128, 64), dropout=0.3,
               lr=3e-4, max_epochs=60, patience=8, batch_size=4096,
               seed=RANDOM_STATE, device=_DEVICE, days=None):
    """Train an MLP; early-stop on a chronological-tail validation AUCPR.
    Returns the CPU model."""
    torch.manual_seed(seed); np.random.seed(seed)
    tr_idx, va_idx = _chrono_split(len(Xtr), days=days)
    Xt = torch.as_tensor(Xtr, dtype=torch.float32)
    yt = torch.as_tensor(ytr, dtype=torch.float32)
    model = _MLP(Xtr.shape[1], hidden, dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    ds = torch.utils.data.TensorDataset(Xt[tr_idx], yt[tr_idx])
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True,
                                     generator=torch.Generator().manual_seed(seed))
    Xva = Xt[va_idx].to(device); yva = ytr[va_idx]
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
        auc = average_precision_score(yva, p) if yva.sum() > 0 else 0.0
        if auc > best_auc:
            best_auc, no_imp = auc, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_imp += 1
            if no_imp >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model.to("cpu").eval(), float(best_auc)


# =============================================================================
# BL-C0 — Logistic Regression
# =============================================================================

# saga is stochastic and converges slowly on poorly conditioned folds. The
# BL-C0 fit is 5 C-values x 3 TimeSeriesSplit folds = 16 fits, so an
# unconverged fold multiplies by 16: the July-2025 rolling fold ran 24 h and
# timed out where June, May and August each finished in 6-21 min. Capping the
# iteration budget and relaxing the tolerance bounds the worst case; both are
# env-overridable so the cap is auditable rather than silent.
LOGREG_MAX_ITER = int(os.environ.get("AP7_LOGREG_MAX_ITER", "1000"))
LOGREG_TOL = float(os.environ.get("AP7_LOGREG_TOL", "1e-3"))


class LogRegClassifier(_BaseBenchmarkClassifier):
    token = "logreg"

    def fit(self, X, y):
        self.features = list(X.columns)
        Xv = self._as_matrix(X)
        yv = np.asarray(y, dtype=int)
        self.scaler = StandardScaler().fit(Xv)
        Xs = self.scaler.transform(Xv)
        C_grid = self.kwargs.get("C_grid", (0.001, 0.01, 0.1, 1, 10))
        best_C, best_auc = 1.0, -np.inf
        tscv = TimeSeriesSplit(n_splits=3)
        for C in C_grid:
            aucs = []
            for tr, va in tscv.split(Xs):
                if yv[tr].sum() == 0 or yv[va].sum() == 0:
                    continue
                m = LogisticRegression(C=C, class_weight="balanced", solver="saga",
                                       max_iter=LOGREG_MAX_ITER, tol=LOGREG_TOL,
                                       n_jobs=-1)
                m.fit(Xs[tr], yv[tr])
                aucs.append(average_precision_score(yv[va], m.predict_proba(Xs[va])[:, 1]))
            score = float(np.mean(aucs)) if aucs else -np.inf
            if score > best_auc:
                best_auc, best_C = score, C
        self.C = best_C
        self.cv_aucpr = best_auc
        self.model = LogisticRegression(C=best_C, class_weight="balanced", solver="saga",
                                        max_iter=LOGREG_MAX_ITER, tol=LOGREG_TOL,
                                        n_jobs=-1).fit(Xs, yv)
        return self

    def predict_proba(self, X):
        Xs = self.scaler.transform(self._as_matrix(X))
        return self.model.predict_proba(Xs)[:, 1]


# =============================================================================
# BL-C1 — Random Forest (balanced_subsample)
# =============================================================================

class RandomForestBench(_BaseBenchmarkClassifier):
    token = "random_forest"

    def fit(self, X, y):
        self.features = list(X.columns)
        Xv = self._as_matrix(X); yv = np.asarray(y, dtype=int)
        grid = self.kwargs.get("grid", [
            {"max_depth": 12, "min_samples_leaf": 5, "max_features": "sqrt"},
            {"max_depth": 16, "min_samples_leaf": 5, "max_features": 0.3},
            {"max_depth": None, "min_samples_leaf": 10, "max_features": "sqrt"},
        ])
        tscv = TimeSeriesSplit(n_splits=3)
        best_p, best_auc = grid[0], -np.inf
        for params in grid:
            aucs = []
            for tr, va in tscv.split(Xv):
                if yv[tr].sum() == 0 or yv[va].sum() == 0:
                    continue
                m = RandomForestClassifier(n_estimators=200, class_weight="balanced_subsample",
                                           n_jobs=-1, random_state=RANDOM_STATE, **params)
                m.fit(Xv[tr], yv[tr])
                aucs.append(average_precision_score(yv[va], m.predict_proba(Xv[va])[:, 1]))
            score = float(np.mean(aucs)) if aucs else -np.inf
            if score > best_auc:
                best_auc, best_p = score, params
        self.params = best_p
        self.cv_aucpr = best_auc
        self.model = RandomForestClassifier(n_estimators=200, class_weight="balanced_subsample",
                                            n_jobs=-1, random_state=RANDOM_STATE,
                                            **best_p).fit(Xv, yv)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(self._as_matrix(X))[:, 1]


# =============================================================================
# BL-C2 — Single-stage XGBoost + SMOTE
# =============================================================================

class XGBSmoteClassifier(_BaseBenchmarkClassifier):
    token = "xgb_smote"

    def fit(self, X, y):
        from imblearn.over_sampling import SMOTE
        self.features = list(X.columns)
        Xv = self._as_matrix(X); yv = np.asarray(y, dtype=int)
        # SMOTE applies ONLY to the training data; validation stays unbalanced.
        sm = SMOTE(random_state=RANDOM_STATE)
        Xr, yr = sm.fit_resample(Xv, yv)
        params = self.kwargs.get("xgb_params", dict(
            max_depth=6, learning_rate=0.05, n_estimators=400, subsample=0.8,
            colsample_bytree=0.8, eval_metric="aucpr", tree_method="hist",
            n_jobs=-1, random_state=RANDOM_STATE))
        self.model = xgb.XGBClassifier(**params)
        self.model.fit(Xr, yr)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(self._as_matrix(X))[:, 1]


# =============================================================================
# BL-C3 — Balanced-bagging XGBoost (Stage-1 alone, no Supervisor)
# =============================================================================

class BalancedBaggingXGB(_BaseBenchmarkClassifier):
    token = "balanced_bagging_xgb"

    def fit(self, X, y):
        self.features = list(X.columns)
        Xv = self._as_matrix(X); yv = np.asarray(y, dtype=int)
        n_members = int(self.kwargs.get("n_members", 10))
        neg_pos_ratio = int(self.kwargs.get("neg_pos_ratio", 5))
        params = self.kwargs.get("xgb_params", dict(
            max_depth=6, eta=0.05, subsample=0.8, colsample_bytree=0.8,
            objective="binary:logistic", eval_metric="aucpr",
            tree_method="hist", seed=RANDOM_STATE))
        rounds = int(self.kwargs.get("num_boost_round", 300))
        pos_ix = np.where(yv == 1)[0]
        neg_ix = np.where(yv == 0)[0]
        rng = np.random.RandomState(RANDOM_STATE)
        self.boosters = []
        for _ in range(n_members):
            samp_neg = rng.choice(neg_ix, size=min(len(neg_ix), len(pos_ix) * neg_pos_ratio),
                                  replace=False)
            bag = np.concatenate([pos_ix, samp_neg]); rng.shuffle(bag)
            d = xgb.DMatrix(Xv[bag], label=yv[bag])
            self.boosters.append(xgb.train(params, d, num_boost_round=rounds,
                                           verbose_eval=False))
        return self

    def predict_proba(self, X):
        d = xgb.DMatrix(self._as_matrix(X))
        p = np.zeros(d.num_row())
        for b in self.boosters:
            p += b.predict(d)
        return p / max(len(self.boosters), 1)


# =============================================================================
# BL-C4 — Two-stage deep-learning cascade (Jin et al., 2023)
# =============================================================================

class TwoStageMLP(_BaseBenchmarkClassifier):
    token = "two_stage_mlp"

    def fit(self, X, y):
        self.features = list(X.columns)
        Xv = self._as_matrix(X); yv = np.asarray(y, dtype=np.float32)
        self.scaler = StandardScaler().fit(Xv)
        Xs = self.scaler.transform(Xv).astype(np.float32)

        # Stage 1: class-weighted BCE filter on all rows.
        n_pos = max(int(yv.sum()), 1); n_neg = max(int((yv == 0).sum()), 1)
        s1_crit = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([n_neg / n_pos], dtype=torch.float32, device=_DEVICE))
        self.stage1, _ = _train_mlp(Xs, yv, criterion=s1_crit,
            **_mlp_kwargs(self.kwargs, X))

        # Stage 1 scores -> alarms (threshold 0.5, NOT swept — reproduces the
        # distribution-mismatch the proposed alignment step is designed to fix).
        with torch.no_grad():
            s1p = torch.sigmoid(self.stage1(torch.as_tensor(Xs))).numpy()
        alarm = s1p >= 0.5
        self.s1_gate = 0.5
        if alarm.sum() > 0 and yv[alarm].sum() > 0 and (yv[alarm] == 0).sum() > 0:
            Xs2, ys2 = Xs[alarm], yv[alarm]
            n_pos2 = max(int(ys2.sum()), 1); n_neg2 = max(int((ys2 == 0).sum()), 1)
            s2_crit = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor([n_neg2 / n_pos2], dtype=torch.float32, device=_DEVICE))
            self.stage2, _ = _train_mlp(Xs2, ys2, criterion=s2_crit,
            **_mlp_kwargs(self.kwargs, X[alarm]))
        else:
            self.stage2 = None
        return self

    def _to_cpu(self):
        self.stage1 = self.stage1.to("cpu")
        if self.stage2 is not None:
            self.stage2 = self.stage2.to("cpu")
        return self

    def predict_proba(self, X):
        Xs = self.scaler.transform(self._as_matrix(X)).astype(np.float32)
        self.stage1.eval()
        with torch.no_grad():
            s1 = torch.sigmoid(self.stage1(torch.as_tensor(Xs))).numpy()
        out = s1.copy()
        if self.stage2 is not None:
            alarm = s1 >= self.s1_gate
            if alarm.sum() > 0:
                self.stage2.eval()
                with torch.no_grad():
                    s2 = torch.sigmoid(self.stage2(torch.as_tensor(Xs[alarm]))).numpy()
                # final score on alarms = stage-2; non-alarms keep low stage-1 score
                out[alarm] = s2
        return out


# =============================================================================
# BL-C5 — MLP with focal loss
# =============================================================================

class MLPFocal(_BaseBenchmarkClassifier):
    token = "mlp_focal"

    def fit(self, X, y):
        self.features = list(X.columns)
        Xv = self._as_matrix(X); yv = np.asarray(y, dtype=np.float32)
        self.scaler = StandardScaler().fit(Xv)
        Xs = self.scaler.transform(Xv).astype(np.float32)
        gamma = float(self.kwargs.get("gamma", 2.0))
        alpha = float(self.kwargs.get("alpha", 0.25))
        self.model, self.val_aucpr = _train_mlp(
            Xs, yv, criterion=_FocalLoss(alpha=alpha, gamma=gamma),
            **_mlp_kwargs(self.kwargs, X))
        return self

    def _to_cpu(self):
        self.model = self.model.to("cpu")
        return self

    def predict_proba(self, X):
        Xs = self.scaler.transform(self._as_matrix(X)).astype(np.float32)
        self.model.eval()
        with torch.no_grad():
            return torch.sigmoid(self.model(torch.as_tensor(Xs))).numpy()


# =============================================================================
# BL-C6 — One-stage GBDT (single class-weighted XGBoost, no cascade)
# =============================================================================

class OneStageGBDT(_BaseBenchmarkClassifier):
    token = "gbdt_one_stage"

    def fit(self, X, y):
        self.features = list(X.columns)
        Xv = self._as_matrix(X); yv = np.asarray(y, dtype=int)
        n_pos = max(int(yv.sum()), 1); n_neg = max(int((yv == 0).sum()), 1)
        params = self.kwargs.get("xgb_params", dict(
            max_depth=6, learning_rate=0.05, n_estimators=400, subsample=0.8,
            colsample_bytree=0.8, eval_metric="aucpr", tree_method="hist",
            scale_pos_weight=n_neg / n_pos, n_jobs=-1, random_state=RANDOM_STATE))
        self.model = xgb.XGBClassifier(**params)
        self.model.fit(Xv, yv)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(self._as_matrix(X))[:, 1]


# =============================================================================
# BL-C7 — Two-stage GBDT cascade (XGB screener -> XGB refiner). Structure
# mirrors BL-C4's two-stage recipe (fixed 0.5 gate, NOT swept) so the contrast
# with the proposed cascade isolates the proposed stage-1 alignment + supervisor
# choice rather than the mere presence of two stages.
# =============================================================================

class TwoStageGBDT(_BaseBenchmarkClassifier):
    token = "gbdt_two_stage"

    def _make_xgb(self, scale_pos_weight):
        params = dict(self.kwargs.get("xgb_params", dict(
            max_depth=6, learning_rate=0.05, n_estimators=400, subsample=0.8,
            colsample_bytree=0.8, eval_metric="aucpr", tree_method="hist",
            n_jobs=-1, random_state=RANDOM_STATE)))
        params.setdefault("scale_pos_weight", scale_pos_weight)
        return xgb.XGBClassifier(**params)

    def fit(self, X, y):
        self.features = list(X.columns)
        Xv = self._as_matrix(X); yv = np.asarray(y, dtype=int)
        # Stage 1: class-weighted screener on all rows.
        n_pos = max(int(yv.sum()), 1); n_neg = max(int((yv == 0).sum()), 1)
        self.stage1 = self._make_xgb(n_neg / n_pos).fit(Xv, yv)
        s1p = self.stage1.predict_proba(Xv)[:, 1]
        alarm = s1p >= 0.5
        self.s1_gate = 0.5
        # Stage 2: refiner trained on stage-1 alarms only (needs both classes).
        if alarm.sum() > 0 and yv[alarm].sum() > 0 and (yv[alarm] == 0).sum() > 0:
            y2 = yv[alarm]
            n_pos2 = max(int(y2.sum()), 1); n_neg2 = max(int((y2 == 0).sum()), 1)
            self.stage2 = self._make_xgb(n_neg2 / n_pos2).fit(Xv[alarm], y2)
        else:
            self.stage2 = None
        return self

    def predict_proba(self, X):
        Xv = self._as_matrix(X)
        s1 = self.stage1.predict_proba(Xv)[:, 1]
        out = s1.copy()
        if self.stage2 is not None:
            alarm = s1 >= self.s1_gate
            if alarm.sum() > 0:
                # final score on alarms = stage-2; non-alarms keep low stage-1 score
                out[alarm] = self.stage2.predict_proba(Xv[alarm])[:, 1]
        return out


# =============================================================================
# BL-C8 — Two-stage MLP with focal loss (BL-C4's cascade recipe, but both
# stages trained with the focal loss that made BL-C5 the best single MLP).
# Same fixed 0.5 gate as BL-C4/C7 so staging remains the only variable.
# =============================================================================

class TwoStageMLPFocal(_BaseBenchmarkClassifier):
    token = "mlp_focal_two_stage"

    def fit(self, X, y):
        self.features = list(X.columns)
        Xv = self._as_matrix(X); yv = np.asarray(y, dtype=np.float32)
        self.scaler = StandardScaler().fit(Xv)
        Xs = self.scaler.transform(Xv).astype(np.float32)

        # Stage 1: focal-loss filter on all rows.
        self.stage1, _ = _train_mlp(Xs, yv, criterion=_FocalLoss(),
                                    **_mlp_kwargs(self.kwargs, X))
        with torch.no_grad():
            s1p = torch.sigmoid(self.stage1(torch.as_tensor(Xs))).numpy()
        alarm = s1p >= 0.5
        self.s1_gate = 0.5
        # Stage 2: focal-loss refiner on stage-1 alarms (needs both classes).
        if alarm.sum() > 0 and yv[alarm].sum() > 0 and (yv[alarm] == 0).sum() > 0:
            self.stage2, _ = _train_mlp(Xs[alarm], yv[alarm],
                                        criterion=_FocalLoss(),
                                        **_mlp_kwargs(self.kwargs, X[alarm]))
        else:
            self.stage2 = None
        return self

    def _to_cpu(self):
        self.stage1 = self.stage1.to("cpu")
        if self.stage2 is not None:
            self.stage2 = self.stage2.to("cpu")
        return self

    def predict_proba(self, X):
        Xs = self.scaler.transform(self._as_matrix(X)).astype(np.float32)
        self.stage1.eval()
        with torch.no_grad():
            s1 = torch.sigmoid(self.stage1(torch.as_tensor(Xs))).numpy()
        out = s1.copy()
        if self.stage2 is not None:
            alarm = s1 >= self.s1_gate
            if alarm.sum() > 0:
                self.stage2.eval()
                with torch.no_grad():
                    s2 = torch.sigmoid(self.stage2(torch.as_tensor(Xs[alarm]))).numpy()
                out[alarm] = s2
        return out


# =============================================================================
# BL-C9 — GBDT screener -> MLP+Focal false-positive supervisor. the proposed system's own
# stage pattern with the supervisor family swapped: stage 1 = class-weighted
# XGBoost over all rows (cheap, imbalance-robust); stage 2 = focal-loss MLP
# on the alarm subset (small, denser, harder — where a neural boundary pays).
# Fixed 0.5 gate as in BL-C4/C7/C8, so the delta to the proposed cascade
# isolates the GBDT cascade's stage-1 threshold alignment + supervisor choice.
# =============================================================================

class GBDTMLPSupervisor(_BaseBenchmarkClassifier):
    token = "gbdt_mlp_supervisor"

    def fit(self, X, y):
        self.features = list(X.columns)
        Xv = self._as_matrix(X); yv = np.asarray(y, dtype=int)
        # Stage 1: class-weighted GBDT screener on raw features.
        n_pos = max(int(yv.sum()), 1); n_neg = max(int((yv == 0).sum()), 1)
        params = dict(self.kwargs.get("xgb_params", dict(
            max_depth=6, learning_rate=0.05, n_estimators=400, subsample=0.8,
            colsample_bytree=0.8, eval_metric="aucpr", tree_method="hist",
            n_jobs=-1, random_state=RANDOM_STATE)))
        params.setdefault("scale_pos_weight", n_neg / n_pos)
        self.stage1 = xgb.XGBClassifier(**params).fit(Xv, yv)
        s1p = self.stage1.predict_proba(Xv)[:, 1]
        alarm = s1p >= 0.5
        self.s1_gate = 0.5
        # Stage 2: focal MLP supervisor on the alarms (scaled on that subset).
        if alarm.sum() > 0 and yv[alarm].sum() > 0 and (yv[alarm] == 0).sum() > 0:
            self.scaler = StandardScaler().fit(Xv[alarm])
            Xs2 = self.scaler.transform(Xv[alarm]).astype(np.float32)
            self.stage2, _ = _train_mlp(Xs2, yv[alarm].astype(np.float32),
                                        criterion=_FocalLoss())
        else:
            self.scaler, self.stage2 = None, None
        return self

    def _to_cpu(self):
        if self.stage2 is not None:
            self.stage2 = self.stage2.to("cpu")
        return self

    def predict_proba(self, X):
        Xv = self._as_matrix(X)
        s1 = self.stage1.predict_proba(Xv)[:, 1]
        out = s1.copy()
        if self.stage2 is not None:
            alarm = s1 >= self.s1_gate
            if alarm.sum() > 0:
                Xs2 = self.scaler.transform(Xv[alarm]).astype(np.float32)
                self.stage2.eval()
                with torch.no_grad():
                    s2 = torch.sigmoid(self.stage2(torch.as_tensor(Xs2))).numpy()
                out[alarm] = s2
        return out


# =============================================================================
# BL-C10 — Static historical risk (Empirical-Bayes hotspot baseline).
# Scores each (pk, hour, weekday) cell by its EB-shrunk historical crash rate
# computed on the TRAINING rows only — no real-time or forecast features.
# This is the Highway-Safety-Manual hotspot-identification paradigm; the gap
# to every real-time model quantifies the value of day-specific information.
# =============================================================================

class HistoricalRiskEB(_BaseBenchmarkClassifier):
    token = "historical_risk"

    _KEYS = ("pk", "hor", "diaSem")

    def _key_frame(self, X):
        cols = {}
        for k in self.keys:
            v = pd.to_numeric(X[k], errors="coerce").fillna(-1)
            if k == "hor" and self._hor_div > 1:
                v = v // self._hor_div          # 5-min slot index -> hour
            cols[k] = v.astype(int)
        return pd.DataFrame(cols, index=X.index if hasattr(X, "index") else None)

    def fit(self, X, y):
        self.features = list(X.columns)
        yv = np.asarray(y, dtype=float)
        self.keys = [k for k in self._KEYS if k in X.columns]
        if not self.keys:
            raise ValueError("historical_risk needs pk/hor/diaSem in the features")
        # coarsen sub-hourly 'hor' encodings to hours so cells stay populated
        self._hor_div = 1
        if "hor" in self.keys:
            n = int(pd.to_numeric(X["hor"], errors="coerce").nunique())
            if n > 30:
                self._hor_div = max(1, int(round(n / 24)))
        kf = self._key_frame(X); kf["y"] = yv
        g = kf.groupby(self.keys)["y"].agg(["sum", "count"])
        self.p0 = float(yv.mean())
        m = float(self.kwargs.get("shrinkage", 200.0))   # EB prior weight (pseudo-obs)
        self.table = ((g["sum"] + m * self.p0) / (g["count"] + m)).rename("rate")
        return self

    def predict_proba(self, X):
        kf = self._key_frame(X)
        merged = kf.join(self.table, on=self.keys)
        return merged["rate"].fillna(self.p0).to_numpy(dtype=float)


# =============================================================================
# BL-C11 — Logistic regression with per-PK fixed effects (heterogeneity-aware
# logit): the econometric nod to unobserved location heterogeneity. Same
# linear model as BL-C0 plus one-hot PK intercepts.
# =============================================================================

class LogRegPKEffects(_BaseBenchmarkClassifier):
    token = "logreg_pk_effects"

    def _design(self, X):
        Xv = self._as_matrix(X)
        pk = pd.to_numeric(pd.DataFrame(X)["pk"], errors="coerce").fillna(-1).astype(int) \
            if "pk" in getattr(X, "columns", []) else pd.Series(-1, index=range(len(Xv)))
        d = pd.get_dummies(pk.astype("category").cat.set_categories(self.pk_cats),
                           dtype=np.float32)
        return np.hstack([Xv, d.to_numpy()])

    def fit(self, X, y):
        self.features = list(X.columns)
        yv = np.asarray(y, dtype=int)
        self.pk_cats = sorted(pd.to_numeric(X["pk"], errors="coerce")
                              .fillna(-1).astype(int).unique()) if "pk" in X.columns else [-1]
        Xd = self._design(X)
        self.scaler = StandardScaler().fit(Xd)
        self.model = LogisticRegression(class_weight="balanced", max_iter=1000,
                                        C=1.0, n_jobs=-1)
        self.model.fit(self.scaler.transform(Xd), yv)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(self.scaler.transform(self._design(X)))[:, 1]


# =============================================================================
# Registry / factory
# =============================================================================

# =============================================================================
# BL-T5 -- Focal-objective GBDT. Section 5.3(e) attributes the neural/tree gap
# to the training objective: the neural arms reshape the loss for a 0.19 %
# positive rate, the tree arms could only be told about imbalance through row
# weights. That is testable, because focal loss is expressible as a custom
# XGBoost objective. This arm carries the same tree family and the same search
# budget as BL-C6 but optimises focal loss instead of a weighted logistic loss,
# and deliberately drops `scale_pos_weight` so the objective is the only
# channel through which imbalance is handled.
#
# The gradient is analytic; the hessian is a central difference of it, which
# avoids a second derivative whose algebra would be one more place to be wrong
# and costs two extra vector evaluations per boosting round.
# =============================================================================

class GBDTFocal(_BaseBenchmarkClassifier):
    token = "gbdt_focal"

    @staticmethod
    def _grad(z, y, alpha, gamma):
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))
        p = np.clip(p, 1e-9, 1.0 - 1e-9)
        pos = (1.0 - p) ** gamma * (gamma * p * np.log(p) - (1.0 - p))
        neg = p ** gamma * (p - gamma * (1.0 - p) * np.log(1.0 - p))
        return np.where(y == 1, alpha * pos, (1.0 - alpha) * neg)

    def fit(self, X, y):
        self.features = list(X.columns)
        Xv = self._as_matrix(X)
        yv = np.asarray(y, dtype=int)
        self.alpha = float(self.kwargs.get("alpha", 0.25))
        self.gamma = float(self.kwargs.get("gamma", 2.0))
        a, g, h = self.alpha, self.gamma, 1e-4

        def obj(predt, dtrain):
            yt = dtrain.get_label()
            grad = self._grad(predt, yt, a, g)
            hess = (self._grad(predt + h, yt, a, g)
                    - self._grad(predt - h, yt, a, g)) / (2.0 * h)
            return grad, np.maximum(hess, 1e-6)

        params = dict(self.kwargs.get("xgb_params", dict(
            max_depth=6, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, tree_method="hist")))
        params.pop("scale_pos_weight", None)      # the point of the arm
        params.pop("eval_metric", None)
        n_rounds = int(params.pop("n_estimators", 400))
        params.update(base_score=0.0, disable_default_eval_metric=1)
        dtrain = xgb.DMatrix(Xv, label=yv)
        self.model = xgb.train(params, dtrain, num_boost_round=n_rounds, obj=obj)
        return self

    def predict_proba(self, X):
        d = xgb.DMatrix(self._as_matrix(X))
        z = self.model.predict(d, output_margin=True)
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


CLASSIFIER_REGISTRY = {
    LogRegClassifier.token: LogRegClassifier,
    RandomForestBench.token: RandomForestBench,
    XGBSmoteClassifier.token: XGBSmoteClassifier,
    BalancedBaggingXGB.token: BalancedBaggingXGB,
    TwoStageMLP.token: TwoStageMLP,
    MLPFocal.token: MLPFocal,
    OneStageGBDT.token: OneStageGBDT,
    TwoStageGBDT.token: TwoStageGBDT,
    TwoStageMLPFocal.token: TwoStageMLPFocal,
    GBDTMLPSupervisor.token: GBDTMLPSupervisor,
    HistoricalRiskEB.token: HistoricalRiskEB,
    LogRegPKEffects.token: LogRegPKEffects,
    GBDTFocal.token: GBDTFocal,
}


# The problem-specific MLP family (BL-C12..BL-C15) lives in its own module and
# subclasses the base defined above, so it is imported here -- after the registry
# literal -- rather than at the top, which would be a circular import.
try:
    from src.models.benchmark_classifiers_mlp_family import MLP_FAMILY as _MLP_FAMILY
    CLASSIFIER_REGISTRY.update(_MLP_FAMILY)
except Exception as _exc:                      # keep the core registry usable
    print(f"[WARN] MLP family not registered: {type(_exc).__name__}: {_exc}")
try:
    from src.models.benchmark_classifiers_tree_family import TREE_FAMILY as _TREE_FAMILY
    CLASSIFIER_REGISTRY.update(_TREE_FAMILY)
except Exception as _exc:
    print(f"[WARN] tree family not registered: {type(_exc).__name__}: {_exc}")


def build_classifier(token: str, **kwargs) -> _BaseBenchmarkClassifier:
    key = (token or "").lower()
    if key not in CLASSIFIER_REGISTRY:
        raise KeyError(f"Unknown classifier token {token!r}. "
                       f"Known: {sorted(CLASSIFIER_REGISTRY)}")
    return CLASSIFIER_REGISTRY[key](**kwargs)
