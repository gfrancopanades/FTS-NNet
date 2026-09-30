"""
End-to-end crash-predictor benchmarks for the 2nd-article comparison (Layer 3)
==============================================================================

These replace the ENTIRE proposed pipeline (Phases 1-3) with a single model trained
end-to-end on crash labels directly from sequences of reconstructed corridor
state — bypassing the forecasting -> classification decomposition.

    BL-E0  TransformerCrashPredictor  — transformer encoder over the state
            window -> crash probability (Abdel-Aty et al., 2024).
    BL-E1  MSGNNCrashPredictor        — multi-structured GNN: spatial-adjacency
            graph + data-driven similarity graph, fused -> crash probability
            (Tran et al., 2023, simplified to two structures for the AP-7).

(BL-E2 GeoLSTM + Phase-3 cascade is just the existing cascade run on the v62
GeoLSTM GNN dir — no model class needed.)

Both are LightningModules with a uniform interface:
    model(x, pk_ids, static_features, sen_directions) -> logits (batch,)
trained with class-weighted BCE and early-stopped on validation AUCPR. The
state window `x` is the raw (Phase-1) corridor state, NOT GNN forecasts.

Author: Gerard Franco
Date:   June 2026
"""

from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import average_precision_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pytorch_lightning as pl
from torch_geometric.nn import GCNConv


class _BaseEnd2End(pl.LightningModule):
    """Shared scaffold: location embedding, class-weighted BCE, val AUCPR."""

    def __init__(self, input_size, num_pks, static_feature_dim, *,
                 pk_embed_dim=16, learning_rate=3e-4, weight_decay=1e-5,
                 pos_weight=1.0, dropout=0.1, **kwargs):
        super().__init__()
        self.input_size = input_size
        self.num_pks = num_pks
        self.static_feature_dim = static_feature_dim
        self.pk_embed_dim = pk_embed_dim
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.dropout_p = dropout
        self.register_buffer("pos_weight", torch.tensor([float(pos_weight)]))
        self.pk_embedding = nn.Embedding(num_pks, pk_embed_dim)
        self.static_projection = nn.Linear(static_feature_dim, pk_embed_dim)
        self.dropout = nn.Dropout(dropout)
        self._val_p, self._val_y = [], []
        self.save_hyperparameters()

    def _loc_ctx(self, pk_ids, static_features, seq_len):
        ctx = torch.cat([self.pk_embedding(pk_ids),
                         self.static_projection(static_features)], dim=-1)
        return ctx.unsqueeze(1).expand(-1, seq_len, -1)

    def build_highway_graph(self, pk_ids, sen_directions, device):
        b = pk_ids.shape[0]
        nbr = torch.where(sen_directions == 0, pk_ids - 1, pk_ids + 1)
        matches = pk_ids.unsqueeze(1) == nbr.unsqueeze(0)
        s, t = matches.nonzero(as_tuple=True)
        if s.numel() > 0:
            return torch.stack([torch.cat([s, t]), torch.cat([t, s])])
        idx = torch.arange(b, device=device)
        return torch.stack([idx, idx])

    def _loss(self, logits, y):
        return F.binary_cross_entropy_with_logits(logits, y, pos_weight=self.pos_weight)

    def training_step(self, batch, _):
        x, y, pk, st, sen = batch
        loss = self._loss(self(x, pk, st, sen), y.float())
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def on_validation_epoch_start(self):
        self._val_p, self._val_y = [], []

    def validation_step(self, batch, _):
        x, y, pk, st, sen = batch
        logits = self(x, pk, st, sen)
        loss = self._loss(logits, y.float())
        self._val_p.append(torch.sigmoid(logits).detach().cpu())
        self._val_y.append(y.detach().cpu())
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        if not self._val_p:
            return
        p = torch.cat(self._val_p).numpy(); y = torch.cat(self._val_y).numpy()
        aucpr = average_precision_score(y, p) if y.sum() > 0 else 0.0
        self.log("val_aucpr", aucpr, prog_bar=True)
        self._val_p, self._val_y = [], []

    def configure_optimizers(self):
        opt = optim.AdamW(self.parameters(), lr=self.learning_rate,
                          weight_decay=self.weight_decay)
        sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3)
        return {"optimizer": opt,
                "lr_scheduler": {"scheduler": sched, "monitor": "val_aucpr"}}

    @torch.no_grad()
    def predict_proba(self, x, pk, st, sen):
        self.eval()
        return torch.sigmoid(self(x, pk, st, sen)).cpu().numpy()


class _PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


# =============================================================================
# BL-E0 — Transformer end-to-end crash predictor (Abdel-Aty et al., 2024)
# =============================================================================

class TransformerCrashPredictor(_BaseEnd2End):
    architecture = "e2e_transformer"

    def __init__(self, input_size, num_pks, static_feature_dim, *,
                 d_model=128, n_heads=8, n_layers=4, dim_ff=512, **kwargs):
        super().__init__(input_size, num_pks, static_feature_dim, **kwargs)
        self.input_proj = nn.Linear(input_size + 2 * self.pk_embed_dim, d_model)
        self.pos_enc = _PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_ff,
            dropout=self.dropout_p, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.head = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.ReLU(),
                                  nn.Dropout(self.dropout_p), nn.Linear(d_model // 2, 1))

    def forward(self, x, pk_ids, static_features, sen_directions=None):
        seq_len = x.shape[1]
        h = self.input_proj(torch.cat([x, self._loc_ctx(pk_ids, static_features, seq_len)], dim=-1))
        h = self.pos_enc(h)
        h = self.encoder(h)
        return self.head(h[:, -1, :]).squeeze(-1)


# =============================================================================
# BL-E3 — LSTM end-to-end crash predictor: the literature-standard deep
# real-time crash prediction baseline (stacked LSTM over the raw state window,
# crash logit from the last hidden state). Same scaffold/loss/protocol as
# BL-E0/E1 so the contrast isolates the architecture family alone.
# =============================================================================

class LSTMCrashPredictor(_BaseEnd2End):
    architecture = "e2e_lstm"

    def __init__(self, input_size, num_pks, static_feature_dim, *,
                 hidden_size=128, num_layers=2, **kwargs):
        super().__init__(input_size, num_pks, static_feature_dim, **kwargs)
        self.lstm = nn.LSTM(input_size + 2 * self.pk_embed_dim, hidden_size,
                            num_layers, batch_first=True,
                            dropout=self.dropout_p if num_layers > 1 else 0)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.head = nn.Sequential(nn.Linear(hidden_size, hidden_size // 2), nn.ReLU(),
                                  nn.Dropout(self.dropout_p),
                                  nn.Linear(hidden_size // 2, 1))

    def forward(self, x, pk_ids, static_features, sen_directions=None):
        seq_len = x.shape[1]
        h, _ = self.lstm(torch.cat(
            [x, self._loc_ctx(pk_ids, static_features, seq_len)], dim=-1))
        return self.head(self.layer_norm(h[:, -1, :])).squeeze(-1)


# =============================================================================
# BL-E1 — MSGNN end-to-end crash predictor (Tran et al., 2023)
# =============================================================================

class MSGNNCrashPredictor(_BaseEnd2End):
    architecture = "e2e_msgnn"

    def __init__(self, input_size, num_pks, static_feature_dim, *,
                 hidden_size=128, sim_topk=8, **kwargs):
        super().__init__(input_size, num_pks, static_feature_dim, **kwargs)
        self.sim_topk = int(sim_topk)
        self.encoder = nn.GRU(input_size + 2 * self.pk_embed_dim, hidden_size,
                              num_layers=2, batch_first=True,
                              dropout=self.dropout_p)
        self.gc_spatial = GCNConv(hidden_size, hidden_size)
        self.gc_similar = GCNConv(hidden_size, hidden_size)
        self.head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size), nn.ReLU(),
            nn.Dropout(self.dropout_p), nn.Linear(hidden_size, 1))

    def _similarity_edges(self, h):
        # cosine-similarity graph over the batch: each node -> its top-k peers.
        hn = F.normalize(h, dim=-1)
        sim = hn @ hn.t()
        b = h.shape[0]
        k = min(self.sim_topk + 1, b)
        idx = sim.topk(k, dim=1).indices  # includes self
        src = torch.arange(b, device=h.device).unsqueeze(1).expand(-1, k).reshape(-1)
        dst = idx.reshape(-1)
        return torch.stack([src, dst])

    def forward(self, x, pk_ids, static_features, sen_directions=None):
        seq_len = x.shape[1]
        device = x.device
        enc, _ = self.encoder(torch.cat([x, self._loc_ctx(pk_ids, static_features, seq_len)], dim=-1))
        h = enc[:, -1, :]
        e_spatial = (self.build_highway_graph(pk_ids, sen_directions, device)
                     if sen_directions is not None
                     else torch.stack([torch.arange(h.shape[0], device=device)] * 2))
        e_similar = self._similarity_edges(h)
        z_sp = F.relu(self.gc_spatial(h, e_spatial))
        z_si = F.relu(self.gc_similar(h, e_similar))
        return self.head(self.dropout(torch.cat([z_sp, z_si], dim=-1))).squeeze(-1)


END2END_REGISTRY = {
    TransformerCrashPredictor.architecture: TransformerCrashPredictor,
    MSGNNCrashPredictor.architecture: MSGNNCrashPredictor,
    LSTMCrashPredictor.architecture: LSTMCrashPredictor,
}


def build_end2end(architecture: str, **kwargs):
    key = (architecture or "").lower()
    if key not in END2END_REGISTRY:
        raise KeyError(f"Unknown end-to-end architecture {architecture!r}. "
                       f"Known: {sorted(END2END_REGISTRY)}")
    return END2END_REGISTRY[key](**kwargs)
