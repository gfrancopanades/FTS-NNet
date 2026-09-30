"""
Benchmark forecasters for the 2nd-article comparison (Stage-1 traffic forecasting)
===========================================================================

This module holds the Layer-1 *forecasting* benchmarks defined in
`benchmark_models_for_2nd_article.md`. Each one is a drop-in replacement for
`SpatioTemporalGNN_LSTM_V5` (the Phase-2 forecaster): same constructor kwargs,
same call signature

    model(x, pk_ids, static_features, sen_directions) -> (batch, output_size)

and the same training scaffold (per-target weighted SmoothL1, RMSE-tracked
validation, AdamW + ReduceLROnPlateau). Because the signature and the saved
`state_dict` format are identical, the existing
`load_pretrained_gnn_model` (made architecture-aware) reconstructs any of them
from the GNN-experiment-dir metadata, and the frozen Phase-3 cascade + the
rolling simulation consume their forecasts unchanged.

Paradigm note
-------------
The GeoLSTM harness treats each (location, time) window as one sample and builds
the corridor graph *within each minibatch* from co-occurring adjacent PKs
(`build_highway_graph`). The spatio-temporal benchmarks below are therefore
adapted to that paradigm — temporal modelling runs inside each sample's
sequence, spatial mixing runs across the batch graph — and every model emits a
single-step prediction per sample, consistent with the Phase-2 autoregressive
rollout protocol (the doc explicitly asks for this single-step-decoder
adaptation).

Models
------
    BL-F1  GeoLSTMForecaster        — stacked LSTM + static geometry, no graph.
    BL-F2  DCRNNForecaster          — GRU + bidirectional diffusion convolution.
    BL-F3  STGCNForecaster          — gated temporal conv (GLU) + Cheb graph conv.
    BL-F4  GraphWaveNetForecaster   — dilated causal TCN + learned adaptive graph.
    BL-F5  ASTGCNForecaster         — spatial + temporal attention + Cheb conv.

(BL-F0 Historical Average is non-parametric and handled outside this module.)

Author: Gerard Franco
Date:   June 2026
Affiliation: Universitat Politecnica de Catalunya
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pytorch_lightning as pl
from torch_geometric.nn import ChebConv, GCNConv
from torch_geometric.utils import degree

# The proposed method (winner) — registered here so the benchmark roster
# (proposed + BL-F0..F6) lives in one place and the arch-aware loader can build
# it uniformly. This is the existing Phase-2 forecaster, unchanged.
from src.models.spatiotemporal_gnn_lstm_v5 import SpatioTemporalGNN_LSTM_V5


# =============================================================================
# Shared base — replicates the V5 training scaffold + location embedding.
# =============================================================================

class BaseBenchmarkForecaster(pl.LightningModule):
    """Common scaffold for every Layer-1 benchmark forecaster.

    Subclasses only implement `forward`. The constructor accepts the *exact*
    kwarg set that `load_pretrained_gnn_model` passes to the V5 model, so the
    arch-aware loader can build any subclass from the same metadata. Kwargs a
    given architecture does not use (e.g. `graph_hidden_dim` for GeoLSTM) are
    accepted and ignored.
    """

    def __init__(self, input_size, hidden_size, num_layers, output_size,
                 dropout_prob, learning_rate, weight_decay,
                 num_pks, static_feature_dim, pk_embed_dim=16,
                 graph_hidden_dim=64, num_graph_layers=2,
                 target_weights=None, **kwargs):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_pks = num_pks
        self.static_feature_dim = static_feature_dim
        self.pk_embed_dim = pk_embed_dim
        self.graph_hidden_dim = graph_hidden_dim
        self.num_graph_layers = num_graph_layers
        self.dropout_prob = dropout_prob

        if target_weights is not None:
            self.register_buffer("target_weights",
                                 torch.tensor(target_weights, dtype=torch.float32))
        else:
            self.register_buffer("target_weights",
                                 torch.ones(output_size, dtype=torch.float32))

        self.val_predictions = []
        self.val_labels = []

        # Location context (identical to V5): PK embedding + static projection,
        # concatenated to the temporal features at every step.
        self.pk_embedding = nn.Embedding(num_pks, pk_embed_dim)
        self.static_projection = nn.Linear(static_feature_dim, pk_embed_dim)

        self.dropout = nn.Dropout(dropout_prob)
        self.criterion = nn.SmoothL1Loss(reduction="none")
        self.save_hyperparameters()

    # -- location helpers -----------------------------------------------------
    def _location_context(self, pk_ids, static_features, seq_len):
        """Return (batch, seq_len, 2*pk_embed_dim) location context."""
        pk_embed = self.pk_embedding(pk_ids)
        static_embed = self.static_projection(static_features)
        ctx = torch.cat([pk_embed, static_embed], dim=-1)
        return ctx.unsqueeze(1).expand(-1, seq_len, -1)

    def build_highway_graph(self, pk_ids, sen_directions, device):
        """Batch graph from consecutive-PK connectivity (copied from V5).

        sen=0 (dec) links PK_i -> PK_{i-1}; sen=1 (cre) links PK_i -> PK_{i+1}.
        Returns an undirected edge_index over the batch samples.
        """
        batch_size = pk_ids.shape[0]
        neighbor_pks = torch.where(sen_directions == 0, pk_ids - 1, pk_ids + 1)
        matches = pk_ids.unsqueeze(1) == neighbor_pks.unsqueeze(0)
        source_indices, target_indices = matches.nonzero(as_tuple=True)
        if source_indices.numel() > 0:
            edge_index = torch.stack([
                torch.cat([source_indices, target_indices]),
                torch.cat([target_indices, source_indices]),
            ])
        else:
            idx = torch.arange(batch_size, device=device)
            edge_index = torch.stack([idx, idx])
        return edge_index

    # -- training scaffold (identical objective to V5/V14) --------------------
    def _compute_weighted_loss(self, outputs, labels):
        per_element = self.criterion(outputs, labels)
        return (per_element * self.target_weights.unsqueeze(0)).mean()

    def training_step(self, batch, batch_idx):
        inputs, labels, pk_ids, static_features, sen_directions = batch
        labels = labels.view(-1, self.output_size)
        outputs = self(inputs, pk_ids, static_features, sen_directions)
        loss = self._compute_weighted_loss(outputs, labels)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def on_validation_epoch_start(self):
        self.val_predictions = []
        self.val_labels = []

    def validation_step(self, batch, batch_idx):
        inputs, labels, pk_ids, static_features, sen_directions = batch
        labels = labels.view(-1, self.output_size)
        outputs = self(inputs, pk_ids, static_features, sen_directions)
        loss = self._compute_weighted_loss(outputs, labels)
        self.val_predictions.append(outputs.detach())
        self.val_labels.append(labels.detach())
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        if len(self.val_predictions) == 0:
            return
        all_preds = torch.cat(self.val_predictions, dim=0).cpu().numpy()
        all_labels = torch.cat(self.val_labels, dim=0).cpu().numpy()
        val_mae = mean_absolute_error(all_labels, all_preds)
        val_r2 = r2_score(all_labels, all_preds)
        val_rmse = np.sqrt(mean_squared_error(all_labels, all_preds))
        for i, name in enumerate(["speed", "intTot", "intP"]):
            if i < all_preds.shape[1]:
                self.log(f"val_r2_{name}", r2_score(all_labels[:, i], all_preds[:, i]),
                         prog_bar=False)
        self.log("val_mae", val_mae, prog_bar=False)
        self.log("val_r2", val_r2, prog_bar=False)
        self.log("val_rmse", val_rmse, prog_bar=False)
        self.val_predictions = []
        self.val_labels = []

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate,
                                weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-7)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss",
                             "interval": "epoch", "frequency": 1},
        }


# =============================================================================
# Spatial helpers (operate on the per-minibatch batch graph)
# =============================================================================

def _normalized_transitions(edge_index, num_nodes, device):
    """Forward / backward random-walk transition matrices for diffusion conv.

    P_F = D_O^{-1} A , P_B = D_I^{-1} A^T, returned as dense (num_nodes, n).
    The batch graph is small (<= a few thousand nodes) so dense is fine.
    """
    A = torch.zeros((num_nodes, num_nodes), device=device)
    A[edge_index[0], edge_index[1]] = 1.0
    out_deg = A.sum(dim=1, keepdim=True).clamp(min=1.0)
    in_deg = A.sum(dim=0, keepdim=True).clamp(min=1.0)
    P_f = A / out_deg
    P_b = (A.t() / in_deg.t())
    return P_f, P_b


class DiffusionConv(nn.Module):
    """K-step bidirectional diffusion convolution (Li et al., 2018)."""

    def __init__(self, in_dim, out_dim, K=2):
        super().__init__()
        self.K = K
        # supports: identity + K forward + K backward
        self.lin = nn.Linear(in_dim * (2 * K + 1), out_dim)

    def forward(self, h, edge_index):
        n = h.shape[0]
        P_f, P_b = _normalized_transitions(edge_index, n, h.device)
        out = [h]
        xf, xb = h, h
        for _ in range(self.K):
            xf = P_f @ xf
            xb = P_b @ xb
            out.append(xf)
            out.append(xb)
        return self.lin(torch.cat(out, dim=-1))


# =============================================================================
# BL-F1 — GeoLSTM (stacked LSTM + static geometry, no graph convolution)
# =============================================================================

class GeoLSTMForecaster(BaseBenchmarkForecaster):
    """Vanilla stacked LSTM with static geometry embedding (no spatial coupling).

    Direct predecessor architecture: identical to V5 minus the graph message
    passing. The gap to the GNN-LSTM isolates the corridor graph's contribution.
    """

    architecture = "geolstm"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        lstm_input = self.input_size + 2 * self.pk_embed_dim
        self.lstm = nn.LSTM(lstm_input, self.hidden_size, self.num_layers,
                            batch_first=True,
                            dropout=self.dropout_prob if self.num_layers > 1 else 0)
        self.layer_norm = nn.LayerNorm(self.hidden_size)
        self.fc1 = nn.Linear(self.hidden_size, self.hidden_size // 2)
        self.fc2 = nn.Linear(self.hidden_size // 2, self.output_size)

    def forward(self, x, pk_ids, static_features, sen_directions=None):
        _, seq_len, _ = x.shape
        ctx = self._location_context(pk_ids, static_features, seq_len)
        x_combined = torch.cat([x, ctx], dim=-1)
        lstm_out, _ = self.lstm(x_combined)
        feats = self.layer_norm(lstm_out[:, -1, :])
        out = self.dropout(feats)
        out = F.relu(self.fc1(out))
        out = self.dropout(out)
        return self.fc2(out)


# =============================================================================
# BL-F2 — DCRNN (GRU temporal encoder + bidirectional diffusion convolution)
# =============================================================================

class DCRNNForecaster(BaseBenchmarkForecaster):
    """Diffusion Convolutional Recurrent Network (Li et al., 2018), adapted.

    GRU encodes each sample's window; the final hidden state is diffused over
    the batch corridor graph with K-step forward+backward random walks, giving a
    directed spatial receptive field appropriate for dual-carriageway corridors.
    """

    architecture = "dcrnn"

    def __init__(self, *args, diffusion_steps=2, **kwargs):
        super().__init__(*args, **kwargs)
        self.K = int(kwargs.get("diffusion_steps", diffusion_steps))
        gru_input = self.input_size + 2 * self.pk_embed_dim
        self.gru = nn.GRU(gru_input, self.hidden_size, self.num_layers,
                          batch_first=True,
                          dropout=self.dropout_prob if self.num_layers > 1 else 0)
        self.layer_norm = nn.LayerNorm(self.hidden_size)
        self.diffusion = DiffusionConv(self.hidden_size, self.hidden_size, K=self.K)
        self.fc1 = nn.Linear(self.hidden_size, self.hidden_size // 2)
        self.fc2 = nn.Linear(self.hidden_size // 2, self.output_size)

    def forward(self, x, pk_ids, static_features, sen_directions=None):
        batch_size, seq_len, _ = x.shape
        device = x.device
        ctx = self._location_context(pk_ids, static_features, seq_len)
        gru_out, _ = self.gru(torch.cat([x, ctx], dim=-1))
        temporal = self.layer_norm(gru_out[:, -1, :])

        if sen_directions is not None:
            edge_index = self.build_highway_graph(pk_ids, sen_directions, device)
        else:
            idx = torch.arange(batch_size, device=device)
            edge_index = torch.stack([idx, idx])

        spatial = F.relu(self.diffusion(temporal, edge_index))
        spatial = self.dropout(spatial)
        combined = temporal + spatial  # residual
        out = F.relu(self.fc1(self.dropout(combined)))
        out = self.dropout(out)
        return self.fc2(out)


# =============================================================================
# BL-F3 — STGCN (gated temporal convolution + Chebyshev graph convolution)
# =============================================================================

class _TemporalGatedConv(nn.Module):
    """1D causal gated convolution with GLU activation over the time axis."""

    def __init__(self, in_ch, out_ch, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        self.pad = kernel_size - 1  # left pad -> causal
        self.conv = nn.Conv1d(in_ch, 2 * out_ch, kernel_size)

    def forward(self, x):  # x: (batch, in_ch, T)
        x = F.pad(x, (self.pad, 0))
        p, q = self.conv(x).chunk(2, dim=1)
        return p * torch.sigmoid(q)


class STGCNForecaster(BaseBenchmarkForecaster):
    """Spatio-Temporal Graph Convolutional Network (Yu et al., 2018), adapted.

    Two ST-Conv blocks: gated temporal conv -> Cheb spatial conv (over the batch
    graph) -> gated temporal conv. Purely convolutional — no recurrent state —
    so it stress-tests error compounding under the long rollout.
    """

    architecture = "stgcn"

    def __init__(self, *args, cheb_k=3, temporal_kernel=3, **kwargs):
        super().__init__(*args, **kwargs)
        self.cheb_k = int(kwargs.get("cheb_k", cheb_k))
        kt = int(kwargs.get("temporal_kernel", temporal_kernel))
        c_in = self.input_size + 2 * self.pk_embed_dim
        c1 = self.graph_hidden_dim
        c2 = self.hidden_size
        # Block 1
        self.t1a = _TemporalGatedConv(c_in, c1, kt)
        self.gc1 = ChebConv(c1, c1, K=self.cheb_k)
        self.t1b = _TemporalGatedConv(c1, c1, kt)
        # Block 2
        self.t2a = _TemporalGatedConv(c1, c2, kt)
        self.gc2 = ChebConv(c2, c2, K=self.cheb_k)
        self.t2b = _TemporalGatedConv(c2, c2, kt)
        self.layer_norm = nn.LayerNorm(c2)
        self.fc = nn.Linear(c2, self.output_size)

    def _spatial(self, gc, h_bt, edge_index, T):
        # h_bt: (batch, C, T) -> apply ChebConv once over all T timesteps at once,
        # by tiling the (identical, per-timestep) batch graph T times block-diagonally.
        b, c, t = h_bt.shape
        h_tbc = h_bt.permute(2, 0, 1).reshape(t * b, c)  # (T*batch, C)
        if edge_index.numel() > 0:
            offsets = (torch.arange(t, device=edge_index.device) * b).repeat_interleave(edge_index.shape[1])
            ei_tiled = edge_index.repeat(1, t) + offsets.unsqueeze(0)
        else:
            ei_tiled = edge_index
        out = gc(h_tbc, ei_tiled)  # (T*batch, C)
        out = out.reshape(t, b, c).permute(1, 2, 0)  # (batch, C, T)
        return F.relu(out)

    def forward(self, x, pk_ids, static_features, sen_directions=None):
        batch_size, seq_len, _ = x.shape
        device = x.device
        ctx = self._location_context(pk_ids, static_features, seq_len)
        h = torch.cat([x, ctx], dim=-1).transpose(1, 2)  # (batch, C, T)

        if sen_directions is not None:
            edge_index = self.build_highway_graph(pk_ids, sen_directions, device)
        else:
            idx = torch.arange(batch_size, device=device)
            edge_index = torch.stack([idx, idx])

        h = self.t1a(h)
        h = self._spatial(self.gc1, h, edge_index, seq_len)
        h = self.dropout(self.t1b(h))
        h = self.t2a(h)
        h = self._spatial(self.gc2, h, edge_index, seq_len)
        h = self.t2b(h)
        feats = self.layer_norm(h[:, :, -1])  # last timestep
        return self.fc(self.dropout(feats))


# =============================================================================
# BL-F4 — Graph WaveNet (dilated causal TCN + learned adaptive adjacency)
# =============================================================================

class _DilatedBlock(nn.Module):
    def __init__(self, ch, dilation, kernel_size=2):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.filt = nn.Conv1d(ch, ch, kernel_size, dilation=dilation)
        self.gate = nn.Conv1d(ch, ch, kernel_size, dilation=dilation)

    def forward(self, x):
        xp = F.pad(x, (self.pad, 0))
        return torch.tanh(self.filt(xp)) * torch.sigmoid(self.gate(xp))


class GraphWaveNetForecaster(BaseBenchmarkForecaster):
    """Graph WaveNet (Wu et al., 2019), adapted to the batch-graph paradigm.

    Dilated causal convolution stacks (1,2,4,8) model the temporal axis; a
    self-adaptive adjacency A~ = softmax(relu(E1 E2^T)) learned over the PK node
    embeddings supplies a data-driven spatial graph, fused with the predefined
    corridor graph. Addresses the AP-7's heterogeneous sensor density.
    """

    architecture = "graphwavenet"

    def __init__(self, *args, adaptive_dim=10, n_blocks=4, **kwargs):
        super().__init__(*args, **kwargs)
        d = int(kwargs.get("adaptive_dim", adaptive_dim))
        ch = self.hidden_size
        c_in = self.input_size + 2 * self.pk_embed_dim
        self.start = nn.Conv1d(c_in, ch, 1)
        dilations = [1, 2, 4, 8] * max(1, int(kwargs.get("n_blocks", n_blocks)) // 4)
        self.blocks = nn.ModuleList([_DilatedBlock(ch, dil) for dil in dilations])
        self.res = nn.ModuleList([nn.Conv1d(ch, ch, 1) for _ in dilations])
        # adaptive adjacency node embeddings (over PKs)
        self.E1 = nn.Parameter(torch.randn(self.num_pks, d) * 0.01)
        self.E2 = nn.Parameter(torch.randn(self.num_pks, d) * 0.01)
        self.gconv_pre = GCNConv(ch, ch)
        self.adapt_lin = nn.Linear(ch, ch)
        self.layer_norm = nn.LayerNorm(ch)
        self.fc = nn.Linear(ch, self.output_size)

    def _adaptive_mix(self, h, pk_ids):
        # h: (batch, ch). Adaptive adjacency restricted to the batch's PKs.
        e1 = self.E1[pk_ids]            # (batch, d)
        e2 = self.E2[pk_ids]            # (batch, d)
        A = F.softmax(F.relu(e1 @ e2.t()), dim=1)  # (batch, batch)
        return self.adapt_lin(A @ h)

    def forward(self, x, pk_ids, static_features, sen_directions=None):
        batch_size, seq_len, _ = x.shape
        device = x.device
        ctx = self._location_context(pk_ids, static_features, seq_len)
        h = torch.cat([x, ctx], dim=-1).transpose(1, 2)  # (batch, C, T)
        h = self.start(h)
        skip = 0
        for block, res in zip(self.blocks, self.res):
            out = block(h)
            h = h + res(out)
            skip = skip + out
        temporal = self.layer_norm((skip if torch.is_tensor(skip) else h)[:, :, -1])

        if sen_directions is not None:
            edge_index = self.build_highway_graph(pk_ids, sen_directions, device)
        else:
            idx = torch.arange(batch_size, device=device)
            edge_index = torch.stack([idx, idx])

        spatial_pre = F.relu(self.gconv_pre(temporal, edge_index))   # predefined graph
        spatial_adp = F.relu(self._adaptive_mix(temporal, pk_ids))   # adaptive graph
        combined = temporal + self.dropout(spatial_pre + spatial_adp)
        return self.fc(self.dropout(combined))


# =============================================================================
# BL-F5 — ASTGCN (spatial + temporal attention + Chebyshev graph convolution)
# =============================================================================

class ASTGCNForecaster(BaseBenchmarkForecaster):
    """Attention-based STGCN (Guo et al., 2019), recent-component adaptation.

    Computes spatial attention (over the batch) and temporal attention (over the
    window) from the input features, applies them around a Cheb graph conv + a
    temporal conv. Periodic (daily/weekly) components are dropped given the
    one-month training windows, per the doc.
    """

    architecture = "astgcn"

    def __init__(self, *args, cheb_k=3, n_heads=8, **kwargs):
        super().__init__(*args, **kwargs)
        self.cheb_k = int(kwargs.get("cheb_k", cheb_k))
        c_in = self.input_size + 2 * self.pk_embed_dim
        ch = self.hidden_size
        self.input_proj = nn.Linear(c_in, ch)
        # temporal attention over the window
        self.t_attn = nn.MultiheadAttention(ch, num_heads=int(kwargs.get("n_heads", n_heads)),
                                             batch_first=True, dropout=self.dropout_prob)
        self.cheb = ChebConv(ch, ch, K=self.cheb_k)
        self.temporal_conv = _TemporalGatedConv(ch, ch, kernel_size=3)
        self.layer_norm = nn.LayerNorm(ch)
        self.fc = nn.Linear(ch, self.output_size)

    def forward(self, x, pk_ids, static_features, sen_directions=None):
        batch_size, seq_len, _ = x.shape
        device = x.device
        ctx = self._location_context(pk_ids, static_features, seq_len)
        h = self.input_proj(torch.cat([x, ctx], dim=-1))  # (batch, T, ch)

        # temporal self-attention over the window
        h_attn, _ = self.t_attn(h, h, h)
        h = h + self.dropout(h_attn)

        # spatial attention across the batch, applied at the last timestep
        last = h[:, -1, :]                                  # (batch, ch)
        s_score = F.softmax(F.relu(last @ last.t()) /
                            (last.shape[-1] ** 0.5), dim=1)  # (batch, batch)
        spatial_ctx = s_score @ last

        if sen_directions is not None:
            edge_index = self.build_highway_graph(pk_ids, sen_directions, device)
        else:
            idx = torch.arange(batch_size, device=device)
            edge_index = torch.stack([idx, idx])
        graph_feat = F.relu(self.cheb(spatial_ctx, edge_index))

        # one temporal gated conv over the attended window for a recent summary
        tconv = self.temporal_conv(h.transpose(1, 2))[:, :, -1]  # (batch, ch)
        feats = self.layer_norm(last + self.dropout(graph_feat) + self.dropout(tconv))
        return self.fc(self.dropout(feats))


# =============================================================================
# BL-F0 — Historical Average (non-parametric, no learned forward)
# =============================================================================

HISTORICAL_AVERAGE_ARCH = "historical_average"


class HistoricalAverageForecaster:
    """Seasonal-naive Historical Average forecaster (BL-F0).

    Definition used here (per the project owner): take the average daily
    *profile* from the same weekdays of the prior-year window, then *correct the
    volume* with the recent weeks. Concretely:

      * PROFILE (shape of all three channels) = mean of (mean_speed, intTot,
        intP) over the prior-year window, keyed by
        (via, sen, pk, diaSem, hor, min) — i.e. one value per location per
        weekday per time-of-day slot.
      * VOLUME CORRECTION = per-(via, sen, pk) scalar
        mean(intTot | recent window) / mean(intTot | prior-year window),
        applied to the two intensity channels (intTot, intP). Speed is taken
        from the profile as-is (it is not a volume).

    This is non-parametric (no torch graph), so it does NOT go through the
    learned `forward()` path. Instead it exposes `predict_dataframe`, which the
    arch-aware `generate_gnn_predictions_on_training_data` calls to fill the
    `*_gnn` forecast columns by a calendar join. Missing profile slots fall back
    hierarchically to the per-location mean, then the global mean, so every row
    receives a prediction (HA needs no warm-up window).
    """

    is_historical_average = True
    architecture = HISTORICAL_AVERAGE_ARCH

    PROFILE_KEYS = ["via", "sen", "pk", "diaSem", "hor", "min"]
    LOC_KEYS = ["via", "sen", "pk"]
    TARGETS = ["mean_speed", "intTot", "intP"]
    VOLUME_CHANNELS = ["intTot", "intP"]
    VOLUME_REF = "intTot"

    def __init__(self):
        self.profile = None          # DataFrame: PROFILE_KEYS -> TARGETS
        self.loc_mean = None         # DataFrame: LOC_KEYS -> TARGETS
        self.global_mean = None      # dict: channel -> float
        self.volume_factor = None    # DataFrame: LOC_KEYS -> 'vol_factor'
        self.meta = {}

    # -- fit ------------------------------------------------------------------
    def fit(self, df_profile: pd.DataFrame, df_recent: pd.DataFrame,
            factor_clip: tuple[float, float] = (0.25, 4.0)) -> "HistoricalAverageForecaster":
        """Build the profile (prior-year) and volume correction (recent weeks)."""
        prof_keys = [k for k in self.PROFILE_KEYS if k in df_profile.columns]
        loc_keys = [k for k in self.LOC_KEYS if k in df_profile.columns]
        targets = [t for t in self.TARGETS if t in df_profile.columns]

        self.profile = (df_profile.groupby(prof_keys, observed=True)[targets]
                        .mean().reset_index())
        self.loc_mean = (df_profile.groupby(loc_keys, observed=True)[targets]
                         .mean().reset_index())
        self.global_mean = {t: float(df_profile[t].mean()) for t in targets}

        # Volume correction: recent level / prior-year level, per location.
        ref = self.VOLUME_REF
        prior_lvl = (df_profile.groupby(loc_keys, observed=True)[ref]
                     .mean().rename("prior_lvl"))
        recent_lvl = (df_recent.groupby(loc_keys, observed=True)[ref]
                      .mean().rename("recent_lvl"))
        vf = pd.concat([prior_lvl, recent_lvl], axis=1).reset_index()
        vf["vol_factor"] = (vf["recent_lvl"] / vf["prior_lvl"]).replace(
            [np.inf, -np.inf], np.nan)
        global_factor = float(np.nanmedian(vf["vol_factor"])) if len(vf) else 1.0
        if not np.isfinite(global_factor) or global_factor <= 0:
            global_factor = 1.0
        vf["vol_factor"] = vf["vol_factor"].fillna(global_factor).clip(*factor_clip)
        self.volume_factor = vf[loc_keys + ["vol_factor"]]

        self._prof_keys = prof_keys
        self._loc_keys = loc_keys
        self._targets = targets
        self.meta = {
            "n_profile_slots": int(len(self.profile)),
            "n_locations": int(len(self.loc_mean)),
            "global_volume_factor": global_factor,
            "factor_clip": list(factor_clip),
            "profile_keys": prof_keys,
            "volume_ref_channel": ref,
        }
        return self

    # -- predict --------------------------------------------------------------
    def predict_dataframe(self, df: pd.DataFrame) -> np.ndarray:
        """Return an (N, 3) array of (mean_speed, intTot, intP) HA forecasts,
        aligned to `df` row order."""
        if self.profile is None:
            raise RuntimeError("HistoricalAverageForecaster.fit was never called.")
        targets = self._targets
        work = df.reset_index(drop=True).copy()
        work["_row"] = np.arange(len(work))

        # 1. profile join on full time-of-week key
        merged = work.merge(self.profile, on=self._prof_keys, how="left")
        # 2. build each channel with hierarchical fallback:
        #    profile slot -> per-location mean -> global mean.
        pred = {}
        loc_lookup = self.loc_mean.rename(columns={t: t + "_loc" for t in targets})
        merged = merged.merge(loc_lookup, on=self._loc_keys, how="left")
        for t in targets:
            col_prof = merged[t] if t in merged else pd.Series(np.nan, index=merged.index)
            col_loc = merged.get(t + "_loc")
            out = col_prof.copy()
            if col_loc is not None:
                out = out.fillna(col_loc)
            out = out.fillna(self.global_mean.get(t, 0.0))
            pred[t] = out

        # 3. volume correction on intensity channels
        merged = merged.merge(self.volume_factor, on=self._loc_keys, how="left")
        vf = merged["vol_factor"].fillna(self.meta.get("global_volume_factor", 1.0))
        for t in self.VOLUME_CHANNELS:
            if t in pred:
                pred[t] = pred[t] * vf

        order = merged["_row"].values
        result = np.zeros((len(work), 3), dtype=np.float32)
        for j, name in enumerate(self.TARGETS):
            if name in pred:
                result[order, j] = pred[name].values.astype(np.float32)
        return result

    # -- persistence ----------------------------------------------------------
    def save(self, path: str) -> None:
        joblib.dump({
            "profile": self.profile,
            "loc_mean": self.loc_mean,
            "global_mean": self.global_mean,
            "volume_factor": self.volume_factor,
            "prof_keys": self._prof_keys,
            "loc_keys": self._loc_keys,
            "targets": self._targets,
            "meta": self.meta,
        }, path)

    @classmethod
    def load(cls, path: str) -> "HistoricalAverageForecaster":
        d = joblib.load(path)
        obj = cls()
        obj.profile = d["profile"]
        obj.loc_mean = d["loc_mean"]
        obj.global_mean = d["global_mean"]
        obj.volume_factor = d["volume_factor"]
        obj._prof_keys = d["prof_keys"]
        obj._loc_keys = d["loc_keys"]
        obj._targets = d["targets"]
        obj.meta = d["meta"]
        return obj


# =============================================================================
# Registry / factory
# =============================================================================

FORECASTER_REGISTRY = {
    # Proposed method (V17 trainer) — GNN-LSTM with graph message passing.
    # GeoLSTM below is the same model minus the graph (the graph-ablation pair).
    "gnn_lstm": SpatioTemporalGNN_LSTM_V5,
    "gnn_lstm_v5": SpatioTemporalGNN_LSTM_V5,
    GeoLSTMForecaster.architecture: GeoLSTMForecaster,
    DCRNNForecaster.architecture: DCRNNForecaster,
    STGCNForecaster.architecture: STGCNForecaster,
    GraphWaveNetForecaster.architecture: GraphWaveNetForecaster,
    ASTGCNForecaster.architecture: ASTGCNForecaster,
}


def build_forecaster(architecture: str, **model_kwargs):
    """Instantiate a benchmark forecaster by its `architecture` token.

    `architecture` is read from the GNN metadata by the arch-aware
    `load_pretrained_gnn_model`. Unknown tokens raise so a misconfigured run
    fails loudly instead of silently loading the wrong model.
    """
    key = (architecture or "").lower()
    if key not in FORECASTER_REGISTRY:
        raise KeyError(
            f"Unknown forecaster architecture {architecture!r}. "
            f"Known: {sorted(FORECASTER_REGISTRY)}"
        )
    return FORECASTER_REGISTRY[key](**model_kwargs)
