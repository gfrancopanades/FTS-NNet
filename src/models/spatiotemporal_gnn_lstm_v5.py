import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import pytorch_lightning as pl
from torch_geometric.nn import GCNConv, GATConv


class SpatioTemporalGNN_LSTM_V5(pl.LightningModule):
    """
    SpatioTemporal GNN-LSTM V5 - Improved highway traffic prediction model.
    
    Changes from V1 (SpatioTemporalGNN_LSTM):
    - Removed final ReLU output activation (was clipping gradients for low-value predictions)
    - Added LayerNorm after LSTM output (stabilizes training, reduces internal covariate shift)
    - Replaced uniform MSELoss with per-target weighted SmoothL1Loss (fixes intensity collapse)
    - Reduced LR scheduler patience (10 -> 3) to allow LR reduction before early stopping
    
    Key features (unchanged from V1):
    - Graph structure respects consecutive PK connections
    - Direction-aware: dec (descending) vs cre (ascending)
    - Graph Convolution for spatial message passing
    - LSTM for temporal processing
    - PK embeddings for location-specific patterns
    - Static geometric features embedded per PK
    
    Highway structure:
    - sen='dec': PK_i connected to PK_{i-1} (descending: 200->199->198...)
    - sen='cre': PK_i connected to PK_{i+1} (ascending: 100->101->102...)
    
    Author: Gerard Franco
    Date: February 2026
    Affiliation: Universitat Politecnica de Catalunya
    """
    
    def __init__(self, input_size, hidden_size, num_layers, output_size, 
                 dropout_prob, learning_rate, weight_decay,
                 num_pks, static_feature_dim, pk_embed_dim=16,
                 graph_hidden_dim=64, num_graph_layers=2,
                 target_weights=None):
        """
        Args:
            input_size: Number of temporal input features
            hidden_size: LSTM hidden state size
            num_layers: Number of LSTM layers
            output_size: Number of output targets
            dropout_prob: Dropout probability
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for regularization
            num_pks: Total number of PK locations
            static_feature_dim: Dimension of static features per PK
            pk_embed_dim: Dimension of PK embeddings
            graph_hidden_dim: Hidden dimension for graph convolution layers
            num_graph_layers: Number of graph convolution layers
            target_weights: Per-target loss weights [speed_w, intTot_w, intP_w].
                            If None, defaults to uniform [1.0, 1.0, 1.0].
        """
        super(SpatioTemporalGNN_LSTM_V5, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_pks = num_pks
        self.pk_embed_dim = pk_embed_dim
        self.graph_hidden_dim = graph_hidden_dim
        self.num_graph_layers = num_graph_layers
        
        # V5: Per-target loss weights (default: uniform)
        if target_weights is not None:
            self.register_buffer(
                'target_weights',
                torch.tensor(target_weights, dtype=torch.float32)
            )
        else:
            self.register_buffer(
                'target_weights',
                torch.ones(output_size, dtype=torch.float32)
            )
        
        # Storage for epoch-level R2 computation (accumulate across batches)
        self.val_predictions = []
        self.val_labels = []
        
        # PK embeddings: Learn location-specific representations
        self.pk_embedding = nn.Embedding(num_pks, pk_embed_dim)
        
        # Static feature projection: Embed geometric features
        self.static_projection = nn.Linear(static_feature_dim, pk_embed_dim)
        
        # LSTM for temporal processing
        # Input: temporal features + PK embedding + static embedding
        lstm_input_size = input_size + 2 * pk_embed_dim
        self.lstm = nn.LSTM(lstm_input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout_prob if num_layers > 1 else 0)
        
        # V5: LayerNorm after LSTM output
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        # Graph Neural Network layers for spatial relationships
        self.graph_layers = nn.ModuleList()
        
        if num_graph_layers == 1:
            self.graph_layers.append(GCNConv(hidden_size, hidden_size))
        else:
            self.graph_layers.append(GCNConv(hidden_size, graph_hidden_dim))
            for _ in range(num_graph_layers - 2):
                self.graph_layers.append(GCNConv(graph_hidden_dim, graph_hidden_dim))
            self.graph_layers.append(GCNConv(graph_hidden_dim, hidden_size))
        
        # Output layers
        self.dropout = nn.Dropout(dropout_prob)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, output_size)
        
        # V5: SmoothL1Loss (Huber) with reduction='none' for per-target weighting
        self.criterion = nn.SmoothL1Loss(reduction='none')
        
        # Save hyperparameters for checkpointing
        self.save_hyperparameters()

    def build_highway_graph(self, pk_ids, sen_directions, device):
        """
        Build graph structure for highway based on PK connectivity and direction.
        OPTIMIZED: Uses pure torch operations without CPU transfers.
        
        Args:
            pk_ids: Tensor of PK IDs in batch (batch_size,)
            sen_directions: Tensor of direction indicators (batch_size,) - 0=dec, 1=cre
            device: Device to create tensors on
        
        Returns:
            edge_index: Edge connections (2, num_edges)
        """
        batch_size = pk_ids.shape[0]
        
        # Calculate neighbor PKs based on direction (all on GPU)
        neighbor_pks = torch.where(sen_directions == 0, pk_ids - 1, pk_ids + 1)
        
        # Create batch indices
        batch_indices = torch.arange(batch_size, device=device)
        
        # Find which neighbors exist in the current batch
        matches = pk_ids.unsqueeze(1) == neighbor_pks.unsqueeze(0)
        
        # Get source and target indices where matches exist
        source_indices, target_indices = matches.nonzero(as_tuple=True)
        
        if source_indices.numel() > 0:
            edge_index = torch.stack([
                torch.cat([source_indices, target_indices]),
                torch.cat([target_indices, source_indices])
            ])
        else:
            edge_index = torch.stack([batch_indices, batch_indices])
        
        return edge_index

    def forward(self, x, pk_ids, static_features, sen_directions=None):
        """
        Forward pass with graph-aware spatial processing.
        
        Args:
            x: Temporal sequences (batch, sequence_length, input_size)
            pk_ids: PK location IDs (batch,)
            static_features: Static geometric features (batch, static_feature_dim)
            sen_directions: Direction indicators (batch,) - 0=dec, 1=cre
        
        Returns:
            output: Predictions (batch, output_size)
        """
        batch_size, seq_len, _ = x.shape
        device = x.device
        
        # 1. Get PK embeddings
        pk_embed = self.pk_embedding(pk_ids)
        
        # 2. Project static features
        static_embed = self.static_projection(static_features)
        
        # 3. Combine embeddings and expand to sequence length
        location_context = torch.cat([pk_embed, static_embed], dim=-1)
        location_context = location_context.unsqueeze(1).expand(-1, seq_len, -1)
        
        # 4. Concatenate temporal features with location context
        x_combined = torch.cat([x, location_context], dim=-1)
        
        # 5. LSTM temporal processing
        lstm_out, (h_n, c_n) = self.lstm(x_combined)
        
        # Take the final timestep output
        temporal_features = lstm_out[:, -1, :]
        
        # V5: Apply LayerNorm after LSTM
        temporal_features = self.layer_norm(temporal_features)
        
        # 6. Build highway graph structure
        if sen_directions is not None:
            edge_index = self.build_highway_graph(pk_ids, sen_directions, device)
        else:
            edge_index = torch.stack([
                torch.arange(batch_size, device=device),
                torch.arange(batch_size, device=device)
            ])
        
        # 7. Apply Graph Convolution for spatial message passing
        graph_features = temporal_features
        
        for i, graph_layer in enumerate(self.graph_layers):
            graph_features = graph_layer(graph_features, edge_index)
            if i < len(self.graph_layers) - 1:
                graph_features = F.relu(graph_features)
                graph_features = self.dropout(graph_features)
        
        # 8. Combine temporal and spatial features with residual connection
        combined_features = temporal_features + graph_features
        
        # 9. Final prediction layers
        x_out = self.dropout(combined_features)
        x_out = F.relu(self.fc1(x_out))
        x_out = self.dropout(x_out)
        
        # V5: No activation on output (removed ReLU)
        # Targets are MinMax-scaled to [0,1]; identity output lets the model
        # predict the full range without gradient clipping near zero.
        output = self.fc2(x_out)
        
        return output

    def _compute_weighted_loss(self, outputs, labels):
        """Compute per-target weighted SmoothL1 loss.
        
        Returns a scalar loss where each target column is weighted by
        self.target_weights before averaging.
        """
        per_element_loss = self.criterion(outputs, labels)  # (batch, output_size)
        weighted = per_element_loss * self.target_weights.unsqueeze(0)
        return weighted.mean()

    def training_step(self, batch, batch_idx):
        inputs, labels, pk_ids, static_features, sen_directions = batch
        labels = labels.view(-1, self.output_size)
        
        outputs = self(inputs, pk_ids, static_features, sen_directions)
        
        # V5: Per-target weighted loss
        loss = self._compute_weighted_loss(outputs, labels)
        
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def on_validation_epoch_start(self):
        """Reset accumulators at the start of each validation epoch."""
        self.val_predictions = []
        self.val_labels = []
    
    def validation_step(self, batch, batch_idx):
        inputs, labels, pk_ids, static_features, sen_directions = batch
        labels = labels.view(-1, self.output_size)
        
        outputs = self(inputs, pk_ids, static_features, sen_directions)
        
        # V5: Per-target weighted loss
        loss = self._compute_weighted_loss(outputs, labels)
        
        # Accumulate predictions and labels for epoch-level R2 computation
        self.val_predictions.append(outputs.detach())
        self.val_labels.append(labels.detach())
        
        self.log('val_loss', loss, prog_bar=True)
        
        return loss
    
    def on_validation_epoch_end(self):
        """Compute epoch-level metrics using all accumulated predictions."""
        if len(self.val_predictions) == 0:
            return
        
        all_preds = torch.cat(self.val_predictions, dim=0).cpu().numpy()
        all_labels = torch.cat(self.val_labels, dim=0).cpu().numpy()
        
        # Compute metrics on the FULL validation set
        val_mae = mean_absolute_error(all_labels, all_preds)
        val_r2 = r2_score(all_labels, all_preds)
        val_rmse = np.sqrt(mean_squared_error(all_labels, all_preds))
        
        # V5: Also compute per-target R2 for diagnostics
        for i, name in enumerate(['speed', 'intTot', 'intP']):
            if i < all_preds.shape[1]:
                target_r2 = r2_score(all_labels[:, i], all_preds[:, i])
                self.log(f'val_r2_{name}', target_r2, prog_bar=False)
        
        self.log('val_mae', val_mae, prog_bar=False)
        self.log('val_r2', val_r2, prog_bar=False)
        self.log('val_rmse', val_rmse, prog_bar=False)
        
        # Clear memory
        self.val_predictions = []
        self.val_labels = []

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        
        # V5: Reduced scheduler patience (10 -> 3) so LR can decrease
        # before early stopping (patience=7) kills the training.
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=3,
            verbose=False,
            min_lr=1e-7
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1
            }
        }
