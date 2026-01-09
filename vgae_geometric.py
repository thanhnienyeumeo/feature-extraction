#!/usr/bin/env python
# coding: utf-8
"""
Variational Graph Auto-Encoder (VGAE) for Session Data using PyTorch Geometric

This module implements VGAE architecture using torch_geometric library for 
learning session embeddings.

Features:
- Learnable item embeddings (like SR-GNN)
- Structural features (degree, position, frequency)
- GCN encoder using torch_geometric.nn.conv.GCNConv
- Built on torch_geometric.nn.models.VGAE

Reference:
- Kipf & Welling, "Variational Graph Auto-Encoders", 2016
- PyTorch Geometric: https://pytorch-geometric.readthedocs.io/
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn.models import VGAE, InnerProductDecoder
from torch_geometric.nn.conv import GCNConv
from torch_geometric.utils import negative_sampling, remove_self_loops, add_self_loops
from torch_geometric.data import Data, Batch


# ============================================================================
# GCN ENCODER
# ============================================================================

class GCNEncoder(nn.Module):
    """
    GCN Encoder for VGAE using PyTorch Geometric.
    
    Architecture:
    - Multiple GCN layers for shared feature extraction
    - Separate GCN layers for mean (mu) and log-variance (logvar)
    """
    def __init__(self, in_channels, hidden_channels, out_channels, 
                 num_layers=2, dropout=0.0):
        super(GCNEncoder, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Build shared GCN layers
        self.convs = nn.ModuleList()
        
        # First layer
        self.convs.append(GCNConv(in_channels, hidden_channels))
        
        # Additional hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        
        # Mean and log-variance layers
        self.conv_mu = GCNConv(hidden_channels, out_channels)
        self.conv_logvar = GCNConv(hidden_channels, out_channels)
    
    def forward(self, x, edge_index):
        """
        Forward pass through encoder.
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges]
        
        Returns:
            mu: Mean of latent distribution [num_nodes, out_channels]
            logvar: Log-variance of latent distribution [num_nodes, out_channels]
        """
        # Pass through shared GCN layers
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            if self.dropout > 0:
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Compute mean and log-variance
        mu = self.conv_mu(x, edge_index)
        logvar = self.conv_logvar(x, edge_index)
        
        return mu, logvar


# ============================================================================
# DEEP VGAE WITH ITEM EMBEDDINGS
# ============================================================================

class SessionVGAE(VGAE):
    """
    VGAE for Session Anomaly Detection with Learnable Item Embeddings.
    
    This model extends PyTorch Geometric's VGAE with:
    1. Learnable item embeddings
    2. Optional structural features
    3. Session-level embedding extraction
    
    Based on: https://github.com/Flawless1202/VGAE_pyG/blob/master/model.py
    """
    def __init__(self, n_items, embedding_dim=64, hidden_dim=64, latent_dim=32,
                 num_gcn_layers=2, dropout=0.0, use_structural_features=True,
                 structural_feature_dim=16):
        """
        Args:
            n_items: Number of unique items in vocabulary
            embedding_dim: Dimension of learnable item embeddings (default: 64)
            hidden_dim: Hidden dimension for GCN layers (default: 64)
            latent_dim: Latent dimension for VAE (default: 32)
            num_gcn_layers: Number of GCN layers in encoder (default: 2)
            dropout: Dropout rate (default: 0.0)
            use_structural_features: Whether to add structural features (default: True)
            structural_feature_dim: Dimension of structural feature projection (default: 16)
        """
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.use_structural_features = use_structural_features
        self.structural_feature_dim = structural_feature_dim
        self.n_structural_features = 6  # Raw structural features
        
        # Calculate input dimension to GCN
        if use_structural_features:
            input_dim = embedding_dim + structural_feature_dim
        else:
            input_dim = embedding_dim
        
        # Create encoder
        encoder = GCNEncoder(
            in_channels=input_dim,
            hidden_channels=hidden_dim,
            out_channels=latent_dim,
            num_layers=num_gcn_layers,
            dropout=dropout
        )
        
        # Initialize VGAE with encoder and default InnerProductDecoder
        super(SessionVGAE, self).__init__(encoder=encoder, decoder=InnerProductDecoder())
        
        # Learnable item embeddings (+1 for padding at index 0)
        self.item_embedding = nn.Embedding(n_items + 1, embedding_dim, padding_idx=0)
        
        # Structural feature projection
        if use_structural_features:
            self.structural_proj = nn.Linear(self.n_structural_features, structural_feature_dim)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters."""
        stdv = 1.0 / math.sqrt(self.embedding_dim)
        self.item_embedding.weight.data.uniform_(-stdv, stdv)
        self.item_embedding.weight.data[0].fill_(0)  # Padding embedding
    
    def get_node_features(self, item_ids, structural_features=None):
        """
        Get node features by combining item embeddings and structural features.
        
        Args:
            item_ids: Item indices [num_nodes]
            structural_features: Optional structural features [num_nodes, n_structural_features]
        
        Returns:
            x: Node features [num_nodes, input_dim]
        """
        # Get item embeddings
        x = self.item_embedding(item_ids)
        
        # Add structural features if provided
        if self.use_structural_features and structural_features is not None:
            struct_emb = self.structural_proj(structural_features)
            x = torch.cat([x, struct_emb], dim=-1)
        
        return x
    
    def forward(self, item_ids, edge_index, structural_features=None):
        """
        Forward pass through VGAE.
        
        Args:
            item_ids: Item indices [num_nodes]
            edge_index: Edge indices [2, num_edges]
            structural_features: Optional structural features [num_nodes, n_structural_features]
        
        Returns:
            adj_pred: Predicted adjacency (all pairs) [num_nodes, num_nodes]
        """
        x = self.get_node_features(item_ids, structural_features)
        z = self.encode(x, edge_index)
        adj_pred = self.decoder.forward_all(z)
        return adj_pred
    
    def compute_loss(self, item_ids, pos_edge_index, structural_features=None, 
                     neg_edge_index=None, num_nodes=None):
        """
        Compute VGAE loss with negative sampling.
        
        Args:
            item_ids: Item indices [num_nodes]
            pos_edge_index: Positive edge indices [2, num_pos_edges]
            structural_features: Optional structural features
            neg_edge_index: Negative edge indices (if None, will be sampled)
            num_nodes: Number of nodes (for negative sampling)
        
        Returns:
            loss: Total loss (reconstruction + KL)
            recon_loss: Reconstruction loss
            kl_loss: KL divergence loss
        """
        x = self.get_node_features(item_ids, structural_features)
        z = self.encode(x, pos_edge_index)
        
        # Positive edge loss
        pos_loss = -torch.log(
            self.decoder(z, pos_edge_index, sigmoid=True) + 1e-15
        ).mean()
        
        # Negative sampling
        if neg_edge_index is None:
            if num_nodes is None:
                num_nodes = item_ids.size(0)
            neg_edge_index = negative_sampling(
                pos_edge_index, 
                num_nodes=num_nodes,
                num_neg_samples=pos_edge_index.size(1)
            )
        
        # Negative edge loss
        neg_loss = -torch.log(
            1 - self.decoder(z, neg_edge_index, sigmoid=True) + 1e-15
        ).mean()
        
        # KL divergence
        kl_loss = (1 / item_ids.size(0)) * self.kl_loss()
        
        # Total loss
        recon_loss = pos_loss + neg_loss
        loss = recon_loss + kl_loss
        
        return loss, recon_loss, kl_loss
    
    def get_embedding(self, item_ids, edge_index, structural_features=None, 
                      batch=None, num_graphs=None):
        """
        Get session embedding from latent representation.
        
        For a single graph: mean pooling over all nodes
        For batched graphs: mean pooling per graph
        
        Args:
            item_ids: Item indices [num_nodes]
            edge_index: Edge indices [2, num_edges]
            structural_features: Optional structural features
            batch: Batch assignment vector [num_nodes] (for batched graphs)
            num_graphs: Number of graphs in batch
        
        Returns:
            embedding: Session embedding [num_graphs, latent_dim] or [latent_dim]
        """
        x = self.get_node_features(item_ids, structural_features)
        z = self.encode(x, edge_index)  # [num_nodes, latent_dim]
        
        if batch is not None and num_graphs is not None:
            # Batched graphs: mean pooling per graph
            from torch_geometric.nn import global_mean_pool
            embedding = global_mean_pool(z, batch)  # [num_graphs, latent_dim]
        else:
            # Single graph: mean pooling
            embedding = z.mean(dim=0, keepdim=True)  # [1, latent_dim]
        
        return embedding
    
    def get_node_embeddings(self, item_ids, edge_index, structural_features=None):
        """
        Get node-level embeddings (latent mean for each node).
        
        Args:
            item_ids: Item indices [num_nodes]
            edge_index: Edge indices [2, num_edges]
            structural_features: Optional structural features
        
        Returns:
            z: Node embeddings [num_nodes, latent_dim]
        """
        x = self.get_node_features(item_ids, structural_features)
        z = self.encode(x, edge_index)
        return z


# ============================================================================
# SESSION GRAPH UTILITIES
# ============================================================================

def compute_structural_features(session, node_ids, node_to_idx, adj_matrix):
    """
    Compute structural features for each node in the session graph.
    
    Features:
    1. In-degree (normalized)
    2. Out-degree (normalized)
    3. First position in session (normalized)
    4. Last position in session (normalized)
    5. Frequency in session (normalized)
    6. Position spread (normalized)
    
    Args:
        session: Original session sequence
        node_ids: List of unique item ids
        node_to_idx: Mapping from item id to node index
        adj_matrix: Adjacency matrix (before making undirected)
    
    Returns:
        features: [n_nodes, 6] structural feature matrix
    """
    n_nodes = len(node_ids)
    session_len = len(session)
    
    features = np.zeros((n_nodes, 6), dtype=np.float32)
    
    for i, item_id in enumerate(node_ids):
        positions = [pos for pos, item in enumerate(session) if item == item_id]
        
        # In-degree and out-degree from adjacency matrix
        in_degree = adj_matrix[:, i].sum() if adj_matrix.shape[0] > 0 else 0
        out_degree = adj_matrix[i, :].sum() if adj_matrix.shape[1] > 0 else 0
        
        features[i, 0] = in_degree / max(n_nodes - 1, 1)
        features[i, 1] = out_degree / max(n_nodes - 1, 1)
        features[i, 2] = positions[0] / max(session_len - 1, 1) if positions else 0
        features[i, 3] = positions[-1] / max(session_len - 1, 1) if positions else 0
        features[i, 4] = len(positions) / session_len
        features[i, 5] = (positions[-1] - positions[0]) / max(session_len - 1, 1) if positions else 0
    
    return features


def build_session_graph_pyg(session, compute_structural=True):
    """
    Build a session graph in PyTorch Geometric format.
    
    Args:
        session: List of item indices
        compute_structural: Whether to compute structural features
    
    Returns:
        data: PyG Data object with:
            - x: Item ids (shifted by 1 for padding) [n_nodes]
            - edge_index: Edge indices [2, n_edges]
            - structural_features: Structural features [n_nodes, 6] (optional)
            - num_nodes: Number of nodes
    """
    # Get unique items (preserve order)
    seen = set()
    node_ids = []
    for item in session:
        if item not in seen:
            seen.add(item)
            node_ids.append(item)
    
    node_to_idx = {node: idx for idx, node in enumerate(node_ids)}
    n_nodes = len(node_ids)
    
    # Build edge list from transitions
    edge_src = []
    edge_dst = []
    adj_matrix = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    
    for i in range(len(session) - 1):
        u = node_to_idx[session[i]]
        v = node_to_idx[session[i + 1]]
        # Directed edge
        adj_matrix[u, v] = 1.0
        edge_src.append(u)
        edge_dst.append(v)
        # Make undirected
        edge_src.append(v)
        edge_dst.append(u)
    
    # Add self-loops
    for i in range(n_nodes):
        edge_src.append(i)
        edge_dst.append(i)
    
    # Remove duplicates
    edge_set = set(zip(edge_src, edge_dst))
    edge_src = [e[0] for e in edge_set]
    edge_dst = [e[1] for e in edge_set]
    
    # Create edge_index tensor
    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
    
    # Item ids (shifted by 1 for padding)
    item_ids = torch.tensor([nid + 1 for nid in node_ids], dtype=torch.long)
    
    # Compute structural features
    structural_features = None
    if compute_structural:
        structural_features = compute_structural_features(
            session, node_ids, node_to_idx, adj_matrix
        )
        structural_features = torch.tensor(structural_features, dtype=torch.float)
    
    # Create PyG Data object
    data = Data(
        x=item_ids,
        edge_index=edge_index,
        num_nodes=n_nodes
    )
    
    if structural_features is not None:
        data.structural_features = structural_features
    
    return data


def collate_session_graphs(data_list):
    """
    Collate multiple session graphs into a batch.
    
    Args:
        data_list: List of PyG Data objects
    
    Returns:
        batch: Batched Data object
    """
    return Batch.from_data_list(data_list)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def trans_to_cuda(variable):
    """Move tensor/module to CUDA if available."""
    if torch.cuda.is_available():
        return variable.cuda()
    return variable


def trans_to_cpu(variable):
    """Move tensor to CPU."""
    if torch.cuda.is_available():
        return variable.cpu()
    return variable


# ============================================================================
# TRAINING UTILITIES
# ============================================================================

class SessionDatasetPyG:
    """
    Dataset class for session data with PyTorch Geometric.
    """
    def __init__(self, sessions, compute_structural=True):
        self.sessions = sessions
        self.compute_structural = compute_structural
        self.n_sessions = len(sessions)
        
        # Pre-build all graphs
        self.graphs = []
        for session in sessions:
            graph = build_session_graph_pyg(session, compute_structural)
            self.graphs.append(graph)
    
    def __len__(self):
        return self.n_sessions
    
    def __getitem__(self, idx):
        return self.graphs[idx]
    
    def get_batch(self, batch_indices):
        """Get a batch of session graphs."""
        batch_graphs = [self.graphs[i] for i in batch_indices]
        return collate_session_graphs(batch_graphs)


def train_vgae_geometric(model, train_sessions, n_items=None, epochs=50, batch_size=32,
                         lr=0.001, early_stopping_patience=10, verbose=True,
                         use_structural_features=True):
    """
    Train VGAE model on session data using PyTorch Geometric.
    
    Args:
        model: SessionVGAE model
        train_sessions: List of training sessions
        n_items: Total number of items (not used, kept for compatibility)
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        early_stopping_patience: Patience for early stopping
        verbose: Print training progress
        use_structural_features: Whether to use structural features
    
    Returns:
        model: Trained model
        history: Training history
    """
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    dataset = SessionDatasetPyG(train_sessions, compute_structural=use_structural_features)
    n_batches = (len(dataset) + batch_size - 1) // batch_size
    
    best_loss = float('inf')
    patience_counter = 0
    history = {'loss': [], 'recon_loss': [], 'kl_loss': []}
    
    for epoch in range(epochs):
        indices = np.random.permutation(len(dataset))
        
        epoch_loss = 0.0
        epoch_recon = 0.0
        epoch_kl = 0.0
        
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(dataset))
            batch_indices = indices[start_idx:end_idx]
            
            # Get batched data
            batch = dataset.get_batch(batch_indices)
            batch = trans_to_cuda(batch)
            
            # Get structural features if available
            structural_features = None
            if use_structural_features and hasattr(batch, 'structural_features'):
                structural_features = batch.structural_features
            
            # Forward pass
            optimizer.zero_grad()
            loss, recon_loss, kl_loss = model.compute_loss(
                batch.x,
                batch.edge_index,
                structural_features=structural_features,
                num_nodes=batch.num_nodes
            )
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_recon += recon_loss.item()
            epoch_kl += kl_loss.item()
        
        epoch_loss /= n_batches
        epoch_recon /= n_batches
        epoch_kl /= n_batches
        
        history['loss'].append(epoch_loss)
        history['recon_loss'].append(epoch_recon)
        history['kl_loss'].append(epoch_kl)
        
        if verbose and (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{epochs}] Loss: {epoch_loss:.6f} '
                  f'(Recon: {epoch_recon:.6f}, KL: {epoch_kl:.6f})')
        
        # Early stopping
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                if verbose:
                    print(f'Early stopping at epoch {epoch+1}')
                break
    
    return model, history


def extract_vgae_embeddings_geometric(model, sessions, n_items=None, batch_size=100,
                                       use_structural_features=True):
    """
    Extract session embeddings from trained VGAE.
    
    Args:
        model: Trained SessionVGAE model
        sessions: List of sessions
        n_items: Not used, kept for compatibility
        batch_size: Batch size for processing
        use_structural_features: Whether to use structural features
    
    Returns:
        embeddings: [n_sessions, latent_dim] session embeddings
    """
    model.eval()
    all_embeddings = []
    
    dataset = SessionDatasetPyG(sessions, compute_structural=use_structural_features)
    n_batches = (len(dataset) + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(dataset))
            batch_indices = list(range(start_idx, end_idx))
            
            # Get batched data
            batch = dataset.get_batch(batch_indices)
            batch = trans_to_cuda(batch)
            
            # Get structural features
            structural_features = None
            if use_structural_features and hasattr(batch, 'structural_features'):
                structural_features = batch.structural_features
            
            # Get embeddings
            embeddings = model.get_embedding(
                batch.x,
                batch.edge_index,
                structural_features=structural_features,
                batch=batch.batch,
                num_graphs=len(batch_indices)
            )
            
            embeddings = trans_to_cpu(embeddings).numpy()
            all_embeddings.append(embeddings)
    
    return np.concatenate(all_embeddings, axis=0)


# ============================================================================
# RECOMMENDED CONFIGURATIONS
# ============================================================================

def get_recommended_config(n_items, dataset_size='medium'):
    """
    Get recommended VGAE configuration based on dataset characteristics.
    
    Args:
        n_items: Number of unique items in vocabulary
        dataset_size: 'small' (<1000 sessions), 'medium' (1000-100000), 'large' (>100000)
    
    Returns:
        config: Dictionary of recommended hyperparameters
    """
    configs = {
        'small': {
            'embedding_dim': 32,
            'hidden_dim': 32,
            'latent_dim': 16,
            'num_gcn_layers': 2,
            'dropout': 0.1,
            'use_structural_features': True,
            'structural_feature_dim': 8,
            'batch_size': 16,
            'epochs': 100,
            'lr': 0.001,
        },
        'medium': {
            'embedding_dim': 64,
            'hidden_dim': 64,
            'latent_dim': 32,
            'num_gcn_layers': 2,
            'dropout': 0.1,
            'use_structural_features': True,
            'structural_feature_dim': 16,
            'batch_size': 32,
            'epochs': 50,
            'lr': 0.001,
        },
        'large': {
            'embedding_dim': 128,
            'hidden_dim': 128,
            'latent_dim': 64,
            'num_gcn_layers': 3,
            'dropout': 0.2,
            'use_structural_features': True,
            'structural_feature_dim': 32,
            'batch_size': 64,
            'epochs': 30,
            'lr': 0.0005,
        }
    }
    
    config = configs.get(dataset_size, configs['medium'])
    
    if n_items > 10000:
        config['embedding_dim'] = max(config['embedding_dim'], 128)
    elif n_items > 1000:
        config['embedding_dim'] = max(config['embedding_dim'], 64)
    
    return config
