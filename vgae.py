#!/usr/bin/env python
# coding: utf-8
"""
Variational Graph Auto-Encoder (VGAE) for Session Data

This module implements VGAE architecture for learning session embeddings.
The VGAE learns to:
1. Encode session graphs into latent representations
2. Reconstruct the graph structure from latent representations

Architecture:
- Item Embedding: Learnable embeddings for each item (like SR-GNN)
- Encoder: GCN-based encoder that outputs mean and log-variance
- Decoder: Inner product decoder for graph reconstruction
- Loss: Reconstruction loss + KL divergence

Feature Options:
1. Learnable item embeddings (recommended, 32-128 dim)
2. Structural features (degree, position, centrality)
3. Combined embeddings + structural features

Reference:
- Kipf & Welling, "Variational Graph Auto-Encoders", 2016
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


# ============================================================================
# GRAPH CONVOLUTION LAYER
# ============================================================================

class GraphConvolution(nn.Module):
    """
    Simple Graph Convolution Layer (GCN).
    
    Performs: H' = σ(D^(-1/2) A D^(-1/2) H W)
    
    For simplicity, we use normalized adjacency: H' = σ(A_norm H W)
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
    
    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
    
    def forward(self, x, adj):
        """
        Args:
            x: Node features [batch, n_nodes, in_features] or [n_nodes, in_features]
            adj: Normalized adjacency matrix [batch, n_nodes, n_nodes] or [n_nodes, n_nodes]
        
        Returns:
            out: Updated node features [batch, n_nodes, out_features] or [n_nodes, out_features]
        """
        # Linear transformation
        support = torch.matmul(x, self.weight)
        
        # Graph convolution (message passing)
        output = torch.matmul(adj, support)
        
        if self.bias is not None:
            output = output + self.bias
        
        return output


# ============================================================================
# VGAE ENCODER
# ============================================================================

class VGAEEncoder(nn.Module):
    """
    Variational Graph Auto-Encoder Encoder.
    
    Uses two GCN layers:
    - First layer: shared feature extraction
    - Second layer: outputs mean (mu) and log-variance (logvar) for latent space
    """
    def __init__(self, input_dim, hidden_dim, latent_dim, num_gcn_layers=2, dropout=0.0):
        super(VGAEEncoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_gcn_layers = num_gcn_layers
        self.dropout = dropout
        
        # Build GCN layers
        self.gcn_layers = nn.ModuleList()
        
        # First layer: input_dim -> hidden_dim
        self.gcn_layers.append(GraphConvolution(input_dim, hidden_dim))
        
        # Additional hidden layers
        for _ in range(num_gcn_layers - 2):
            self.gcn_layers.append(GraphConvolution(hidden_dim, hidden_dim))
        
        # Mean and log-variance GCN layers
        self.gc_mu = GraphConvolution(hidden_dim, latent_dim)
        self.gc_logvar = GraphConvolution(hidden_dim, latent_dim)
    
    def forward(self, x, adj):
        """
        Encode node features into latent space.
        
        Args:
            x: Node features [batch, n_nodes, input_dim]
            adj: Normalized adjacency [batch, n_nodes, n_nodes]
        
        Returns:
            mu: Mean of latent distribution [batch, n_nodes, latent_dim]
            logvar: Log-variance of latent distribution [batch, n_nodes, latent_dim]
        """
        # Pass through GCN layers
        hidden = x
        for gcn in self.gcn_layers:
            hidden = F.relu(gcn(hidden, adj))
            if self.dropout > 0:
                hidden = F.dropout(hidden, p=self.dropout, training=self.training)
        
        # Mean and log-variance
        mu = self.gc_mu(hidden, adj)
        logvar = self.gc_logvar(hidden, adj)
        
        return mu, logvar


# ============================================================================
# VGAE DECODER
# ============================================================================

class InnerProductDecoder(nn.Module):
    """
    Inner Product Decoder for graph reconstruction.
    
    Reconstructs adjacency matrix: A_hat = sigmoid(Z @ Z^T)
    """
    def __init__(self):
        super(InnerProductDecoder, self).__init__()
    
    def forward(self, z):
        """
        Decode latent representations to adjacency matrix.
        
        Args:
            z: Latent representations [batch, n_nodes, latent_dim]
        
        Returns:
            adj_recon: Reconstructed adjacency [batch, n_nodes, n_nodes]
        """
        # Inner product: Z @ Z^T
        adj_recon = torch.matmul(z, z.transpose(-2, -1))
        return adj_recon


# ============================================================================
# FULL VGAE MODEL WITH LEARNABLE EMBEDDINGS
# ============================================================================

class VGAE(nn.Module):
    """
    Variational Graph Auto-Encoder for Session Data with Learnable Item Embeddings.
    
    This model includes:
    1. Learnable item embeddings (like SR-GNN) - captures item semantics
    2. Optional structural features - captures graph structure
    3. GCN encoder - processes graph structure
    4. Inner product decoder - reconstructs adjacency
    
    Pipeline:
    1. Look up item embeddings + add structural features
    2. Encode session graph → latent distribution (mu, logvar)
    3. Sample from latent distribution using reparameterization trick
    4. Decode latent representation → reconstructed adjacency
    
    Loss:
    - Reconstruction loss: BCE between original and reconstructed adjacency
    - KL divergence: regularization to keep latent space close to N(0,1)
    
    Recommended embedding_dim:
    - Small datasets: 32-64
    - Medium datasets: 64-128
    - Large datasets: 128-256
    """
    def __init__(self, n_items, embedding_dim=64, hidden_dim=64, latent_dim=32,
                 num_gcn_layers=2, dropout=0.0, use_structural_features=True,
                 structural_feature_dim=8):
        """
        Args:
            n_items: Number of unique items in vocabulary
            embedding_dim: Dimension of learnable item embeddings (default: 64)
            hidden_dim: Hidden dimension for GCN layers (default: 64)
            latent_dim: Latent dimension for VAE (default: 32)
            num_gcn_layers: Number of GCN layers in encoder (default: 2)
            dropout: Dropout rate (default: 0.0)
            use_structural_features: Whether to add structural features (default: True)
            structural_feature_dim: Dimension of structural feature projection (default: 8)
        """
        super(VGAE, self).__init__()
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.use_structural_features = use_structural_features
        self.structural_feature_dim = structural_feature_dim
        
        # Learnable item embeddings (like SR-GNN)
        # +1 for padding token at index 0
        self.item_embedding = nn.Embedding(n_items + 1, embedding_dim, padding_idx=0)
        
        # Structural feature projection (if used)
        # Raw structural features: [in_degree, out_degree, position, frequency, ...]
        self.n_structural_features = 6  # Number of raw structural features
        if use_structural_features:
            self.structural_proj = nn.Linear(self.n_structural_features, structural_feature_dim)
            input_dim = embedding_dim + structural_feature_dim
        else:
            input_dim = embedding_dim
        
        self.input_dim = input_dim
        
        # Encoder and decoder
        self.encoder = VGAEEncoder(input_dim, hidden_dim, latent_dim, num_gcn_layers, dropout)
        self.decoder = InnerProductDecoder()
        
        # Initialize weights
        self.reset_parameters()
    
    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.embedding_dim)
        self.item_embedding.weight.data.uniform_(-stdv, stdv)
        # Set padding embedding to zeros
        self.item_embedding.weight.data[0].fill_(0)
    
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick: z = mu + std * epsilon
        
        This allows gradients to flow through the sampling operation.
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + std * eps
        else:
            return mu
    
    def forward(self, item_ids, adj, structural_features=None):
        """
        Forward pass through VGAE.
        
        Args:
            item_ids: Item indices [batch, n_nodes] (LongTensor)
            adj: Normalized adjacency [batch, n_nodes, n_nodes]
            structural_features: Optional structural features [batch, n_nodes, n_structural_features]
        
        Returns:
            adj_recon: Reconstructed adjacency [batch, n_nodes, n_nodes]
            mu: Latent mean [batch, n_nodes, latent_dim]
            logvar: Latent log-variance [batch, n_nodes, latent_dim]
        """
        # Get item embeddings
        x = self.item_embedding(item_ids)  # [batch, n_nodes, embedding_dim]
        
        # Add structural features if provided
        if self.use_structural_features and structural_features is not None:
            struct_emb = self.structural_proj(structural_features)  # [batch, n_nodes, structural_feature_dim]
            x = torch.cat([x, struct_emb], dim=-1)  # [batch, n_nodes, embedding_dim + structural_feature_dim]
        
        # Encode
        mu, logvar = self.encoder(x, adj)
        
        # Sample latent representation
        z = self.reparameterize(mu, logvar)
        
        # Decode
        adj_recon = self.decoder(z)
        
        return adj_recon, mu, logvar
    
    def get_embedding(self, item_ids, adj, structural_features=None, mask=None):
        """
        Get session embedding (use mean of latent distribution).
        
        For anomaly detection, we use the mean (mu) as the embedding.
        
        Args:
            item_ids: Item indices [batch, n_nodes]
            adj: Normalized adjacency [batch, n_nodes, n_nodes]
            structural_features: Optional structural features [batch, n_nodes, n_structural_features]
            mask: Node mask [batch, n_nodes] for proper averaging
        
        Returns:
            embedding: Session embedding [batch, latent_dim]
        """
        # Get item embeddings
        x = self.item_embedding(item_ids)
        
        # Add structural features if provided
        if self.use_structural_features and structural_features is not None:
            struct_emb = self.structural_proj(structural_features)
            x = torch.cat([x, struct_emb], dim=-1)
        
        # Encode
        mu, _ = self.encoder(x, adj)
        
        # Aggregate node embeddings to get session embedding
        if mask is not None:
            # Masked mean pooling
            mask_expanded = mask.unsqueeze(-1)  # [batch, n_nodes, 1]
            sum_embeddings = (mu * mask_expanded).sum(dim=1)  # [batch, latent_dim]
            count = mask.sum(dim=1, keepdim=True).clamp(min=1)  # [batch, 1]
            embedding = sum_embeddings / count
        else:
            # Simple mean pooling
            embedding = mu.mean(dim=1)  # [batch, latent_dim]
        
        return embedding
    
    def get_node_embeddings(self, item_ids, adj, structural_features=None):
        """
        Get node-level embeddings (latent mean for each node).
        
        Args:
            item_ids: Item indices [batch, n_nodes]
            adj: Normalized adjacency [batch, n_nodes, n_nodes]
            structural_features: Optional structural features
        
        Returns:
            mu: Node embeddings [batch, n_nodes, latent_dim]
        """
        x = self.item_embedding(item_ids)
        
        if self.use_structural_features and structural_features is not None:
            struct_emb = self.structural_proj(structural_features)
            x = torch.cat([x, struct_emb], dim=-1)
        
        mu, _ = self.encoder(x, adj)
        return mu


# ============================================================================
# SESSION GRAPH UTILITIES
# ============================================================================

def compute_structural_features(session, node_ids, node_to_idx, adj):
    """
    Compute structural features for each node in the session graph.
    
    Features:
    1. In-degree (normalized)
    2. Out-degree (normalized)
    3. First position in session (normalized)
    4. Last position in session (normalized)
    5. Frequency in session (normalized)
    6. Betweenness-like centrality (simplified)
    
    Args:
        session: Original session sequence
        node_ids: List of unique item ids
        node_to_idx: Mapping from item id to node index
        adj: Adjacency matrix (before normalization)
    
    Returns:
        features: [n_nodes, 6] structural feature matrix
    """
    n_nodes = len(node_ids)
    session_len = len(session)
    
    features = np.zeros((n_nodes, 6), dtype=np.float32)
    
    for i, item_id in enumerate(node_ids):
        # Get positions of this item in session
        positions = [pos for pos, item in enumerate(session) if item == item_id]
        
        # 1. In-degree (normalized by max possible)
        in_degree = adj[:, i].sum() if adj.shape[0] > 0 else 0
        features[i, 0] = in_degree / max(n_nodes - 1, 1)
        
        # 2. Out-degree (normalized)
        out_degree = adj[i, :].sum() if adj.shape[1] > 0 else 0
        features[i, 1] = out_degree / max(n_nodes - 1, 1)
        
        # 3. First position (normalized)
        first_pos = positions[0] if positions else 0
        features[i, 2] = first_pos / max(session_len - 1, 1)
        
        # 4. Last position (normalized)
        last_pos = positions[-1] if positions else 0
        features[i, 3] = last_pos / max(session_len - 1, 1)
        
        # 5. Frequency (normalized)
        freq = len(positions)
        features[i, 4] = freq / session_len
        
        # 6. Position spread (last - first, normalized) - indicates item importance
        pos_spread = (last_pos - first_pos) / max(session_len - 1, 1) if positions else 0
        features[i, 5] = pos_spread
    
    return features


def build_session_graph_for_vgae(session, n_items=None, compute_structural=True):
    """
    Build a session graph from a sequence of items for VGAE.
    
    For a session [1, 2, 3, 2], the graph has:
    - Nodes: unique items in the session
    - Edges: transitions between consecutive items
    
    Args:
        session: list of item indices
        n_items: total number of items in vocabulary
        compute_structural: whether to compute structural features
    
    Returns:
        node_ids: [n_nodes] original item ids (for embedding lookup)
        adj_normalized: [n_nodes, n_nodes] normalized adjacency matrix
        adj_label: [n_nodes, n_nodes] original adjacency (for reconstruction target)
        structural_features: [n_nodes, 6] structural features (if compute_structural=True)
    """
    # Get unique items (preserve order of first appearance)
    seen = set()
    node_ids = []
    for item in session:
        if item not in seen:
            seen.add(item)
            node_ids.append(item)
    
    node_to_idx = {node: idx for idx, node in enumerate(node_ids)}
    n_nodes = len(node_ids)
    
    # Build adjacency matrix from transitions
    adj = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    for i in range(len(session) - 1):
        u = node_to_idx[session[i]]
        v = node_to_idx[session[i + 1]]
        adj[u][v] = 1.0
        adj[v][u] = 1.0  # Make undirected for VGAE
    
    # Compute structural features before normalization
    structural_features = None
    if compute_structural:
        structural_features = compute_structural_features(session, node_ids, node_to_idx, adj)
    
    # Add self-loops
    adj_with_self = adj + np.eye(n_nodes, dtype=np.float32)
    
    # Normalize adjacency: D^(-1/2) A D^(-1/2)
    degree = adj_with_self.sum(axis=1)
    degree_inv_sqrt = np.power(degree, -0.5)
    degree_inv_sqrt[np.isinf(degree_inv_sqrt)] = 0.0
    D_inv_sqrt = np.diag(degree_inv_sqrt)
    adj_normalized = D_inv_sqrt @ adj_with_self @ D_inv_sqrt
    
    # Adjacency label (original adjacency for reconstruction loss)
    adj_label = adj + np.eye(n_nodes, dtype=np.float32)  # Include self-loops in target
    
    return node_ids, adj_normalized, adj_label, structural_features


def process_sessions_for_vgae(sessions, n_items=None, max_nodes=None, compute_structural=True):
    """
    Process multiple sessions into batched format for VGAE with learnable embeddings.
    
    Args:
        sessions: list of sessions
        n_items: total number of items
        max_nodes: maximum number of nodes (for padding)
        compute_structural: whether to compute structural features
    
    Returns:
        item_ids: [batch, max_nodes] item indices for embedding lookup
        adj_normalized: [batch, max_nodes, max_nodes]
        adj_label: [batch, max_nodes, max_nodes]
        structural_features: [batch, max_nodes, 6] or None
        masks: [batch, max_nodes] binary mask for valid nodes
    """
    batch_size = len(sessions)
    
    # Process each session
    node_ids_list = []
    adj_norm_list = []
    adj_label_list = []
    struct_feat_list = []
    n_nodes_list = []
    
    for session in sessions:
        node_ids, adj_norm, adj_label, struct_feat = build_session_graph_for_vgae(
            session, n_items, compute_structural
        )
        node_ids_list.append(node_ids)
        adj_norm_list.append(adj_norm)
        adj_label_list.append(adj_label)
        struct_feat_list.append(struct_feat)
        n_nodes_list.append(len(node_ids))
    
    # Find max nodes for padding
    if max_nodes is None:
        max_nodes = max(n_nodes_list)
    
    # Create padded arrays
    item_ids = np.zeros((batch_size, max_nodes), dtype=np.int64)  # 0 is padding
    adj_normalized = np.zeros((batch_size, max_nodes, max_nodes), dtype=np.float32)
    adj_label = np.zeros((batch_size, max_nodes, max_nodes), dtype=np.float32)
    masks = np.zeros((batch_size, max_nodes), dtype=np.float32)
    
    if compute_structural:
        structural_features = np.zeros((batch_size, max_nodes, 6), dtype=np.float32)
    else:
        structural_features = None
    
    for i, (nids, adj_n, adj_l, struct_f, n) in enumerate(
        zip(node_ids_list, adj_norm_list, adj_label_list, struct_feat_list, n_nodes_list)
    ):
        # Item ids (+1 to reserve 0 for padding)
        item_ids[i, :n] = [nid + 1 for nid in nids]  # Shift by 1
        adj_normalized[i, :n, :n] = adj_n
        adj_label[i, :n, :n] = adj_l
        masks[i, :n] = 1.0
        
        if compute_structural and struct_f is not None:
            structural_features[i, :n, :] = struct_f
    
    return item_ids, adj_normalized, adj_label, structural_features, masks


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

def vgae_loss(adj_recon, adj_label, mu, logvar, mask=None, beta=1.0):
    """
    Compute VGAE loss: Reconstruction + KL divergence.
    
    Args:
        adj_recon: Reconstructed adjacency [batch, n_nodes, n_nodes]
        adj_label: Target adjacency [batch, n_nodes, n_nodes]
        mu: Latent mean [batch, n_nodes, latent_dim]
        logvar: Latent log-variance [batch, n_nodes, latent_dim]
        mask: Node mask [batch, n_nodes] for handling padding
        beta: Weight for KL divergence term
    
    Returns:
        loss: Total loss
        recon_loss: Reconstruction loss
        kl_loss: KL divergence loss
    """
    batch_size = adj_recon.size(0)
    
    # Apply mask to adjacency matrices if provided
    if mask is not None:
        # Create 2D mask [batch, n_nodes, n_nodes]
        mask_2d = mask.unsqueeze(-1) * mask.unsqueeze(-2)
        adj_recon = adj_recon * mask_2d
        adj_label = adj_label * mask_2d
        
        # Create mask for mu/logvar
        mask_3d = mask.unsqueeze(-1)  # [batch, n_nodes, 1]
        mu = mu * mask_3d
        logvar = logvar * mask_3d
    
    # Reconstruction loss (binary cross entropy)
    adj_recon_flat = adj_recon.view(batch_size, -1)
    adj_label_flat = adj_label.view(batch_size, -1)
    
    # Use BCE with logits for numerical stability
    recon_loss = F.binary_cross_entropy_with_logits(
        adj_recon_flat, adj_label_flat, reduction='mean'
    )
    
    # KL divergence: -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Total loss
    loss = recon_loss + beta * kl_loss
    
    return loss, recon_loss, kl_loss


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def trans_to_cuda(variable):
    """Move tensor to CUDA if available."""
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

class SessionDataset:
    """
    Dataset class for session data with learnable embeddings.
    """
    def __init__(self, sessions, n_items=None, compute_structural=True):
        self.sessions = sessions
        self.n_items = n_items
        self.n_sessions = len(sessions)
        self.compute_structural = compute_structural
    
    def __len__(self):
        return self.n_sessions
    
    def get_batch(self, batch_indices):
        """Get a batch of sessions."""
        batch_sessions = [self.sessions[i] for i in batch_indices]
        return process_sessions_for_vgae(
            batch_sessions, self.n_items, compute_structural=self.compute_structural
        )


def train_vgae(model, train_sessions, n_items=None, epochs=50, batch_size=32, 
               lr=0.001, beta=1.0, early_stopping_patience=10, verbose=True,
               use_structural_features=True):
    """
    Train VGAE model on session data.
    
    Args:
        model: VGAE model
        train_sessions: list of training sessions
        n_items: total number of items
        epochs: number of training epochs
        batch_size: batch size
        lr: learning rate
        beta: KL divergence weight
        early_stopping_patience: patience for early stopping
        verbose: print training progress
        use_structural_features: whether to use structural features
    
    Returns:
        model: trained model
        history: training history
    """
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    dataset = SessionDataset(train_sessions, n_items, compute_structural=use_structural_features)
    n_batches = (len(dataset) + batch_size - 1) // batch_size
    
    best_loss = float('inf')
    patience_counter = 0
    history = {'loss': [], 'recon_loss': [], 'kl_loss': []}
    
    for epoch in range(epochs):
        # Shuffle data
        indices = np.random.permutation(len(dataset))
        
        epoch_loss = 0.0
        epoch_recon = 0.0
        epoch_kl = 0.0
        
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(dataset))
            batch_indices = indices[start_idx:end_idx]
            
            # Get batch data
            item_ids, adj_norm, adj_label, structural_features, masks = dataset.get_batch(batch_indices)
            
            # Convert to tensors
            item_ids = trans_to_cuda(torch.LongTensor(item_ids))
            adj_norm = trans_to_cuda(torch.FloatTensor(adj_norm))
            adj_label = trans_to_cuda(torch.FloatTensor(adj_label))
            masks = trans_to_cuda(torch.FloatTensor(masks))
            
            if structural_features is not None:
                structural_features = trans_to_cuda(torch.FloatTensor(structural_features))
            
            # Forward pass
            optimizer.zero_grad()
            adj_recon, mu, logvar = model(item_ids, adj_norm, structural_features)
            
            # Compute loss
            loss, recon_loss, kl_loss = vgae_loss(
                adj_recon, adj_label, mu, logvar, masks, beta
            )
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_recon += recon_loss.item()
            epoch_kl += kl_loss.item()
        
        # Average losses
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


def extract_vgae_embeddings(model, sessions, n_items=None, batch_size=100,
                            use_structural_features=True):
    """
    Extract session embeddings from trained VGAE.
    
    Args:
        model: trained VGAE model
        sessions: list of sessions
        n_items: total number of items
        batch_size: batch size for processing
        use_structural_features: whether to use structural features
    
    Returns:
        embeddings: [n_sessions, latent_dim] session embeddings
    """
    model.eval()
    all_embeddings = []
    
    n_sessions = len(sessions)
    n_batches = (n_sessions + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_sessions)
            batch_sessions = sessions[start_idx:end_idx]
            
            # Process sessions
            item_ids, adj_norm, _, structural_features, masks = process_sessions_for_vgae(
                batch_sessions, n_items, compute_structural=use_structural_features
            )
            
            # Convert to tensors
            item_ids = trans_to_cuda(torch.LongTensor(item_ids))
            adj_norm = trans_to_cuda(torch.FloatTensor(adj_norm))
            masks = trans_to_cuda(torch.FloatTensor(masks))
            
            if structural_features is not None:
                structural_features = trans_to_cuda(torch.FloatTensor(structural_features))
            
            # Get embeddings with proper mask handling
            embeddings = model.get_embedding(item_ids, adj_norm, structural_features, masks)
            
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
    
    # Adjust embedding_dim based on n_items
    if n_items > 10000:
        config['embedding_dim'] = max(config['embedding_dim'], 128)
    elif n_items > 1000:
        config['embedding_dim'] = max(config['embedding_dim'], 64)
    
    return config
