#!/usr/bin/env python
# coding: utf-8
"""
Variational Graph Auto-Encoder (VGAE) for Session Data

This module implements VGAE architecture for learning session embeddings.
The VGAE learns to:
1. Encode session graphs into latent representations
2. Reconstruct the graph structure from latent representations

Architecture:
- Encoder: GCN-based encoder that outputs mean and log-variance
- Decoder: Inner product decoder for graph reconstruction
- Loss: Reconstruction loss + KL divergence

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
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VGAEEncoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # Shared GCN layer
        self.gc1 = GraphConvolution(input_dim, hidden_dim)
        
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
        # Shared hidden representation
        hidden = F.relu(self.gc1(x, adj))
        
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
# FULL VGAE MODEL
# ============================================================================

class VGAE(nn.Module):
    """
    Variational Graph Auto-Encoder for Session Data.
    
    Pipeline:
    1. Encode session graph → latent distribution (mu, logvar)
    2. Sample from latent distribution using reparameterization trick
    3. Decode latent representation → reconstructed adjacency
    
    Loss:
    - Reconstruction loss: BCE between original and reconstructed adjacency
    - KL divergence: regularization to keep latent space close to N(0,1)
    """
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VGAE, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        self.encoder = VGAEEncoder(input_dim, hidden_dim, latent_dim)
        self.decoder = InnerProductDecoder()
    
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
    
    def forward(self, x, adj):
        """
        Forward pass through VGAE.
        
        Args:
            x: Node features [batch, n_nodes, input_dim]
            adj: Normalized adjacency [batch, n_nodes, n_nodes]
        
        Returns:
            adj_recon: Reconstructed adjacency [batch, n_nodes, n_nodes]
            mu: Latent mean [batch, n_nodes, latent_dim]
            logvar: Latent log-variance [batch, n_nodes, latent_dim]
        """
        # Encode
        mu, logvar = self.encoder(x, adj)
        
        # Sample latent representation
        z = self.reparameterize(mu, logvar)
        
        # Decode
        adj_recon = self.decoder(z)
        
        return adj_recon, mu, logvar
    
    def get_embedding(self, x, adj):
        """
        Get session embedding (use mean of latent distribution).
        
        For anomaly detection, we use the mean (mu) as the embedding.
        
        Args:
            x: Node features [batch, n_nodes, input_dim]
            adj: Normalized adjacency [batch, n_nodes, n_nodes]
        
        Returns:
            embedding: Session embedding [batch, latent_dim]
        """
        mu, _ = self.encoder(x, adj)
        
        # Aggregate node embeddings to get session embedding
        # Option 1: Mean pooling over nodes
        # Option 2: Sum pooling over nodes
        # Option 3: Last node (for sequence)
        # We use mean pooling
        embedding = mu.mean(dim=-2)  # [batch, latent_dim]
        
        return embedding
    
    def get_node_embeddings(self, x, adj):
        """
        Get node-level embeddings (latent mean for each node).
        
        Args:
            x: Node features [batch, n_nodes, input_dim]
            adj: Normalized adjacency [batch, n_nodes, n_nodes]
        
        Returns:
            mu: Node embeddings [batch, n_nodes, latent_dim]
        """
        mu, _ = self.encoder(x, adj)
        return mu


# ============================================================================
# SESSION GRAPH UTILITIES
# ============================================================================

def build_session_graph_for_vgae(session, n_items=None):
    """
    Build a session graph from a sequence of items for VGAE.
    
    For a session [1, 2, 3, 2], the graph has:
    - Nodes: unique items in the session
    - Edges: transitions between consecutive items
    
    Args:
        session: list of item indices
        n_items: total number of items in vocabulary (for one-hot features)
    
    Returns:
        node_features: [n_nodes, feature_dim] node feature matrix
        adj: [n_nodes, n_nodes] normalized adjacency matrix
        adj_label: [n_nodes, n_nodes] original adjacency (for reconstruction target)
        node_ids: list of original item ids
    """
    # Get unique items
    node_ids = list(set(session))
    node_to_idx = {node: idx for idx, node in enumerate(node_ids)}
    n_nodes = len(node_ids)
    
    # Build adjacency matrix from transitions
    adj = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    for i in range(len(session) - 1):
        u = node_to_idx[session[i]]
        v = node_to_idx[session[i + 1]]
        adj[u][v] = 1.0
        adj[v][u] = 1.0  # Make undirected for VGAE
    
    # Add self-loops
    adj_with_self = adj + np.eye(n_nodes, dtype=np.float32)
    
    # Normalize adjacency: D^(-1/2) A D^(-1/2)
    degree = adj_with_self.sum(axis=1)
    degree_inv_sqrt = np.power(degree, -0.5)
    degree_inv_sqrt[np.isinf(degree_inv_sqrt)] = 0.0
    D_inv_sqrt = np.diag(degree_inv_sqrt)
    adj_normalized = D_inv_sqrt @ adj_with_self @ D_inv_sqrt
    
    # Node features: one-hot encoding based on position in session or item id
    # Option 1: Use item id as feature (if n_items provided)
    # Option 2: Use position encoding
    # Option 3: Use degree as feature
    # We'll use a combination of item id (scaled) and degree
    if n_items is not None and n_items > 0:
        # Normalized item id as feature
        node_features = np.array([[node_ids[i] / n_items, degree[i] / n_nodes] 
                                   for i in range(n_nodes)], dtype=np.float32)
    else:
        # Just use degree and node index
        node_features = np.array([[i / n_nodes, degree[i] / n_nodes] 
                                   for i in range(n_nodes)], dtype=np.float32)
    
    # Adjacency label (original adjacency for reconstruction loss)
    adj_label = adj + np.eye(n_nodes, dtype=np.float32)  # Include self-loops in target
    
    return node_features, adj_normalized, adj_label, node_ids


def process_sessions_for_vgae(sessions, n_items=None, max_nodes=None):
    """
    Process multiple sessions into batched format for VGAE.
    
    Args:
        sessions: list of sessions
        n_items: total number of items
        max_nodes: maximum number of nodes (for padding)
    
    Returns:
        node_features: [batch, max_nodes, feature_dim]
        adj_normalized: [batch, max_nodes, max_nodes]
        adj_label: [batch, max_nodes, max_nodes]
        masks: [batch, max_nodes] binary mask for valid nodes
    """
    batch_size = len(sessions)
    
    # Process each session
    features_list = []
    adj_norm_list = []
    adj_label_list = []
    n_nodes_list = []
    
    for session in sessions:
        features, adj_norm, adj_label, _ = build_session_graph_for_vgae(session, n_items)
        features_list.append(features)
        adj_norm_list.append(adj_norm)
        adj_label_list.append(adj_label)
        n_nodes_list.append(len(features))
    
    # Find max nodes for padding
    if max_nodes is None:
        max_nodes = max(n_nodes_list)
    
    feature_dim = features_list[0].shape[1]
    
    # Create padded arrays
    node_features = np.zeros((batch_size, max_nodes, feature_dim), dtype=np.float32)
    adj_normalized = np.zeros((batch_size, max_nodes, max_nodes), dtype=np.float32)
    adj_label = np.zeros((batch_size, max_nodes, max_nodes), dtype=np.float32)
    masks = np.zeros((batch_size, max_nodes), dtype=np.float32)
    
    for i, (feat, adj_n, adj_l, n) in enumerate(zip(features_list, adj_norm_list, adj_label_list, n_nodes_list)):
        node_features[i, :n, :] = feat
        adj_normalized[i, :n, :n] = adj_n
        adj_label[i, :n, :n] = adj_l
        masks[i, :n] = 1.0
    
    return node_features, adj_normalized, adj_label, masks


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
    Dataset class for session data.
    """
    def __init__(self, sessions, n_items=None):
        self.sessions = sessions
        self.n_items = n_items
        self.n_sessions = len(sessions)
    
    def __len__(self):
        return self.n_sessions
    
    def get_batch(self, batch_indices):
        """Get a batch of sessions."""
        batch_sessions = [self.sessions[i] for i in batch_indices]
        return process_sessions_for_vgae(batch_sessions, self.n_items)


def train_vgae(model, train_sessions, n_items=None, epochs=50, batch_size=32, 
               lr=0.001, beta=1.0, early_stopping_patience=10, verbose=True):
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
    
    Returns:
        model: trained model
        history: training history
    """
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    dataset = SessionDataset(train_sessions, n_items)
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
            node_features, adj_norm, adj_label, masks = dataset.get_batch(batch_indices)
            
            # Convert to tensors
            node_features = trans_to_cuda(torch.FloatTensor(node_features))
            adj_norm = trans_to_cuda(torch.FloatTensor(adj_norm))
            adj_label = trans_to_cuda(torch.FloatTensor(adj_label))
            masks = trans_to_cuda(torch.FloatTensor(masks))
            
            # Forward pass
            optimizer.zero_grad()
            adj_recon, mu, logvar = model(node_features, adj_norm)
            
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


def extract_vgae_embeddings(model, sessions, n_items=None, batch_size=100):
    """
    Extract session embeddings from trained VGAE.
    
    Args:
        model: trained VGAE model
        sessions: list of sessions
        n_items: total number of items
        batch_size: batch size for processing
    
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
            node_features, adj_norm, _, masks = process_sessions_for_vgae(
                batch_sessions, n_items
            )
            
            # Convert to tensors
            node_features = trans_to_cuda(torch.FloatTensor(node_features))
            adj_norm = trans_to_cuda(torch.FloatTensor(adj_norm))
            masks = trans_to_cuda(torch.FloatTensor(masks))
            
            # Get embeddings
            embeddings = model.get_embedding(node_features, adj_norm)
            
            # Apply mask for proper mean (accounting for padding)
            # Note: get_embedding already does mean pooling, but we could refine this
            
            embeddings = trans_to_cpu(embeddings).numpy()
            all_embeddings.append(embeddings)
    
    return np.concatenate(all_embeddings, axis=0)
