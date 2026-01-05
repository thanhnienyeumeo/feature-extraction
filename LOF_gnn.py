#!/usr/bin/env python
# coding: utf-8
"""
Anomaly Detection using GNN Hidden Representations + LOF Pipeline - Supports CSV and PKL input

This script uses GNN's hidden layer output as input features for LOF.
Pipeline: Raw Features → Build Sample Graph → GNN → Hidden Representations → LOF → Anomaly Score

Key difference from AutoEncoder approach:
- AutoEncoder: Uses reconstruction error (how different the output is from input)
- GNN: Uses hidden representations (learned features that capture sample relationships in a graph)

How GNN hidden representations work with LOF:
1. We build a k-NN graph where samples are nodes, connected to their similar neighbors
2. GNN performs message passing: each node aggregates information from its neighbors
3. After message passing, each node has a "hidden representation" that encodes:
   - Its own features
   - Information from its neighborhood (context)
4. Normal samples cluster together in this hidden space (similar neighborhoods)
5. Anomalies have different hidden representations (different neighborhood structure)
6. LOF detects these anomalies based on local density in the hidden space

This script can work with:
1. Original CSV files (e.g., weekr4.2.csv, dayr4.2.csv)
2. Pickle files with percentile transformation (e.g., week-r5.2-percentile30.pkl)

Usage:
    python LOF_gnn.py /path/to/weekr4.2.csv
    python LOF_gnn.py /path/to/week-r5.2-percentile30.pkl
"""

import pandas as pd
import numpy as np
import sys
import math
import torch
from torch import nn
from torch.nn import Module, Parameter
import torch.nn.functional as F
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.preprocessing import StandardScaler

print('=' * 80)
print('Anomaly Detection using GNN Hidden Representations + LOF Pipeline')
print('Pipeline: Raw Features → Sample Graph → GNN → Hidden Repr → LOF')
print('Supports both CSV and PKL input files')
print('=' * 80)

# ============================================================================
# GNN COMPONENTS (adapted from gnn.py for tabular data)
# ============================================================================

class GNN(Module):
    """
    Graph Neural Network module that performs message passing on a graph.
    
    For anomaly detection:
    - Input: feature vectors as initial hidden states, adjacency matrix from k-NN graph
    - Output: refined hidden representations that capture both features and graph structure
    """
    def __init__(self, hidden_size, step=1):
        super(GNN, self).__init__()
        self.step = step
        self.hidden_size = hidden_size
        self.input_size = hidden_size * 2
        self.gate_size = 3 * hidden_size
        self.w_ih = Parameter(torch.Tensor(self.gate_size, self.input_size))
        self.w_hh = Parameter(torch.Tensor(self.gate_size, self.hidden_size))
        self.b_ih = Parameter(torch.Tensor(self.gate_size))
        self.b_hh = Parameter(torch.Tensor(self.gate_size))
        self.b_iah = Parameter(torch.Tensor(self.hidden_size))
        self.b_oah = Parameter(torch.Tensor(self.hidden_size))

        self.linear_edge_in = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_out = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

    def GNNCell(self, A, hidden):
        """
        Single GNN cell computation.
        
        A: Adjacency matrix [batch, n_nodes, 2*n_nodes] containing in-edges and out-edges
        hidden: Node hidden states [batch, n_nodes, hidden_size]
        """
        # Message passing from incoming edges
        input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah
        # Message passing from outgoing edges
        input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah
        # Concatenate both directions
        inputs = torch.cat([input_in, input_out], 2)
        
        # GRU-style gating mechanism
        gi = F.linear(inputs, self.w_ih, self.b_ih)
        gh = F.linear(hidden, self.w_hh, self.b_hh)
        i_r, i_i, i_n = gi.chunk(3, 2)
        h_r, h_i, h_n = gh.chunk(3, 2)
        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + resetgate * h_n)
        hy = newgate + inputgate * (hidden - newgate)
        return hy

    def forward(self, A, hidden):
        """
        Forward pass: apply GNN cell for 'step' iterations.
        
        Returns: Updated hidden representations after message passing
        """
        for i in range(self.step):
            hidden = self.GNNCell(A, hidden)
        return hidden


class TabularGNNEncoder(Module):
    """
    GNN-based encoder for tabular data.
    
    This module:
    1. Projects input features to hidden dimension
    2. Applies GNN message passing using a sample-similarity graph
    3. Returns hidden representations suitable for anomaly detection
    
    The key insight: normal samples form tight clusters in the graph, so their
    hidden representations (after message passing) will be similar. Anomalies
    have different neighborhood structures, resulting in distinct hidden representations.
    """
    def __init__(self, input_dim, hidden_dim, gnn_steps=1):
        super(TabularGNNEncoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Project input features to hidden dimension
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # GNN for message passing
        self.gnn = GNN(hidden_dim, step=gnn_steps)
        
        # Output projection (optional, for reconstruction objective)
        self.output_projection = nn.Linear(hidden_dim, input_dim)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
    
    def forward(self, x, A):
        """
        Forward pass through the GNN encoder.
        
        Args:
            x: Input features [batch_size, n_samples, input_dim]
            A: Adjacency matrix [batch_size, n_samples, 2*n_samples]
        
        Returns:
            hidden: Hidden representations [batch_size, n_samples, hidden_dim]
            reconstructed: Reconstructed input (for training) [batch_size, n_samples, input_dim]
        """
        # Project to hidden dimension
        hidden = self.input_projection(x)
        hidden = F.relu(hidden)
        
        # Apply GNN message passing
        hidden = self.gnn(A, hidden)
        
        # Reconstruct for training objective
        reconstructed = self.output_projection(hidden)
        
        return hidden, reconstructed
    
    def get_hidden(self, x, A):
        """Get only the hidden representations (for inference)."""
        hidden = self.input_projection(x)
        hidden = F.relu(hidden)
        hidden = self.gnn(A, hidden)
        return hidden


def build_knn_adjacency(X, k=10):
    """
    Build a k-NN adjacency matrix from feature vectors.
    
    This creates a graph where:
    - Each sample is a node
    - Edges connect each sample to its k most similar neighbors
    - Edge weights are based on similarity (closer = higher weight)
    
    For GNN, we need both in-edges and out-edges (same in undirected case).
    
    Args:
        X: Feature matrix [n_samples, n_features]
        k: Number of nearest neighbors
    
    Returns:
        A: Adjacency matrix [1, n_samples, 2*n_samples] for GNN
    """
    n_samples = X.shape[0]
    k = min(k, n_samples - 1)  # Can't have more neighbors than samples
    
    # Find k-nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(X)
    distances, indices = nbrs.kneighbors(X)
    
    # Build adjacency matrix (excluding self-loops from k-NN, but we'll normalize)
    # Create sparse-like representation first, then convert
    A_in = np.zeros((n_samples, n_samples), dtype=np.float32)
    
    for i in range(n_samples):
        for j_idx in range(1, k+1):  # Skip first (self)
            j = indices[i, j_idx]
            # Weight by inverse distance (closer = stronger connection)
            weight = 1.0 / (distances[i, j_idx] + 1e-6)
            A_in[i, j] = weight
    
    # Normalize rows
    row_sums = A_in.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    A_in = A_in / row_sums
    
    # For undirected graph, out-edges = in-edges transposed
    A_out = A_in.T.copy()
    row_sums = A_out.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    A_out = A_out / row_sums
    
    # Concatenate for GNN format [in_edges, out_edges]
    A = np.concatenate([A_in, A_out], axis=1)
    
    # Add batch dimension
    A = A[np.newaxis, :, :]
    
    return A


def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


def train_gnn_encoder(model, X_train, A_train, epochs=50, lr=0.001, batch_size=None):
    """
    Train the GNN encoder using reconstruction loss.
    
    The model learns to:
    1. Project features to hidden space
    2. Aggregate information from neighbors via message passing
    3. Reconstruct the original features
    
    This encourages the hidden representations to capture meaningful patterns.
    Normal samples with similar neighbors will have similar hidden representations.
    """
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    X_tensor = trans_to_cuda(torch.FloatTensor(X_train))
    A_tensor = trans_to_cuda(torch.FloatTensor(A_train))
    
    # Add batch dimension if needed
    if X_tensor.dim() == 2:
        X_tensor = X_tensor.unsqueeze(0)
    
    best_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        hidden, reconstructed = model(X_tensor, A_tensor)
        
        # Reconstruction loss
        loss = F.mse_loss(reconstructed, X_tensor)
        
        loss.backward()
        optimizer.step()
        
        # Early stopping
        if loss.item() < best_loss:
            best_loss = loss.item()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}')
    
    return model


def extract_hidden_representations(model, X, A):
    """
    Extract hidden representations from trained GNN.
    
    These hidden representations capture:
    - The sample's own features (via input projection)
    - Information from similar samples (via message passing)
    
    For anomaly detection:
    - Normal samples: similar hidden representations (cluster together)
    - Anomalies: distinct hidden representations (isolated)
    """
    model.eval()
    
    X_tensor = trans_to_cuda(torch.FloatTensor(X))
    A_tensor = trans_to_cuda(torch.FloatTensor(A))
    
    if X_tensor.dim() == 2:
        X_tensor = X_tensor.unsqueeze(0)
    
    with torch.no_grad():
        hidden = model.get_hidden(X_tensor, A_tensor)
    
    # Remove batch dimension and convert to numpy
    hidden = trans_to_cpu(hidden).squeeze(0).numpy()
    
    return hidden


# ============================================================================
# STEP 1: Load data (CSV or PKL)
# ============================================================================
if len(sys.argv) < 2:
    print("\nUsage: python LOF_gnn.py <path_to_data>")
    print("Example: python LOF_gnn.py /home/user/r4.2/ExtractedData/weekr4.2.csv")
    sys.exit(1)

path = sys.argv[1]
print(f'\nLoading data from: {path}')

# Detect file type and load accordingly
if path.endswith('.pkl') or path.endswith('.pickle'):
    data = pd.read_pickle(path)
    print(f'Loaded pickle file')
elif path.endswith('.csv') or path.endswith('.csv.gz'):
    data = pd.read_csv(path)
    print(f'Loaded CSV file')
else:
    # Try CSV first, then pickle
    try:
        data = pd.read_csv(path)
        print(f'Loaded as CSV file')
    except:
        data = pd.read_pickle(path)
        print(f'Loaded as pickle file')

print(f'Data shape: {data.shape[0]} rows, {data.shape[1]} columns')

# ============================================================================
# STEP 2: Identify feature columns vs metadata columns
# ============================================================================
# These columns should NOT be used as features (metadata/labels)
removed_cols = [
    'user', 'day', 'week', 'starttime', 'endtime', 'sessionid', 'insider',
    'isweekday', 'isweekend',  # For day-level data
    'role', 'b_unit', 'f_unit', 'dept', 'team', 'project',  # Categorical user info
    'ITAdmin', 'O', 'C', 'E', 'A', 'N',  # User attributes
]
# Get feature columns (all numeric columns except removed ones)
x_cols = [col for col in data.columns if col not in removed_cols]

# Further filter to only numeric columns
numeric_cols = data[x_cols].select_dtypes(include=[np.number]).columns.tolist()
x_cols = numeric_cols

print(f'\nUsing {len(x_cols)} features for anomaly detection')
print(f'Sample features: {x_cols[:5]}...')

# ============================================================================
# STEP 3: Split data by time (first half = train, second half = test)
# ============================================================================
run = 1
np.random.seed(run)
torch.manual_seed(run)

max_week = data['week'].max()
data1stHalf = data[data.week <= max_week / 2]
dataTest = data[data.week > max_week / 2]

print(f'\nTrain period: weeks 0-{int(max_week/2)} ({len(data1stHalf)} rows)')
print(f'Test period: weeks {int(max_week/2)+1}-{int(max_week)} ({len(dataTest)} rows)')

# ============================================================================
# STEP 4: Select training users (200 random users)
# ============================================================================
all_train_users = list(set(data1stHalf.user))
nUsers = np.random.permutation(all_train_users)
trainUsers = nUsers[:min(200000, len(nUsers))]  # Use up to 200 users

print(f'\nTraining on {len(trainUsers)} randomly selected users')

# ============================================================================
# STEP 5: Prepare training and test data
# ============================================================================
xTrain = data1stHalf[data1stHalf.user.isin(trainUsers)][x_cols].values
yTrain = data1stHalf[data1stHalf.user.isin(trainUsers)]['insider'].values
xTrain_not_insider = data1stHalf[data1stHalf['insider'] == 0]
xTrain_not_insider = xTrain_not_insider[xTrain_not_insider.user.isin(trainUsers)]
xTrain_not_insider = xTrain_not_insider[x_cols].values
yTrainBin_not_insider = 0
yTrainBin = yTrain > 0

# Test on only second half
xTest = dataTest[x_cols].values
yTest = dataTest['insider'].values
yTestBin = yTest > 0
print(f'Training samples if insider in training: {len(xTrain)} (insiders in train: {yTrainBin.sum()})')
print(f'Training samples if insider not in training: {len(xTrain_not_insider)} (not insiders in train)')
print(f'Test samples: {len(xTest)} (total insiders: {yTestBin.sum()})')

# Handle any NaN or infinite values
xTrain = np.nan_to_num(xTrain, nan=0.0, posinf=0.0, neginf=0.0)
xTrain_not_insider = np.nan_to_num(xTrain_not_insider, nan=0.0, posinf=0.0, neginf=0.0)
xTest = np.nan_to_num(xTest, nan=0.0, posinf=0.0, neginf=0.0)

# ============================================================================
# STEP 6: Standardize features
# ============================================================================
print('\nStandardizing features...')
scaler = StandardScaler()
xTrain_not_insider = scaler.fit_transform(xTrain_not_insider)
xTest = scaler.transform(xTest)

# Also prepare scaled data for insider in training scenario
scaler_with_insider = StandardScaler()
xTrain_scaled = scaler_with_insider.fit_transform(xTrain)
xTest_scaled_with_insider = scaler_with_insider.transform(dataTest[x_cols].values)
xTest_scaled_with_insider = np.nan_to_num(xTest_scaled_with_insider, nan=0.0, posinf=0.0, neginf=0.0)

# ============================================================================
# STEP 7: GNN Architecture Configuration
# ============================================================================
n_features = len(x_cols)
# Hidden dimension for GNN (similar to AutoEncoder bottleneck)
hidden_dim = max(16, 1 << (int(n_features / 4).bit_length()))
gnn_steps = 2  # Number of message passing iterations
k_neighbors = 20  # Number of neighbors for k-NN graph

print(f'\nGNN configuration:')
print(f'  Input dimension: {n_features}')
print(f'  Hidden dimension: {hidden_dim}')
print(f'  GNN steps: {gnn_steps}')
print(f'  k-NN neighbors: {k_neighbors}')

# ============================================================================
# STEP 8: PIPELINE FOR NOT INSIDER IN TRAINING
# ============================================================================
print('\n' + '=' * 80)
print('PIPELINE: NOT INSIDER IN TRAINING (GNN Hidden + LOF)')
print('=' * 80)

# Step 8.1: Build k-NN graph from training data
print('\n***** Building k-NN graph from training data...')
A_train_not_insider = build_knn_adjacency(xTrain_not_insider, k=k_neighbors)
print(f'Adjacency matrix shape: {A_train_not_insider.shape}')

# Step 8.2: Create and train GNN encoder
print('\n***** Training GNN Encoder...')
gnn_encoder_not_insider = TabularGNNEncoder(
    input_dim=n_features,
    hidden_dim=hidden_dim,
    gnn_steps=gnn_steps
)
gnn_encoder_not_insider = trans_to_cuda(gnn_encoder_not_insider)

gnn_encoder_not_insider = train_gnn_encoder(
    gnn_encoder_not_insider, 
    xTrain_not_insider, 
    A_train_not_insider,
    epochs=50,
    lr=0.001
)
print('GNN Encoder training completed')

# Step 8.3: Extract hidden representations for training data
print('\nExtracting hidden representations for training data...')
train_hidden_not_insider = extract_hidden_representations(
    gnn_encoder_not_insider, 
    xTrain_not_insider, 
    A_train_not_insider
)
print(f'Training hidden representations shape: {train_hidden_not_insider.shape}')

# Step 8.4: Build k-NN graph and extract hidden representations for test data
print('Building k-NN graph for test data...')
A_test = build_knn_adjacency(xTest, k=k_neighbors)

print('Extracting hidden representations for test data...')
test_hidden_not_insider = extract_hidden_representations(
    gnn_encoder_not_insider,
    xTest,
    A_test
)
print(f'Test hidden representations shape: {test_hidden_not_insider.shape}')

# Step 8.5: Train LOF on hidden representations
print('\n***** Training LOF on GNN hidden representations...')
lof_not_insider = LocalOutlierFactor(
    n_neighbors=20,
    contamination='auto',
    novelty=True,
    n_jobs=-1
)
lof_not_insider.fit(train_hidden_not_insider)
print('LOF training completed')

# Step 8.6: Get anomaly scores from LOF
print('Calculating anomaly scores...')
anomaly_scores_not_insiders = -lof_not_insider.score_samples(test_hidden_not_insider)

# ============================================================================
# STEP 9: PIPELINE FOR INSIDER IN TRAINING
# ============================================================================
print('\n' + '=' * 80)
print('PIPELINE: INSIDER IN TRAINING (GNN Hidden + LOF)')
print('=' * 80)

# Step 9.1: Build k-NN graph from training data (with insider)
print('\n***** Building k-NN graph from training data...')
A_train_insider = build_knn_adjacency(xTrain_scaled, k=k_neighbors)
print(f'Adjacency matrix shape: {A_train_insider.shape}')

# Step 9.2: Create and train GNN encoder
print('\n***** Training GNN Encoder...')
gnn_encoder_insider = TabularGNNEncoder(
    input_dim=n_features,
    hidden_dim=hidden_dim,
    gnn_steps=gnn_steps
)
gnn_encoder_insider = trans_to_cuda(gnn_encoder_insider)

gnn_encoder_insider = train_gnn_encoder(
    gnn_encoder_insider, 
    xTrain_scaled, 
    A_train_insider,
    epochs=50,
    lr=0.001
)
print('GNN Encoder training completed')

# Step 9.3: Extract hidden representations for training data
print('\nExtracting hidden representations for training data...')
train_hidden_insider = extract_hidden_representations(
    gnn_encoder_insider, 
    xTrain_scaled, 
    A_train_insider
)
print(f'Training hidden representations shape: {train_hidden_insider.shape}')

# Step 9.4: Build k-NN graph and extract hidden representations for test data
print('Building k-NN graph for test data...')
A_test_insider = build_knn_adjacency(xTest_scaled_with_insider, k=k_neighbors)

print('Extracting hidden representations for test data...')
test_hidden_insider = extract_hidden_representations(
    gnn_encoder_insider,
    xTest_scaled_with_insider,
    A_test_insider
)
print(f'Test hidden representations shape: {test_hidden_insider.shape}')

# Step 9.5: Train LOF on hidden representations
print('\n***** Training LOF on GNN hidden representations...')
lof_insider = LocalOutlierFactor(
    n_neighbors=20,
    contamination='auto',
    novelty=True,
    n_jobs=-1
)
lof_insider.fit(train_hidden_insider)
print('LOF training completed')

# Step 9.6: Get anomaly scores from LOF
print('Calculating anomaly scores...')
anomaly_scores_insiders = -lof_insider.score_samples(test_hidden_insider)

# ============================================================================
# STEP 10: Evaluate results
# ============================================================================
from utils import evaluate_results

print('\n' + '=' * 80)
print('RESULTS FOR GNN HIDDEN + LOF (INSIDER IN TRAINING)')
print('=' * 80)
evaluate_results(anomaly_scores_insiders, yTestBin)

print('\n' + '=' * 80)
print('RESULTS FOR GNN HIDDEN + LOF (NOT INSIDER IN TRAINING)')
print('=' * 80)
evaluate_results(anomaly_scores_not_insiders, yTestBin)

# ============================================================================
# EXPLANATION: How GNN Hidden Representations Work with LOF
# ============================================================================
print('\n' + '=' * 80)
print('HOW GNN HIDDEN REPRESENTATIONS WORK WITH LOF')
print('=' * 80)
print('''
The GNN + LOF pipeline works as follows:

1. BUILD SAMPLE GRAPH (k-NN):
   - Each sample (row) becomes a node in a graph
   - Edges connect each sample to its k most similar neighbors (based on features)
   - This captures the "neighborhood structure" of the data

2. GNN MESSAGE PASSING:
   - GNN iteratively updates each node's representation by aggregating information
     from its neighbors
   - After message passing, each node's hidden representation encodes:
     a) Its own features (via input projection)
     b) Information from its neighborhood (via message passing)

3. WHY THIS HELPS ANOMALY DETECTION:
   - Normal samples: Have normal neighbors → Their hidden representations
     cluster together in a consistent pattern
   - Anomalies: Have unusual neighborhood structure → Their hidden representations
     are different from the normal cluster

4. LOF ON HIDDEN SPACE:
   - LOF measures local density: how dense is the neighborhood of a point
   - In the GNN hidden space:
     - Normal samples: Dense neighborhoods (similar hidden representations)
     - Anomalies: Sparse neighborhoods (isolated hidden representations)
   - LOF score indicates how "outlying" each sample is

COMPARISON WITH AUTOENCODER APPROACH:
- AutoEncoder: Uses reconstruction error (how well input can be reconstructed)
  → Anomalies have high reconstruction error (model hasn't seen them)
- GNN: Uses learned representations that capture neighborhood structure
  → Anomalies have distinct representations (different neighborhood context)

The GNN approach can capture more complex relationships between samples
through the graph structure, potentially detecting anomalies that are
hard to find with reconstruction-based methods.
''')
