#!/usr/bin/env python
# coding: utf-8
"""
Anomaly Detection using Transformer AutoEncoder + LOF Pipeline - Supports CSV and PKL input

This script uses Transformer-based AutoEncoder's reconstruction error as input features for LOF.
Pipeline: Raw Features → Transformer AutoEncoder → Reconstruction Error → LOF → Anomaly Score

This script can work with:
1. Original CSV files (e.g., weekr4.2.csv, dayr4.2.csv)
2. Pickle files with percentile transformation (e.g., week-r5.2-percentile30.pkl)

Usage:
    python LOF_Transformer.py /path/to/weekr4.2.csv
    python LOF_Transformer.py /path/to/week-r5.2-percentile30.pkl
"""

import pandas as pd
import numpy as np
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('=' * 80)
print('Anomaly Detection using Transformer AutoEncoder + LOF Pipeline')
print('Pipeline: Raw Features → Transformer AutoEncoder → Reconstruction Error → LOF')
print(f'Using device: {device}')
print('Supports both CSV and PKL input files')
print('=' * 80)


# ============================================================================
# Transformer AutoEncoder Model
# ============================================================================
class TransformerAutoEncoder(nn.Module):
    """
    Transformer-based AutoEncoder for tabular data.
    
    Architecture:
    - Input projection: projects each feature to embedding dimension
    - Positional encoding: adds positional information
    - Transformer encoder: processes the sequence of feature embeddings
    - Output projection: reconstructs original features
    """
    
    def __init__(self, n_features, d_model=64, nhead=4, num_encoder_layers=2, 
                 dim_feedforward=128, dropout=0.1):
        super(TransformerAutoEncoder, self).__init__()
        
        self.n_features = n_features
        self.d_model = d_model
        
        # Input projection: each feature becomes a token
        self.input_projection = nn.Linear(1, d_model)
        
        # Positional encoding (learnable)
        self.positional_encoding = nn.Parameter(torch.randn(1, n_features, d_model))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        # Bottleneck (compression)
        self.bottleneck = nn.Sequential(
            nn.Linear(d_model * n_features, d_model * n_features // 4),
            nn.ReLU(),
            nn.Linear(d_model * n_features // 4, d_model * n_features),
            nn.ReLU()
        )
        
        # Output projection: reconstruct original features
        self.output_projection = nn.Linear(d_model, 1)
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Reshape input: (batch, n_features) -> (batch, n_features, 1)
        x = x.unsqueeze(-1)
        
        # Project each feature to d_model dimensions: (batch, n_features, d_model)
        x = self.input_projection(x)
        
        # Add positional encoding
        x = x + self.positional_encoding
        
        # Pass through transformer encoder
        x = self.transformer_encoder(x)
        
        # Bottleneck (flatten, compress, expand)
        x = x.reshape(batch_size, -1)
        x = self.bottleneck(x)
        x = x.reshape(batch_size, self.n_features, self.d_model)
        
        # Project back to original feature dimension
        x = self.output_projection(x)
        
        # Reshape output: (batch, n_features, 1) -> (batch, n_features)
        x = x.squeeze(-1)
        
        return x


def train_transformer_autoencoder(model, train_data, epochs=50, batch_size=256, lr=0.001):
    """Train the Transformer AutoEncoder."""
    model.train()
    model.to(device)
    
    # Create DataLoader
    train_tensor = torch.FloatTensor(train_data).to(device)
    dataset = TensorDataset(train_tensor, train_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    best_loss = float('inf')
    patience_counter = 0
    max_patience = 10
    
    for epoch in range(epochs):
        total_loss = 0
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        scheduler.step(avg_loss)
        
        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                print(f'  Early stopping at epoch {epoch + 1}')
                break
        
        if (epoch + 1) % 10 == 0:
            print(f'  Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.6f}')
    
    return model


def get_reconstruction_errors(model, data, batch_size=256):
    """Get reconstruction errors for the given data."""
    model.eval()
    model.to(device)
    
    data_tensor = torch.FloatTensor(data).to(device)
    dataset = TensorDataset(data_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    all_errors = []
    
    with torch.no_grad():
        for batch in dataloader:
            batch_x = batch[0]
            output = model(batch_x)
            # Calculate squared reconstruction error per feature
            errors = (batch_x - output) ** 2
            all_errors.append(errors.cpu().numpy())
    
    return np.vstack(all_errors)


# ============================================================================
# STEP 1: Load data (CSV or PKL)
# ============================================================================
if len(sys.argv) < 2:
    print("\nUsage: python LOF_Transformer.py <path_to_data>")
    print("Example: python LOF_Transformer.py /home/user/r4.2/ExtractedData/weekr4.2.csv")
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
# removed_cols.extend([col for col in data.columns if 'afterhour' in col or 'weekend' in col])
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

# Test on ALL data to get full picture
xTest_all = data[x_cols].values
yTest_all = data['insider'].values
yTestBin_all = yTest_all > 0
#Test on only second half
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
# STEP 7: Build Transformer AutoEncoder architecture
# ============================================================================
n_features = len(x_cols)

# Transformer hyperparameters (adaptive based on feature count)
d_model = max(32, min(128, 2 ** int(np.log2(n_features))))  # Embedding dimension
nhead = max(2, min(8, d_model // 16))  # Number of attention heads (must divide d_model)
# Ensure d_model is divisible by nhead
while d_model % nhead != 0:
    nhead -= 1
num_encoder_layers = 2
dim_feedforward = d_model * 2

print(f'\nTransformer AutoEncoder architecture:')
print(f'  Input features: {n_features}')
print(f'  Embedding dimension (d_model): {d_model}')
print(f'  Number of attention heads: {nhead}')
print(f'  Number of encoder layers: {num_encoder_layers}')
print(f'  Feedforward dimension: {dim_feedforward}')

# ============================================================================
# STEP 8: PIPELINE FOR NOT INSIDER IN TRAINING
# ============================================================================
print('\n' + '=' * 80)
print('PIPELINE: NOT INSIDER IN TRAINING')
print('=' * 80)

# Step 8.1: Train Transformer AutoEncoder on training data (not insider)
print('\n*****Training Transformer AutoEncoder...')
ae_not_insider = TransformerAutoEncoder(
    n_features=n_features,
    d_model=d_model,
    nhead=nhead,
    num_encoder_layers=num_encoder_layers,
    dim_feedforward=dim_feedforward,
    dropout=0.1
)
ae_not_insider = train_transformer_autoencoder(ae_not_insider, xTrain_not_insider, epochs=50)
print('Transformer AutoEncoder training completed')

# Step 8.2: Get reconstruction error for training data (to train LOF)
print('\nCalculating reconstruction errors for training data...')
train_reconstruction_errors = get_reconstruction_errors(ae_not_insider, xTrain_not_insider)
print(f'Training reconstruction error shape: {train_reconstruction_errors.shape}')

# Step 8.3: Get reconstruction error for test data
print('Calculating reconstruction errors for test data...')
test_reconstruction_errors = get_reconstruction_errors(ae_not_insider, xTest)
print(f'Test reconstruction error shape: {test_reconstruction_errors.shape}')

# Step 8.4: Train LOF on reconstruction errors
print('\n*****Training LOF on reconstruction errors...')
lof_not_insider = LocalOutlierFactor(
    n_neighbors=20,
    contamination='auto',
    novelty=True,
    n_jobs=-1
)
lof_not_insider.fit(train_reconstruction_errors)
print('LOF training completed')

# Step 8.5: Get anomaly scores from LOF
print('Calculating anomaly scores...')
anomaly_scores_not_insiders = -lof_not_insider.score_samples(test_reconstruction_errors)

# ============================================================================
# STEP 9: PIPELINE FOR INSIDER IN TRAINING
# ============================================================================
print('\n' + '=' * 80)
print('PIPELINE: INSIDER IN TRAINING')
print('=' * 80)

# Step 9.1: Train Transformer AutoEncoder on training data (with insider)
print('\n*****Training Transformer AutoEncoder...')
ae_insider = TransformerAutoEncoder(
    n_features=n_features,
    d_model=d_model,
    nhead=nhead,
    num_encoder_layers=num_encoder_layers,
    dim_feedforward=dim_feedforward,
    dropout=0.1
)
ae_insider = train_transformer_autoencoder(ae_insider, xTrain_scaled, epochs=50)
print('Transformer AutoEncoder training completed')

# Step 9.2: Get reconstruction error for training data (to train LOF)
print('\nCalculating reconstruction errors for training data...')
train_insider_reconstruction_errors = get_reconstruction_errors(ae_insider, xTrain_scaled)
print(f'Training reconstruction error shape: {train_insider_reconstruction_errors.shape}')

# Step 9.3: Get reconstruction error for test data
print('Calculating reconstruction errors for test data...')
test_insider_reconstruction_errors = get_reconstruction_errors(ae_insider, xTest_scaled_with_insider)
print(f'Test reconstruction error shape: {test_insider_reconstruction_errors.shape}')

# Step 9.4: Train LOF on reconstruction errors
print('\n*****Training LOF on reconstruction errors...')
lof_insider = LocalOutlierFactor(
    n_neighbors=20,
    contamination='auto',
    novelty=True,
    n_jobs=-1
)
lof_insider.fit(train_insider_reconstruction_errors)
print('LOF training completed')

# Step 9.5: Get anomaly scores from LOF
print('Calculating anomaly scores...')
anomaly_scores_insiders = -lof_insider.score_samples(test_insider_reconstruction_errors)

# ============================================================================
# STEP 10: Evaluate results
# ============================================================================
from utils import evaluate_results

print('\n' + '=' * 80)
print('RESULTS FOR TRANSFORMER AUTOENCODER + LOF (INSIDER IN TRAINING)')
print('=' * 80)
evaluate_results(anomaly_scores_insiders, yTestBin)

print('\n' + '=' * 80)
print('RESULTS FOR TRANSFORMER AUTOENCODER + LOF (NOT INSIDER IN TRAINING)')
print('=' * 80)
evaluate_results(anomaly_scores_not_insiders, yTestBin)
