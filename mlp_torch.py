#!/usr/bin/env python
# coding: utf-8
"""
Anomaly Detection using Isolation Forest - Supports CSV and PKL input

This script can work with:
1. Original CSV files (e.g., weekr4.2.csv, dayr4.2.csv)
2. Pickle files with percentile transformation (e.g., week-r5.2-percentile30.pkl)

Usage:
    python example_anomaly_detection_iforest.py /path/to/weekr4.2.csv
    python example_anomaly_detection_iforest.py /path/to/week-r5.2-percentile30.pkl
"""

import pandas as pd
import numpy as np
import sys
from sklearn.metrics.pairwise import paired_distances
print('=' * 80)
print('Anomaly Detection using Isolation Forest')
print('Supports both CSV and PKL input files')
print('=' * 80)

# ============================================================================
# STEP 1: Load data (CSV or PKL)
# ============================================================================
if len(sys.argv) < 2:
    print("\nUsage: mlp_torch.py <path_to_data>")
    print("Example: mlp_torch.py /home/user/r4.2/ExtractedData/weekr4.2.csv")
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
print(f'Training samples if insider ịn training: {len(xTrain)} (insiders in train: {yTrainBin.sum()})')
print(f'Training samples if insider not in training: {len(xTrain_not_insider)} (not insiders in train')
print(f'Test samples: {len(xTest)} (total insiders: {yTestBin.sum()})')

# Handle any NaN or infinite values
xTrain = np.nan_to_num(xTrain, nan=0.0, posinf=0.0, neginf=0.0)
xTrain_not_insider = np.nan_to_num(xTrain_not_insider, nan=0.0, posinf=0.0, neginf=0.0)
xTest = np.nan_to_num(xTest, nan=0.0, posinf=0.0, neginf=0.0)


import torch
import torch.nn as nn

class AutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        """
        input_dim: số chiều input
        hidden_dims: list hoặc tuple, ví dụ [128, 64, 32]
        """
        super().__init__()

        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, h))
            encoder_layers.append(nn.ReLU())
            prev_dim = h
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder (đảo ngược)
        decoder_layers = []
        hidden_dims_rev = list(hidden_dims[::-1])
        for i in range(len(hidden_dims_rev) - 1):
            decoder_layers.append(nn.Linear(hidden_dims_rev[i], hidden_dims_rev[i+1]))
            decoder_layers.append(nn.ReLU())
        decoder_layers.append(nn.Linear(hidden_dims_rev[-1], input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat
class TorchAE:
    def __init__(
        self,
        input_dim,
        hidden_dims,
        lr=1e-3,
        batch_size=256,
        epochs=50,
        device=None
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoEncoder(input_dim, hidden_dims).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        self.criterion = nn.MSELoss()
        self.batch_size = batch_size
        self.epochs = epochs

    def fit(self, X):
        X = torch.tensor(X, dtype=torch.float32)
        dataset = torch.utils.data.TensorDataset(X)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            for (batch,) in loader:
                batch = batch.to(self.device)

                self.optimizer.zero_grad()
                recon = self.model(batch)
                loss = self.criterion(recon, batch)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item() * batch.size(0)

            avg_loss = total_loss / len(dataset)
            print(f"Epoch [{epoch+1}/{self.epochs}] - Recon Loss: {avg_loss:.6f}")

    def reconstruct(self, X):
        self.model.eval()
        with torch.no_grad():
            X = torch.tensor(X, dtype=torch.float32).to(self.device)
            recon = self.model(X)
        return recon.cpu().numpy()

    def transform(self, X):
        """Trả về latent vector (encoder output)"""
        self.model.eval()
        with torch.no_grad():
            X = torch.tensor(X, dtype=torch.float32).to(self.device)
            z = self.model.encoder(X)
        return z.cpu().numpy()

    def reconstruction_error(self, X):
        """MSE theo từng sample (dùng cho anomaly detection)"""
        self.model.eval()
        with torch.no_grad():
            X = torch.tensor(X, dtype=torch.float32).to(self.device)
            recon = self.model(X)
            error = torch.mean((recon - X) ** 2, dim=1)
        return error.cpu().numpy()

    def paired_distances(self, X):
        self.model.eval()
        with torch.no_grad():
            X = torch.tensor(X, dtype=torch.float32).to(self.device)
            recon = self.model(X)
            X = X.cpu().numpy()
            recon = recon.cpu().numpy()
            distances = paired_distances(X, recon)
        return distances

n_features = len(x_cols)
hidden1 = max(10, 1 << (int(n_features / 4).bit_length()))  # Encoder
hidden2 = max(5, 1 << (int(n_features / 8).bit_length()))   # Bottleneck
hidden3 = max(10, 1 << (int(n_features / 4).bit_length()))  # Decoder
input_dim = xTrain.shape[1]
hidden_dims = [hidden1, hidden2, hidden3]
print(f'\nAutoencoder architecture: {n_features} → {hidden1} → {hidden2} → {hidden3} → {n_features}')
ae = TorchAE(
    input_dim=input_dim,
    hidden_dims=hidden_dims,
    epochs=50,
    batch_size=256
)
print('device: ', ae.device)
ae.fit(xTrain)

# Reconstruction
# x_recon = ae.reconstruct(xTest)

# # Latent representation
# z = ae.transform(xTest)

# Reconstruction error (anomaly score)
# scores = ae.reconstruction_error(xTest)
scores = ae.paired_distances(xTest)
ae = TorchAE(
    input_dim=input_dim,
    hidden_dims=hidden_dims,
    epochs=50,
    batch_size=512
)
ae.fit(xTrain_not_insider)
# scores_not_insider = ae.reconstruction_error(xTest)
scores_not_insider = ae.paired_distances(xTest)
print(f'Anomaly scores: {scores}')

# ============================================================================
# STEP 6: Evaluate results
# ============================================================================
from utils import evaluate_results
print('\n*****Evaluating results...')
print('=' * 80)
print('RESULTS FOR NOT INSIDER IN TRAINING')
print('=' * 80)
evaluate_results(scores_not_insider, yTestBin)

print('\n' + '=' * 80)
print('RESULTS FOR INSIDER IN TRAINING')
print('=' * 80)
evaluate_results(scores, yTestBin)
