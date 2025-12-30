#!/usr/bin/env python
# coding: utf-8
"""
Anomaly Detection using AutoEncoder + LOF Pipeline - Supports CSV and PKL input

This script uses AutoEncoder's reconstruction error as input features for LOF.
Pipeline: Raw Features → AutoEncoder → Reconstruction Error → LOF → Anomaly Score

This script can work with:
1. Original CSV files (e.g., weekr4.2.csv, dayr4.2.csv)
2. Pickle files with percentile transformation (e.g., week-r5.2-percentile30.pkl)

Usage:
    python LOF_AutoEncoder.py /path/to/weekr4.2.csv
    python LOF_AutoEncoder.py /path/to/week-r5.2-percentile30.pkl
"""

import pandas as pd
import numpy as np
import sys
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics.pairwise import paired_distances

print('=' * 80)
print('Anomaly Detection using AutoEncoder + LOF Pipeline')
print('Pipeline: Raw Features → AutoEncoder → Reconstruction Error → LOF')
print('Supports both CSV and PKL input files')
print('=' * 80)

# ============================================================================
# STEP 1: Load data (CSV or PKL)
# ============================================================================
if len(sys.argv) < 2:
    print("\nUsage: python LOF_AutoEncoder.py <path_to_data>")
    print("Example: python LOF_AutoEncoder.py /home/user/r4.2/ExtractedData/weekr4.2.csv")
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
# STEP 7: Build AutoEncoder architecture
# ============================================================================
n_features = len(x_cols)
hidden1 = max(10, 1 << (int(n_features / 4).bit_length()))  # Encoder
hidden2 = max(5, 1 << (int(n_features / 8).bit_length()))   # Bottleneck
hidden3 = max(10, 1 << (int(n_features / 4).bit_length()))  # Decoder

print(f'\nAutoencoder architecture: {n_features} → {hidden1} → {hidden2} → {hidden3} → {n_features}')

# ============================================================================
# STEP 8: PIPELINE FOR NOT INSIDER IN TRAINING
# ============================================================================
print('\n' + '=' * 80)
print('PIPELINE: NOT INSIDER IN TRAINING')
print('=' * 80)

# Step 8.1: Train AutoEncoder on training data (not insider)
print('\n*****Training AutoEncoder...')
ae_not_insider = MLPRegressor(
    hidden_layer_sizes=(hidden1, hidden2, hidden3),
    max_iter=50,
    random_state=10,
    early_stopping=True,
    validation_fraction=0.1,
    verbose=False
)
ae_not_insider.fit(xTrain_not_insider, xTrain_not_insider)
print('AutoEncoder training completed')

# Step 8.2: Get reconstruction error for training data (to train LOF)
print('\nCalculating reconstruction errors for training data...')
xTrain_reconstructed = ae_not_insider.predict(xTrain_not_insider)
# Calculate per-sample reconstruction error (MSE per feature, resulting in n_features dimensional vector)
train_reconstruction_errors = (xTrain_not_insider - xTrain_reconstructed) ** 2
print(f'Training reconstruction error shape: {train_reconstruction_errors.shape}')

# Step 8.3: Get reconstruction error for test data
print('Calculating reconstruction errors for test data...')
xTest_reconstructed = ae_not_insider.predict(xTest)
test_reconstruction_errors = (xTest - xTest_reconstructed) ** 2
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

# Step 9.1: Train AutoEncoder on training data (with insider)
print('\n*****Training AutoEncoder...')
ae_insider = MLPRegressor(
    hidden_layer_sizes=(hidden1, hidden2, hidden3),
    max_iter=50,
    random_state=10,
    early_stopping=True,
    validation_fraction=0.1,
    verbose=False
)
ae_insider.fit(xTrain_scaled, xTrain_scaled)
print('AutoEncoder training completed')

# Step 9.2: Get reconstruction error for training data (to train LOF)
print('\nCalculating reconstruction errors for training data...')
xTrain_insider_reconstructed = ae_insider.predict(xTrain_scaled)
train_insider_reconstruction_errors = (xTrain_scaled - xTrain_insider_reconstructed) ** 2
print(f'Training reconstruction error shape: {train_insider_reconstruction_errors.shape}')

# Step 9.3: Get reconstruction error for test data
print('Calculating reconstruction errors for test data...')
xTest_insider_reconstructed = ae_insider.predict(xTest_scaled_with_insider)
test_insider_reconstruction_errors = (xTest_scaled_with_insider - xTest_insider_reconstructed) ** 2
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
print('RESULTS FOR AUTOENCODER + LOF (INSIDER IN TRAINING)')
print('=' * 80)
evaluate_results(anomaly_scores_insiders, yTestBin)

print('\n' + '=' * 80)
print('RESULTS FOR AUTOENCODER + LOF (NOT INSIDER IN TRAINING)')
print('=' * 80)
evaluate_results(anomaly_scores_not_insiders, yTestBin)
