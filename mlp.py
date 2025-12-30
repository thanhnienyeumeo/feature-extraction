#!/usr/bin/env python
# coding: utf-8
"""
Anomaly Detection using Autoencoder - Modified for CSV input

This script can work with:
1. Original CSV files (e.g., weekr4.2.csv, dayr4.2.csv)
2. Pickle files with percentile transformation (e.g., week-r5.2-percentile30.pkl)

Usage:
    python example_anomaly_detection_csv.py /path/to/weekr4.2.csv
    python example_anomaly_detection_csv.py /path/to/week-r5.2-percentile30.pkl
"""

import pandas as pd
import numpy as np
import sys
from sklearn.neural_network import MLPRegressor
from sklearn.metrics.pairwise import paired_distances
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

print('=' * 80)
print('Anomaly Detection using Autoencoder (MLPRegressor)')
print('Supports both CSV and PKL input files')
print('=' * 80)

# ============================================================================
# STEP 1: Load data (CSV or PKL)
# ============================================================================
if len(sys.argv) < 2:
    print("\nUsage: python example_anomaly_detection_csv.py <path_to_data>")
    print("Example: python example_anomaly_detection_csv.py /home/user/r4.2/ExtractedData/weekr4.2.csv")
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
# data = data[data['isweekday'] == 1]
removed_cols = [
    'user', 'day', 'week', 'starttime', 'endtime', 'sessionid', 'insider',
    'isweekday', 'isweekend',  # For day-level data
    'role', 'b_unit', 'f_unit', 'dept', 'team', 'project',  # Categorical user info
    'ITAdmin', 'O', 'C', 'E', 'A', 'N',  # User attributes (keep as features or remove)
]
removed_cols.extend([col for col in data.columns if 'afterhour' in col or 'weekend' in col])
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
# STEP 4: Select training users (200 random normal users)
# ============================================================================
# Get users from first half
all_train_users = list(set(data1stHalf.user))
nUsers = np.random.permutation(all_train_users)
trainUsers = nUsers[:min(20000, len(nUsers))]  # Use up to 200 users

print(f'\nTraining on {len(trainUsers)} randomly selected users')

# ============================================================================
# STEP 5: Prepare training and test data
# ============================================================================
xTrain = data1stHalf[data1stHalf.user.isin(trainUsers)][x_cols].values
yTrain = data1stHalf[data1stHalf.user.isin(trainUsers)]['insider'].values
yTrainBin = yTrain > 0

# Test on ALL data (not just second half) to get full picture
xTest_all = data[x_cols].values
yTest_all = data['insider'].values
yTestBin_all = yTest_all > 0
#Test on only second half
xTest = dataTest[x_cols].values
yTest = dataTest['insider'].values
yTestBin = yTest > 0
print(f'Training samples: {len(xTrain)} (insiders in train: {yTrainBin.sum()})')
print(f'Test samples: {len(xTest)} (total insiders: {yTestBin.sum()})')

# Handle any NaN or infinite values
xTrain = np.nan_to_num(xTrain, nan=0.0, posinf=0.0, neginf=0.0)
xTest = np.nan_to_num(xTest, nan=0.0, posinf=0.0, neginf=0.0)

# ============================================================================
# STEP 6: Standardize features (IMPORTANT for raw CSV data!)
# ============================================================================
print('\nStandardizing features...')
scaler = StandardScaler()
xTrain = scaler.fit_transform(xTrain)
xTest = scaler.transform(xTest)

# ============================================================================
# STEP 7: Build and train Autoencoder
# ============================================================================
# Architecture: input → compress → bottleneck → decompress → output
n_features = len(x_cols)
hidden1 = max(10, 1 << (int(n_features / 4).bit_length()))  # Encoder
hidden2 = max(5, 1 << (int(n_features / 8).bit_length()))   # Bottleneck
hidden3 = max(10, 1 << (int(n_features / 4).bit_length()))  # Decoder

print(f'\nAutoencoder architecture: {n_features} → {hidden1} → {hidden2} → {hidden3} → {n_features}')

ae = MLPRegressor(
    hidden_layer_sizes=(hidden1, hidden2, hidden3),
    max_iter=50,  # More iterations for better convergence
    random_state=10,
    early_stopping=True,  # Stop if validation score doesn't improve
    validation_fraction=0.1,
    verbose=False
)

print('Training autoencoder...')
ae.fit(xTrain, xTrain)  # Train to reconstruct input
print(f'Training completed in {ae.n_iter_} iterations')

# ============================================================================
# STEP 8: Calculate reconstruction error (anomaly score)
# ============================================================================
print('\nCalculating reconstruction errors...')
predictions = ae.predict(xTest)
reconstructionError = paired_distances(xTest, predictions)

# ============================================================================
# STEP 9: Evaluate results
# ============================================================================
print('\n' + '=' * 80)
print('RESULTS')
print('=' * 80)

# AUC Score
if yTestBin.sum() > 0:  # Only if there are actual insiders
    auc = roc_auc_score(yTestBin, reconstructionError)
    print(f'\nAUC Score: {auc:.4f}')
    print('(AUC > 0.5 means model is better than random, AUC = 1.0 is perfect)')
else:
    print('\nNo insider labels found in data - cannot compute AUC')

# Detection rate at different budgets
print('\nMetrics at different investigation budgets:')
print('-' * 80)
print(f'{"Budget":<10} {"TP":<8} {"FP":<8} {"FN":<8} {"TN":<8} {"Precision":<12} {"Recall":<12} {"F1":<10}')
print('-' * 80)

for ib in [0.001, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2]:
    threshold = np.percentile(reconstructionError, 100 - 100 * ib)
    predicted = reconstructionError > threshold
    
    # Calculate confusion matrix
    TP = np.sum((predicted == True) & (yTestBin == True))
    FP = np.sum((predicted == True) & (yTestBin == False))
    FN = np.sum((predicted == False) & (yTestBin == True))
    TN = np.sum((predicted == False) & (yTestBin == False))
    
    # Calculate metrics
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f'{100*ib:>6.1f}%   {TP:<8} {FP:<8} {FN:<8} {TN:<8} {precision:<12.4f} {recall:<12.4f} {f1:<10.4f}')

print('-' * 80)
print('\nBudget explanation:')
print('  - Budget = percentage of samples flagged as suspicious')
print('  - 1% budget means investigating top 1% highest reconstruction errors')
print('  - Higher recall at lower budget = better model')
