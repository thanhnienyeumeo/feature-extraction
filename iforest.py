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
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

print('=' * 80)
print('Anomaly Detection using Isolation Forest')
print('Supports both CSV and PKL input files')
print('=' * 80)

# ============================================================================
# STEP 1: Load data (CSV or PKL)
# ============================================================================
if len(sys.argv) < 2:
    print("\nUsage: python example_anomaly_detection_iforest.py <path_to_data>")
    print("Example: python example_anomaly_detection_iforest.py /home/user/r4.2/ExtractedData/weekr4.2.csv")
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
trainUsers = nUsers[:min(20000, len(nUsers))]  # Use up to 200 users

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

# ============================================================================
# STEP 6: Standardize features
# ============================================================================
print('\nStandardizing features...')
scaler = StandardScaler()
# xTrain = scaler.fit_transform(xTrain)
xTrain_not_insider = scaler.fit_transform(xTrain_not_insider)
# xTest = scaler.transform(xTest)
xTest = scaler.transform(xTest)
# ============================================================================
# STEP 7: Train Isolation Forest
# ============================================================================
print('\n*****Training Isolation Forest with insider not in training...')
# Isolation Forest model
# contamination='auto' lets the model determine threshold automatically
# n_estimators: number of trees in the forest
# max_samples: number of samples to draw to train each tree
iforest = IsolationForest(
    n_estimators=200, 
    max_samples='auto', 
    contamination='auto', 
    random_state=10, 
    n_jobs=-1
)

# iforest.fit(xTrain)
iforest.fit(xTrain_not_insider)
print('Training completed')

# ============================================================================
# STEP 8: Calculate anomaly scores
# ============================================================================
# Get anomaly scores (lower/more negative = more anomalous)
# We negate the scores so that higher values = more anomalous (for consistency with AUC)
print('\nCalculating anomaly scores...')
anomaly_scores_not_insiders = -iforest.score_samples(xTest)
# Do STEP 7 and STEP 8 again but with insider in training
# Use different variable names to avoid overwriting xTest (which is already scaled)
scaler_with_insider = StandardScaler()
xTrain_scaled = scaler_with_insider.fit_transform(xTrain)
xTest_scaled_with_insider = scaler_with_insider.transform(dataTest[x_cols].values)
xTest_scaled_with_insider = np.nan_to_num(xTest_scaled_with_insider, nan=0.0, posinf=0.0, neginf=0.0)
print('\n*****Training Isolation Forest with insider in training...')

iforest = IsolationForest(
    n_estimators=200, 
    max_samples='auto', 
    contamination='auto', 
    random_state=10, 
    n_jobs=-1
)
iforest.fit(xTrain_scaled)
print('Training completed for insider in training')
anomaly_scores_insiders = -iforest.score_samples(xTest_scaled_with_insider)

# ============================================================================
# STEP 9: Evaluate results
# ============================================================================
from utils import evaluate_results
print('\n' + '=' * 80)
print('RESULTS FOR INSIDER IN TRAINING')
print('=' * 80)

evaluate_results(anomaly_scores_insiders, yTestBin)

print('\n' + '=' * 80)
print('RESULTS FOR NOT INSIDER IN TRAINING')
print('=' * 80)

evaluate_results(anomaly_scores_not_insiders, yTestBin)


# # STEP 10: Build MLP Regressor
# from sklearn.neural_network import MLPRegressor
# n_features = len(x_cols)
# hidden1 = max(10, 1 << (int(n_features / 4).bit_length()))  # Encoder
# hidden2 = max(5, 1 << (int(n_features / 8).bit_length()))   # Bottleneck
# hidden3 = max(10, 1 << (int(n_features / 4).bit_length()))  # Decoder

# print(f'\nAutoencoder architecture: {n_features} → {hidden1} → {hidden2} → {hidden3} → {n_features}')

# ae = MLPRegressor(
#     hidden_layer_sizes=(hidden1, hidden2, hidden3),
#     max_iter=50,  # More iterations for better convergence
#     random_state=10,
#     early_stopping=True,  # Stop if validation score doesn't improve
#     validation_fraction=0.1,
#     verbose=False
# )

# print('*****Training autoencoder...')
# from sklearn.metrics.pairwise import paired_distances
# ae.fit(xTrain_not_insider, xTrain_not_insider)
# print('Training completed for MLP Regressor for insider not in training')
# anomaly_scores_not_insiders_mlp = ae.predict(xTest)
# anomaly_scores_not_insiders_mlp = paired_distances(xTest, anomaly_scores_not_insiders_mlp)

# ae = MLPRegressor(
#     hidden_layer_sizes=(hidden1, hidden2, hidden3),
#     max_iter=50,  # More iterations for better convergence
#     random_state=10,
#     early_stopping=True,  # Stop if validation score doesn't improve
#     validation_fraction=0.1,
#     verbose=False
# )
# ae.fit(xTrain_scaled, xTrain_scaled)
# print('Training completed for MLP Regressor for insider in training')
# anomaly_scores_insiders_mlp = ae.predict(xTest_scaled_with_insider)
# anomaly_scores_insiders_mlp = paired_distances(xTest_scaled_with_insider, anomaly_scores_insiders_mlp)

# # ============================================================================
# # STEP 11: Evaluate results
# # ============================================================================

# print('\n' + '=' * 80)
# print('RESULTS FOR INSIDER IN TRAINING')
# print('=' * 80)

# evaluate_results(anomaly_scores_insiders_mlp, yTestBin)

# print('\n' + '=' * 80)
# print('RESULTS FOR INSIDER NOT IN TRAINING')
# print('=' * 80)

# # AUC Score
# evaluate_results(anomaly_scores_not_insiders_mlp, yTestBin)