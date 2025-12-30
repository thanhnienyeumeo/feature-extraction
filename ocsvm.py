#!/usr/bin/env python
# coding: utf-8
"""
Anomaly Detection using One-Class SVM (OCSVM) - Supports CSV and PKL input

This script can work with:
1. Original CSV files (e.g., weekr4.2.csv, dayr4.2.csv)
2. Pickle files with percentile transformation (e.g., week-r5.2-percentile30.pkl)

Usage:
    python example_anomaly_detection_ocsvm.py /path/to/weekr4.2.csv
    python example_anomaly_detection_ocsvm.py /path/to/week-r5.2-percentile30.pkl

One-Class SVM learns a decision boundary around normal data points.
Points outside this boundary are considered anomalies.
"""

import pandas as pd
import numpy as np
import sys
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

print('=' * 80)
print('Anomaly Detection using One-Class SVM (OCSVM)')
print('Supports both CSV and PKL input files')
print('=' * 80)

# ============================================================================
# STEP 1: Load data (CSV or PKL)
# ============================================================================
if len(sys.argv) < 2:
    print("\nUsage: python example_anomaly_detection_ocsvm.py <path_to_data>")
    print("Example: python example_anomaly_detection_ocsvm.py /home/user/r4.2/ExtractedData/weekr4.2.csv")
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
# STEP 4: Select training users
# ============================================================================
all_train_users = list(set(data1stHalf.user))
nUsers = np.random.permutation(all_train_users)
trainUsers = nUsers[:min(200000, len(nUsers))]  # Use up to 200000 users

print(f'\nTraining on {len(trainUsers)} randomly selected users')

# ============================================================================
# STEP 5: Prepare training and test data
# ============================================================================
xTrain = data1stHalf[data1stHalf.user.isin(trainUsers)][x_cols].values
yTrain = data1stHalf[data1stHalf.user.isin(trainUsers)]['insider'].values
yTrainBin = yTrain > 0

# Test on ALL data
xTest_all = data[x_cols].values
yTest_all = data['insider'].values
yTestBin_all = yTest_all > 0

# Test on only second half
xTest = dataTest[x_cols].values
yTest = dataTest['insider'].values
yTestBin = yTest > 0

print(f'Training samples: {len(xTrain)} (insiders in train: {yTrainBin.sum()})')
print(f'Test samples: {len(xTest)} (total insiders: {yTestBin.sum()})')

# Handle any NaN or infinite values
xTrain = np.nan_to_num(xTrain, nan=0.0, posinf=0.0, neginf=0.0)
xTest = np.nan_to_num(xTest, nan=0.0, posinf=0.0, neginf=0.0)

# ============================================================================
# STEP 6: Standardize features (CRITICAL for SVM!)
# ============================================================================
print('\nStandardizing features...')
scaler = StandardScaler()
xTrain = scaler.fit_transform(xTrain)
xTest = scaler.transform(xTest)

# ============================================================================
# STEP 7: Train One-Class SVM
# ============================================================================
print('\nTraining One-Class SVM...')
print('(This may take a while for large datasets)')
import time
t1 =time.time()
# One-Class SVM parameters:
# - kernel: 'rbf' (Radial Basis Function) works well for most cases
# - nu: upper bound on fraction of outliers, lower bound on fraction of support vectors
# - gamma: kernel coefficient ('scale' = 1/(n_features * X.var()))
ocsvm = OneClassSVM(
    kernel='rbf',
    nu=0.1,  # Expect ~10% anomalies (adjust based on your data)
    gamma='scale',  # Automatic scaling
    cache_size=1000,  # MB of cache for faster training
    verbose=False
)

# For very large datasets, consider subsampling for training
if len(xTrain) > 100000:
    print(f'Large dataset detected. Subsampling to 100000 samples for training...')
    subsample_idx = np.random.choice(len(xTrain), 100000, replace=False)
    xTrain_sub = xTrain[subsample_idx]
    ocsvm.fit(xTrain_sub)
else:
    ocsvm.fit(xTrain)
t2 = time.time()
print(f'Training time: {t2 - t1} seconds')
print('Training completed')
print(f'Number of support vectors: {len(ocsvm.support_)}')

# ============================================================================
# STEP 8: Calculate anomaly scores
# ============================================================================
print('\nCalculating anomaly scores...')

# decision_function returns signed distance to the separating hyperplane
# Negative values = anomalies, Positive values = normal
# We negate so that higher values = more anomalous (for AUC consistency)
anomaly_scores = -ocsvm.decision_function(xTest)

# Also get predictions (-1 = anomaly, 1 = normal)
predictions_raw = ocsvm.predict(xTest)

# ============================================================================
# STEP 9: Evaluate results
# ============================================================================
print('\n' + '=' * 80)
print('RESULTS')
print('=' * 80)

# AUC Score
if yTestBin.sum() > 0:  # Only if there are actual insiders
    auc = roc_auc_score(yTestBin, anomaly_scores)
    print(f'\nAUC Score: {auc:.4f}')
    print('(AUC > 0.5 means model is better than random, AUC = 1.0 is perfect)')
else:
    print('\nNo insider labels found in data - cannot compute AUC')

# OCSVM native predictions (based on nu parameter)
print('\n--- OCSVM Native Predictions (based on nu parameter) ---')
predicted_native = predictions_raw == -1  # -1 means anomaly
TP_native = np.sum((predicted_native == True) & (yTestBin == True))
FP_native = np.sum((predicted_native == True) & (yTestBin == False))
FN_native = np.sum((predicted_native == False) & (yTestBin == True))
TN_native = np.sum((predicted_native == False) & (yTestBin == False))

precision_native = TP_native / (TP_native + FP_native) if (TP_native + FP_native) > 0 else 0
recall_native = TP_native / (TP_native + FN_native) if (TP_native + FN_native) > 0 else 0
f1_native = 2 * precision_native * recall_native / (precision_native + recall_native) if (precision_native + recall_native) > 0 else 0

print(f'TP={TP_native}, FP={FP_native}, FN={FN_native}, TN={TN_native}')
print(f'Precision={precision_native:.4f}, Recall={recall_native:.4f}, F1={f1_native:.4f}')
print(f'Flagged as anomaly: {predicted_native.sum()} ({100*predicted_native.sum()/len(predicted_native):.2f}%)')

# Detection rate at different budgets
print('\n--- Metrics at different investigation budgets ---')
print('-' * 80)
print(f'{"Budget":<10} {"TP":<8} {"FP":<8} {"FN":<8} {"TN":<8} {"Precision":<12} {"Recall":<12} {"F1":<10}')
print('-' * 80)

for ib in [0.001, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2]:
    threshold = np.percentile(anomaly_scores, 100 - 100 * ib)
    predicted = anomaly_scores > threshold
    
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
print('  - 1% budget means investigating top 1% highest anomaly scores')
print('  - Higher recall at lower budget = better model')

print('\nOCSVM explanation:')
print('  - OCSVM learns a tight boundary around normal data')
print('  - Points outside boundary are anomalies')
print('  - nu parameter controls expected fraction of outliers')
