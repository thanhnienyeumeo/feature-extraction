#!/usr/bin/env python
# coding: utf-8
"""
Per-User Anomaly Detection using One-Class SVM (OCSVM)

Two training pipelines:
1. Insider IN training: Train on all user data (may include malicious behavior)
2. Insider NOT in training: Train only on non-insider samples (cleaner baseline)

Score normalization using percentile ranking for cross-user comparison.

Usage:
    python ocsvm_per_user.py /path/to/weekr4.2.csv
    python ocsvm_per_user.py /path/to/week-r5.2-percentile30.pkl
"""

import pandas as pd
import numpy as np
import sys
import time
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from collections import defaultdict
from scipy import stats

print('=' * 80)
print('Per-User OCSVM with Two Training Pipelines')
print('1. Insider IN training  2. Insider NOT in training')
print('=' * 80)

# ============================================================================
# STEP 1: Load data (CSV or PKL)
# ============================================================================
if len(sys.argv) < 2:
    print("\nUsage: python ocsvm_per_user.py <path_to_data>")
    print("Example: python ocsvm_per_user.py /home/user/r4.2/ExtractedData/weekr4.2.csv")
    sys.exit(1)

path = sys.argv[1]
print(f'\nLoading data from: {path}')

if path.endswith('.pkl') or path.endswith('.pickle'):
    data = pd.read_pickle(path)
    print(f'Loaded pickle file')
elif path.endswith('.csv') or path.endswith('.csv.gz'):
    data = pd.read_csv(path)
    print(f'Loaded CSV file')
else:
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
removed_cols = [
    'user', 'day', 'week', 'starttime', 'endtime', 'sessionid', 'insider',
    'isweekday', 'isweekend',
    'role', 'b_unit', 'f_unit', 'dept', 'team', 'project',
    'ITAdmin', 'O', 'C', 'E', 'A', 'N',
]

x_cols = [col for col in data.columns if col not in removed_cols]
numeric_cols = data[x_cols].select_dtypes(include=[np.number]).columns.tolist()
x_cols = numeric_cols

print(f'\nUsing {len(x_cols)} features for anomaly detection')
print(f'Sample features: {x_cols[:5]}...')

# ============================================================================
# STEP 3: Split data by time
# ============================================================================
run = 1
np.random.seed(run)

max_week = data['week'].max()
data1stHalf = data[data.week <= max_week / 2]
dataTest = data[data.week > max_week / 2]

# Also create training data WITHOUT insiders
data1stHalf_no_insider = data1stHalf[data1stHalf['insider'] == 0]

print(f'\nTrain period: weeks 0-{int(max_week/2)} ({len(data1stHalf)} rows)')
print(f'Train period (no insider): {len(data1stHalf_no_insider)} rows')
print(f'Test period: weeks {int(max_week/2)+1}-{int(max_week)} ({len(dataTest)} rows)')

# ============================================================================
# STEP 4: Get all unique users
# ============================================================================
train_users = data1stHalf['user'].unique()
test_users = dataTest['user'].unique()

print(f'\nUsers in train period: {len(train_users)}')
print(f'Users in test period: {len(test_users)}')

# ============================================================================
# STEP 5: Configuration
# ============================================================================
MIN_SAMPLES_PER_USER = 5
NU_PARAM = 0.005

print(f'\nConfiguration:')
print(f'  - Minimum samples per user: {MIN_SAMPLES_PER_USER}')
print(f'  - OCSVM nu parameter: {NU_PARAM}')

# ============================================================================
# STEP 6: Prepare test data
# ============================================================================
xTest = dataTest[x_cols].values
yTest = dataTest['insider'].values
yTestBin = yTest > 0
test_users_list = dataTest['user'].values
xTest = np.nan_to_num(xTest, nan=0.0, posinf=0.0, neginf=0.0)

print(f'\nTest samples: {len(xTest)} (total insiders: {yTestBin.sum()})')

# ============================================================================
# Helper function to evaluate results
# ============================================================================
def evaluate_results(anomaly_scores, yTestBin, title=""):
    """Evaluate and print results for given anomaly scores."""
    if title:
        print(f'\n{title}')
    
    # AUC Score
    if yTestBin.sum() > 0:
        auc = roc_auc_score(yTestBin, anomaly_scores)
        print(f'\nAUC Score: {auc:.4f}')
        print('(AUC > 0.5 means model is better than random, AUC = 1.0 is perfect)')
    else:
        print('\nNo insider labels found - cannot compute AUC')
        return
    
    # Metrics at different budgets
    print('\nMetrics at different investigation budgets:')
    print('-' * 80)
    print(f'{"Budget":<10} {"TP":<8} {"FP":<8} {"FN":<8} {"TN":<8} {"Precision":<12} {"Recall":<12} {"F1":<10}')
    print('-' * 80)
    
    for ib in [0.001, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2]:
        threshold = np.percentile(anomaly_scores, 100 - 100 * ib)
        predicted = anomaly_scores > threshold
        
        TP = np.sum((predicted == True) & (yTestBin == True))
        FP = np.sum((predicted == True) & (yTestBin == False))
        FN = np.sum((predicted == False) & (yTestBin == True))
        TN = np.sum((predicted == False) & (yTestBin == False))
        
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f'{100*ib:>6.1f}%   {TP:<8} {FP:<8} {FN:<8} {TN:<8} {precision:<12.4f} {recall:<12.4f} {f1:<10.4f}')
    
    return auc

# ============================================================================
# Helper function to train per-user models
# ============================================================================
def train_per_user_models(train_data, train_users, x_cols, min_samples, nu_param, description=""):
    """Train OCSVM model for each user and return models with scalers."""
    print(f'\n{"="*80}')
    print(f'Training per-user OCSVM models {description}...')
    print('=' * 80)
    
    t1 = time.time()
    
    user_models = {}
    user_scalers = {}
    users_with_model = []
    users_using_fallback = []
    
    train_grouped = train_data.groupby('user')
    
    for i, user in enumerate(train_users):
        if (i + 1) % 100 == 0 or i == 0:
            print(f'  Processing user {i+1}/{len(train_users)}...', end='\r')
        
        if user not in train_grouped.groups:
            users_using_fallback.append(user)
            continue
            
        user_train_data = train_grouped.get_group(user)
        
        if len(user_train_data) < min_samples:
            users_using_fallback.append(user)
            continue
        
        xTrain_user = user_train_data[x_cols].values
        xTrain_user = np.nan_to_num(xTrain_user, nan=0.0, posinf=0.0, neginf=0.0)
        
        if xTrain_user.std(axis=0).sum() == 0:
            users_using_fallback.append(user)
            continue
        
        try:
            scaler = StandardScaler()
            xTrain_user_scaled = scaler.fit_transform(xTrain_user)
            xTrain_user_scaled = np.nan_to_num(xTrain_user_scaled, nan=0.0, posinf=0.0, neginf=0.0)
            
            ocsvm = OneClassSVM(
                kernel='rbf',
                nu=min(nu_param, 0.5),
                gamma='scale',
                cache_size=200,
                verbose=False
            )
            ocsvm.fit(xTrain_user_scaled)
            
            user_models[user] = ocsvm
            user_scalers[user] = scaler
            users_with_model.append(user)
            
        except Exception as e:
            users_using_fallback.append(user)
    
    t2 = time.time()
    print(f'\nTraining completed in {t2 - t1:.2f} seconds')
    print(f'Users with dedicated model: {len(users_with_model)}')
    print(f'Users using fallback: {len(users_using_fallback)}')
    
    return user_models, user_scalers, users_with_model, users_using_fallback

# ============================================================================
# Helper function to compute scores with per-user models
# ============================================================================
def compute_per_user_scores(xTest, test_users_list, user_models, user_scalers, 
                           global_model, global_scaler, description=""):
    """Compute anomaly scores using per-user models with global fallback."""
    print(f'\nCalculating anomaly scores {description}...')
    
    raw_scores = np.zeros(len(xTest))
    stats_count = {'personal': 0, 'fallback': 0}
    
    for i in range(len(xTest)):
        if (i + 1) % 10000 == 0:
            print(f'  Processing sample {i+1}/{len(xTest)}...', end='\r')
        
        user = test_users_list[i]
        sample = xTest[i:i+1]
        
        if user in user_models:
            scaler = user_scalers[user]
            model = user_models[user]
            
            sample_scaled = scaler.transform(sample)
            sample_scaled = np.nan_to_num(sample_scaled, nan=0.0, posinf=0.0, neginf=0.0)
            
            raw_scores[i] = -model.decision_function(sample_scaled)[0]
            stats_count['personal'] += 1
        else:
            sample_scaled = global_scaler.transform(sample)
            sample_scaled = np.nan_to_num(sample_scaled, nan=0.0, posinf=0.0, neginf=0.0)
            
            raw_scores[i] = -global_model.decision_function(sample_scaled)[0]
            stats_count['fallback'] += 1
    
    print(f'\n  Personal model: {stats_count["personal"]}, Fallback: {stats_count["fallback"]}')
    
    # Normalize scores using percentile ranking within each user
    # This makes scores comparable across users
    normalized_scores = np.zeros(len(xTest))
    
    # Group by user and compute percentile within each user's scores
    user_indices = defaultdict(list)
    for i, user in enumerate(test_users_list):
        user_indices[user].append(i)
    
    for user, indices in user_indices.items():
        user_raw = raw_scores[indices]
        # Convert to percentile rank (0-100)
        user_percentile = stats.rankdata(user_raw, method='average') / len(user_raw) * 100
        for j, idx in enumerate(indices):
            normalized_scores[idx] = user_percentile[j]
    
    return raw_scores, normalized_scores

# ============================================================================
# PIPELINE 1: INSIDER IN TRAINING
# ============================================================================
print('\n' + '#' * 80)
print('# PIPELINE 1: INSIDER IN TRAINING')
print('#' * 80)

# Train global fallback model (with insiders)
print('\nTraining global fallback model (with insiders)...')
xTrain_global = data1stHalf[x_cols].values
xTrain_global = np.nan_to_num(xTrain_global, nan=0.0, posinf=0.0, neginf=0.0)

global_scaler_with = StandardScaler()
xTrain_global_scaled = global_scaler_with.fit_transform(xTrain_global)

if len(xTrain_global_scaled) > 50000:
    subsample_idx = np.random.choice(len(xTrain_global_scaled), 50000, replace=False)
    xTrain_sub = xTrain_global_scaled[subsample_idx]
else:
    xTrain_sub = xTrain_global_scaled

global_ocsvm_with = OneClassSVM(kernel='rbf', nu=NU_PARAM, gamma='scale', cache_size=500)
global_ocsvm_with.fit(xTrain_sub)
print(f'Global model trained with {len(xTrain_sub)} samples')

# Train per-user models (with insiders)
user_models_with, user_scalers_with, _, _ = train_per_user_models(
    data1stHalf, train_users, x_cols, MIN_SAMPLES_PER_USER, NU_PARAM,
    "(with insiders)"
)

# Compute scores
raw_scores_with, norm_scores_with = compute_per_user_scores(
    xTest, test_users_list, user_models_with, user_scalers_with,
    global_ocsvm_with, global_scaler_with, "(with insiders)"
)

# Evaluate
print('\n' + '=' * 80)
print('RESULTS FOR INSIDER IN TRAINING (Per-User OCSVM)')
print('=' * 80)
evaluate_results(norm_scores_with, yTestBin)

# ============================================================================
# PIPELINE 2: INSIDER NOT IN TRAINING
# ============================================================================
print('\n' + '#' * 80)
print('# PIPELINE 2: INSIDER NOT IN TRAINING')
print('#' * 80)

# Train global fallback model (without insiders)
print('\nTraining global fallback model (without insiders)...')
xTrain_global_no = data1stHalf_no_insider[x_cols].values
xTrain_global_no = np.nan_to_num(xTrain_global_no, nan=0.0, posinf=0.0, neginf=0.0)

global_scaler_without = StandardScaler()
xTrain_global_no_scaled = global_scaler_without.fit_transform(xTrain_global_no)

if len(xTrain_global_no_scaled) > 50000:
    subsample_idx = np.random.choice(len(xTrain_global_no_scaled), 50000, replace=False)
    xTrain_sub_no = xTrain_global_no_scaled[subsample_idx]
else:
    xTrain_sub_no = xTrain_global_no_scaled

global_ocsvm_without = OneClassSVM(kernel='rbf', nu=NU_PARAM, gamma='scale', cache_size=500)
global_ocsvm_without.fit(xTrain_sub_no)
print(f'Global model trained with {len(xTrain_sub_no)} samples')

# Train per-user models (without insiders)
user_models_without, user_scalers_without, _, _ = train_per_user_models(
    data1stHalf_no_insider, train_users, x_cols, MIN_SAMPLES_PER_USER, NU_PARAM,
    "(without insiders)"
)

# Compute scores
raw_scores_without, norm_scores_without = compute_per_user_scores(
    xTest, test_users_list, user_models_without, user_scalers_without,
    global_ocsvm_without, global_scaler_without, "(without insiders)"
)

# Evaluate
print('\n' + '=' * 80)
print('RESULTS FOR INSIDER NOT IN TRAINING (Per-User OCSVM)')
print('=' * 80)
evaluate_results(norm_scores_without, yTestBin)

# ============================================================================
# BONUS: Global OCSVM (for comparison)
# ============================================================================
print('\n' + '#' * 80)
print('# COMPARISON: GLOBAL OCSVM (single model for all users)')
print('#' * 80)

# Scale test data with both scalers
xTest_scaled_with = global_scaler_with.transform(xTest)
xTest_scaled_with = np.nan_to_num(xTest_scaled_with, nan=0.0, posinf=0.0, neginf=0.0)

xTest_scaled_without = global_scaler_without.transform(xTest)
xTest_scaled_without = np.nan_to_num(xTest_scaled_without, nan=0.0, posinf=0.0, neginf=0.0)

# Get global scores
global_scores_with = -global_ocsvm_with.decision_function(xTest_scaled_with)
global_scores_without = -global_ocsvm_without.decision_function(xTest_scaled_without)

print('\n' + '=' * 80)
print('GLOBAL OCSVM - INSIDER IN TRAINING')
print('=' * 80)
evaluate_results(global_scores_with, yTestBin)

print('\n' + '=' * 80)
print('GLOBAL OCSVM - INSIDER NOT IN TRAINING')
print('=' * 80)
evaluate_results(global_scores_without, yTestBin)

# ============================================================================
# BONUS: Hybrid approach - combine per-user percentile with global score
# ============================================================================
print('\n' + '#' * 80)
print('# HYBRID: Per-User Percentile + Global Score')
print('#' * 80)

# Combine per-user percentile with global raw score
# This captures both "unusual for this user" and "unusual globally"
hybrid_scores_with = norm_scores_with * 0.5 + stats.rankdata(global_scores_with) / len(global_scores_with) * 100 * 0.5
hybrid_scores_without = norm_scores_without * 0.5 + stats.rankdata(global_scores_without) / len(global_scores_without) * 100 * 0.5

print('\n' + '=' * 80)
print('HYBRID (Per-User + Global) - INSIDER IN TRAINING')
print('=' * 80)
evaluate_results(hybrid_scores_with, yTestBin)

print('\n' + '=' * 80)
print('HYBRID (Per-User + Global) - INSIDER NOT IN TRAINING')
print('=' * 80)
evaluate_results(hybrid_scores_without, yTestBin)

print('\n' + '=' * 80)
print('Summary:')
print('  - Per-User: Each user has their own OCSVM model')
print('  - Global: Single OCSVM for all users')
print('  - Hybrid: Combines per-user deviation + global anomaly')
print('  - "Not in training" typically performs better (cleaner baseline)')
print('=' * 80)
