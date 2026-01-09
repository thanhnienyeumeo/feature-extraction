#!/usr/bin/env python
# coding: utf-8
"""
Session Anomaly Detection using VGAE + LOF Pipeline (PyTorch Geometric Version)

This script uses Variational Graph Auto-Encoder (VGAE) implemented with PyTorch Geometric
to learn session embeddings, then applies LOF to detect anomalous sessions.

Pipeline:
1. TRAINING: Load training sessions → Build session graphs → Train VGAE
2. TESTING: Load test sessions → Extract embeddings from trained VGAE → LOF → Anomaly Detection

This script includes:
- Training phase: Train VGAE on session data (no labels needed)
- Testing phase: Use trained VGAE embeddings with LOF for anomaly detection

Data format (pickle file):
- Training: [sessions] - list of sessions, each session is a list of item indices
  Example: [[1,2,3], [1,2,3,4], [2,6,4]]
- Testing: [[sessions], [labels]] - sessions and binary labels
  Example: [[[1,2,3], [1,2,3,4], [2,6,4]], [1, 1, 0]]

Usage:
    python VGAE_LOF_geometric.py --train_data train.pkl --test_data test.pkl
    python VGAE_LOF_geometric.py --train_data train.pkl --test_data test.pkl --embedding_dim 128
"""

import argparse
import pickle
import numpy as np
import torch
from sklearn.neighbors import LocalOutlierFactor

from vgae_geometric import (
    SessionVGAE,
    train_vgae_geometric,
    extract_vgae_embeddings_geometric,
    trans_to_cuda,
    trans_to_cpu,
    get_recommended_config
)
from utils import evaluate_results

print('=' * 80)
print('Session Anomaly Detection using VGAE + LOF Pipeline')
print('PyTorch Geometric Version')
print('Pipeline: Training Sessions → VGAE → Embeddings → LOF → Anomaly Detection')
print('Features: Learnable Item Embeddings + Structural Features')
print('=' * 80)


# ============================================================================
# DATA LOADING
# ============================================================================

def load_train_data(data_path):
    """
    Load training data from a pickle file.
    
    Training data format: [sessions] - list of sessions (no labels)
    Each session is a list of item indices.
    
    Example: [[1,2,3], [1,2,3,4], [2,6,4]]
    
    Args:
        data_path: path to the pickle file
    
    Returns:
        sessions: list of sessions
        max_item: maximum item index
    """
    print(f'\nLoading training data from: {data_path}')
    
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    # Handle both formats: just sessions or [sessions, labels]
    if isinstance(data, list) and len(data) == 2 and isinstance(data[1], list):
        if isinstance(data[0][0], list):
            sessions = data[0]
            print('  Note: Training data has labels, but they will be ignored')
        else:
            sessions = data
    else:
        sessions = data
    
    print(f'  - Number of training sessions: {len(sessions)}')
    
    session_lengths = [len(s) for s in sessions]
    print(f'  - Session lengths: min={min(session_lengths)}, max={max(session_lengths)}, avg={np.mean(session_lengths):.1f}')
    
    max_item = max(max(s) for s in sessions)
    print(f'  - Max item index: {max_item}')
    
    return sessions, max_item


def load_test_data(data_path):
    """
    Load test data from a pickle file.
    
    Test data format: [[sessions], [labels]]
    - sessions: list of sessions, each session is a list of item indices
    - labels: list of binary labels (0=normal, 1=anomaly)
    
    Example: [[[1,2,3], [1,2,3,4], [2,6,4]], [1, 1, 0]]
    
    Args:
        data_path: path to the pickle file
    
    Returns:
        sessions: list of sessions
        labels: numpy array of binary labels
        max_item: maximum item index
    """
    print(f'\nLoading test data from: {data_path}')
    
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    sessions = data[0]
    labels = np.array(data[1])
    
    print(f'  - Number of test sessions: {len(sessions)}')
    print(f'  - Number of labels: {len(labels)}')
    print(f'  - Anomalous sessions: {labels.sum()} ({100*labels.mean():.2f}%)')
    
    if len(sessions) != len(labels):
        raise ValueError(f'Number of sessions ({len(sessions)}) does not match number of labels ({len(labels)})')
    
    session_lengths = [len(s) for s in sessions]
    print(f'  - Session lengths: min={min(session_lengths)}, max={max(session_lengths)}, avg={np.mean(session_lengths):.1f}')
    
    max_item = max(max(s) for s in sessions)
    print(f'  - Max item index: {max_item}')
    
    return sessions, labels, max_item


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Session Anomaly Detection using VGAE + LOF (PyTorch Geometric)')
    
    # Data arguments
    parser.add_argument('--train_data', type=str, required=True,
                        help='Path to training data pickle file')
    parser.add_argument('--test_data', type=str, required=True,
                        help='Path to test data pickle file')
    
    # VGAE architecture arguments
    parser.add_argument('--embedding_dim', type=int, default=64,
                        help='Item embedding dimension (default: 64, recommended: 32-128)')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='Hidden dimension for VGAE encoder (default: 64)')
    parser.add_argument('--latent_dim', type=int, default=32,
                        help='Latent dimension for VGAE (default: 32)')
    parser.add_argument('--num_gcn_layers', type=int, default=2,
                        help='Number of GCN layers in encoder (default: 2)')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate (default: 0.1)')
    parser.add_argument('--no_structural_features', action='store_true',
                        help='Disable structural features (use only item embeddings)')
    parser.add_argument('--structural_feature_dim', type=int, default=16,
                        help='Structural feature projection dimension (default: 16)')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs (default: 50)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training (default: 32)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--early_stopping', type=int, default=10,
                        help='Early stopping patience (default: 10)')
    
    # LOF arguments
    parser.add_argument('--lof_neighbors', type=int, default=20,
                        help='Number of neighbors for LOF (default: 20)')
    
    # Model saving arguments
    parser.add_argument('--save_model', type=str, default=None,
                        help='Path to save trained VGAE model')
    parser.add_argument('--load_model', type=str, default=None,
                        help='Path to load pre-trained VGAE model (skip training)')
    
    # Auto configuration
    parser.add_argument('--auto_config', action='store_true',
                        help='Use recommended configuration based on dataset size')
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # =========================================================================
    # STEP 1: Load training data
    # =========================================================================
    train_sessions, train_max_item = load_train_data(args.train_data)
    
    # =========================================================================
    # STEP 2: Load test data
    # =========================================================================
    test_sessions, labels, test_max_item = load_test_data(args.test_data)
    
    # Determine total number of items
    n_items = max(train_max_item, test_max_item) + 1
    print(f'\nTotal number of items: {n_items}')
    
    # Determine use_structural_features
    use_structural_features = not args.no_structural_features
    
    # Auto configuration if requested
    if args.auto_config:
        dataset_size = 'small' if len(train_sessions) < 1000 else ('large' if len(train_sessions) > 100000 else 'medium')
        config = get_recommended_config(n_items, dataset_size)
        print(f'\nUsing auto-configuration for {dataset_size} dataset:')
        for key, value in config.items():
            print(f'  - {key}: {value}')
        
        args.embedding_dim = config['embedding_dim']
        args.hidden_dim = config['hidden_dim']
        args.latent_dim = config['latent_dim']
        args.num_gcn_layers = config['num_gcn_layers']
        args.dropout = config['dropout']
        args.structural_feature_dim = config['structural_feature_dim']
        use_structural_features = config['use_structural_features']
    
    # =========================================================================
    # STEP 3: Create or load VGAE model
    # =========================================================================
    print('\n' + '=' * 80)
    print('VGAE MODEL CONFIGURATION (PyTorch Geometric)')
    print('=' * 80)
    print(f'  - Number of items: {n_items}')
    print(f'  - Item embedding dimension: {args.embedding_dim}')
    print(f'  - Hidden dimension: {args.hidden_dim}')
    print(f'  - Latent dimension: {args.latent_dim}')
    print(f'  - Number of GCN layers: {args.num_gcn_layers}')
    print(f'  - Dropout: {args.dropout}')
    print(f'  - Use structural features: {use_structural_features}')
    if use_structural_features:
        print(f'  - Structural feature dimension: {args.structural_feature_dim}')
    
    if use_structural_features:
        total_input_dim = args.embedding_dim + args.structural_feature_dim
    else:
        total_input_dim = args.embedding_dim
    print(f'  - Total input dimension to GCN: {total_input_dim}')
    
    model = SessionVGAE(
        n_items=n_items,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        num_gcn_layers=args.num_gcn_layers,
        dropout=args.dropout,
        use_structural_features=use_structural_features,
        structural_feature_dim=args.structural_feature_dim
    )
    model = trans_to_cuda(model)
    
    # =========================================================================
    # STEP 4: Training phase (or load pre-trained model)
    # =========================================================================
    if args.load_model:
        print('\n' + '=' * 80)
        print('LOADING PRE-TRAINED VGAE MODEL')
        print('=' * 80)
        print(f'\nLoading model from: {args.load_model}')
        
        checkpoint = torch.load(args.load_model, map_location='cpu', weights_only=False)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
            if 'use_structural_features' in checkpoint:
                use_structural_features = checkpoint['use_structural_features']
        else:
            model.load_state_dict(checkpoint)
        model = trans_to_cuda(model)
        print('Model loaded successfully')
    else:
        print('\n' + '=' * 80)
        print('TRAINING VGAE ON SESSION DATA (PyTorch Geometric)')
        print('=' * 80)
        print(f'\nTraining configuration:')
        print(f'  - Epochs: {args.epochs}')
        print(f'  - Batch size: {args.batch_size}')
        print(f'  - Learning rate: {args.lr}')
        print(f'  - Early stopping patience: {args.early_stopping}')
        
        print('\nTraining VGAE...')
        model, history = train_vgae_geometric(
            model=model,
            train_sessions=train_sessions,
            n_items=n_items,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            early_stopping_patience=args.early_stopping,
            verbose=True,
            use_structural_features=use_structural_features
        )
        print('VGAE training completed')
        
        # Save model if requested
        if args.save_model:
            print(f'\nSaving model to: {args.save_model}')
            torch.save({
                'state_dict': model.state_dict(),
                'n_items': n_items,
                'embedding_dim': args.embedding_dim,
                'hidden_dim': args.hidden_dim,
                'latent_dim': args.latent_dim,
                'num_gcn_layers': args.num_gcn_layers,
                'dropout': args.dropout,
                'use_structural_features': use_structural_features,
                'structural_feature_dim': args.structural_feature_dim
            }, args.save_model)
            print('Model saved')
    
    # =========================================================================
    # STEP 5: Extract embeddings for training data (for LOF fitting)
    # =========================================================================
    print('\n' + '=' * 80)
    print('EXTRACTING SESSION EMBEDDINGS')
    print('=' * 80)
    
    print('\nExtracting embeddings for training sessions...')
    train_embeddings = extract_vgae_embeddings_geometric(
        model, train_sessions, n_items, batch_size=args.batch_size,
        use_structural_features=use_structural_features
    )
    print(f'Training embeddings shape: {train_embeddings.shape}')
    
    print('Extracting embeddings for test sessions...')
    test_embeddings = extract_vgae_embeddings_geometric(
        model, test_sessions, n_items, batch_size=args.batch_size,
        use_structural_features=use_structural_features
    )
    print(f'Test embeddings shape: {test_embeddings.shape}')
    
    # Handle NaN or infinite values
    train_embeddings = np.nan_to_num(train_embeddings, nan=0.0, posinf=0.0, neginf=0.0)
    test_embeddings = np.nan_to_num(test_embeddings, nan=0.0, posinf=0.0, neginf=0.0)
    
    # =========================================================================
    # STEP 6: Train LOF on training embeddings
    # =========================================================================
    print('\n' + '=' * 80)
    print('TRAINING LOF ON VGAE EMBEDDINGS')
    print('=' * 80)
    
    print(f'\nFitting LOF on training embeddings (n_neighbors={args.lof_neighbors})...')
    lof = LocalOutlierFactor(
        n_neighbors=args.lof_neighbors,
        contamination='auto',
        novelty=True,
        n_jobs=-1
    )
    lof.fit(train_embeddings)
    print('LOF training completed')
    
    # =========================================================================
    # STEP 7: Get anomaly scores for test data
    # =========================================================================
    print('\n' + '=' * 80)
    print('ANOMALY DETECTION ON TEST DATA')
    print('=' * 80)
    
    print('\nCalculating anomaly scores for test sessions...')
    anomaly_scores = -lof.score_samples(test_embeddings)
    
    print(f'\nAnomaly scores statistics:')
    print(f'  - Min: {anomaly_scores.min():.4f}')
    print(f'  - Max: {anomaly_scores.max():.4f}')
    print(f'  - Mean: {anomaly_scores.mean():.4f}')
    print(f'  - Std: {anomaly_scores.std():.4f}')
    
    # =========================================================================
    # STEP 8: Evaluate results
    # =========================================================================
    print('\n' + '=' * 80)
    print('EVALUATION RESULTS: VGAE + LOF SESSION ANOMALY DETECTION')
    print('=' * 80)
    
    yTestBin = labels > 0
    evaluate_results(anomaly_scores, yTestBin)
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print('\n' + '=' * 80)
    print('SUMMARY')
    print('=' * 80)
    print(f'''
Pipeline completed (PyTorch Geometric Version):

1. TRAINING PHASE:
   - Loaded training sessions from: {args.train_data}
   - Number of training sessions: {len(train_sessions)}
   - Trained VGAE model:
     * Item embedding dimension: {args.embedding_dim}
     * Hidden dimension: {args.hidden_dim}
     * Latent dimension: {args.latent_dim}
     * GCN layers: {args.num_gcn_layers}
     * Structural features: {use_structural_features}
     * Epochs: {args.epochs}

2. TESTING PHASE:
   - Loaded test sessions from: {args.test_data}
   - Number of test sessions: {len(test_sessions)}
   - Anomalous sessions: {labels.sum()} ({100*labels.mean():.2f}%)

3. ANOMALY DETECTION:
   - Extracted VGAE embeddings (dimension: {args.latent_dim})
   - Applied LOF with {args.lof_neighbors} neighbors
   - See evaluation metrics above

FEATURE DIMENSIONS EXPLAINED:
- Item embeddings ({args.embedding_dim}D): Learnable representations for each item
- Structural features ({args.structural_feature_dim if use_structural_features else 0}D): In/out degree, position, frequency
- Total input to GCN: {args.embedding_dim + (args.structural_feature_dim if use_structural_features else 0)}D
- Final session embedding: {args.latent_dim}D

ADVANTAGES OF PyTorch Geometric:
- Native support for batched graphs
- Efficient sparse operations
- Built-in negative sampling
- Cleaner edge_index format
''')
    
    return anomaly_scores, labels


if __name__ == '__main__':
    main()
