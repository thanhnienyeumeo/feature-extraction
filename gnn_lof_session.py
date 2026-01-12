#!/usr/bin/env python
# coding: utf-8
"""
Session Anomaly Detection using GNN + LOF Pipeline (TESTING ONLY)

This script uses a pre-trained GNN (SessionGraph) to extract session embeddings,
then applies LOF to detect anomalous sessions.

Pipeline: Session Data → Pre-trained GNN → Session Embeddings → LOF → Anomaly Score

This script is for TESTING ONLY - no training is performed.
- Must provide a path to pre-trained GNN weights (.pt file)
- Must provide a path to test data pickle file

Test data format (pickle file): [[sessions], [labels]]
- sessions: list of sessions, each session is a list of item indices
  Example: [[1,2,3], [1,2,3,4], [2,6,4]]
- labels: list of binary labels (0=normal, 1=anomaly)
  Example: [1, 1, 0]

Usage:
    python gnn_lof_session.py --model model.pt --data test_data.pkl
    python gnn_lof_session.py --model model.pt --data test_data.pkl --lof_neighbors 20
"""

import argparse
import pickle
import numpy as np
import torch
from sklearn.neighbors import LocalOutlierFactor

from gnn import SessionGraph, GNN, trans_to_cuda, trans_to_cpu
from utils import evaluate_results

print('=' * 80)
print('Session Anomaly Detection using GNN + LOF Pipeline (TESTING ONLY)')
print('Pipeline: Session Data → Pre-trained GNN → Session Embeddings → LOF')
print('=' * 80)


# ============================================================================
# SESSION DATA PROCESSING UTILITIES
# ============================================================================

def build_session_graph(session):
    """
    Build a session graph from a sequence of items.
    
    For a session [1, 2, 3, 2], the graph has:
    - Nodes: unique items {1, 2, 3}
    - Edges: transitions (1→2, 2→3, 3→2)
    
    Returns:
        items: list of unique items in the session (re-indexed)
        A: adjacency matrix [n_items, 2*n_items] with in-edges and out-edges
        alias: mapping from session positions to unique item indices
    """
    # Get unique items and create mapping
    node_set = list(set(session))
    node_to_idx = {node: idx for idx, node in enumerate(node_set)}
    
    n_nodes = len(node_set)
    
    # Build adjacency matrix
    # A_in[i][j] = number of edges from j to i
    # A_out[i][j] = number of edges from i to j
    A_in = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    A_out = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    
    for i in range(len(session) - 1):
        u = node_to_idx[session[i]]
        v = node_to_idx[session[i + 1]]
        A_out[u][v] += 1
        A_in[v][u] += 1
    
    # Normalize (row-wise)
    A_in_sum = A_in.sum(axis=1, keepdims=True)
    A_in_sum[A_in_sum == 0] = 1
    A_in = A_in / A_in_sum
    
    A_out_sum = A_out.sum(axis=1, keepdims=True)
    A_out_sum[A_out_sum == 0] = 1
    A_out = A_out / A_out_sum
    
    # Concatenate in and out edges
    A = np.concatenate([A_in, A_out], axis=1)
    
    # Create alias (position in session → index in unique items)
    alias = [node_to_idx[item] for item in session]
    
    return node_set, A, alias


def process_sessions_batch(sessions, max_len=None):
    """
    Process a batch of sessions into the format required by SessionGraph.
    
    Args:
        sessions: list of sessions, each is a list of item indices
        max_len: maximum session length (for padding), if None use max in batch
    
    Returns:
        items: [batch, max_n_nodes] - unique items in each session (padded)
        A: [batch, max_n_nodes, 2*max_n_nodes] - adjacency matrices
        alias_inputs: [batch, max_len] - position to item mapping
        mask: [batch, max_len] - valid positions (1 for valid, 0 for padding)
    """
    batch_size = len(sessions)
    
    # Process each session
    items_list = []
    A_list = []
    alias_list = []
    
    for session in sessions:
        items, A, alias = build_session_graph(session)
        items_list.append(items)
        A_list.append(A)
        alias_list.append(alias)
    
    # Find max dimensions for padding
    max_n_nodes = max(len(items) for items in items_list)
    if max_len is None:
        max_len = max(len(session) for session in sessions)
    
    # Create padded arrays
    items_padded = np.zeros((batch_size, max_n_nodes), dtype=np.int64)
    A_padded = np.zeros((batch_size, max_n_nodes, 2 * max_n_nodes), dtype=np.float32)
    alias_padded = np.zeros((batch_size, max_len), dtype=np.int64)
    mask = np.zeros((batch_size, max_len), dtype=np.int64)
    
    for i, (items, A, alias, session) in enumerate(zip(items_list, A_list, alias_list, sessions)):
        n_nodes = len(items)
        seq_len = len(session)
        
        items_padded[i, :n_nodes] = items
        A_padded[i, :n_nodes, :n_nodes] = A[:, :n_nodes]  # in-edges
        A_padded[i, :n_nodes, max_n_nodes:max_n_nodes + n_nodes] = A[:, n_nodes:]  # out-edges
        alias_padded[i, :seq_len] = alias
        mask[i, :seq_len] = 1
    
    return items_padded, A_padded, alias_padded, mask


def extract_session_embeddings(model, sessions, batch_size=100):
    """
    Extract session embeddings from a pre-trained SessionGraph model.
    
    The embedding for each session is computed by:
    1. Getting hidden states for all items via GNN forward pass
    2. Aggregating hidden states using attention mechanism (similar to compute_scores)
    
    Args:
        model: Pre-trained SessionGraph model
        sessions: list of sessions
        batch_size: batch size for processing
    
    Returns:
        embeddings: [n_sessions, hidden_size] session embeddings
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
            
            # Process sessions into batch format
            items, A, alias_inputs, mask = process_sessions_batch(batch_sessions)
            
            # Move to device
            items = trans_to_cuda(torch.LongTensor(items))
            A = trans_to_cuda(torch.FloatTensor(A))
            alias_inputs = trans_to_cuda(torch.LongTensor(alias_inputs))
            mask = trans_to_cuda(torch.LongTensor(mask))
            
            # Forward pass through GNN
            hidden = model(items, A)  # [batch, max_n_nodes, hidden_size]
            
            # Get sequence hidden states using alias
            # For each sample, gather the hidden states in sequence order
            batch_embeddings = []
            for i in range(len(batch_sessions)):
                # Get hidden states in sequence order
                seq_hidden = hidden[i][alias_inputs[i]]  # [max_len, hidden_size]
                
                # Apply attention-like aggregation (similar to compute_scores)
                # Get the last valid hidden state
                valid_len = mask[i].sum().item()
                ht = seq_hidden[valid_len - 1]  # Last item hidden state
                
                # Attention over all items
                q1 = model.linear_one(ht).unsqueeze(0)  # [1, hidden_size]
                q2 = model.linear_two(seq_hidden)  # [max_len, hidden_size]
                alpha = model.linear_three(torch.sigmoid(q1 + q2))  # [max_len, 1]
                
                # Weighted sum with mask
                mask_float = mask[i].float().unsqueeze(1)  # [max_len, 1]
                session_repr = (alpha * seq_hidden * mask_float).sum(dim=0)  # [hidden_size]
                
                # Combine with last item if hybrid mode
                if not model.nonhybrid:
                    session_repr = model.linear_transform(torch.cat([session_repr, ht], dim=0))
                
                batch_embeddings.append(session_repr)
            
            # Stack batch embeddings
            batch_embeddings = torch.stack(batch_embeddings)  # [batch, hidden_size]
            batch_embeddings = trans_to_cpu(batch_embeddings).numpy()
            all_embeddings.append(batch_embeddings)
    
    # Concatenate all batches
    embeddings = np.concatenate(all_embeddings, axis=0)
    
    return embeddings


# ============================================================================
# MODEL LOADING
# ============================================================================

class ModelOptions:
    """
    Options class to hold model hyperparameters.
    This mimics the opt object used in SessionGraph initialization.
    """
    def __init__(self, hidden_size=100, batch_size=100, nonhybrid=False, 
                 step=1, lr=0.001, l2=1e-5, lr_dc_step=3, lr_dc=0.1):
        self.hiddenSize = hidden_size
        self.batchSize = batch_size
        self.nonhybrid = nonhybrid
        self.step = step
        self.lr = lr
        self.l2 = l2
        self.lr_dc_step = lr_dc_step
        self.lr_dc = lr_dc


def load_model(model_path, n_node, opt=None):
    """
    Load a pre-trained SessionGraph model from a .pt file.
    
    Args:
        model_path: path to the .pt file
        n_node: number of nodes (items) in the model
        opt: ModelOptions object (if None, will try to load from checkpoint)
    
    Returns:
        model: loaded SessionGraph model
    """
    print(f'\nLoading model from: {model_path}')
    
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    # Try to extract options from checkpoint
    if opt is None:
        if 'opt' in checkpoint:
            opt = checkpoint['opt']
        elif 'options' in checkpoint:
            opt = checkpoint['options']
        else:
            # Try to infer from state dict
            state_dict = checkpoint if isinstance(checkpoint, dict) and 'state_dict' not in checkpoint else checkpoint.get('state_dict', checkpoint)
            if 'embedding.weight' in state_dict:
                hidden_size = state_dict['embedding.weight'].shape[1]
                opt = ModelOptions(hidden_size=hidden_size)
                print(f'Inferred hidden_size={hidden_size} from checkpoint')
            else:
                print('Warning: Could not infer options, using defaults')
                opt = ModelOptions()
    
    # Get n_node from checkpoint if available
    if 'n_node' in checkpoint:
        n_node = checkpoint['n_node']
    
    # Create model
    model = SessionGraph(opt, n_node)
    
    # Load state dict
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # Assume checkpoint is the state dict itself
        model.load_state_dict(checkpoint)
    
    model = trans_to_cuda(model)
    model.eval()
    
    print(f'Model loaded successfully')
    print(f'  - Hidden size: {opt.hiddenSize}')
    print(f'  - Number of nodes: {n_node}')
    print(f'  - Nonhybrid: {opt.nonhybrid}')
    
    return model, opt


# ============================================================================
# DATA LOADING
# ============================================================================

def load_test_data(data_path):
    """
    Load test data from a pickle file.
    
    Expected format: [[sessions], [labels]]
    - sessions: list of sessions, each session is a list of item indices
    - labels: list of binary labels (0=normal, 1=anomaly)
    
    Example: [[[1,2,3], [1,2,3,4], [2,6,4]], [1, 1, 0]]
    
    Args:
        data_path: path to the pickle file
    
    Returns:
        sessions: list of sessions
        labels: numpy array of binary labels
    """
    print(f'\nLoading test data from: {data_path}')
    
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    sessions = data[0]
    labels = np.array(data[1])
    
    print(f'  - Number of sessions: {len(sessions)}')
    print(f'  - Number of labels: {len(labels)}')
    print(f'  - Anomalous sessions: {labels.sum()} ({100*labels.mean():.2f}%)')
    
    # Validate data
    if len(sessions) != len(labels):
        raise ValueError(f'Number of sessions ({len(sessions)}) does not match number of labels ({len(labels)})')
    
    # Get statistics
    session_lengths = [len(s) for s in sessions]
    print(f'  - Session lengths: min={min(session_lengths)}, max={max(session_lengths)}, avg={np.mean(session_lengths):.1f}')
    
    # Get max item index to determine n_node
    max_item = max(max(s) for s in sessions)
    print(f'  - Max item index: {max_item}')
    
    return sessions, labels, max_item


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Session Anomaly Detection using GNN + LOF (Testing Only)')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to pre-trained model weights (.pt file)')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to test data pickle file')
    parser.add_argument('--lof_neighbors', type=int, default=20,
                        help='Number of neighbors for LOF (default: 20)')
    parser.add_argument('--lof_fitted', type=str, default=None,
                        help='Path to fitted LOF model pickle (optional)')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='Batch size for embedding extraction (default: 100)')
    parser.add_argument('--hidden_size', type=int, default=None,
                        help='Hidden size (if not in checkpoint)')
    parser.add_argument('--n_node', type=int, default=None,
                        help='Number of nodes/items (if not in checkpoint)')
    
    args = parser.parse_args()
    
    # =========================================================================
    # STEP 1: Validate inputs
    # =========================================================================
    if not args.model:
        raise ValueError('ERROR: Must provide path to pre-trained model weights (--model)')
    
    if not args.data:
        raise ValueError('ERROR: Must provide path to test data (--data)')
    
    # =========================================================================
    # STEP 2: Load test data
    # =========================================================================
    sessions, labels, max_item = load_test_data(args.data)
    
    # Determine n_node (number of items in vocabulary)
    # n_node should be at least max_item + 1 (0-indexed) + 1 (for padding/unknown)
    n_node = args.n_node if args.n_node else max_item + 2
    
    # =========================================================================
    # STEP 3: Load pre-trained model
    # =========================================================================
    opt = None
    if args.hidden_size:
        opt = ModelOptions(hidden_size=args.hidden_size)
    
    model, opt = load_model(args.model, n_node, opt)
    
    # =========================================================================
    # STEP 4: Extract session embeddings using GNN
    # =========================================================================
    print('\n' + '=' * 80)
    print('EXTRACTING SESSION EMBEDDINGS')
    print('=' * 80)
    
    print('\nExtracting embeddings from pre-trained GNN...')
    embeddings = extract_session_embeddings(model, sessions, batch_size=args.batch_size)
    print(f'Session embeddings shape: {embeddings.shape}')
    
    # Handle any NaN or infinite values
    embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=0.0, neginf=0.0)
    
    # =========================================================================
    # STEP 5: Apply LOF for anomaly detection
    # =========================================================================
    print('\n' + '=' * 80)
    print('APPLYING LOF FOR ANOMALY DETECTION')
    print('=' * 80)
    
    if args.lof_fitted:
        # Load pre-fitted LOF model
        print(f'\nLoading fitted LOF model from: {args.lof_fitted}')
        with open(args.lof_fitted, 'rb') as f:
            lof = pickle.load(f)
        print('LOF model loaded')
    else:
        # Fit LOF on test embeddings (novelty=False for transductive anomaly detection)
        print(f'\nFitting LOF on session embeddings (n_neighbors={args.lof_neighbors})...')
        lof = LocalOutlierFactor(
            n_neighbors=args.lof_neighbors,
            contamination='auto',
            novelty=False,  # Transductive mode for test-only scenario
            n_jobs=-1
        )
        # fit_predict returns -1 for outliers, 1 for inliers
        lof.fit(embeddings)
        print('LOF fitting completed')
    
    # Get anomaly scores
    print('Calculating anomaly scores...')
    # negative_outlier_factor_ gives the opposite of LOF scores (higher = more anomalous)
    anomaly_scores = -lof.negative_outlier_factor_
    
    print(f'\nAnomaly scores statistics:')
    print(f'  - Min: {anomaly_scores.min():.4f}')
    print(f'  - Max: {anomaly_scores.max():.4f}')
    print(f'  - Mean: {anomaly_scores.mean():.4f}')
    print(f'  - Std: {anomaly_scores.std():.4f}')
    
    # =========================================================================
    # STEP 6: Evaluate results
    # =========================================================================
    print('\n' + '=' * 80)
    print('EVALUATION RESULTS: GNN + LOF SESSION ANOMALY DETECTION')
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
Pipeline completed:
1. Loaded pre-trained GNN model from: {args.model}
2. Loaded test sessions from: {args.data}
   - Total sessions: {len(sessions)}
   - Anomalous sessions: {labels.sum()} ({100*labels.mean():.2f}%)
3. Extracted session embeddings using GNN forward pass
   - Embedding dimension: {embeddings.shape[1]}
4. Applied LOF for anomaly detection
   - LOF neighbors: {args.lof_neighbors}
5. Evaluated results (see metrics above)

Note: This is TESTING ONLY - no model training was performed.
The GNN model was used purely as an embedding extractor.
''')
    
    return anomaly_scores, labels


if __name__ == '__main__':
    main()
