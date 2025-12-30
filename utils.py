import numpy as np
from sklearn.metrics import roc_auc_score
def evaluate_results(scores, yTestBin=None):
    # AUC Score
    if yTestBin is not None and yTestBin.sum() > 0:  # Only if there are actual insiders
        auc = roc_auc_score(yTestBin, scores)
        print(f'\nAUC Score: {auc:.4f}')
        print('(AUC > 0.5 means model is better than random, AUC = 1.0 is perfect)')
    else:
        print('\nNo insider labels found in data - cannot compute AUC')

    # Detection rate at different budgets
    print('\nMetrics at different investigation budgets:')
    print('-' * 80)
    print(f'{"Budget":<10} {"TP":<8} {"FP":<8} {"FN":<8} {"TN":<8} {"Precision":<12} {"Recall":<12} {"F1":<10}')
    print('-' * 80)

    for ib in [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2]:
        threshold = np.percentile(scores, 100 - 100 * ib)
        predicted = scores > threshold
        
        # Calculate confusion matrix
        TP = np.sum((predicted == True) & (yTestBin == True))
        FP = np.sum((predicted == True) & (yTestBin == False))
        FN = np.sum((predicted == False) & (yTestBin == True))
        TN = np.sum((predicted == False) & (yTestBin == False))
        
        # Calculate metrics
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        fpr= FP / (FP + TN)
        print(f'{100*ib:>6.1f}%   {TP:<8} {FP:<8} {FN:<8} {TN:<8} {precision:<12.4f} {recall:<12.4f} {f1:<10.4f} {fpr:<10.4f}')