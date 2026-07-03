"""
correlation_pruning.py — Stage 3: drop highly-correlated redundant features.
"""

import numpy as np


def prune_correlated_features(df_features, meta_cols, threshold=0.9):
    """Drops one of each pair of features with pairwise |correlation| > threshold."""
    feature_cols = df_features.drop(columns=meta_cols)
    corr_matrix = feature_cols.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [c for c in upper.columns if any(upper[c] > threshold)]
    df_reduced = feature_cols.drop(columns=to_drop)
    print(f"Dropped {len(to_drop)} redundant features (>{threshold} corr). Remaining:", df_reduced.shape[1])
    return df_reduced
