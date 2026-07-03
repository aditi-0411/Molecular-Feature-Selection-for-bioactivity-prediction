"""
cv_feature_selection.py — Stage 4: leakage-free CV feature selection.

For each CV fold, the variance/mutual-information filter and LightGBM
importance ranking are computed using ONLY that fold's training split —
never the validation fold or the full dataset — to avoid feature-selection
leakage. Features that are consistently selected across folds are kept.
"""

from collections import Counter

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import StratifiedKFold


def select_stable_features(
    df_reduced,
    target_col="BACE1_activity",
    n_splits=5,
    var_threshold=0.01,
    mi_threshold=0.01,
    top_pct=0.50,
    min_keep=500,
    min_fold_count=3,
    seed=42,
):
    X = df_reduced.drop(columns=[target_col])
    y = df_reduced[target_col]

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    selected_feature_sets = []

    print(f"Performing {n_splits}-fold leakage-free feature selection...")
    for fold, (train_idx, _) in enumerate(skf.split(X, y), 1):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]

        variances = X_train.var()
        mi_scores = mutual_info_classif(X_train, y_train, discrete_features="auto", random_state=seed)
        mi_df = pd.DataFrame({"Feature": X.columns, "Variance": variances.values, "MI": mi_scores})
        mask = (mi_df["Variance"] > var_threshold) | (mi_df["MI"] > mi_threshold)
        X_filtered = X_train[mi_df.loc[mask, "Feature"]]
        print(f"Fold {fold}: after variance/MI filter: {X_filtered.shape[1]} features")

        model = lgb.LGBMClassifier(
            n_estimators=500, learning_rate=0.05, num_leaves=64,
            n_jobs=-1, random_state=seed, verbosity=-1,
        )
        model.fit(X_filtered, y_train)
        importances = pd.Series(model.feature_importances_, index=X_filtered.columns).sort_values(ascending=False)

        top_k = max(min_keep, int(np.ceil(top_pct * len(importances))))
        top_k = min(top_k, len(importances))
        top_features_fold = importances.head(top_k).index.tolist()
        print(f"Fold {fold}: selected {len(top_features_fold)} top features")
        selected_feature_sets.append(top_features_fold)

    all_features = [feat for sublist in selected_feature_sets for feat in sublist]
    feature_counts = Counter(all_features)
    stable_features = [feat for feat, count in feature_counts.items() if count >= min_fold_count]
    print(f"Stable features across folds: {len(stable_features)}")

    df_reduced_final = df_reduced[stable_features + [target_col]]
    print("df_reduced_final shape:", df_reduced_final.shape)
    return df_reduced_final, stable_features
