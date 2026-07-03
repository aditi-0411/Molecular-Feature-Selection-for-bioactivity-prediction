"""
stability_selection.py — Stage 8: bootstrap stability selection.

Runs many bootstrap resamples, fits a LightGBM model on each, and keeps
features that are consistently ranked as important across resamples —
distinguishing genuinely predictive features from ones that only looked
important on one particular fit.
"""

from collections import Counter

import lightgbm as lgb
import numpy as np
import pandas as pd
from tqdm import tqdm


def bootstrap_stability_selection(
    df_shap_final,
    target_col="BACE1_activity",
    smiles_col="SMILES",
    n_bootstrap=50,
    top_frac=0.4,
    freq_threshold=0.25,
    seed=42,
):
    drop_cols = [c for c in [smiles_col, target_col] if c in df_shap_final.columns]
    X_data = df_shap_final.drop(columns=drop_cols)
    y_data = df_shap_final[target_col]

    bootstrap_top_k = int(top_frac * X_data.shape[1])
    selected_counts = Counter()

    for i in tqdm(range(n_bootstrap), desc="Bootstrap runs"):
        idx = np.random.choice(len(X_data), len(X_data), replace=True)
        Xb, yb = X_data.iloc[idx], y_data.iloc[idx]
        model = lgb.LGBMClassifier(
            n_estimators=300, learning_rate=0.05, num_leaves=64,
            random_state=seed + i, verbosity=-1,
        )
        model.fit(Xb, yb)
        imps = pd.Series(model.feature_importances_, index=X_data.columns)
        top_feats = imps.sort_values(ascending=False).head(bootstrap_top_k).index.tolist()
        selected_counts.update(top_feats)

    freq_df = pd.DataFrame({
        "Feature": list(selected_counts.keys()),
        "Count": [selected_counts[f] for f in selected_counts.keys()],
    })
    freq_df["Frequency"] = freq_df["Count"] / n_bootstrap
    freq_df = freq_df.sort_values("Frequency", ascending=False)

    robust_features = freq_df[freq_df["Frequency"] >= freq_threshold]["Feature"].tolist()
    print(f"Stability selection complete. Robust features (>= {freq_threshold}): {len(robust_features)}")

    return freq_df, robust_features, X_data, y_data
