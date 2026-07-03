"""
shap_selection.py — Stage 6: SHAP-based interpretability refinement.

Includes the fixed "final eval" step: the original notebook referenced
undefined X_train2/y_train2/X_test2/y_test2 here. This version correctly
splits X_shap/y_shap (the actual SHAP-trimmed feature matrix).
"""

import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


def shap_refine(df_lasso_final_or_reduced, target_col="BACE1_activity", top_k=150, seed=42):
    df = df_lasso_final_or_reduced.copy()
    X = df.drop(columns=[target_col])
    y = df[target_col]
    print(f"Data for SHAP: {X.shape[0]} samples x {X.shape[1]} features")

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    model = LGBMClassifier(
        n_estimators=500, learning_rate=0.05, num_leaves=64,
        subsample=0.8, colsample_bytree=0.8,
        random_state=seed, n_jobs=-1, verbosity=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_pred)
    print(f"Validation ROC-AUC before SHAP refinement: {auc:.4f}")

    explainer = shap.TreeExplainer(model)
    sample_idx = np.random.choice(X_train.shape[0], size=min(1000, X_train.shape[0]), replace=False)
    X_sample = X_train.iloc[sample_idx]
    shap_values_raw = explainer.shap_values(X_sample)

    if isinstance(shap_values_raw, list):
        shap_values = shap_values_raw[1] if len(shap_values_raw) > 1 else shap_values_raw[0]
    else:
        shap_values = shap_values_raw

    shap_importance = pd.DataFrame({
        "Feature": X.columns,
        "MeanAbsSHAP": np.abs(shap_values).mean(axis=0),
    }).sort_values(by="MeanAbsSHAP", ascending=False).reset_index(drop=True)

    selected_features = shap_importance.head(top_k)["Feature"].tolist()

    df_shap_final = pd.concat([
        df[selected_features].reset_index(drop=True),
        df[target_col].reset_index(drop=True),
    ], axis=1)

    print(f"Kept after SHAP ranking: {len(selected_features)}")
    return df_shap_final, selected_features, shap_importance


def shap_final_eval(df_shap_final, target_col="BACE1_activity", seed=42):
    """FIX: uses the actual SHAP-trimmed X_shap/y_shap split, not the
    undefined X_train2/y_train2/X_test2/y_test2 from the original notebook."""
    X_shap = df_shap_final.drop(columns=[target_col])
    y_shap = df_shap_final[target_col]

    X_train_shap, X_val_shap, y_train_shap, y_val_shap = train_test_split(
        X_shap, y_shap, test_size=0.2, stratify=y_shap, random_state=42
    )

    model_shap = LGBMClassifier(
        n_estimators=500, learning_rate=0.05, num_leaves=64, random_state=seed, verbosity=-1
    )
    model_shap.fit(X_train_shap, y_train_shap)
    y_pred2 = model_shap.predict_proba(X_val_shap)[:, 1]
    auc_shap = roc_auc_score(y_val_shap, y_pred2)
    print(f"Final validation ROC-AUC using SHAP-selected features: {auc_shap:.4f}")
    return auc_shap, model_shap
