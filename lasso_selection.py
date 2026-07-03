"""
lasso_selection.py — Stage 5: L1-penalized logistic regression shrinkage.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def lasso_shrinkage(df_reduced_final, target_col="BACE1_activity", seed=42):
    X = df_reduced_final.drop(columns=[target_col])
    y = df_reduced_final[target_col]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    lasso = LogisticRegressionCV(
        Cs=10, penalty="l1", solver="saga", scoring="roc_auc",
        cv=5, max_iter=5000, n_jobs=-1, random_state=seed, refit=True,
    )
    lasso.fit(X_train_scaled, y_train)
    coef = np.abs(lasso.coef_).ravel()
    selected = X.columns[coef > 1e-6].tolist()

    y_pred_proba = lasso.predict_proba(X_val_scaled)[:, 1]
    auc = roc_auc_score(y_val, y_pred_proba)
    print(f"LASSO validation ROC-AUC: {auc:.4f}")

    df_lasso_final = pd.concat([X[selected].reset_index(drop=True), y.reset_index(drop=True)], axis=1)
    print(f"Final LASSO-selected features: {len(selected)} | Dropped: {len(X.columns) - len(selected)}")
    return df_lasso_final, selected, auc
