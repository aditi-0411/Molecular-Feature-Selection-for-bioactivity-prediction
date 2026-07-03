"""
pipeline.py — runs the full 9-stage molecular feature selection pipeline
end-to-end and returns a dict of key results.
"""

import os
import random

import numpy as np

from src.correlation_pruning import prune_correlated_features
from src.cv_feature_selection import select_stable_features
from src.data_loading import load_and_clean
from src.enrichment_factor import enrichment_report
from src.fingerprints import build_feature_matrix
from src.lasso_selection import lasso_shrinkage
from src.scaffold_split import evaluate_scaffold_split
from src.shap_selection import shap_final_eval, shap_refine
from src.stability_selection import bootstrap_stability_selection

SEED = 42


def set_seed(seed=SEED):
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def run_pipeline(csv_path, seed=SEED):
    set_seed(seed)

    # 1. Load & clean
    df = load_and_clean(csv_path)

    # 2. Fingerprints + descriptors
    df_features, df_meta, meta_cols = build_feature_matrix(df)

    # 3. Correlation pruning
    df_reduced = prune_correlated_features(df_features, meta_cols)

    # 4. Leakage-free CV feature selection
    df_reduced_final, stable_features = select_stable_features(df_reduced, seed=seed)

    # 5. LASSO shrinkage
    df_lasso_final, lasso_selected, lasso_auc = lasso_shrinkage(df_reduced_final, seed=seed)

    # 6. SHAP refinement
    df_shap_final, shap_selected, shap_importance = shap_refine(df_lasso_final, seed=seed)
    if "SMILES" in df_meta.columns:
        df_shap_final = df_shap_final.join(df_meta[["SMILES"]])
    shap_auc, _ = shap_final_eval(df_shap_final.drop(columns=["SMILES"], errors="ignore"), seed=seed)

    # 7. Scaffold split evaluation
    scaffold_results = evaluate_scaffold_split(df_shap_final, seed=seed)

    # 8. Bootstrap stability selection
    freq_df, robust_features, X_data, y_data = bootstrap_stability_selection(df_shap_final, seed=seed)

    # 9. Enrichment factor (using scaffold-split test predictions as the scored set)
    ef_df = enrichment_report(
        scaffold_results["y_test"],
        {"Scaffold-split model": scaffold_results["model"].predict_proba(scaffold_results["X_test"])[:, 1]},
    )

    return {
        "n_stable_cv_features": len(stable_features),
        "lasso_auc": lasso_auc,
        "n_lasso_features": len(lasso_selected),
        "n_shap_features": len(shap_selected),
        "shap_final_auc": shap_auc,
        "scaffold_split_auc": scaffold_results["roc_auc"],
        "scaffold_split_pr_auc": scaffold_results["pr_auc"],
        "n_robust_features": len(robust_features),
        "robust_features": robust_features,
        "stability_freq_df": freq_df,
        "enrichment_factor_df": ef_df,
    }


if __name__ == "__main__":
    results = run_pipeline("molecule_classification_dataset.csv")
    for k, v in results.items():
        if not hasattr(v, "shape"):  # skip printing full DataFrames
            print(f"{k}: {v}")
