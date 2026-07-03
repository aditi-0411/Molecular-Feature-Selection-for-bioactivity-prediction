"""
scaffold_split.py — Stage 7: Murcko scaffold split evaluation.

A random train/test split lets near-identical analogs of the same molecular
scaffold appear in both splits, inflating validation metrics. Scaffold
splitting groups molecules by their Murcko scaffold first, then assigns
whole scaffold groups to train or test — a much more honest test of
generalization to genuinely new chemotypes.
"""

import lightgbm as lgb
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score


def scaffold_split(df, smiles_col="SMILES", test_frac=0.2, seed=42):
    scaffolds = {}
    for i, smi in enumerate(df[smiles_col]):
        mol = Chem.MolFromSmiles(smi)
        if mol:
            scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
            scaffolds.setdefault(scaf, []).append(i)

    scaffold_sets = sorted(scaffolds.values(), key=lambda x: len(x), reverse=True)
    train_idx, test_idx = [], []
    total = len(df)
    for s in scaffold_sets:
        if len(test_idx) + len(s) < test_frac * total:
            test_idx.extend(s)
        else:
            train_idx.extend(s)
    return train_idx, test_idx


def evaluate_scaffold_split(df_shap_final, target_col="BACE1_activity", smiles_col="SMILES", seed=42):
    train_idx, test_idx = scaffold_split(df_shap_final, smiles_col=smiles_col, seed=seed)
    train_df, test_df = df_shap_final.iloc[train_idx], df_shap_final.iloc[test_idx]

    X_train = train_df.drop(columns=[smiles_col, target_col])
    y_train = train_df[target_col]
    X_test = test_df.drop(columns=[smiles_col, target_col])
    y_test = test_df[target_col]

    model = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=64, random_state=seed, verbosity=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, y_pred)
    prec, rec, _ = precision_recall_curve(y_test, y_pred)
    pr_auc = auc(rec, prec)

    print(f"Scaffold-split ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f}")
    return {"model": model, "roc_auc": roc_auc, "pr_auc": pr_auc, "X_test": X_test, "y_test": y_test}
