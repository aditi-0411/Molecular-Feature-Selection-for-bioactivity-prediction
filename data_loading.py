"""
data_loading.py — Stage 1: load and clean the raw molecule dataset.

Expects a CSV with at minimum: SMILES, Protein, and optionally descriptor
columns (LogP, HBA, HBD, TPSA) and a ChEMBL ID.
"""

import pandas as pd


def load_and_clean(csv_path):
    """Loads the raw CSV, imputes descriptor NaNs, and derives the binary
    BACE1_activity label from the 'Protein' column."""
    data = pd.read_csv(csv_path)
    df = data.copy()
    print("Initial shape:", df.shape)

    if "Name" in df.columns:
        df = df.drop("Name", axis=1)

    df.columns = df.columns.str.strip()

    if "ChEMBL ID" in df.columns:
        df["ChEMBL ID"] = df["ChEMBL ID"].astype(str).str.replace("CHEMBL", "", regex=False)

    for col in ["LogP", "HBA", "HBD", "TPSA"]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mean())

    if "Protein" in df.columns:
        df["BACE1_activity"] = (df["Protein"] == "BACE1").astype(int)
    else:
        raise KeyError("Column 'Protein' not found.")

    print("Class balance:", df["BACE1_activity"].value_counts(normalize=True).to_dict())
    return df
