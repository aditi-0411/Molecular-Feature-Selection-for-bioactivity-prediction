"""
fingerprints.py — Stage 2: Morgan fingerprints + descriptor feature matrix.
"""

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import DataStructs, rdFingerprintGenerator

POSSIBLE_DESCRIPTORS = [
    "Molecular Weight", "MolecularWeight", "MolWeight",
    "LogP", "HBA", "HBD", "TPSA", "BACE1_activity",
]
META_COLUMNS = ["ChEMBL ID", "SMILES"]


def smiles_to_morgan_fp(smiles, radius=2, n_bits=1024):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fp = morgan_gen.GetFingerprint(mol)
    arr = np.zeros((n_bits,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def build_feature_matrix(df, radius=2, n_bits=1024):
    """Parses SMILES into Morgan fingerprints and concatenates with
    descriptor + metadata columns. Drops rows with unparsable SMILES."""
    fps, valid_idx = [], []
    for idx, smi in df["SMILES"].astype(str).items():
        arr = smiles_to_morgan_fp(smi, radius=radius, n_bits=n_bits)
        if arr is None:
            continue
        fps.append(arr)
        valid_idx.append(idx)

    print(f"Parsed {len(fps)} valid molecules | Dropped {len(df) - len(fps)} invalid SMILES")
    if len(fps) == 0:
        raise ValueError("No valid SMILES parsed.")

    desc_cols = [c for c in POSSIBLE_DESCRIPTORS if c in df.columns]
    meta_cols = [c for c in META_COLUMNS if c in df.columns]

    fp_array = np.vstack(fps)
    fp_cols = [f"fp_{i}" for i in range(fp_array.shape[1])]
    df_fp = pd.DataFrame(fp_array, columns=fp_cols).reset_index(drop=True)
    df_desc = df.loc[valid_idx, desc_cols].reset_index(drop=True)
    df_meta = df.loc[valid_idx, meta_cols].reset_index(drop=True)

    df_features = pd.concat([df_meta, df_desc, df_fp], axis=1)
    print("Final feature matrix shape:", df_features.shape)
    return df_features, df_meta, meta_cols
