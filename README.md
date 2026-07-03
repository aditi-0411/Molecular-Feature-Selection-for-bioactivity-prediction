# Molecular Feature Selection Framework for BACE1 Activity Classification

A multi-stage, leakage-aware feature selection pipeline for molecular machine learning, applied to classifying BACE1 (β-secretase 1, an Alzheimer's drug target) activity from SMILES strings. The framework takes a molecule dataset from ~1,000 raw Morgan-fingerprint + descriptor features down to a compact, statistically stable feature set — validated with scaffold-based splitting and a virtual-screening enrichment metric, not just a random train/test split.


---

## Why this is harder than a normal tabular ML problem

Molecular datasets have failure modes that a generic `train_test_split` + `SelectKBest` pipeline will walk straight into:

- **Structural leakage** — random splits let near-identical analogs of the same scaffold land in both train and test, inflating validation metrics.
- **Feature-selection leakage** — picking features using the full dataset (including validation/test rows) before splitting silently leaks label information.
- **Fingerprint bits are high-dimensional and mostly redundant** — 1024-bit Morgan fingerprints are extremely correlated and mostly uninformative for any single dataset.
- **Feature importance instability** — a single train/test split's "top features" often don't replicate on a different resample; without checking that, you're reporting noise as insight.

This pipeline addresses each of those directly (see stages below).

## Pipeline

```
Raw CSV (SMILES + descriptors)
      │
1. Load & clean (impute descriptor NaNs, derive BACE1 binary label)
      │
2. Morgan fingerprints (radius=2, 1024 bits) + RDKit descriptors
      │
3. Correlation pruning (drop features with pairwise |r| > 0.9)
      │
4. Leakage-free CV feature selection
   — 5-fold StratifiedKFold, variance + mutual-info filter computed
     PER FOLD on the training split only, then LightGBM importance
     ranking; keep features stable across ≥3 of 5 folds
      │
5. LASSO shrinkage (L1-penalized LogisticRegressionCV, 5-fold, scoring=ROC-AUC)
      │
6. SHAP interpretability refinement (TreeExplainer on LightGBM, keep top-150 |SHAP|)
      │
7. Scaffold-split evaluation (Murcko scaffold split, not random —
   the real test of whether the model generalizes to new chemotypes)
      │
8. Bootstrap stability selection (50 bootstrap resamples, keep features
   selected in ≥25% of resamples → robust, non-noise features)

```

## Results

| Stage                              | Features |
|-------------------------------------|:--------:|
| Raw fingerprint + descriptor matrix | 1,032    |
| After correlation pruning           | 1,030    |
| After CV-stable variance/MI/LightGBM filter | 495 |
| After LASSO shrinkage               | 313      | 
| After SHAP refinement (top-150)     | 150      | 
| After bootstrap stability selection | **73**   | 

**Scaffold-split evaluation** (held-out unseen chemotypes, not a random split): ROC-AUC 0.96, PR-AUC 0.959

**Key finding:** going from 150 SHAP-selected features down to 73 stability-selected features costs essentially nothing (ΔAUC = -0.0002 — i.e. the smaller set is marginally *better*), and stability frequency correlates only moderately with SHAP importance (r = 0.435). That gap is the interesting result: SHAP tells you what mattered on one fit; bootstrap stability tells you what reliably matters across resamples. They agree partially, not fully — worth knowing before treating either ranking as ground truth.


## Project structure

```
.
├── README.md
├── requirements.txt
├── notebooks/
│   └── molecular_feature_selection.ipynb   # full annotated pipeline (bug-fixed)
├── src/
│   ├── data_loading.py            # stage 1: load, clean, label
│   ├── fingerprints.py            # stage 2: Morgan fingerprints + descriptors
│   ├── correlation_pruning.py     # stage 3
│   ├── cv_feature_selection.py    # stage 4: leakage-free CV selection
│   ├── lasso_selection.py         # stage 5
│   ├── shap_selection.py          # stage 6
│   ├── scaffold_split.py          # stage 7: Murcko scaffold split + eval
│   ├── stability_selection.py     # stage 8: bootstrap stability selection
│   └── pipeline.py                # runs all stages end-to-end
└── results/
    └── (CSV outputs, stability plots land here)
```

## Setup

```bash
pip install -r requirements.txt
```

You'll need `molecule_classification_dataset.csv` (SMILES + `Protein` + descriptor columns) in your working directory — this repo doesn't redistribute the dataset itself.

## Usage

```python
from src.pipeline import run_pipeline

results = run_pipeline("molecule_classification_dataset.csv")
print(results["stable_model_auc"], results["scaffold_split_auc"])
```

Or run stage-by-stage (useful for debugging / inspecting intermediate shapes — this is why the original notebook was split into one cell per stage):

```python
from src.data_loading import load_and_clean
from src.fingerprints import build_feature_matrix
from src.cv_feature_selection import select_stable_features
# ...
```


## Tech stack

RDKit (SMILES parsing, Morgan fingerprints, Murcko scaffolds) · LightGBM · scikit-learn (LASSO, mutual information, stratified CV) · SHAP · pandas / numpy · matplotlib / seaborn

## Future work

- [ ] Replace the fixed 0.9-correlation and 0.25-stability thresholds with a small validation sweep
- [ ] Extend Enrichment Factor evaluation to a properly imbalanced screening scenario (down-sample actives to <5%) to make the metric meaningful on this dataset
- [ ] Compare LightGBM importance-based selection against permutation importance as a cross-check
- [ ] Package the pipeline as a scikit-learn-compatible `Pipeline`/`Transformer` for reuse on other targets

## Author

Aditi — M.Sc.-M.Tech. Data and Computational Science, IIT Jodhpur
