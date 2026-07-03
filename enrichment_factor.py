"""
enrichment_factor.py — Stage 9: Enrichment Factor (EF), a virtual-screening
utility metric.

EF@X% answers: if we only screened the top X% of molecules ranked by the
model, how many more true actives would we find versus picking randomly?

    EF = (hit rate in top X%) / (baseline hit rate in full set)

IMPORTANT: EF is designed for the classic virtual-screening scenario where
true actives are RARE (e.g. 1-5% of a compound library). If actives are the
majority class (as in this BACE1 dataset, ~85.6% positive), the theoretical
maximum EF is capped near 1/baseline_rate — close to 1.0-1.2x regardless of
model quality. Don't present a near-1.0x EF on an imbalanced-toward-positive
dataset as if it were a strong virtual-screening result; state the ceiling
explicitly.
"""

import numpy as np
import pandas as pd


def enrichment_factor(y_true, y_scores, top_fraction):
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)
    n_total = len(y_true)
    n_top = max(1, int(np.ceil(top_fraction * n_total)))

    order = np.argsort(y_scores)[::-1]  # descending by predicted score
    top_idx = order[:n_top]

    hits_in_top = y_true[top_idx].sum()
    hit_rate_top = hits_in_top / n_top
    baseline_rate = y_true.sum() / n_total
    ef = hit_rate_top / baseline_rate if baseline_rate > 0 else np.nan
    return ef, hit_rate_top, baseline_rate


def enrichment_report(y_test, model_predictions, fractions=(0.01, 0.05, 0.10, 0.20)):
    """model_predictions: dict of {model_label: y_pred_scores}."""
    y_test_arr = y_test.values if hasattr(y_test, "values") else np.asarray(y_test)
    rows = []

    for label, y_scores in model_predictions.items():
        print("=" * 70)
        print(f"ENRICHMENT FACTOR  |  {label}")
        print("=" * 70)
        for frac in fractions:
            ef, hit_rate, baseline = enrichment_factor(y_test_arr, y_scores, frac)
            rows.append({
                "Top %": f"{frac*100:.0f}%", "Model": label,
                "Hit Rate": hit_rate, "Baseline Rate": baseline, "EF": ef,
            })
            print(f"Top {frac*100:>5.1f}%  |  Hit Rate: {hit_rate:.4f}  |  "
                  f"Baseline: {baseline:.4f}  |  EF: {ef:.2f}x")
        print()

    ef_df = pd.DataFrame(rows)

    baseline_rate = y_test_arr.sum() / len(y_test_arr)
    if baseline_rate > 0.3:
        ceiling = 1.0 / baseline_rate
        print(f"NOTE: Baseline positive rate is high (~{baseline_rate*100:.1f}%), "
              f"so EF has a low ceiling (~{ceiling:.2f}x max) here. "
              "EF is most meaningful when actives are rare.")

    return ef_df
