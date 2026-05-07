"""FSL evaluation: centroid + 1NN classification + weighted metric.

The Siamese model exposes `.test_all(X, y)` returning {'centroid', 'nn'} test
accuracies via shared anchor embeddings. This module wraps that for use across
multiple eval datasets and adds the paper's Weighted Metric (Section 4.6 (Model Evaluation Metric)):

    WM = 0.5 * (a_0 + (1 / (s(s+1)/2)) * sum_{i=1..s} (s - i + 1) * a_i)

where a_i is the test accuracy at attack severity X = i (a_0 = benign-side
accuracy) and s is the maximum mantissa severity (23).
"""

from __future__ import annotations

from typing import Dict, Iterable, Literal, Mapping, Tuple

import numpy as np


S_MANTISSA = 23


def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    k: int = 1,
    metric: Literal["euclidean", "cosine", "cityblock"] = "euclidean",
) -> Dict[str, float]:
    """One-dataset wrapper around Siamese.test_all -> {'centroid', 'nn'}."""
    return model.test_all(
        X_test, y_test, is_print=False, k=k, metric=metric, return_acc=True,
    )


def evaluate_on_datasets(
    model,
    datasets: Mapping[str, Tuple[np.ndarray, np.ndarray]],
    *,
    k: int = 1,
    metric: Literal["euclidean", "cosine", "cityblock"] = "euclidean",
) -> Dict[str, Dict[str, float]]:
    """Evaluate one Siamese model on every (X, y) in `datasets`.

    Returns {dataset_name: {'centroid': acc, 'nn': acc}}.
    """
    out: Dict[str, Dict[str, float]] = {}
    for name, (X, y) in datasets.items():
        out[name] = evaluate_model(model, X, y, k=k, metric=metric)
    return out


def weighted_metric(per_x_accuracies: Mapping[int, float], s: int = S_MANTISSA) -> float:
    """Paper's Weighted Metric (Section 4.6 (Model Evaluation Metric)).

    `per_x_accuracies` must contain at minimum keys 0..s; missing entries
    default to 0.0 (treated as a complete miss for the missing severity).
    """
    a_0 = float(per_x_accuracies.get(0, 0.0))
    denom = s * (s + 1) // 2
    weighted_sum = sum((s - i + 1) * float(per_x_accuracies.get(i, 0.0)) for i in range(1, s + 1))
    return 0.5 * (a_0 + weighted_sum / denom)


def per_x_accuracy_from_split(
    benign_scores_to_label: Iterable[Tuple[float, int]],
    threshold: float,
) -> float:
    """Threshold-based accuracy over (score, label) pairs (label 0 = benign, 1 = mal).

    Used by threshold-calibrated baselines (B4-B7) when computing per-X test accuracy.
    """
    pairs = list(benign_scores_to_label)
    if not pairs:
        return 0.0
    correct = sum(1 for s, l in pairs if (s > threshold) == (l == 1))
    return correct / len(pairs)
