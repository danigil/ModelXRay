"""B1: Gilkarov et al. (2023) reproduction.

Trains an XGBoost classifier (shared config in Section sec:xgb_config) directly
on the flattened float32 weight vector of each model. This is the reproduction
target of the prior work; it is the primary historical comparison point in
Experiment 1 (SCZ STL10 zoo, 40k train / 10k test).
"""

from __future__ import annotations

import numpy as np

from model_xray.baselines.shared import make_xgb


def fit_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    *,
    xgb_kwargs: dict | None = None,
):
    """Fit an XGBoost on flattened float32 weights and return (model, y_pred).

    X_*: (N, n_weights) float32. y_train: (N,) {0, 1}.
    """
    assert X_train.ndim == 2 and X_test.ndim == 2
    clf = make_xgb(**(xgb_kwargs or {}))
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return clf, y_pred
