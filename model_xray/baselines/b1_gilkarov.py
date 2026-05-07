"""B1: Gilkarov et al. (2023) reproduction.

Trains a supervised classifier directly on the flattened float32 weight vector
of each model. The original Gilkarov code (github.com/ArielCyber/AI_Model_Steganalysis,
classification.py) uses sklearn's `HistGradientBoostingClassifier` and
`RandomForestClassifier` for the weights-feature path. The paper's reproduction
swaps in XGBoost (Section 4.1 (Baseline)) so the three XGBoost-based methods
(B1, B2, GF + XGBoost) share an identical classifier configuration.

`fit_predict` defaults to XGBoost (matches the paper's claim) but accepts
`classifier="hgb"` or `"rf"` to reproduce the original ArielCyber protocol.

Stratified 80/20 train/test split with random_state=0 mirrors classification.py.
"""

from __future__ import annotations

from typing import Literal, Tuple

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import train_test_split

from model_xray.baselines.shared import make_xgb


ClassifierName = Literal["xgboost", "hgb", "rf"]
_CTORS = {
    "xgboost": lambda **kw: make_xgb(**kw),
    "hgb":     lambda **_: HistGradientBoostingClassifier(),
    "rf":      lambda **_: RandomForestClassifier(),
}


def fit_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    *,
    classifier: ClassifierName = "xgboost",
    xgb_kwargs: dict | None = None,
):
    """Fit one classifier on flattened float32 weights, return (clf, y_pred)."""
    assert X_train.ndim == 2 and X_test.ndim == 2
    if classifier not in _CTORS:
        raise ValueError(f"unknown classifier {classifier!r}; pick from {list(_CTORS)}")
    clf = _CTORS[classifier](**(xgb_kwargs or {}))
    clf.fit(X_train, y_train)
    return clf, clf.predict(X_test)


def split_and_score(
    X: np.ndarray,
    y: np.ndarray,
    *,
    classifier: ClassifierName = "xgboost",
    test_size: float = 0.2,
    random_state: int = 0,
    xgb_kwargs: dict | None = None,
) -> dict:
    """80/20 stratified split + fit + score; mirrors classification.py:generate_result.

    Returns a dict with keys: classifier, accuracy, recall, precision, f1.
    """
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    _, y_pred = fit_predict(Xtr, ytr, Xte, classifier=classifier, xgb_kwargs=xgb_kwargs)
    return {
        "classifier": classifier,
        "accuracy":  float(accuracy_score(yte, y_pred)),
        "recall":    float(recall_score(yte, y_pred, zero_division=0.0)),
        "precision": float(precision_score(yte, y_pred, zero_division=0.0)),
        "f1":        float(f1_score(yte, y_pred, zero_division=0.0)),
    }
