"""Shared XGBoost classifier configuration (Section 4.1 (Baseline)).

All three XGBoost-based methods in the paper - B1 (Gilkarov), B2 (Yin), and the
"Ours - GF + XGBoost" ablation in Experiment 4 - use the same classifier:
`tree_method="hist"`, `device="cuda"`, all other parameters at xgboost defaults
(100 boosted trees, max_depth=6, lr=0.3, full subsampling, lambda=1, binary
logistic objective). The three methods differ only in the input feature vector.

Set `MODELXRAY_XGB_DEVICE=cpu` to fall back to CPU (useful when the GPU is
already saturated by TF/torch contexts, or for CI without a GPU).
"""

from __future__ import annotations

import os
from typing import Any

try:
    import xgboost
except ImportError as e:  # pragma: no cover
    xgboost = None
    _import_err: Exception | None = e
else:
    _import_err = None


def make_xgb(**overrides: Any):
    """Construct an xgboost.XGBClassifier with the paper's shared config."""
    if xgboost is None:
        raise ImportError(
            "xgboost is required for B1, B2, and the GF + XGBoost ablation. "
            f"pip install xgboost. Original error: {_import_err!r}"
        )
    params = dict(tree_method="hist", device=os.environ.get("MODELXRAY_XGB_DEVICE", "cuda"))
    params.update(overrides)
    return xgboost.XGBClassifier(**params)
