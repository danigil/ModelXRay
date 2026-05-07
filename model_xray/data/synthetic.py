"""Synthetic float32 tensors used by the runtime/memory study (D6).

The deployment-feasibility tables (Table 4, Table 5) measure GF
feature-extraction cost on synthetic random tensors at parameter-count ladder
10**2, 10**3, ..., 10**8. No real model is needed — only the byte/bit
distribution matters, and uniform random float32 drawn from N(0, 1) matches the
benign-weight distribution closely enough for timing/memory measurement.
"""

from __future__ import annotations

import numpy as np


PARAM_LADDER = [10 ** k for k in range(2, 9)]  # 100, 1k, 10k, ..., 100M


def gen_random_weights(n_weights: int, *, seed: int | None = None) -> np.ndarray:
    """Return a (1, n_weights) float32 array of N(0, 1) samples."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal(size=(1, n_weights)).astype(np.float32)
