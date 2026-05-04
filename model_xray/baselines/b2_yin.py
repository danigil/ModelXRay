"""B2: Yin et al. (2022) NIST randomness-test feature extraction.

Implements the four bit-plane statistics (phi_1..phi_4) that Yin et al. designed
for LSB neural-weight steganalysis: frequency (monobit), serial, approximate
entropy, cumulative sums. The paper extends Yin's original design by computing
all four statistics on each of the 23 mantissa bit positions and concatenating
the results into a 92-dimensional feature vector (`calc_phis_all`), then training
an XGBoost classifier on top (see `model_xray.baselines.shared.make_xgb`).

Yin et al. did not publish their code; this is our reproduction.
"""

from __future__ import annotations

import numpy as np

from model_xray.utils.general_utils import ndarray_to_bytes_arr


def calc_frequency_monobit_2d(arr_bits: np.ndarray) -> np.ndarray:
    assert arr_bits.ndim == 2, f"expected 2D, got {arr_bits.ndim}D"
    n_weights = arr_bits.shape[1]
    n_ones = np.sum(arr_bits, axis=1, keepdims=True)
    n_zeros = n_weights - n_ones
    return np.abs(n_ones - n_zeros) / np.sqrt(n_weights)


def get_bits_arr(arr: np.ndarray) -> np.ndarray:
    assert arr.ndim == 2, f"expected 2D, got {arr.ndim}D"
    n, m = arr.shape
    bitamount = arr.itemsize * 8
    arr_bytes = ndarray_to_bytes_arr(arr)
    return np.unpackbits(arr_bytes, bitorder="big").reshape(n, m, bitamount)


def get_bits_arr_2d(arr: np.ndarray, left_idx: int = None, right_idx: int = None) -> np.ndarray:
    arr_bits = get_bits_arr(arr)
    if left_idx is None or right_idx is None:
        return arr_bits
    arr_bits = arr_bits[..., left_idx:right_idx]
    return arr_bits.reshape(arr.shape[0], -1)


def calc_phi1(arr: np.ndarray, left_idx: int, right_idx: int, arr_bits=None) -> np.ndarray:
    assert left_idx < right_idx
    if arr_bits is None:
        arr_bits = get_bits_arr_2d(arr, left_idx, right_idx)
    return calc_frequency_monobit_2d(arr_bits)


def _compute_psi_single(seq: np.ndarray, m: int) -> float:
    """psi_m = (2^m / n) * sum(frequency^2) - n over overlapping m-bit patterns."""
    n = seq.size
    if n < m:
        raise ValueError("Sequence length must be at least m")
    seq = np.concatenate((seq, seq[: m - 1]))
    n_new = seq.size
    shape = (n_new - m + 1, m)
    strides = (seq.strides[0], seq.strides[0])
    windows = np.lib.stride_tricks.as_strided(seq, shape=shape, strides=strides)
    powers = 2 ** np.arange(m - 1, -1, -1)
    pattern_ints = windows.dot(powers)
    counts = np.bincount(pattern_ints, minlength=2 ** m)
    sum_sq = np.sum(counts.astype(np.float64) ** 2)
    return ((2 ** m / n) * sum_sq) - n


def _calc_phi2_single(seq: np.ndarray, p: int = 8) -> float:
    """Phi_2 = psi_p - psi_{p-1} over a binary sequence."""
    seq = np.asarray(seq).flatten()
    if seq.size < p:
        raise ValueError("Sequence length must be at least p")
    return _compute_psi_single(seq, p) - _compute_psi_single(seq, p - 1)


def calc_phi2(arr: np.ndarray, left_idx: int, right_idx: int, p: int = 8, arr_bits=None) -> np.ndarray:
    assert arr.ndim == 2
    if arr_bits is None:
        arr_bits = get_bits_arr_2d(arr, left_idx, right_idx)
    f = lambda x: _calc_phi2_single(x, p)
    return np.apply_along_axis(f, 1, arr_bits).reshape(-1, 1)


def _compute_entropy_single(seq: np.ndarray, m: int) -> float:
    n = seq.size
    if n < m:
        raise ValueError("Sequence length must be at least m")
    seq = np.concatenate((seq, seq[: m - 1]))
    n_new = seq.size
    shape = (n_new - m + 1, m)
    strides = (seq.strides[0], seq.strides[0])
    windows = np.lib.stride_tricks.as_strided(seq, shape=shape, strides=strides)
    powers = 2 ** np.arange(m - 1, -1, -1)
    pattern_ints = windows.dot(powers)
    counts = np.bincount(pattern_ints, minlength=2 ** m)
    p = counts / n
    upper = 2 * m - 1
    p = p[:upper]
    return float(np.dot(p, np.log2(p + 1e-10)))


def _compute_apen(seq: np.ndarray, m: int) -> float:
    return _compute_entropy_single(seq, m) - _compute_entropy_single(seq, m + 1)


def _calc_phi3_single(seq: np.ndarray, p: int = 8) -> float:
    """Phi_3 = 2n (log2(2) - apen_p)."""
    seq = np.asarray(seq).flatten()
    n = seq.size
    if n < p:
        raise ValueError("Sequence length must be at least p")
    apen = _compute_apen(seq, p)
    return 2 * n * (np.log2(2) - apen)


def calc_phi3(arr: np.ndarray, left_idx: int, right_idx: int, p: int = 8, arr_bits=None) -> np.ndarray:
    assert arr.ndim == 2
    if arr_bits is None:
        arr_bits = get_bits_arr_2d(arr, left_idx, right_idx)
    f = lambda x: _calc_phi3_single(x, p)
    return np.apply_along_axis(f, 1, arr_bits).reshape(-1, 1)


def calc_phi4(arr: np.ndarray, left_idx: int, right_idx: int, arr_bits=None) -> np.ndarray:
    assert left_idx < right_idx
    if arr_bits is None:
        arr_bits = get_bits_arr_2d(arr, left_idx, right_idx)
    cumsum_1s = np.cumsum(arr_bits, axis=1)
    cumsum_0s = np.cumsum(1 - arr_bits, axis=1)
    s_i = np.abs(cumsum_1s - cumsum_0s)
    return np.max(s_i, axis=1, keepdims=True)


def calc_phis(arr: np.ndarray, left_idx: int, right_idx: int, p: int = 8, arr_bits=None) -> np.ndarray:
    """Stack phi_1..phi_4 into an (n_models, 4) feature matrix for ONE bit window."""
    assert arr.ndim == 2
    phi1 = calc_phi1(arr, left_idx, right_idx, arr_bits=arr_bits)
    phi2 = calc_phi2(arr, left_idx, right_idx, p=p, arr_bits=arr_bits)
    phi3 = calc_phi3(arr, left_idx, right_idx, p=p, arr_bits=arr_bits)
    phi4 = calc_phi4(arr, left_idx, right_idx, arr_bits=arr_bits)
    return np.hstack((phi1, phi2, phi3, phi4))


def calc_phis_all(arr: np.ndarray, p: int = 8) -> np.ndarray:
    """Concat phi_1..phi_4 across all 23 mantissa bit positions -> (n_models, 92).

    This is the 92-dim feature vector used by B2 in the paper.
    """
    assert arr.ndim == 2
    arr_bits_all = get_bits_arr_2d(arr)
    out = []
    for x_curr in range(1, 24):
        x_curr = 32 - x_curr
        bits = arr_bits_all[..., x_curr : x_curr + 1].reshape(arr.shape[0], -1)
        out.append(calc_phis(arr, left_idx=x_curr, right_idx=x_curr + 1, p=p, arr_bits=bits))
    return np.concatenate(out, axis=1)
