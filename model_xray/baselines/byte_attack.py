"""Byte-space X-LSB-Attack-Fill and IEEE-754 conversions.

The canonical X-LSB-Fill implementation in `model_xray.procedures.embedding_procs`
operates on weight tensors via the full ZenML-era pipeline. The threshold detectors
(B4-B7) and the per-experiment runners benefit from a tighter, NumPy-only path that
goes flat-float32 -> (n, 4) uint8 big-endian [MSB..LSB] and back, matching the byte
layout used by `model_xray.utils.general_utils.ndarray_to_bytes_arr`.

The two implementations are equivalent on float32 inputs and produce identical
attacked weights for any (X, payload).
"""

from __future__ import annotations

import math

import numpy as np


def float32_to_bytes(weights: np.ndarray) -> np.ndarray:
    """Flat float32 -> (n, 4) uint8 big-endian [MSB..LSB]."""
    assert weights.dtype == np.float32, f"expected float32, got {weights.dtype}"
    assert weights.ndim == 1, f"expected 1-D, got {weights.shape}"
    raw = np.frombuffer(weights.tobytes(order="C"), dtype=np.uint8)
    return np.ascontiguousarray(raw.reshape(-1, 4)[:, ::-1])


def bytes_to_float32(bytes_arr: np.ndarray) -> np.ndarray:
    """(n, 4) uint8 big-endian [MSB..LSB] -> flat float32."""
    assert bytes_arr.dtype == np.uint8
    assert bytes_arr.ndim == 2 and bytes_arr.shape[1] == 4
    little = np.ascontiguousarray(bytes_arr[:, ::-1])
    return np.frombuffer(little.tobytes(), dtype=np.float32).copy()


def load_malware_bits(payload_path: str, n_bits: int) -> np.ndarray:
    """Read payload, unpack to bits big-endian-within-byte, tile/truncate to n_bits."""
    with open(payload_path, "rb") as f:
        data = f.read()
    payload = np.frombuffer(data, dtype=np.uint8)
    all_bits = np.unpackbits(payload, bitorder="big")
    if len(all_bits) == 0:
        raise ValueError(f"payload file is empty: {payload_path}")
    if len(all_bits) >= n_bits:
        return all_bits[:n_bits].copy()
    reps = math.ceil(n_bits / len(all_bits))
    return np.tile(all_bits, reps)[:n_bits].copy()


def xlsb_attack_fill_bytes(
    bytes_arr: np.ndarray,
    x: int,
    malware_bits: np.ndarray,
    chunk_weights: int = 2_000_000,
) -> np.ndarray:
    """Apply X-LSB-Attack-Fill on (n, 4) uint8 big-endian.

    Overwrites the X lowest mantissa bits of every weight with malware_bits.
    Returns a new attacked array; chunked to bound memory.
    """
    assert bytes_arr.dtype == np.uint8
    assert bytes_arr.ndim == 2 and bytes_arr.shape[1] == 4
    assert 1 <= x <= 23, f"X must be in [1, 23] (mantissa bits), got {x}"
    n = bytes_arr.shape[0]
    assert malware_bits.shape == (n * x,), (
        f"malware_bits shape {malware_bits.shape} != expected ({n * x},)"
    )
    assert malware_bits.dtype == np.uint8

    out = bytes_arr.copy()
    for i in range(0, n, chunk_weights):
        j = min(i + chunk_weights, n)
        # Mantissa lives across bytes 1 (lower 7 bits), 2, 3. Unpack bytes 1..3
        # into a 24-bit big-endian sequence per weight; the LAST X bits are the X LSBs.
        chunk = out[i:j, 1:4]
        bits = np.unpackbits(chunk, axis=-1, bitorder="big")
        bits[:, -x:] = malware_bits[i * x : j * x].reshape(j - i, x)
        out[i:j, 1:4] = np.packbits(bits, axis=-1, bitorder="big")
    return out


def attacked_weights(
    benign_weights: np.ndarray,
    x: int,
    malware_bits_or_path,
    chunk_weights: int = 2_000_000,
    *,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Convenience: float32 benign -> float32 attacked at severity X.

    `malware_bits_or_path` may be:
      - a path-like (str/bytes) to a payload file (lazy-loaded),
      - a preloaded bit array of shape (n*x,),
      - or None to use a uniform pseudo-random payload (Experiment 4 setting).
    """
    n = benign_weights.shape[0]
    needed_bits = n * x
    if isinstance(malware_bits_or_path, (str, bytes)):
        mbits = load_malware_bits(str(malware_bits_or_path), needed_bits)
    elif malware_bits_or_path is None:
        rng = rng if rng is not None else np.random.default_rng()
        mbits = rng.integers(0, 2, size=needed_bits, dtype=np.uint8)
    else:
        mbits = malware_bits_or_path
        assert mbits.shape == (needed_bits,), f"expected ({needed_bits},), got {mbits.shape}"

    b_benign = float32_to_bytes(benign_weights)
    b_attacked = xlsb_attack_fill_bytes(b_benign, x, mbits, chunk_weights=chunk_weights)
    return bytes_to_float32(b_attacked)
