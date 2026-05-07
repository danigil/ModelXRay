"""B4-B7: threshold-based steganalysis detectors (Section 4.1 (Baseline)).

Each detector:
  - operates on a flat float32 weight array,
  - fits a reference statistic from a list of benign weight arrays (`fit`),
  - produces a scalar anomaly score (`score`),
  - uses the shared `find_threshold` grid-search to pick a calibrated threshold
    on a (benign_scores, malicious_scores) split.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence, Tuple

import numpy as np

from model_xray.baselines.byte_attack import float32_to_bytes


class BaseDetector(ABC):
    name: str = "base"

    @abstractmethod
    def fit(self, benign_weights_list: Sequence[np.ndarray]) -> None: ...

    @abstractmethod
    def score(self, weights: np.ndarray) -> float: ...

    def classify(self, weights: np.ndarray, threshold: float) -> int:
        return 1 if self.score(weights) > threshold else 0

    def find_threshold(
        self,
        benign_scores: Sequence[float],
        malicious_scores: Sequence[float],
    ) -> Tuple[float, float]:
        """Grid-search the threshold maximizing training accuracy.

        Candidates are midpoints between sorted unique scores plus sentinels
        below all and above all scores.
        """
        all_scores = list(benign_scores) + list(malicious_scores)
        all_labels = [0] * len(benign_scores) + [1] * len(malicious_scores)
        sorted_unique = sorted(set(all_scores))
        candidates = [sorted_unique[0] - 1.0]
        for a, b in zip(sorted_unique[:-1], sorted_unique[1:]):
            candidates.append(0.5 * (a + b))
        candidates.append(sorted_unique[-1] + 1.0)

        best_t, best_acc = candidates[0], -1.0
        for t in candidates:
            preds = [1 if s > t else 0 for s in all_scores]
            acc = sum(p == l for p, l in zip(preds, all_labels)) / len(all_labels)
            if acc > best_acc:
                best_acc = acc
                best_t = t
        return best_t, best_acc


def _shannon_entropy(counts: np.ndarray) -> float:
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts.astype(np.float64) / float(total)
    nz = p[p > 0]
    return float(-np.sum(nz * np.log2(nz)))


class ByteEntropyDetector(BaseDetector):
    """B5: Shannon entropy of each of the 4 byte positions."""

    name = "byte_entropy"

    def __init__(self):
        self.mu: np.ndarray | None = None

    def _features(self, weights: np.ndarray) -> np.ndarray:
        b = float32_to_bytes(weights)
        out = np.empty(4, dtype=np.float64)
        for j in range(4):
            counts = np.bincount(b[:, j], minlength=256)
            out[j] = _shannon_entropy(counts)
        return out

    def fit(self, benign_weights_list):
        feats = np.stack([self._features(w) for w in benign_weights_list])
        self.mu = feats.mean(axis=0)

    def score(self, weights):
        assert self.mu is not None, "call fit() first"
        f = self._features(weights)
        return float(np.linalg.norm(f - self.mu))


class HistogramKLDetector(BaseDetector):
    """B6: KL(P_test || P_ref) on byte-j histogram with Laplace smoothing.

    Defaults to byte position 3 (LSB, most affected by low-X attacks).
    """

    name = "histogram_kl"

    def __init__(self, byte_positions: Tuple[int, ...] = (3,)):
        self.byte_positions = tuple(byte_positions)
        self.refs: dict[int, np.ndarray] | None = None

    def _smoothed_hist(self, weights: np.ndarray, j: int) -> np.ndarray:
        b = float32_to_bytes(weights)
        counts = np.bincount(b[:, j], minlength=256).astype(np.float64)
        counts += 1.0  # Laplace smoothing
        return counts / counts.sum()

    def fit(self, benign_weights_list):
        self.refs = {}
        for j in self.byte_positions:
            hs = np.stack([self._smoothed_hist(w, j) for w in benign_weights_list])
            m = hs.mean(axis=0)
            self.refs[j] = m / m.sum()

    def score(self, weights):
        assert self.refs is not None, "call fit() first"
        total = 0.0
        for j in self.byte_positions:
            p = self._smoothed_hist(weights, j)
            q = self.refs[j]
            total += float(np.sum(p * (np.log(p) - np.log(q))))
        return total


class WeightValueDistributionDetector(BaseDetector):
    """B7: summary stats over float weight VALUES.

    Features: mean, std, skew, kurtosis, min, max, near-zero ratio,
    plus a 100-bin histogram over [-2, 2] with overflow clipping. 107-dim total.
    """

    name = "weight_value_dist"

    _BIN_EDGES = np.linspace(-2.0, 2.0, 101)
    _N_BINS = 100
    _SUMMARY_LEN = 7

    def __init__(self):
        self.mu: np.ndarray | None = None
        self.sigma: np.ndarray | None = None

    def _features(self, weights: np.ndarray) -> np.ndarray:
        w = weights.astype(np.float64, copy=False)
        # Guard against NaN/Inf introduced by attacks that happen to set exponent=0xFF, etc.
        # Fill-attack only touches mantissa, so this shouldn't happen for real models, but be safe.
        finite = np.isfinite(w)
        if not finite.all():
            w = w[finite]
        mean = np.mean(w) if w.size else 0.0
        std = np.std(w) if w.size else 0.0
        if std > 0:
            z = (w - mean) / std
            skew = float(np.mean(z ** 3))
            kurt = float(np.mean(z ** 4) - 3.0)
        else:
            skew = 0.0
            kurt = 0.0
        min_v = float(np.min(w)) if w.size else 0.0
        max_v = float(np.max(w)) if w.size else 0.0
        near_zero = float(np.mean(np.abs(w) < 0.01)) if w.size else 0.0
        clipped = np.clip(w, self._BIN_EDGES[0], self._BIN_EDGES[-1])
        hist, _ = np.histogram(clipped, bins=self._BIN_EDGES)
        hist = hist.astype(np.float64) / max(w.size, 1)
        return np.concatenate([[mean, std, skew, kurt, min_v, max_v, near_zero], hist])

    def fit(self, benign_weights_list):
        feats = np.stack([self._features(w) for w in benign_weights_list])
        self.mu = feats.mean(axis=0)
        self.sigma = feats.std(axis=0) + 1e-9

    def score(self, weights):
        assert self.mu is not None, "call fit() first"
        f = self._features(weights)
        return float(np.linalg.norm((f - self.mu) / self.sigma))


class ByteAutocorrelationDetector(BaseDetector):
    """B4: lag-1 Pearson autocorrelation of byte-j sequence (default byte 3)."""

    name = "byte_autocorr"

    def __init__(self, byte_positions: Tuple[int, ...] = (3,)):
        self.byte_positions = tuple(byte_positions)
        self.mu: np.ndarray | None = None

    def _features(self, weights: np.ndarray) -> np.ndarray:
        b = float32_to_bytes(weights)
        out = np.empty(len(self.byte_positions), dtype=np.float64)
        for i, j in enumerate(self.byte_positions):
            x = b[:, j].astype(np.float64)
            if x.size < 2:
                out[i] = 0.0
                continue
            a = x[:-1]
            c = x[1:]
            ma = a.mean()
            mc = c.mean()
            num = float(np.mean((a - ma) * (c - mc)))
            den = float(np.sqrt(a.var() * c.var())) + 1e-12
            out[i] = num / den
        return out

    def fit(self, benign_weights_list):
        feats = np.stack([self._features(w) for w in benign_weights_list])
        self.mu = feats.mean(axis=0)

    def score(self, weights):
        assert self.mu is not None, "call fit() first"
        f = self._features(weights)
        return float(np.linalg.norm(f - self.mu))


ALL_DETECTORS = [
    ByteEntropyDetector,
    HistogramKLDetector,
    WeightValueDistributionDetector,
    ByteAutocorrelationDetector,
]
