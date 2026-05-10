"""Regenerate exp2_id_al.png — Experiment 2 AL view (paper Figure 6).

The AL plot shows the Weighted Metric (Section 4.6 (Model Evaluation Metric)) per anchor
X_hat:
    WM(X_hat) = 0.5 * (a_0 + sum_{i=1..23} (24 - i) * a_i / 276)

For FSL we read the per-(run, train_X) test_acc CSVs and assume a_0 = 1.0 (the
CSVs do not record benign-test rows; matches the previously published PNG).

For B4-B7 we use the cached per-(arch, baseline) feature .npz to recompute
threshold-calibrated WM curves with proper benign-test accuracy.
"""

from __future__ import annotations

import argparse
import os
from typing import Dict

import numpy as np
import pandas as pd

from model_xray.data.pretrained_models import FAMOUS_SMALL_ALL
from model_xray.plots._style import (
    FSL_STYLES_EXP2, MALCONV_STYLE, NAIVE_STYLES,
    add_ci, default_out_dir, finalize, init_axes, repo_root, save_fig,
)


RESULTS_DIR = os.path.join(repo_root(), "results", "exp2")
SMALL_CACHE = os.path.join(RESULTS_DIR, "threshold_features_small.npz")

S_MANTISSA = 23
WM_DENOM = S_MANTISSA * (S_MANTISSA + 1) // 2  # 276
N_SEEDS = 30
X_RANGE = list(range(1, S_MANTISSA + 1))
BASELINE_NAMES = ["byte_entropy", "histogram_kl", "weight_value_dist", "byte_autocorr"]
SMALL_ARCHS = list(FAMOUS_SMALL_ALL)


def _wm(a0: float, a_x: np.ndarray) -> float:
    weights = np.arange(S_MANTISSA, 0, -1)
    return 0.5 * (a0 + float((weights * a_x).sum()) / WM_DENOM)


def _l2(f, mu):
    return np.sqrt(((f - mu) ** 2).sum(axis=-1))


def _weighted_l2(f, mu, sigma):
    return np.sqrt((((f - mu) / sigma) ** 2).sum(axis=-1))


def _kl(p, q):
    p = np.clip(p, 1e-12, None); q = np.clip(q, 1e-12, None)
    p = p / p.sum(axis=-1, keepdims=True)
    return (p * (np.log(p) - np.log(q))).sum(axis=-1)


def _score(name: str, features: np.ndarray, ref: np.ndarray) -> np.ndarray:
    if name in ("byte_entropy", "byte_autocorr"):
        return _l2(features, ref.mean(axis=0))
    if name == "weight_value_dist":
        mu, sigma = ref.mean(axis=0), ref.std(axis=0) + 1e-9
        return _weighted_l2(features, mu, sigma)
    if name == "histogram_kl":
        avg = ref.mean(axis=0)
        q = avg / avg.sum()
        p = features / features.sum(axis=-1, keepdims=True)
        return _kl(p, q[None, :])
    raise ValueError(name)


def _find_threshold(b: np.ndarray, m: np.ndarray) -> float:
    s = np.concatenate([b, m]); l = np.concatenate([np.zeros_like(b), np.ones_like(m)])
    cands = np.unique(s)
    cands = np.concatenate([[cands[0] - 1.0], 0.5 * (cands[:-1] + cands[1:]), [cands[-1] + 1.0]])
    accs = np.array([float(np.mean((s > t) == l)) for t in cands])
    return float(cands[int(np.argmax(accs))])


def _load_cache(path: str) -> Dict[str, Dict[str, np.ndarray]]:
    if not os.path.exists(path):
        return {}
    z = np.load(path)
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for key in z.files:
        arch, baseline = key.split("__")
        out.setdefault(arch, {})[baseline] = z[key]
    return out


def _naive_al_curve(dataset_key: str) -> pd.DataFrame:
    cache = _load_cache(SMALL_CACHE)
    if not cache:
        return pd.DataFrame()
    n_small = len(SMALL_ARCHS)
    rows = []
    for seed in range(N_SEEDS):
        train_idx = np.sort(np.random.default_rng(seed).choice(n_small, 3, replace=False))
        test_idx = np.array([i for i in range(n_small) if i not in train_idx])
        for b in BASELINE_NAMES:
            per_arch = np.stack([cache[a][b] for a in SMALL_ARCHS])  # (n, 24, F)
            ref = per_arch[train_idx, 0, :]
            small_scores = np.stack([_score(b, per_arch[:, x, :], ref) for x in range(24)])  # (24, n)
            for x_hat in X_RANGE:
                t = _find_threshold(small_scores[0, train_idx], small_scores[x_hat, train_idx])
                a0 = float(np.mean(small_scores[0, test_idx] <= t))
                a_x = np.array([float(np.mean(small_scores[x, test_idx] > t)) for x in X_RANGE])
                rows.append({"seed": seed, "baseline": b, "X_hat": x_hat, "WM": _wm(a0, a_x)})
    df = (pd.DataFrame(rows).groupby(["baseline", "X_hat"])["WM"]
          .agg(mean="mean", std="std", n_seeds="count").reset_index())
    return add_ci(df)


def _fsl_al_curve(csv_path: str, eval_set: str, eval_col: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "eval_set" in df.columns:
        df = df[df["eval_set"] == eval_set]
    weights = np.arange(S_MANTISSA, 0, -1)
    rows = []
    for run in df["repeat"].unique():
        sub = df[df["repeat"] == run]
        for train_x in sorted(sub["X_hat"].unique()):
            if train_x < 1 or train_x > 23:
                continue
            block = sub[sub["X_hat"] == train_x].set_index("X")
            a0 = float(block.loc[0, eval_col]) if 0 in block.index else 1.0
            a_x = np.array([float(block.loc[x, eval_col]) if x in block.index else 0.0 for x in X_RANGE])
            rows.append({"run": run, "X_hat": int(train_x),
                         "WM": 0.5 * (a0 + (weights * a_x).sum() / WM_DENOM)})
    out = (pd.DataFrame(rows).groupby("X_hat")["WM"]
           .agg(mean="mean", std="std", n_seeds="count").reset_index())
    return add_ci(out)


def _malconv_al_curve(suffix: str) -> pd.DataFrame:
    p = os.path.join(RESULTS_DIR, f"b3_malconv_crossx_{suffix}.csv")
    if not os.path.exists(p):
        return pd.DataFrame()
    df = pd.read_csv(p)
    weights = np.arange(S_MANTISSA, 0, -1)
    rows = []
    for (seed, x_hat), grp in df.groupby(["repeat", "X_hat"]):
        block = grp.set_index("X")
        a0 = float(block.loc[0, "accuracy"]) if 0 in block.index else 1.0
        a_x = np.array([float(block.loc[x, "accuracy"]) if x in block.index else 0.0 for x in X_RANGE])
        rows.append({"seed": seed, "X_hat": int(x_hat),
                     "WM": 0.5 * (a0 + (weights * a_x).sum() / WM_DENOM)})
    out = (pd.DataFrame(rows).groupby("X_hat")["WM"]
           .agg(mean="mean", std="std", n_seeds="count").reset_index())
    return add_ci(out)


def _plot_curve(ax, x, c: pd.DataFrame, style, *, lw, alpha, ms=0):
    plot_kwargs = {k: v for k, v in style.items() if k != "marker"}
    ax.plot(c[x], c["mean"], linewidth=lw, markersize=ms,
            marker=style.get("marker", None), **plot_kwargs)
    if c["half_ci"].sum() > 0:
        lo = (c["mean"] - c["half_ci"]).clip(lower=0, upper=1)
        hi = (c["mean"] + c["half_ci"]).clip(lower=0, upper=1)
        ax.fill_between(c[x], lo, hi, color=style["color"], alpha=alpha, linewidth=0)


def plot_one(dataset_key: str, dataset_label: str, suffix: str, out_path: str):
    fig, ax = init_axes()

    for arch_label, arch_suffix in (("OSL CNN", "osl"), ("SRNet", "srnet")):
        path = os.path.join(RESULTS_DIR, f"fsl_{arch_suffix}_{suffix}.csv")
        if not os.path.exists(path):
            print(f"  skip missing FSL {path}")
            continue
        for eval_label, col in (("Centroid", "centroid"), ("1NN", "nn")):
            curve = _fsl_al_curve(path, eval_set=dataset_key, eval_col=col)
            _plot_curve(ax, "X_hat", curve, FSL_STYLES_EXP2[f"{arch_label} ({eval_label})"], lw=2.0, alpha=0.10)

    mc = _malconv_al_curve(suffix)
    if not mc.empty:
        mc = mc.sort_values("X_hat")
        _plot_curve(ax, "X_hat", mc, MALCONV_STYLE, lw=1.3, alpha=0.12, ms=4)

    nv = _naive_al_curve(dataset_key)
    for baseline_name, style in NAIVE_STYLES.items():
        sub = nv[nv["baseline"] == baseline_name].sort_values("X_hat")
        if sub.empty:
            continue
        _plot_curve(ax, "X_hat", sub, style, lw=1.3, alpha=0.12, ms=4)

    finalize(ax, title=f"Model Collection = {dataset_label}", ylim=(0.45, 1.02), ylabel="Metric")
    save_fig(fig, out_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--id-out", default=os.path.join(default_out_dir(), "exp2_id_al.png"))
    args = parser.parse_args()

    plot_one("famous_le_10m", "Famous Small CNNs", "small", args.id_out)


if __name__ == "__main__":
    main()
