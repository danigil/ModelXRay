"""Experiment 4: ResNet18-TinyImageNet, our methods vs B2/B3/B5/B7 (Figure 8).

Six-way comparison on the homogeneous ResNet18-TinyImageNet zoo (D5) under
matched 5-fold x 5-repeat stratified CV with random-bit payloads:

    Ours - GF + XGBoost   (XGBoost on Grayscale-Fourpart pixels)
    Ours - GF + 1NN       (1-NN on Grayscale-Fourpart pixels)
    B2 - Yin et al.       (XGBoost on 92-d phi features)
    B3 - MalConv-lite     (raw-byte 1D-CNN, 512 KB window)
    B5 - Byte Entropy     (threshold detector, no calibration here per paper)
    B7 - Weight-Value Distribution (threshold detector)

B4 and B6 are intentionally omitted (paper Section 5.3.2: both stuck at 50%
with random-bit payloads on this homogeneous zoo).

Inputs:
    $MODELXRAY_RESNET_MZ_ROOT/tiny-imagenet_resnet18/weights.npy   (from script 05)

Outputs (under `results/exp4/`):
    resnet18_gf_xgboost_per_x.csv
    resnet18_gf_1nn_per_x.csv
    resnet18_b2_yin_per_x.csv
    resnet18_b3_malconv_per_x.csv
    resnet18_b5_b7_threshold_per_x.csv
"""

from __future__ import annotations

import argparse
import os
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neighbors import KNeighborsClassifier

from model_xray.baselines.b2_yin import calc_phis_all
from model_xray.baselines.b3_malconv import TrainConfig as MalConvCfg, train_and_eval as malconv_train
from model_xray.baselines.byte_attack import attacked_weights, float32_to_bytes
from model_xray.baselines.shared import make_xgb
from model_xray.baselines.threshold import ByteEntropyDetector, WeightValueDistributionDetector
from model_xray.data import paths as _paths
# `img_pp_xlsb_attack` is imported lazily inside `_gf_pixels` (the only function
# that uses it). At top-level, it pulls model_xray.procedures.embedding_procs,
# which in turn pulls model_xray.configs.types -> tensorflow.keras + transformers,
# costing ~10s of TF init for runs that don't need any of it (thresholds, B2 Yin).


X_RANGE = list(range(1, 24))
MALCONV_BYTE_WINDOW = 512 * 1024  # 512 KB


def _load_resnet18(mz_name: str = "tiny-imagenet_resnet18") -> np.ndarray:
    path = os.path.join(_paths.get_resnet_mz_root(), mz_name, "weights.npy")
    return np.load(path).astype(np.float32)


def _attack_block(weights: np.ndarray, x: int) -> np.ndarray:
    """Return (n, n_weights) random-payload attacked weights at severity X."""
    return np.stack([attacked_weights(w, x=x, malware_bits_or_path=None) for w in weights])


_GF_FEATURE_CACHE: dict[int, np.ndarray] = {}
_B2_FEATURE_CACHE: dict[int, np.ndarray] = {}


def _b2_yin_features(weights: np.ndarray, x: int) -> np.ndarray:
    """(n_models, 92) Yin et al. NIST phi_1..phi_4 features per model.

    Per-model processing -- avoids the prior `_attack_block` path that built a
    (n_models, n_weights) attacked tensor in one shot (5+ GB on ResNet18,
    the bit-unpacking inside calc_phis_all then doubled it -> OOM-killed).

    Cached per X, like the GF cache, so multiple downstream consumers share
    one feature computation per attack severity.
    """
    if x in _B2_FEATURE_CACHE:
        return _B2_FEATURE_CACHE[x]
    n = weights.shape[0]
    out = np.zeros((n, 92), dtype=np.float64)
    for i in range(n):
        ws_i = weights[i] if x == 0 else attacked_weights(weights[i], x=x, malware_bits_or_path=None)
        out[i] = calc_phis_all(ws_i.reshape(1, -1))[0]
    _B2_FEATURE_CACHE[x] = out
    return out


def _gf_pixels(weights: np.ndarray, x: int, imsize: int = 50) -> np.ndarray:
    """(n, imsize*imsize) flattened GF images at severity X (X=0 = benign).

    Uses PIL Image.BOX resampling (area-averaging) for the 6720x6720 -> 50x50
    downsample -- ~0.02 s/model, matching the paper's run_gf_xgboost_resnet18.py.
    The previous skimage anti-aliased path was ~10 min/model.

    Cached per X so gf_xgb and gf_1nn share one computation per attack severity.
    """
    if x in _GF_FEATURE_CACHE:
        return _GF_FEATURE_CACHE[x]

    from PIL import Image
    from model_xray.procedures.image_rep_procs import _grayscale_fourpart
    n = weights.shape[0]
    out = np.zeros((n, imsize * imsize), dtype=np.float32)
    for i in range(n):
        ws_i = weights[i] if x == 0 else attacked_weights(weights[i], x=x, malware_bits_or_path=None)
        img = _grayscale_fourpart(ws_i.reshape(1, -1))[0]  # (H, W) uint8
        img = Image.fromarray(img).resize((imsize, imsize), Image.BOX)
        out[i] = np.asarray(img, dtype=np.float32).reshape(-1)
    _GF_FEATURE_CACHE[x] = out
    return out


def _byte_window(weights: np.ndarray, x: int, window: int = MALCONV_BYTE_WINDOW) -> np.ndarray:
    """Take a deterministic 512-KB byte window at offset 0 (paper Section 4.1 (Baseline))."""
    if x > 0:
        weights = _attack_block(weights, x)
    out = np.zeros((weights.shape[0], window), dtype=np.uint8)
    for i, w in enumerate(weights):
        b = float32_to_bytes(w).reshape(-1)
        out[i, : min(window, b.size)] = b[:window]
    return out


def _cv_folds(n: int, n_splits: int, n_repeats: int, seed: int):
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    y_dummy = np.zeros(n)  # dummy stratification (binary attack labels added later per fold)
    return list(rskf.split(np.zeros(n), y_dummy))


# -------------------- per-method runners --------------------

def run_xgb_on_features(weights: np.ndarray, *, x_range, n_splits, n_repeats, seed,
                        feature_fn, label: str) -> pd.DataFrame:
    """Generic CV runner: feature_fn(weights, x) -> (n, d) feature matrix."""
    rows = []
    for x in x_range:
        feat_b = feature_fn(weights, 0)
        feat_m = feature_fn(weights, x)
        X = np.concatenate([feat_b, feat_m]); y = np.concatenate([np.zeros(len(feat_b)), np.ones(len(feat_m))])
        for fold_i, (tr, te) in enumerate(_cv_folds(len(X), n_splits, n_repeats, seed)):
            print(f"[exp4 {label}] x={x} fold={fold_i}")
            clf = make_xgb()
            clf.fit(X[tr], y[tr])
            test_acc = float((clf.predict(X[te]) == y[te]).mean())
            train_acc = float((clf.predict(X[tr]) == y[tr]).mean())
            rows.append({"repeat": fold_i, "baseline": label, "X": x, "fold": fold_i,
                         "accuracy": test_acc})
    return pd.DataFrame(rows)


def run_1nn_on_gf(weights: np.ndarray, *, x_range, n_splits, n_repeats, seed) -> pd.DataFrame:
    rows = []
    for x in x_range:
        feat_b = _gf_pixels(weights, 0)
        feat_m = _gf_pixels(weights, x)
        X = np.concatenate([feat_b, feat_m]); y = np.concatenate([np.zeros(len(feat_b)), np.ones(len(feat_m))])
        for fold_i, (tr, te) in enumerate(_cv_folds(len(X), n_splits, n_repeats, seed)):
            print(f"[exp4 GF+1NN] x={x} fold={fold_i}")
            knn = KNeighborsClassifier(n_neighbors=1)
            knn.fit(X[tr], y[tr])
            test_acc = float((knn.predict(X[te]) == y[te]).mean())
            train_acc = float((knn.predict(X[tr]) == y[tr]).mean())
            rows.append({"repeat": fold_i, "baseline": "GF+1NN", "X": x, "fold": fold_i,
                         "accuracy": test_acc})
    return pd.DataFrame(rows)


def run_malconv(weights: np.ndarray, *, x_range, n_splits, n_repeats, seed) -> pd.DataFrame:
    rows = []
    for x in x_range:
        Xb = _byte_window(weights, 0)
        Xm = _byte_window(weights, x)
        X = np.concatenate([Xb, Xm]); y = np.concatenate([np.zeros(len(Xb)), np.ones(len(Xm))]).astype(np.float32)
        for fold_i, (tr, te) in enumerate(_cv_folds(len(X), n_splits, n_repeats, seed)):
            print(f"[exp4 MalConv] x={x} fold={fold_i}")
            res = malconv_train(X[tr], y[tr], X[te], y[te], MalConvCfg(seed=fold_i))
            rows.append({"repeat": fold_i, "baseline": "B3-MalConv", "X": x, "fold": fold_i,
                         "accuracy": res["test_acc"]})
    return pd.DataFrame(rows)


def run_threshold_b5_b7(weights: np.ndarray, *, x_range, n_splits, n_repeats, seed) -> pd.DataFrame:
    """Paper-faithful threshold-baseline CV.

    Key protocol detail: KFold over n_models (NOT stratified-k-fold over the
    doubled (benign, attacked) population). Each fold's `tr` indices are then
    used in PAIRED fashion -- the same model index supplies both a train
    benign sample and a train attacked sample. This:
      (a) matches the paper's per-fold n_train ~ 0.8 * n_models,
      (b) prevents the model-identity leakage that the prior independent-split
          protocol allowed (train benign and train attacked drawn from
          disjoint model subsets makes the threshold over-fit to model
          identity rather than the attack signature, e.g. B7 hitting ~1.0
          at X=16 instead of the paper's ~0.51).
    """
    from sklearn.model_selection import KFold, RepeatedKFold
    n = len(weights)
    if n_repeats <= 1:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        folds = list(kf.split(np.arange(n)))
    else:
        rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
        folds = list(rkf.split(np.arange(n)))

    rows = []
    for x in x_range:
        attacked_cache = np.stack([attacked_weights(weights[i], x=x, malware_bits_or_path=None)
                                    for i in range(n)])
        for fold_i, (tr, te) in enumerate(folds):
            for cls, name in [(ByteEntropyDetector, "byte_entropy"),
                              (WeightValueDistributionDetector, "weight_value_dist")]:
                det = cls()
                det.fit([weights[i] for i in tr])
                bs_tr = [det.score(weights[i]) for i in tr]
                ms_tr = [det.score(attacked_cache[i]) for i in tr]
                t, _ = det.find_threshold(bs_tr, ms_tr)
                bs_te = [det.score(weights[i]) for i in te]
                ms_te = [det.score(attacked_cache[i]) for i in te]
                tn = sum(1 for s in bs_te if s <= t); tp = sum(1 for s in ms_te if s > t)
                acc = (tn + tp) / max(1, len(bs_te) + len(ms_te))
                rows.append({"repeat": fold_i, "baseline": name, "X": x, "fold": fold_i,
                             "accuracy": acc})
        del attacked_cache
    return pd.DataFrame(rows)


# -------------------- main --------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--mz-name", default="tiny-imagenet_resnet18")
    parser.add_argument("--n-models", type=int, default=None,
                        help="Subset of benign models to use (default: all). "
                             "Useful for memory-bound smoke tests on b2_yin / b3_malconv.")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-repeats", type=int, default=5)
    parser.add_argument("--x-range", type=int, nargs="+", default=X_RANGE)
    parser.add_argument("--methods", nargs="+",
                        default=["gf_xgb", "gf_1nn", "b2_yin", "b3_malconv", "thresholds"])
    parser.add_argument("--quick", action="store_true",
                        help="Smoke test: 2x2 CV, x in [4, 12, 20].")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default=os.path.join(_paths.results_dir(), "exp4"))
    args = parser.parse_args()

    if args.quick:
        args.n_splits = 2; args.n_repeats = 2; args.x_range = [4, 12, 20]

    weights = _load_resnet18(args.mz_name)
    if args.n_models is not None:
        weights = weights[: args.n_models]
    print(f"Loaded ResNet18 zoo: {weights.shape}")
    os.makedirs(args.out_dir, exist_ok=True)

    if "gf_xgb" in args.methods:
        df = run_xgb_on_features(weights, x_range=args.x_range, n_splits=args.n_splits,
                                 n_repeats=args.n_repeats, seed=args.seed,
                                 feature_fn=_gf_pixels, label="GF+XGBoost")
        df.to_csv(os.path.join(args.out_dir, "gf_xgboost.csv"), index=False)
    if "gf_1nn" in args.methods:
        df = run_1nn_on_gf(weights, x_range=args.x_range, n_splits=args.n_splits,
                           n_repeats=args.n_repeats, seed=args.seed)
        df.to_csv(os.path.join(args.out_dir, "gf_1nn.csv"), index=False)
    if "b2_yin" in args.methods:
        df = run_xgb_on_features(weights, x_range=args.x_range, n_splits=args.n_splits,
                                 n_repeats=args.n_repeats, seed=args.seed,
                                 feature_fn=_b2_yin_features,
                                 label="B2-Yin")
        df.to_csv(os.path.join(args.out_dir, "b2_yin.csv"), index=False)
    if "b3_malconv" in args.methods:
        df = run_malconv(weights, x_range=args.x_range, n_splits=args.n_splits,
                         n_repeats=args.n_repeats, seed=args.seed)
        df.to_csv(os.path.join(args.out_dir, "b3_malconv.csv"), index=False)
    if "thresholds" in args.methods:
        df = run_threshold_b5_b7(weights, x_range=args.x_range, n_splits=args.n_splits,
                                 n_repeats=args.n_repeats, seed=args.seed)
        df.to_csv(os.path.join(args.out_dir, "b5_b7_threshold.csv"), index=False)
    print(f"All requested methods complete; CSVs under {args.out_dir}")


if __name__ == "__main__":
    main()
