"""Experiment 2: Famous CNNs ID + OOD (figs:exp2_id_oml, exp2_id_al, exp2_ood_oml).

Trains FSL detectors (OSL CNN at 100x100 + SRNet at 256x256) on 6 models drawn
from the small-CNN train split (Section 4.7.1), then evaluates per X across:
  - in-distribution test split (other small-CNN architectures)
  - out-of-distribution test split (large CNNs, Section 4.7.4)

Also runs the threshold detectors (B4-B7) and B3 (MalConv-lite). The XGBoost
academic baselines (B1, B2) are NOT in this experiment per the paper:
B1 is the SCZ-only Gilkarov reproduction; B2 (Yin) is too slow at this scale
(Section 5.2: 173s/model at 1e8 params).

Inputs (from `scripts/data_creation/02_create_famous_small_cnns.py` and `03_*`):
    $MODELXRAY_GHRP_DIR/famous_le_10m/mcwa.h5   keyed by arch (D2)
    $MODELXRAY_GHRP_DIR/famous_le_100m/mcwa.h5  keyed by arch (D3)

Outputs (under `results/exp2/`):
    fsl_<arch>_id_per_x.csv, fsl_<arch>_ood_per_x.csv
    b3_malconv_id_per_x.csv, b3_malconv_ood_per_x.csv
    b4_b7_threshold_id_per_x.csv, b4_b7_threshold_ood_per_x.csv
"""

from __future__ import annotations

import argparse
import gc
import os
from typing import Dict, Sequence

import h5py
import numpy as np
import pandas as pd

from model_xray.baselines.b3_malconv import TrainConfig as MalConvCfg, train_and_eval as malconv_train
from model_xray.baselines.byte_attack import attacked_weights, float32_to_bytes
from model_xray.baselines.threshold import ALL_DETECTORS
from model_xray.data import paths as _paths
from model_xray.data.attack_pipeline import img_pp_xlsb_attack
from model_xray.data.pretrained_models import LARGE_TEST, SMALL_TEST, SMALL_TRAIN
from model_xray.fsl.train import train_fsl
from model_xray.fsl.evaluate import evaluate_model


X_RANGE = list(range(1, 24))


def _load_collection(path: str, arch_names: Sequence[str]) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    with h5py.File(path, "r") as f:
        for n in arch_names:
            if n not in f:
                raise KeyError(f"{n!r} not in {path}; available: {list(f.keys())}")
            out[n] = np.asarray(f[n][...]).reshape(-1).astype(np.float32)
    return out


def _attack_dict(arch_w: Dict[str, np.ndarray], x: int, payload) -> Dict[str, np.ndarray]:
    return {n: attacked_weights(w, x=x, malware_bits_or_path=payload) for n, w in arch_w.items()}


def _stack_imgs(arch_w: Dict[str, np.ndarray], imsize: int, payload, x: int) -> np.ndarray:
    """Stack per-arch GF images at given X (X=0 means benign)."""
    return np.stack([img_pp_xlsb_attack(w[np.newaxis, :], imsize=imsize, x=x, payload_filepath=payload)[0]
                     for w in arch_w.values()])


# -------------------- FSL --------------------

def run_fsl(small_train, small_test, large_test, *, model_arch, imsize, mode, n_repeats, x_range,
            payload, seed) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for r in range(n_repeats):
        for x in x_range:
            print(f"[exp2 FSL/{model_arch}] repeat={r} x={x}")
            X_train_b = _stack_imgs(small_train, imsize, payload, x=0)
            X_train_m = _stack_imgs(small_train, imsize, payload, x=x)
            X_train = np.concatenate([X_train_b, X_train_m])
            y_train = np.concatenate([np.zeros(len(X_train_b)), np.ones(len(X_train_m))])
            try:
                model = train_fsl(X_train, y_train, model_arch=model_arch, imsize=imsize, mode=mode)
                # ID test: small_test
                X_id_b = _stack_imgs(small_test, imsize, payload, x=0)
                X_id_m = _stack_imgs(small_test, imsize, payload, x=x)
                id_res = evaluate_model(model, np.concatenate([X_id_b, X_id_m]),
                                        np.concatenate([np.zeros(len(X_id_b)), np.ones(len(X_id_m))]))
                # OOD test: large_test
                X_oo_b = _stack_imgs(large_test, imsize, payload, x=0)
                X_oo_m = _stack_imgs(large_test, imsize, payload, x=x)
                oo_res = evaluate_model(model, np.concatenate([X_oo_b, X_oo_m]),
                                        np.concatenate([np.zeros(len(X_oo_b)), np.ones(len(X_oo_m))]))
                rows.append({"repeat": r, "X": x, "split": "id", **id_res})
                rows.append({"repeat": r, "X": x, "split": "ood", **oo_res})
            except Exception as e:
                print(f"[exp2 FSL/{model_arch}] FAILED repeat={r} x={x}: {e!r}")
            finally:
                gc.collect()
    return pd.DataFrame(rows)


# -------------------- B3 / thresholds --------------------

def run_b3(small_train, small_test, large_test, *, x_range, payload, seed) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    def _bytes_uint8(arr_dict, x):
        if x == 0:
            return np.stack([float32_to_bytes(w).reshape(-1) for w in arr_dict.values()]).astype(np.uint8)
        return np.stack([float32_to_bytes(attacked_weights(w, x=x, malware_bits_or_path=payload)).reshape(-1)
                         for w in arr_dict.values()]).astype(np.uint8)
    for x in x_range:
        print(f"[exp2 B3 MalConv] x={x}")
        Xtr = np.concatenate([_bytes_uint8(small_train, 0), _bytes_uint8(small_train, x)])
        ytr = np.concatenate([np.zeros(len(small_train)), np.ones(len(small_train))]).astype(np.float32)
        for split_name, test_dict in [("id", small_test), ("ood", large_test)]:
            Xte = np.concatenate([_bytes_uint8(test_dict, 0), _bytes_uint8(test_dict, x)])
            yte = np.concatenate([np.zeros(len(test_dict)), np.ones(len(test_dict))]).astype(np.float32)
            res = malconv_train(Xtr, ytr, Xte, yte, MalConvCfg(seed=seed))
            rows.append({"X": x, "split": split_name, "test_acc": res["test_acc"]})
    return pd.DataFrame(rows)


def run_thresholds(small_train, small_test, large_test, *, x_range, payload) -> pd.DataFrame:
    rows = []
    for x in x_range:
        print(f"[exp2 B4-B7] x={x}")
        # Fit each detector on the 3 train benign weights, calibrate on train benign+attacked
        train_b = list(small_train.values())
        train_m = [attacked_weights(w, x=x, malware_bits_or_path=payload) for w in small_train.values()]
        for cls in ALL_DETECTORS:
            det = cls()
            det.fit(train_b)
            bs_tr = [det.score(w) for w in train_b]
            ms_tr = [det.score(w) for w in train_m]
            t, _ = det.find_threshold(bs_tr, ms_tr)
            for split_name, test_dict in [("id", small_test), ("ood", large_test)]:
                test_b = list(test_dict.values())
                test_m = [attacked_weights(w, x=x, malware_bits_or_path=payload) for w in test_dict.values()]
                bs_te = [det.score(w) for w in test_b]
                ms_te = [det.score(w) for w in test_m]
                tn = sum(1 for s in bs_te if s <= t)
                tp = sum(1 for s in ms_te if s > t)
                acc = (tn + tp) / (len(bs_te) + len(ms_te))
                rows.append({"X": x, "split": split_name, "baseline": det.name, "test_acc": acc})
    return pd.DataFrame(rows)


# -------------------- main --------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--small-h5", default=None,
                        help="Path to famous_le_10m/mcwa.h5 (default: $MODELXRAY_GHRP_DIR/famous_le_10m/mcwa.h5).")
    parser.add_argument("--large-h5", default=None)
    parser.add_argument("--n-repeats", type=int, default=30)
    parser.add_argument("--x-range", type=int, nargs="+", default=X_RANGE)
    parser.add_argument("--payload-file", default=None)
    parser.add_argument("--mode", default="ub")
    parser.add_argument("--methods", nargs="+", default=["fsl_osl", "fsl_srnet", "b3", "thresholds"])
    parser.add_argument("--quick", action="store_true",
                        help="Smoke test: 2 repeats, x in [1, 8, 16, 23].")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default=os.path.join(_paths.results_dir(), "exp2"))
    args = parser.parse_args()

    if args.quick:
        args.n_repeats = 2
        args.x_range = [1, 8, 16, 23]

    payload = args.payload_file or _paths.get_payload_file()
    if args.small_h5 is None:
        args.small_h5 = os.path.join(_paths.get_ghrp_dir(), "famous_le_10m", "mcwa.h5")
    if args.large_h5 is None:
        args.large_h5 = os.path.join(_paths.get_ghrp_dir(), "famous_le_100m", "mcwa.h5")

    print(f"Loading {args.small_h5}, {args.large_h5} ...")
    small_train = _load_collection(args.small_h5, SMALL_TRAIN)
    small_test = _load_collection(args.small_h5, SMALL_TEST)
    large_test = _load_collection(args.large_h5, LARGE_TEST)
    print(f"Train(small)={len(small_train)}  test(small)={len(small_test)}  test(large)={len(large_test)}")
    os.makedirs(args.out_dir, exist_ok=True)

    if "fsl_osl" in args.methods:
        df = run_fsl(small_train, small_test, large_test, model_arch="osl_siamese_cnn",
                     imsize=100, mode=args.mode, n_repeats=args.n_repeats, x_range=args.x_range,
                     payload=payload, seed=args.seed)
        df.to_csv(os.path.join(args.out_dir, "fsl_osl_per_x.csv"), index=False)
    if "fsl_srnet" in args.methods:
        df = run_fsl(small_train, small_test, large_test, model_arch="srnet",
                     imsize=256, mode=args.mode, n_repeats=args.n_repeats, x_range=args.x_range,
                     payload=payload, seed=args.seed)
        df.to_csv(os.path.join(args.out_dir, "fsl_srnet_per_x.csv"), index=False)
    if "b3" in args.methods:
        df = run_b3(small_train, small_test, large_test, x_range=args.x_range, payload=payload, seed=args.seed)
        df.to_csv(os.path.join(args.out_dir, "b3_malconv_per_x.csv"), index=False)
    if "thresholds" in args.methods:
        df = run_thresholds(small_train, small_test, large_test, x_range=args.x_range, payload=payload)
        df.to_csv(os.path.join(args.out_dir, "b4_b7_threshold_per_x.csv"), index=False)
    print(f"All requested methods complete; CSVs under {args.out_dir}")


if __name__ == "__main__":
    main()
