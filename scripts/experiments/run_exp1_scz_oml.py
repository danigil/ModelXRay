"""Experiment 1: SCZ STL10 OML (Figure 4).

Trains the FSL detector (OSL CNN, centroid + 1-NN) on 6 randomly drawn SCZ
STL10 models per repeat (3 benign + 3 attacked), and evaluates per-X test
accuracy across X in [1, 23]. Overlays B1 (Gilkarov XGBoost), B3 (MalConv-lite),
and B4-B7 (threshold detectors) trained with the matched Gilkarov-style budget
(4000 train / 1000 test) for fair comparison.

Inputs (from `scripts/data_creation/01_create_scz_stl10.py`):
    $MODELXRAY_GHRP_DIR/stl10/weights.npy  shape=(n_models, n_weights) float32

Outputs (under `results/exp1/`):
    fsl_per_x.csv          OSL CNN centroid+1NN per X, per run
    b1_xgb_per_x.csv       Gilkarov reproduction
    b3_malconv_per_x.csv   MalConv-lite raw-byte 1D-CNN
    b4_b7_threshold_per_x.csv  Naive baselines, per X, per run

The plot script in `scripts/plots/exp1_id_oml.py` consumes these.
"""

from __future__ import annotations

import argparse
import gc
import os
from typing import Sequence

import numpy as np
import pandas as pd

from model_xray.baselines.b1_gilkarov import fit_predict as b1_fit_predict
from model_xray.baselines.b2_yin import calc_phis_all  # noqa: F401 (reserved for B2 ablation)
from model_xray.baselines.b3_malconv import TrainConfig as MalConvCfg, train_and_eval as malconv_train
from model_xray.baselines.byte_attack import attacked_weights, float32_to_bytes, load_malware_bits
from model_xray.baselines.threshold import ALL_DETECTORS
from model_xray.data import paths as _paths
from model_xray.data.attack_pipeline import img_pp_xlsb_attack
from model_xray.data.ghrp_zoos import load_mz_weights
from model_xray.fsl.train import train_fsl
from model_xray.fsl.evaluate import evaluate_model


X_RANGE = list(range(1, 24))


def _load_scz(mz_name: str = "stl10") -> np.ndarray:
    return load_mz_weights(mz_name).astype(np.float32)


# -------------------- FSL --------------------

def run_fsl(weights: np.ndarray, *, n_repeats: int, x_range: Sequence[int],
            payload_filepath: str | None, imsize: int, mode: str, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    n_models = weights.shape[0]
    for r in range(n_repeats):
        # Random 6 train models (3 will be attacked at the run's anchor X)
        train_idx = rng.choice(n_models, size=6, replace=False)
        test_idx = rng.choice(np.setdiff1d(np.arange(n_models), train_idx),
                              size=min(300, n_models - 6), replace=False)
        for x in x_range:
            print(f"[exp1 FSL] repeat={r} x={x}")
            X_train_b = img_pp_xlsb_attack(weights[train_idx[:3]], imsize=imsize, x=0,
                                           payload_filepath=payload_filepath)
            X_train_m = img_pp_xlsb_attack(weights[train_idx[3:]], imsize=imsize, x=x,
                                           payload_filepath=payload_filepath)
            X_train = np.concatenate([X_train_b, X_train_m], axis=0)
            y_train = np.array([0] * 3 + [1] * 3)
            X_test_b = img_pp_xlsb_attack(weights[test_idx], imsize=imsize, x=0,
                                          payload_filepath=payload_filepath)
            X_test_m = img_pp_xlsb_attack(weights[test_idx], imsize=imsize, x=x,
                                          payload_filepath=payload_filepath)
            X_test = np.concatenate([X_test_b, X_test_m], axis=0)
            y_test = np.concatenate([np.zeros(len(X_test_b)), np.ones(len(X_test_m))])
            try:
                model = train_fsl(X_train, y_train, model_arch="osl_siamese_cnn",
                                  imsize=imsize, mode=mode)
                res = evaluate_model(model, X_test, y_test)
                rows.append({"repeat": r, "X_hat": x, "X": x,
                             "centroid": res["centroid"], "nn": res["nn"]})
            except Exception as e:
                print(f"[exp1 FSL] FAILED repeat={r} x={x}: {e!r}")
            finally:
                # GPU memory hygiene: drop the Keras session and the trained
                # Siamese weights so the next (X, repeat) iteration doesn't
                # accumulate ResourceExhaustedError on small GPUs.
                try:
                    del model
                except NameError:
                    pass
                try:
                    import tensorflow as tf
                    tf.keras.backend.clear_session()
                except Exception:
                    pass
                gc.collect()
    return pd.DataFrame(rows)


# -------------------- B1 / B3 / threshold --------------------

def _split(weights: np.ndarray, n_train: int, n_test: int, rng) -> tuple[np.ndarray, np.ndarray]:
    n = weights.shape[0]
    perm = rng.permutation(n)
    return perm[:n_train], perm[n_train : n_train + n_test]


def run_b1_xgb(weights: np.ndarray, *, x_range, payload_filepath, n_train, n_test,
               n_repeats, seed) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for r in range(n_repeats):
        tr, te = _split(weights, n_train, n_test, rng)
        for x in x_range:
            print(f"[exp1 B1 XGB] repeat={r} x={x}")
            attacker = lambda idx: np.stack([attacked_weights(weights[i], x=x,
                                                              malware_bits_or_path=payload_filepath)
                                              for i in idx])
            X_tr = np.concatenate([weights[tr], attacker(tr)])
            y_tr = np.concatenate([np.zeros(len(tr)), np.ones(len(tr))])
            X_te = np.concatenate([weights[te], attacker(te)])
            y_te = np.concatenate([np.zeros(len(te)), np.ones(len(te))])
            _, y_pred = b1_fit_predict(X_tr.astype(np.float32), y_tr.astype(int),
                                       X_te.astype(np.float32))
            acc = float((y_pred == y_te).mean())
            rows.append({"repeat": r, "baseline": "xgboost", "X": x, "accuracy": acc})
    return pd.DataFrame(rows)


def run_b3_malconv(weights: np.ndarray, *, x_range, payload_filepath, n_train, n_test,
                   n_repeats, seed) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for r in range(n_repeats):
        tr, te = _split(weights, n_train, n_test, rng)
        for x in x_range:
            print(f"[exp1 B3 MalConv] repeat={r} x={x}")
            def to_bytes(arr2d):
                return np.stack([float32_to_bytes(w).reshape(-1) for w in arr2d]).astype(np.uint8)
            X_tr_b = to_bytes(weights[tr])
            X_tr_m = to_bytes(np.stack([attacked_weights(weights[i], x=x, malware_bits_or_path=payload_filepath) for i in tr]))
            X_te_b = to_bytes(weights[te])
            X_te_m = to_bytes(np.stack([attacked_weights(weights[i], x=x, malware_bits_or_path=payload_filepath) for i in te]))
            X_tr = np.concatenate([X_tr_b, X_tr_m]); y_tr = np.concatenate([np.zeros(len(tr)), np.ones(len(tr))])
            X_te = np.concatenate([X_te_b, X_te_m]); y_te = np.concatenate([np.zeros(len(te)), np.ones(len(te))])
            res = malconv_train(X_tr, y_tr.astype(np.float32), X_te, y_te.astype(np.float32),
                                MalConvCfg(seed=r))
            rows.append({"repeat": r, "baseline": "malconv_lite", "X": x, "accuracy": res["test_acc"]})
    return pd.DataFrame(rows)


def run_thresholds(weights: np.ndarray, *, x_range, payload_filepath, n_train, n_test,
                   n_repeats, seed) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for r in range(n_repeats):
        tr, te = _split(weights, n_train, n_test, rng)
        for x in x_range:
            print(f"[exp1 B4-B7] repeat={r} x={x}")
            attacker = lambda idx: [attacked_weights(weights[i], x=x, malware_bits_or_path=payload_filepath) for i in idx]
            train_attacked = attacker(tr)
            test_attacked = attacker(te)
            for cls in ALL_DETECTORS:
                det = cls()
                det.fit([weights[i] for i in tr])
                bs_tr = [det.score(weights[i]) for i in tr]
                ms_tr = [det.score(w) for w in train_attacked]
                t, _ = det.find_threshold(bs_tr, ms_tr)
                bs_te = [det.score(weights[i]) for i in te]
                ms_te = [det.score(w) for w in test_attacked]
                tn = sum(1 for s in bs_te if s <= t)
                tp = sum(1 for s in ms_te if s > t)
                acc = (tn + tp) / (len(bs_te) + len(ms_te))
                rows.append({"repeat": r, "baseline": det.name, "X": x, "accuracy": acc})
    return pd.DataFrame(rows)


# -------------------- main --------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--mz-name", default="stl10")
    parser.add_argument("--n-repeats", type=int, default=30)
    parser.add_argument("--x-range", type=int, nargs="+", default=X_RANGE)
    parser.add_argument("--payload-file", default=None,
                        help="Path to malware payload (default: $MODELXRAY_PAYLOAD_FILE).")
    parser.add_argument("--imsize", type=int, default=100)
    parser.add_argument("--mode", default="ub")
    parser.add_argument("--n-train-baselines", type=int, default=4000)
    parser.add_argument("--n-test-baselines", type=int, default=1000)
    parser.add_argument("--quick", action="store_true",
                        help="Smoke test: 2 repeats, x in [1, 8, 16, 23], 200/100 baseline split.")
    parser.add_argument("--methods", nargs="+", default=["fsl", "b1", "b3", "thresholds"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default=os.path.join(_paths.results_dir(), "exp1"))
    args = parser.parse_args()

    if args.quick:
        args.n_repeats = 2
        args.x_range = [1, 8, 16, 23]
        args.n_train_baselines = 200
        args.n_test_baselines = 100

    payload = args.payload_file or _paths.get_payload_file()
    weights = _load_scz(args.mz_name)
    print(f"Loaded SCZ {args.mz_name}: {weights.shape}")
    os.makedirs(args.out_dir, exist_ok=True)

    if "fsl" in args.methods:
        df = run_fsl(weights, n_repeats=args.n_repeats, x_range=args.x_range,
                     payload_filepath=payload, imsize=args.imsize, mode=args.mode, seed=args.seed)
        df.to_csv(os.path.join(args.out_dir, "fsl_osl.csv"), index=False)
    if "b1" in args.methods:
        df = run_b1_xgb(weights, x_range=args.x_range, payload_filepath=payload,
                        n_train=args.n_train_baselines, n_test=args.n_test_baselines,
                        n_repeats=args.n_repeats, seed=args.seed)
        df.to_csv(os.path.join(args.out_dir, "b1_gilkarov_per_x.csv"), index=False)
    if "b3" in args.methods:
        df = run_b3_malconv(weights, x_range=args.x_range, payload_filepath=payload,
                            n_train=args.n_train_baselines, n_test=args.n_test_baselines,
                            n_repeats=args.n_repeats, seed=args.seed)
        df.to_csv(os.path.join(args.out_dir, "b3_malconv.csv"), index=False)
    if "thresholds" in args.methods:
        df = run_thresholds(weights, x_range=args.x_range, payload_filepath=payload,
                            n_train=args.n_train_baselines, n_test=args.n_test_baselines,
                            n_repeats=args.n_repeats, seed=args.seed)
        df.to_csv(os.path.join(args.out_dir, "b4_b7_threshold.csv"), index=False)
    print(f"All requested methods complete; CSVs under {args.out_dir}")


if __name__ == "__main__":
    main()
