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
    """Stack per-arch GF images at given X (X=0 means benign).

    Uses PIL Image.BOX for the (huge GF intermediate) -> (imsize, imsize)
    downsample. Famous large CNNs reach ~89M params (NASNetLarge), whose GF
    intermediate is ~9450x9450 -- the canonical img_pp_xlsb_attack path goes
    through skimage_resize(anti_aliasing=True) which takes minutes per arch.
    PIL Image.BOX area-resampling matches the paper's run_gf_xgboost_resnet18.py
    fast-path: ~0.05 s per 89M-param model.
    """
    from PIL import Image
    from model_xray.baselines.byte_attack import attacked_weights
    from model_xray.procedures.image_rep_procs import _grayscale_fourpart
    out = []
    for w in arch_w.values():
        ws = w if x == 0 else attacked_weights(w, x=x, malware_bits_or_path=payload)
        img = _grayscale_fourpart(ws.reshape(1, -1))[0]  # (H, W) uint8
        img = Image.fromarray(img).resize((imsize, imsize), Image.BOX)
        out.append(np.asarray(img, dtype=np.float32))
    return np.stack(out)


# -------------------- FSL --------------------

def _fsl_one_xhat_subproc(q, small_train, small_test, large_test, *,
                          model_arch, imsize, mode, x_hat, x_range, payload,
                          crossx, repeat):
    """Train the FSL model and evaluate ONE x_hat anchor across requested
    eval Xs and eval sets, push the resulting rows back via `q`.

    Lives in its own process so that TF's accumulating session/graph state
    is reclaimed when the process exits — calling tf.keras.backend.clear_session()
    + gc.collect() between trainings does not actually return GPU memory to
    the OS, leading to ResourceExhaustedError after ~20 trainings (verified
    overnight on the 23-X_hat crossx sweep).
    """
    rows: list[dict] = []
    try:
        from model_xray.fsl.train import train_fsl
        from model_xray.fsl.evaluate import evaluate_model

        X_train_b = _stack_imgs(small_train, imsize, payload, x=0)
        X_train_m = _stack_imgs(small_train, imsize, payload, x=x_hat)
        X_train = np.concatenate([X_train_b, X_train_m])
        y_train = np.concatenate([np.zeros(len(X_train_b)), np.ones(len(X_train_m))])

        model = train_fsl(X_train, y_train, model_arch=model_arch, imsize=imsize, mode=mode)

        eval_xs = list(x_range) if crossx else [x_hat]
        for eval_x in eval_xs:
            for eval_set, test_dict in (("famous_le_10m", small_test),
                                        ("famous_le_100m", large_test)):
                X_b = _stack_imgs(test_dict, imsize, payload, x=0)
                X_m = _stack_imgs(test_dict, imsize, payload, x=eval_x)
                res = evaluate_model(model, np.concatenate([X_b, X_m]),
                                     np.concatenate([np.zeros(len(X_b)), np.ones(len(X_m))]))
                rows.append({"repeat": repeat, "X_hat": x_hat, "X": eval_x,
                             "eval_set": eval_set, **res})
    except Exception as e:
        print(f"[exp2 FSL/{model_arch} subproc] FAILED repeat={repeat} x_hat={x_hat}: {e!r}")
    finally:
        q.put(rows)


def run_fsl(small_train, small_test, large_test, *, model_arch, imsize, mode, n_repeats, x_range,
            payload, seed, crossx: bool = False, x_hat_range=None) -> pd.DataFrame:
    """Train + evaluate the FSL detector across X (per-x_hat in a fresh subprocess).

    OML mode (`crossx=False`): each (repeat, X) trains one model at anchor X and
    evaluates against ID + OOD at the same X (canonical schema rows have
    `X_hat == X`).

    AL mode (`crossx=True`): each (repeat, X_hat) trains one model at anchor
    X_hat and evaluates against ID + OOD across *every* X in `x_range`
    (cross-X sweep, used by the AL Figure 6 weighted-metric curve).

    `x_hat_range` (optional): restrict the training-anchor sweep to a subset
    of `x_range`. Used for resuming a partial crossx run (e.g. retry only
    X_hat ∈ {20,21,22,23} after an OOM at the high end). When None, use
    `x_range` for both anchor and eval.

    Each x_hat is run in its own multiprocessing.Process so that TF's
    accumulating GPU session state is reclaimed by the OS at process exit
    (clear_session + gc are not sufficient for long sweeps).
    """
    import multiprocessing as mp

    # Use spawn to ensure each child gets a fresh interpreter (the parent has
    # already imported TF for compute-graph construction in earlier x_hats
    # of a re-run; spawn avoids inheriting any stale CUDA state).
    ctx = mp.get_context("spawn")
    rows: list[dict] = []
    anchor_xs = list(x_hat_range) if x_hat_range is not None else list(x_range)
    for r in range(n_repeats):
        for x_hat in anchor_xs:
            print(f"[exp2 FSL/{model_arch}] repeat={r} x_hat={x_hat} crossx={crossx} (subproc)")
            q = ctx.Queue()
            p = ctx.Process(
                target=_fsl_one_xhat_subproc,
                args=(q, small_train, small_test, large_test),
                kwargs=dict(model_arch=model_arch, imsize=imsize, mode=mode,
                            x_hat=x_hat, x_range=list(x_range), payload=payload,
                            crossx=crossx, repeat=r),
            )
            p.start()
            try:
                xhat_rows = q.get()  # block until subprocess has put something
                rows.extend(xhat_rows)
            finally:
                p.join()
    return pd.DataFrame(rows)


# -------------------- B3 / thresholds --------------------

def run_b3(small_train, small_test, large_test, *, x_range, payload, seed,
           crossx: bool = False) -> pd.DataFrame:
    """B3 (MalConv-lite) baseline.

    Architectures in `small_*` / `large_*` have wildly different parameter
    counts (MobileNetV2 ~3.5M floats vs VGG16 ~138M), so the raw byte
    sequences cannot be stacked. Extract a fixed 512 KB byte window
    (= the first 131072 floats) from each model so all per-arch byte
    sequences share a common length; all architectures used in Exp 2 have
    at least that many parameters.
    """
    rows = []
    WINDOW_BYTES = 512 * 1024
    floats_per_window = WINDOW_BYTES // 4

    def _bytes_uint8(arr_dict, x):
        out = np.empty((len(arr_dict), WINDOW_BYTES), dtype=np.uint8)
        for i, w in enumerate(arr_dict.values()):
            if w.shape[0] < floats_per_window:
                raise ValueError(f"arch with only {w.shape[0]} floats < window {floats_per_window}")
            ws_window = w[:floats_per_window].astype(np.float32, copy=False)
            if x > 0:
                ws_window = attacked_weights(ws_window, x=x, malware_bits_or_path=payload)
            out[i] = float32_to_bytes(ws_window).reshape(-1)
        return out
    for x_hat in x_range:
        print(f"[exp2 B3 MalConv] x_hat={x_hat} crossx={crossx}")
        Xtr = np.concatenate([_bytes_uint8(small_train, 0), _bytes_uint8(small_train, x_hat)])
        ytr = np.concatenate([np.zeros(len(small_train)), np.ones(len(small_train))]).astype(np.float32)
        eval_xs = x_range if crossx else [x_hat]
        for eval_x in eval_xs:
            for eval_set, test_dict in (("famous_le_10m", small_test),
                                        ("famous_le_100m", large_test)):
                Xte = np.concatenate([_bytes_uint8(test_dict, 0), _bytes_uint8(test_dict, eval_x)])
                yte = np.concatenate([np.zeros(len(test_dict)), np.ones(len(test_dict))]).astype(np.float32)
                res = malconv_train(Xtr, ytr, Xte, yte, MalConvCfg(seed=seed))
                rows.append({"repeat": seed, "baseline": "malconv_lite",
                             "X_hat": x_hat, "X": eval_x, "eval_set": eval_set,
                             "accuracy": res["test_acc"]})
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
            for eval_set, test_dict in (("famous_le_10m", small_test),
                                        ("famous_le_100m", large_test)):
                test_b = list(test_dict.values())
                test_m = [attacked_weights(w, x=x, malware_bits_or_path=payload) for w in test_dict.values()]
                bs_te = [det.score(w) for w in test_b]
                ms_te = [det.score(w) for w in test_m]
                tn = sum(1 for s in bs_te if s <= t)
                tp = sum(1 for s in ms_te if s > t)
                acc = (tn + tp) / (len(bs_te) + len(ms_te))
                rows.append({"repeat": 0, "baseline": det.name, "X": x,
                             "eval_set": eval_set, "accuracy": acc})
    return pd.DataFrame(rows)


# -------------------- main --------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--small-h5", default=None,
                        help="Path to famous_le_10m/mcwa.h5 (default: $MODELXRAY_GHRP_DIR/famous_le_10m/mcwa.h5).")
    parser.add_argument("--large-h5", default=None)
    parser.add_argument("--small-train-archs", nargs="+", default=list(SMALL_TRAIN),
                        help="Override training arch list (default: paper's 3 small CNNs).")
    parser.add_argument("--small-test-archs", nargs="+", default=list(SMALL_TEST),
                        help="Override ID test arch list (default: paper's 5 small CNNs).")
    parser.add_argument("--large-test-archs", nargs="+", default=list(LARGE_TEST),
                        help="Override OOD test arch list (default: paper's 16 large CNNs).")
    parser.add_argument("--n-repeats", type=int, default=30)
    parser.add_argument("--x-range", type=int, nargs="+", default=X_RANGE)
    parser.add_argument("--x-hat-range", type=int, nargs="+", default=None,
                        help="FSL-only: restrict training-anchor X to this subset "
                             "of --x-range. Default: same as --x-range. Useful for "
                             "resuming a partial crossx sweep without re-running "
                             "anchors that already succeeded.")
    parser.add_argument("--payload-file", default=None)
    parser.add_argument("--mode", default="ub")
    parser.add_argument("--methods", nargs="+", default=["fsl_osl", "fsl_srnet", "b3", "thresholds"])
    parser.add_argument("--train-set", choices=["small", "large"], default="small",
                        help="Training set tag — controls output filename suffix and which arch list "
                             "is used to train the FSL detector. Default `small` reproduces paper "
                             "Figures 5/7 (small-CNN training). `large` is for Exp 2's secondary "
                             "large-train configuration.")
    parser.add_argument("--crossx", action="store_true",
                        help="AL Figure 6: also evaluate trained model across every X in --x-range "
                             "(produces extra rows with X_hat != X). Output filename suffix becomes "
                             "fsl_*_crossx_<train_set>.csv.")
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
    small_train = _load_collection(args.small_h5, args.small_train_archs)
    small_test = _load_collection(args.small_h5, args.small_test_archs)
    large_test = _load_collection(args.large_h5, args.large_test_archs)
    print(f"Train(small)={len(small_train)}  test(small)={len(small_test)}  test(large)={len(large_test)}")
    os.makedirs(args.out_dir, exist_ok=True)

    crossx_tag = "_crossx" if args.crossx else ""
    suf = args.train_set
    if "fsl_osl" in args.methods:
        df = run_fsl(small_train, small_test, large_test, model_arch="osl_siamese_cnn",
                     imsize=100, mode=args.mode, n_repeats=args.n_repeats, x_range=args.x_range,
                     payload=payload, seed=args.seed, crossx=args.crossx,
                     x_hat_range=args.x_hat_range)
        df.to_csv(os.path.join(args.out_dir, f"fsl_osl{crossx_tag}_{suf}.csv"), index=False)
    if "fsl_srnet" in args.methods:
        df = run_fsl(small_train, small_test, large_test, model_arch="srnet",
                     imsize=256, mode=args.mode, n_repeats=args.n_repeats, x_range=args.x_range,
                     payload=payload, seed=args.seed, crossx=args.crossx,
                     x_hat_range=args.x_hat_range)
        df.to_csv(os.path.join(args.out_dir, f"fsl_srnet{crossx_tag}_{suf}.csv"), index=False)
    if "b3" in args.methods:
        df = run_b3(small_train, small_test, large_test, x_range=args.x_range,
                    payload=payload, seed=args.seed, crossx=args.crossx)
        df.to_csv(os.path.join(args.out_dir, f"b3_malconv{crossx_tag}_{suf}.csv"), index=False)
    if "thresholds" in args.methods:
        df = run_thresholds(small_train, small_test, large_test, x_range=args.x_range, payload=payload)
        df.to_csv(os.path.join(args.out_dir, f"b4_b7_threshold_{suf}.csv"), index=False)
    print(f"All requested methods complete; CSVs under {args.out_dir}")


if __name__ == "__main__":
    main()
