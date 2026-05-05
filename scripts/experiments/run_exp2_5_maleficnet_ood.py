"""Experiment 2.5: MaleficNet OOD evaluation (tab:malefic).

Reuses the FSL detectors trained in Experiment 2 (script run_exp2_famous_cnns.py)
and evaluates them on the MaleficNet attacked-image dataset (D4) - a
spread-spectrum attack family that the detectors were NEVER trained on. The
table reports {centroid, 1-NN} test accuracy + Weighted Metric per
(arch x payload) variant.

Approach: rather than re-implementing Experiment 2's training, this script
re-trains FSL on (small_train benign, small_train attacked at X=1) - the same
recipe as Experiment 2 - because the trained Keras Siamese checkpoints are not
checkpointed to disk in the public-artifact pipeline. The X=1 anchor is the
top-row entry of tab:malefic (paper: WM 87, OOD acc 80, OSL CNN, centroid).

Inputs:
    $MODELXRAY_GHRP_DIR/famous_le_10m/mcwa.h5  (D2, for FSL training)
    $MODELXRAY_MALEFICNET_DIR/maleficnet_imgs<gf50>.npy   (D4)

Outputs (results/exp2_5/):
    maleficnet_ood_results.csv   per (arch, X_anchor, repeat) -> centroid/nn acc
"""

from __future__ import annotations

import argparse
import os
from typing import Sequence

import h5py
import numpy as np
import pandas as pd

from model_xray.data import paths as _paths
from model_xray.data.attack_pipeline import img_pp_xlsb_attack
from model_xray.data.maleficnet import ret_maleficnet_data
from model_xray.data.pretrained_models import SMALL_TRAIN
from model_xray.fsl.train import train_fsl
from model_xray.fsl.evaluate import evaluate_model


def _load_archs(h5_path: str, arch_names: Sequence[str]):
    out = {}
    with h5py.File(h5_path, "r") as f:
        for n in arch_names:
            out[n] = np.asarray(f[n][...]).reshape(-1).astype(np.float32)
    return out


def _stack_imgs(arch_w, imsize, payload, x):
    return np.stack([img_pp_xlsb_attack(w[np.newaxis, :], imsize=imsize, x=x, payload_filepath=payload)[0]
                     for w in arch_w.values()])


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--small-h5", default=None)
    parser.add_argument("--mz-imsize", type=int, default=50,
                        help="MaleficNet image size to load (must match the cache built by 04_*).")
    parser.add_argument("--mz-image-rep", default="gf", choices=["gf", "rgb", "s"],
                        help="MaleficNet cached image rep (default: gf; pre-existing caches may be rgb).")
    parser.add_argument("--fsl-imsize", type=int, default=100,
                        help="FSL training imsize (paper: 100 for OSL CNN).")
    parser.add_argument("--model-arch", default="osl_siamese_cnn",
                        choices=["osl_siamese_cnn", "srnet"])
    parser.add_argument("--x-anchors", type=int, nargs="+", default=[1])
    parser.add_argument("--n-repeats", type=int, default=10)
    parser.add_argument("--payload-file", default=None)
    parser.add_argument("--mode", default="ub")
    parser.add_argument("--out-dir", default=os.path.join(_paths.results_dir(), "exp2_5"))
    parser.add_argument("--quick", action="store_true",
                        help="Smoke test: 2 repeats, single anchor X=1.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.quick:
        args.n_repeats = 2
        args.x_anchors = [1]

    if args.small_h5 is None:
        args.small_h5 = os.path.join(_paths.get_ghrp_dir(), "famous_le_10m", "mcwa.h5")

    payload = args.payload_file or _paths.get_payload_file()
    print(f"Loading FSL training set from {args.small_h5} ...")
    small_train = _load_archs(args.small_h5, SMALL_TRAIN)
    print(f"Loading MaleficNet OOD test set (imsize={args.mz_imsize}) ...")
    X_oo, y_oo = ret_maleficnet_data(imsize=args.mz_imsize, image_rep=args.mz_image_rep,
                                     split_benign_mal=False, flatten_imgs=False)
    # The FSL detector was trained on (n, fsl_imsize, fsl_imsize, 1) grayscale
    # GF images. Coerce the cached MaleficNet test set to that shape:
    #   - if it is RGB (n, h, w, 3), collapse channels by mean-averaging,
    #   - then resize each (h, w) frame to (fsl_imsize, fsl_imsize) via PIL.
    if X_oo.ndim == 4 and X_oo.shape[-1] == 3:
        X_oo = X_oo.mean(axis=-1)  # collapse RGB -> grayscale
    if args.mz_imsize != args.fsl_imsize:
        from PIL import Image
        X_resized = np.zeros((X_oo.shape[0], args.fsl_imsize, args.fsl_imsize), dtype=X_oo.dtype)
        for i in range(X_oo.shape[0]):
            img = Image.fromarray(X_oo[i].astype(np.uint8) if X_oo.dtype != np.uint8 else X_oo[i])
            X_resized[i] = np.asarray(img.resize((args.fsl_imsize, args.fsl_imsize), Image.BICUBIC))
        X_oo = X_resized

    rows = []
    for x_anchor in args.x_anchors:
        for r in range(args.n_repeats):
            print(f"[exp2.5] anchor X={x_anchor} repeat={r}")
            X_train_b = _stack_imgs(small_train, args.fsl_imsize, payload, x=0)
            X_train_m = _stack_imgs(small_train, args.fsl_imsize, payload, x=x_anchor)
            X_train = np.concatenate([X_train_b, X_train_m])
            y_train = np.concatenate([np.zeros(len(X_train_b)), np.ones(len(X_train_m))])
            model = train_fsl(X_train, y_train, model_arch=args.model_arch,
                              imsize=args.fsl_imsize, mode=args.mode)
            ood_res = evaluate_model(model, X_oo, y_oo)
            rows.append({"X_anchor": x_anchor, "repeat": r,
                         "model_arch": args.model_arch, **ood_res})

    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.DataFrame(rows)
    out_path = os.path.join(args.out_dir, "maleficnet_ood_results.csv")
    df.to_csv(out_path, index=False)
    print(df)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
