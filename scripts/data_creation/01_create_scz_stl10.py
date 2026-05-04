"""D1: build the SCZ STL10 benign-weight cache.

Reads the GHRP STL10 small-CNN zoo (downloaded by `download_zoos.sh`) and
materializes a single `weights.npy` of shape (n_models, n_weights) under
$MODELXRAY_GHRP_DIR/stl10/.

Per-X attacked variants and image representations are built on demand by the
Experiment 1 runner (scripts/experiments/run_exp1_scz_oml.py). This split keeps
the on-disk dataset small and lets reruns sweep alternative X values without
re-extracting weights.
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from model_xray.data import paths as _paths
from model_xray.data.ghrp_zoos import compile_mz_weights


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--mz-name", default="stl10",
                        help="GHRP zoo subdir name (default: stl10).")
    parser.add_argument("--force", action="store_true",
                        help="Recompile even if weights.npy already exists.")
    args = parser.parse_args()

    out_path = _paths.ghrp_mz_weights_path(args.mz_name)
    if os.path.exists(out_path) and not args.force:
        print(f"[01] {out_path} already exists; pass --force to recompile.")
        return

    print(f"[01] Compiling SCZ {args.mz_name} weights from {_paths.get_ghrp_dir()} ...")
    weights = compile_mz_weights(args.mz_name)
    print(f"[01] Compiled shape: {weights.shape}, dtype={weights.dtype}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.save(out_path, weights)
    print(f"[01] Wrote {out_path}")


if __name__ == "__main__":
    main()
