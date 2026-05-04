"""D2: build the Famous Small CNNs (le-10M params) MCWA HDF5.

Loads each Keras pretrained model in SMALL_TRAIN + SMALL_TEST (Section 4.7.1),
extracts a flat float32 weight vector per architecture, and writes them as
named datasets in a single HDF5 file under $MODELXRAY_GHRP_DIR/famous_le_10m/.

Per-X attacks are applied at experiment time (run_exp2_famous_cnns.py).
"""

from __future__ import annotations

import argparse
import os

import h5py
import numpy as np

from model_xray.data import paths as _paths
from model_xray.data.pretrained_models import FAMOUS_SMALL_ALL, keras_weights


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--collection", default="famous_le_10m",
                        help="Subdir name under MODELXRAY_GHRP_DIR (default famous_le_10m).")
    parser.add_argument("--archs", nargs="*", default=list(FAMOUS_SMALL_ALL),
                        help="Architecture names to ingest (default: paper's 8 small CNNs).")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    out_dir = os.path.join(_paths.get_ghrp_dir(), args.collection)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "mcwa.h5")

    if os.path.exists(out_path) and not args.force:
        print(f"[02] {out_path} already exists; pass --force to overwrite.")
        return

    print(f"[02] Building {out_path} from {len(args.archs)} archs ...")
    with h5py.File(out_path, "w") as f:
        for name in args.archs:
            print(f"[02]   loading {name} ...")
            w = keras_weights(name).astype(np.float32)
            f.create_dataset(name, data=w[np.newaxis, :])  # (1, n) for run_experiment compatibility
            print(f"[02]   {name}: shape={w.shape}, dtype={w.dtype}")

    print(f"[02] Wrote {out_path}")


if __name__ == "__main__":
    main()
