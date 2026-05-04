"""D3: build the Famous Large CNNs (le-100M params) MCWA HDF5.

Identical shape/format to the small-CNN cache (script 02), but on the
LARGE_TEST architecture list (Section 4.7.4). Used as the OOD test set for
Experiment 2 OOD plots and as additional benign references for Experiment 2.5.
"""

from __future__ import annotations

import argparse
import os

import h5py
import numpy as np

from model_xray.data import paths as _paths
from model_xray.data.pretrained_models import FAMOUS_LARGE_ALL, keras_weights


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--collection", default="famous_le_100m",
                        help="Subdir name under MODELXRAY_GHRP_DIR (default famous_le_100m).")
    parser.add_argument("--archs", nargs="*", default=list(FAMOUS_LARGE_ALL),
                        help="Architecture names to ingest (default: paper's 16 large CNNs).")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    out_dir = os.path.join(_paths.get_ghrp_dir(), args.collection)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "mcwa.h5")

    if os.path.exists(out_path) and not args.force:
        print(f"[03] {out_path} already exists; pass --force to overwrite.")
        return

    print(f"[03] Building {out_path} from {len(args.archs)} archs ...")
    with h5py.File(out_path, "w") as f:
        for name in args.archs:
            print(f"[03]   loading {name} ...")
            w = keras_weights(name).astype(np.float32)
            f.create_dataset(name, data=w[np.newaxis, :])
            print(f"[03]   {name}: shape={w.shape}, dtype={w.dtype}")

    print(f"[03] Wrote {out_path}")


if __name__ == "__main__":
    main()
