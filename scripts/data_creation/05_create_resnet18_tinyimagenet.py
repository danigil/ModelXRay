"""D5: build the ResNet18-TinyImageNet benign-weight cache (Exp 4 / Yin acc).

Walks the GHRP ResNet18-TinyImageNet checkpoint tree under
$MODELXRAY_RESNET_MZ_ROOT, takes the last `--last-n-checkpoints` per model
(default 1, matching the original ingestion), flattens each to float32, and
stacks the lot into a single .npy of shape (n_models, n_weights).

Per-X attacks and image reps are applied at experiment time
(run_exp4_resnet18_yin.py). The benign cache makes the repeated 5-fold x 5-rep
CV sweep over X reasonably fast.
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from model_xray.data import paths as _paths
from model_xray.data.ghrp_zoos import iter_resnet_mz_checkpoints


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--root-dirs", nargs="+",
                        default=["tiny-imagenet_resnet18_kaiming_uniform_subset"],
                        help="Subdirs under MODELXRAY_RESNET_MZ_ROOT to scan.")
    parser.add_argument("--checkpoint-dirnames", nargs="*", default=["checkpoint_000060"],
                        help="Checkpoint subdir names to load per model (default: checkpoint_000060).")
    parser.add_argument("--last-n-checkpoints", type=int, default=1)
    parser.add_argument("--mz-name", default="tiny-imagenet_resnet18",
                        help="Output subdir name under MODELXRAY_RESNET_MZ_ROOT.")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    out_dir = os.path.join(_paths.get_resnet_mz_root(), args.mz_name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "weights.npy")

    if os.path.exists(out_path) and not args.force:
        print(f"[05] {out_path} already exists; pass --force to recompile.")
        return

    print(f"[05] Walking checkpoint dirs {args.root_dirs} ...")
    weights = list(iter_resnet_mz_checkpoints(
        args.root_dirs,
        checkpoint_dirnames=args.checkpoint_dirnames,
        last_n_checkpoints=args.last_n_checkpoints,
    ))
    if not weights:
        raise RuntimeError("No checkpoint files found; verify --root-dirs and "
                           "--checkpoint-dirnames against your zoo layout.")
    arr = np.stack(weights, axis=0).astype(np.float32)
    print(f"[05] Stacked weight tensor shape={arr.shape}, dtype={arr.dtype}")

    np.save(out_path, arr)
    print(f"[05] Wrote {out_path}")


if __name__ == "__main__":
    main()
