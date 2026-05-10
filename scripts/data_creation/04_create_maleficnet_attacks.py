"""D4: regenerate the MaleficNet attacked-model image dataset (Exp 2.5 OOD).

REQUIRES the isolated `maleficnet` env (see requirements-maleficnet.txt). The
upstream MaleficNet implementation pinned in external_code/maleficnet/
conflicts with the primary env's TF/Torch stack.

Pipeline:
  1. For each (architecture, payload) in MAL_OPTIONS, run the vendored
     MaleficNet injector (external_code/maleficnet/) to produce an attacked
     checkpoint .pt under $MODELXRAY_MALEFICNET_DOWNLOADS.
  2. Load every attacked + benign checkpoint, extract its float32 weight
     vector, render Grayscale-Fourpart images at imsize=50 (or --imsize).
  3. Persist the stack as $MODELXRAY_MALEFICNET_DIR/maleficnet_imgs<gf50>.npy
     plus a parallel metadata array.

Once produced, the image dataset is consumed by the Exp 2.5 runner in the
PRIMARY env via model_xray.data.maleficnet.ret_maleficnet_data.
"""

from __future__ import annotations

import argparse
import itertools
import os
from typing import Optional

import numpy as np

from model_xray.data import paths as _paths
from model_xray.data.maleficnet import MAL_OPTIONS


def _ingest_one(model_name: str, dataset: str, payload_name: str,
                imsize: int, image_rep: Optional[str], checkpoints_dir: str):
    import torch  # local import: only valid in the maleficnet env
    from model_xray.data.attack_pipeline import image_rep_ws
    from model_xray.data.ghrp_zoos import extract_weights_pytorch

    cp_path = os.path.join(checkpoints_dir, f"{model_name}_{dataset}_{payload_name}_model.pt")
    sd = torch.load(cp_path, weights_only=True, map_location="cpu")
    ws = extract_weights_pytorch(sd)
    imgs = image_rep_ws(np.expand_dims(ws, 0) if ws.ndim == 1 else ws,
                        imsize=imsize, image_rep=image_rep)
    if imgs.ndim == 2:
        imgs = np.expand_dims(imgs, 0)
    return imgs, {
        "model_name": model_name,
        "dataset": dataset,
        "payload_name": payload_name,
        "imsize": imsize,
        "image_rep": image_rep,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--datasets", nargs="*", default=["cifar10", "imagenet12"])
    parser.add_argument("--archs", nargs="*", default=list(MAL_OPTIONS.keys()),
                        help="Architectures to ingest. Defaults to MAL_OPTIONS.keys().")
    parser.add_argument("--imsize", type=int, default=50)
    parser.add_argument("--image-rep", default="gf",
                        help="Image representation backend (default: gf).")
    parser.add_argument("--checkpoints-dir", default=None,
                        help="Where injector .pt files live "
                             "(default: $MODELXRAY_MALEFICNET_DOWNLOADS/checkpoints).")
    parser.add_argument("--skip-attack-generation", action="store_true",
                        help="Assume attacked .pt files already exist; skip the inject step.")
    args = parser.parse_args()

    if args.checkpoints_dir is None:
        from model_xray.options import MALEFICNET_DATASET_DOWNLOAD_DIR
        args.checkpoints_dir = os.path.join(MALEFICNET_DATASET_DOWNLOAD_DIR, "checkpoints")

    if not args.skip_attack_generation:
        print("[04] Attack-generation step is delegated to external_code/maleficnet.")
        print("[04] Ensure the maleficnet injector has produced "
              f"{args.checkpoints_dir}/<arch>_<dataset>_<payload>_model.pt files.")
        print("[04] Re-run with --skip-attack-generation once generation is complete.")
        return

    imgs_all, metas_all = [], []
    for arch in args.archs:
        payloads = ["pre"] + MAL_OPTIONS.get(arch, [])
        for dataset, payload in itertools.product(args.datasets, payloads):
            print(f"[04] ingesting {arch}:{dataset}:{payload}")
            try:
                imgs, meta = _ingest_one(arch, dataset, payload,
                                         args.imsize, args.image_rep, args.checkpoints_dir)
                imgs_all.append(imgs)
                metas_all.append(meta)
            except FileNotFoundError as e:
                print(f"[04]   missing {e}; skipping.")

    imgs_final = np.concatenate(imgs_all, axis=0)
    out_imgs = _paths.maleficnet_imgs_path(imsize=args.imsize, image_rep=args.image_rep)
    out_meta = _paths.maleficnet_metadata_path(imsize=args.imsize, image_rep=args.image_rep)
    os.makedirs(os.path.dirname(out_imgs), exist_ok=True)
    np.save(out_imgs, imgs_final)
    np.save(out_meta, metas_all)
    print(f"[04] Wrote {out_imgs} (shape={imgs_final.shape}) + {out_meta}")


if __name__ == "__main__":
    main()
