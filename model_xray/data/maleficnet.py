"""MaleficNet attacked-model ingestion + loaders (D4).

MaleficNet (Pagiux et al.) embeds an LDPC-coded payload into model weights via a
spread-spectrum attack. We use the upstream implementation vendored under
`external_code/maleficnet/` to generate per-(architecture, payload) checkpoints,
then convert them to GF image representations for OOD evaluation in Experiment 2.5.

D4 generation runs in a SEPARATE Python env (requirements-maleficnet.txt) because
the upstream maleficnet pinned deps conflict with the primary env's TF/Torch
stack. Loading the cached .npy/.npz files (this module's `ret_maleficnet_data`)
works fine in the primary env.
"""

from __future__ import annotations

from typing import Literal, Optional

import numpy as np

from model_xray.data import paths as _paths


# Architecture x payload matrix from external_code/maleficnet/ run defaults.
# vgg16 row is intentionally commented out below; that arch wasn't used in the
# paper's Exp 2.5 because its attacked checkpoint was unstable in our runs.
MAL_OPTIONS: dict[str, list[str]] = {
    "densenet121": ["stuxnet", "destover"],
    "resnet50":    ["stuxnet", "destover", "asprox", "bladabindi"],
    "resnet101":   ["stuxnet", "destover", "asprox", "bladabindi", "zeus-bank", "eq", "kovter", "cerber"],
    "vgg11":       ["stuxnet", "destover", "asprox", "bladabindi", "zeus-bank", "eq", "kovter", "cerber", "ardamax"],
    # "vgg16":     [...],
}


def ret_maleficnet_data(
    *,
    imsize: int = 50,
    image_rep: Optional[Literal["gf", "rgb", "s"]] = "gf",
    split_benign_mal: bool = False,
    flatten_imgs: bool = False,
    return_metadata: bool = False,
):
    """Load the cached MaleficNet image dataset and binary labels.

    With `return_metadata=True`, also return the per-row metadata array
    (cols `model_name, dataset, payload_name, imsize, image_rep`) so callers
    can group by (architecture, payload) for paper Table 2.
    """
    imgs = np.load(_paths.maleficnet_imgs_path(imsize=imsize, image_rep=image_rep))
    metadata = np.load(_paths.maleficnet_metadata_path(imsize=imsize, image_rep=image_rep), allow_pickle=True)

    y = np.array([0 if m["payload_name"] == "pre" else 1 for m in metadata])

    if flatten_imgs:
        imgs = imgs.reshape(imgs.shape[0], -1)
    if imgs.ndim > 2 and imgs.shape[1] == 1:
        imgs = np.squeeze(imgs, axis=1)

    if split_benign_mal:
        b = np.where(y == 0)[0]
        m = np.where(y == 1)[0]
        if return_metadata:
            return imgs[b], y[b], imgs[m], y[m], metadata[b], metadata[m]
        return imgs[b], y[b], imgs[m], y[m]
    if return_metadata:
        return imgs, y, metadata
    return imgs, y


def get_maleficnet_eval_datas(
    *,
    imsize: int = 50,
    image_rep: Optional[Literal["gf", "rgb", "s"]] = "gf",
    flatten_imgs: bool = False,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return {"maleficnet_benign": (X, y), "maleficnet_mal": (X, y)} for eval."""
    Xb, yb, Xm, ym = ret_maleficnet_data(
        imsize=imsize, image_rep=image_rep, split_benign_mal=True, flatten_imgs=flatten_imgs,
    )
    return {"maleficnet_benign": (Xb, yb), "maleficnet_mal": (Xm, ym)}
