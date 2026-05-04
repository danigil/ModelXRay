"""Filesystem path resolution for the public reproduction artifact.

All large data lives outside the repo (model zoos, attacked models, payload binaries).
Locations are resolved from environment variables so the repo itself stays portable.

Required envs (only the ones touched by your reproduction step are checked):

    MODELXRAY_GHRP_DIR        : directory containing GHRP zoo subdirs (D1, D5)
    MODELXRAY_RESNET_MZ_ROOT  : directory containing the ResNet18-TinyImageNet
                                checkpoints + per-X .npz output dir (D5)
    MODELXRAY_MALEFICNET_DIR  : directory of MaleficNet attacked images (D4)
    MODELXRAY_PAYLOAD_FILE    : optional path to the malware payload (e.g. m_9d_ed_6d).
                                If unset, attacks default to a uniform pseudo-random
                                payload (the Experiment-4 random-payload setting).
"""

from __future__ import annotations

import os
import re
from typing import Literal, Optional


def _require_env(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        raise EnvironmentError(
            f"Environment variable {name} is required for this operation. "
            f"See model_xray/data/paths.py for the full env-var list."
        )
    return val


def get_ghrp_dir() -> str:
    return _require_env("MODELXRAY_GHRP_DIR")


def get_resnet_mz_root() -> str:
    return _require_env("MODELXRAY_RESNET_MZ_ROOT")


def get_maleficnet_dir() -> str:
    return _require_env("MODELXRAY_MALEFICNET_DIR")


def get_payload_file() -> Optional[str]:
    return os.environ.get("MODELXRAY_PAYLOAD_FILE") or None


def repo_root() -> str:
    """Repo root resolved from this file's location (model_xray/data/paths.py)."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))


def results_dir() -> str:
    """Cached-CSV results directory under the repo (committed)."""
    return os.path.join(repo_root(), "results")


def get_payload_name(payload_filepath: Optional[str]) -> str:
    return f'{payload_filepath.split("/")[-1]}' if payload_filepath else "rand"


def image_rep_str(image_rep: Optional[Literal["gf", "rgb", "bb", "bb1d", "bits", "s", "phis", "fcm"]] = None,
                  imsize: Optional[int] = 100) -> str:
    if imsize is None or imsize <= 0:
        imsize = ""
    return f"{image_rep}{imsize}" if image_rep is not None else "none"


def reverse_image_rep_str(s: str):
    if s == "none":
        return None, None
    m = re.match(r"([a-zA-Z]+)(\d+)", s)
    if not m:
        raise ValueError(f"Invalid image representation string: {s!r}")
    return m.group(1), int(m.group(2))


def ghrp_mz_weights_path(mz_name: str) -> str:
    """Cached float32 weights for one GHRP zoo (e.g. cifar10/mnist/stl10/svhn)."""
    return os.path.join(get_ghrp_dir(), mz_name, "weights.npy")


def resnet_mz_npz_path(
    mz_name: str,
    *,
    imsize: int = 50,
    payload_filepath: Optional[str] = None,
    image_rep: Optional[str] = "rgb",
    savename_prefix: str = "",
) -> str:
    """Per-X attacked image .npz output path for a ResNet18-TinyImageNet ingestion."""
    save_dir = os.path.join(get_resnet_mz_root(), mz_name)
    os.makedirs(save_dir, exist_ok=True)
    return os.path.join(
        save_dir,
        f"{savename_prefix}imgs_{image_rep_str(image_rep, imsize)}_{get_payload_name(payload_filepath)}.npz",
    )


def maleficnet_imgs_path(*, imsize: int = 50, image_rep: Optional[str] = "gf") -> str:
    return os.path.join(get_maleficnet_dir(), f"maleficnet_imgs{image_rep_str(image_rep, imsize)}.npy")


def maleficnet_metadata_path(*, imsize: int = 50, image_rep: Optional[str] = "gf") -> str:
    return os.path.join(get_maleficnet_dir(), f"maleficnet_metadata{image_rep_str(image_rep, imsize)}.npy")
