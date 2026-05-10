"""GHRP model-zoo loaders for D1 (SCZ STL10) and D5 (ResNet18-TinyImageNet).

The GHRP authors release each zoo as a `dataset.pt` containing per-checkpoint
state dicts. We flatten each checkpoint to a 1-D float32 vector, cache the
concatenation as `weights.npy`, and (for ResNet18-TinyImageNet) ingest the
attack+image-rep grid into per-X .npz files.
"""

from __future__ import annotations

import gc
import glob
import os
from typing import Callable, Dict, List, Literal, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T

from model_xray.data import paths as _paths


# -------------------- weight extraction --------------------


def flatten_sd(sd) -> torch.Tensor:
    """Concatenate every tensor in a state dict to a single 1-D tensor."""
    return torch.cat([v.flatten() for v in sd.values()]).ravel()


def extract_weights_pytorch(sd) -> np.ndarray:
    """Concatenate the float32 leaves of a state dict as a flat numpy array."""
    ws = [w.cpu().detach().numpy().flatten() for w in sd.values() if w.dtype == torch.float32]
    return np.concatenate(ws)


# -------------------- GHRP small-CNN zoos (D1) --------------------


def compile_mz_weights(mz_name: str, ghrp_dir: Optional[str] = None) -> np.ndarray:
    """Concatenate all train/test/val weights from a GHRP `dataset.pt`."""
    base = ghrp_dir or _paths.get_ghrp_dir()
    zoo_dir = os.path.join(base, mz_name)
    if not os.path.exists(zoo_dir):
        raise FileNotFoundError(f"GHRP zoo dir not found: {zoo_dir}")
    dataset_path = os.path.join(zoo_dir, "dataset.pt")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"GHRP dataset.pt not found: {dataset_path}")
    dataset = torch.load(dataset_path, map_location="cpu")
    parts = [
        dataset["trainset"].__get_weights__(),
        dataset["testset"].__get_weights__(),
    ]
    valset = dataset.get("valset", None)
    if valset is not None:
        parts.append(valset.__get_weights__())
    return torch.cat(parts, 0).numpy()


def load_mz_weights(mz_name: str, *, ghrp_dir: Optional[str] = None, fallback_to_compile: bool = True) -> np.ndarray:
    """Load the cached weights.npy; if missing, optionally compile from dataset.pt."""
    base = ghrp_dir or _paths.get_ghrp_dir()
    cache = os.path.join(base, mz_name, "weights.npy")
    if not os.path.exists(cache):
        if not fallback_to_compile:
            raise FileNotFoundError(cache)
        w = compile_mz_weights(mz_name, ghrp_dir=base)
        np.save(cache, w)
    return np.load(cache)


# -------------------- ResNet18-TinyImageNet ingestion (D5) --------------------


def iter_resnet_mz_checkpoints(
    root_dirs: Sequence[str],
    *,
    checkpoint_dirnames: Optional[Sequence[str]] = None,
    checkpoint_filename: str = "checkpoints",
    last_n_checkpoints: int = 1,
    resnet_mz_root: Optional[str] = None,
    float32_only: bool = True,
):
    """Yield flat numpy weight vectors for each model checkpoint.

    `root_dirs` are subdirectories under `MODELXRAY_RESNET_MZ_ROOT` (e.g.
    `tiny-imagenet_resnet18_kaiming_uniform_subset`). Each model directory
    underneath contains one or more `checkpoint_<N>/checkpoints` files.

    With `float32_only=True` (default) the yielded vector contains only the
    float32 leaves of the state dict -- this matches the paper's
    cache_resnet18.py / extract_weights_pytorch protocol and is what the
    threshold detectors B4-B7 expect. PyTorch BatchNorm modules also store
    an int64 `num_batches_tracked` counter (values up to ~50k) as part of
    the state dict; including those would poison the `max` / histogram
    features of the WeightValueDistribution detector.
    """
    base = resnet_mz_root or _paths.get_resnet_mz_root()
    for root in root_dirs:
        root_path = os.path.join(base, root)
        for model_dir in sorted(os.listdir(root_path)):
            model_path = os.path.join(root_path, model_dir)
            cps = checkpoint_dirnames
            if cps is None:
                cps = [os.path.basename(p) for p in glob.glob(os.path.join(model_path, "checkpoint_*"))]
                if last_n_checkpoints and last_n_checkpoints > 0:
                    cps = sorted(cps, key=lambda x: int(x.split("_")[-1]))[-last_n_checkpoints:]
            for cp in cps:
                cp_file = os.path.join(model_path, cp, checkpoint_filename)
                if not os.path.isfile(cp_file):
                    continue
                # Checkpoints were saved with CUDA tensors; map to CPU so this
                # loader works in a CPU-only torch venv (the primary GPU env
                # uses tensorflow[and-cuda] + torch+cpu — TF holds the GPU).
                sd = torch.load(cp_file, weights_only=True, map_location="cpu")
                if float32_only:
                    w = extract_weights_pytorch(sd)
                else:
                    w = flatten_sd(sd).detach().cpu().numpy()
                yield w
                del sd, w
                gc.collect()


# -------------------- Cover/stego PyTorch dataset (FSL training) --------------------


class StegoDataset(Dataset):
    """Pairs (cover_img, stego_img) with binary labels for triplet/contrastive FSL."""

    def __init__(self, cover_imgs: np.ndarray, stego_imgs: np.ndarray, transform: Optional[Callable] = None):
        self.cover_imgs = cover_imgs
        self.stego_imgs = stego_imgs
        self.transform = transform

    def __len__(self):
        return min(len(self.cover_imgs), len(self.stego_imgs))

    def __getitem__(self, idx):
        cover = self.cover_imgs[idx]
        stego = self.stego_imgs[idx]
        if self.transform:
            cover = self.transform(cover)
            stego = self.transform(stego)
        return {
            "cover": cover,
            "stego": stego,
            "label": [torch.tensor(0, dtype=torch.long), torch.tensor(1, dtype=torch.long)],
        }


_TRANSFORM_RGB = T.Compose([
    T.ToTensor(),
    T.ConvertImageDtype(torch.float),
    T.Normalize(mean=[0.5], std=[0.5]),
])

_TRANSFORM_1D = T.Compose([
    T.Lambda(lambda x: torch.from_numpy(x)),
    T.Lambda(lambda x: torch.swapaxes(x, 0, 1)),
])


def _transform_for(kind: Literal["rgb", "1d"]):
    return _TRANSFORM_1D if kind == "1d" else _TRANSFORM_RGB


def get_train_loader(cover_imgs, stego_imgs, batch_size: int = 4, transform_type: Literal["rgb", "1d"] = "rgb"):
    return DataLoader(
        StegoDataset(cover_imgs, stego_imgs, _transform_for(transform_type)),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=False,
    )


def get_val_loader(cover_imgs, stego_imgs, batch_size: int = 4, transform_type: Literal["rgb", "1d"] = "rgb"):
    return DataLoader(
        StegoDataset(cover_imgs, stego_imgs, _transform_for(transform_type)),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=False,
        drop_last=False,
    )
