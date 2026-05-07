"""Famous pretrained CNN loaders for D2 / D3.

D2 = "Famous Small CNNs" (le-10M params), D3 = "Famous Large CNNs" (le-100M),
matching the paper's Section 4.7 splits. The constants below are the exact
architecture lists used in Experiments 2 and 2.5.

For Keras-side weight extraction we use the canonical `model_xray.procedures.
cover_data_procs.pretrained_model` helper. The ingest_*_dataset entry point
(scripts/data_creation/02_create_famous_small_cnns.py) loops these constants,
extracts flat float32 weight vectors, and persists them as a single MCWA HDF5
file keyed by architecture name.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch


# Section 4.7.1 + 4.7.4 splits used by Experiments 2 and 2.5.
SMALL_TRAIN = ("MobileNet", "NASNetMobile", "MobileNetV3Large")
SMALL_TEST = (
    "DenseNet121",
    "EfficientNetV2B0",
    "EfficientNetV2B1",
    "MobileNetV2",
    "MobileNetV3Small",
)

# ConvNeXt is intentionally omitted to match the paper line 4.7.4.
LARGE_TEST = (
    "DenseNet169",
    "DenseNet201",
    "EfficientNetV2B2",
    "EfficientNetV2B3",
    "EfficientNetV2S",
    "EfficientNetV2M",
    "InceptionV3",
    "InceptionResNetV2",
    "NASNetLarge",
    "ResNet50",
    "ResNet50V2",
    "ResNet101",
    "ResNet101V2",
    "ResNet152",
    "ResNet152V2",
    "Xception",
)

FAMOUS_SMALL_ALL = SMALL_TRAIN + SMALL_TEST
FAMOUS_LARGE_ALL = LARGE_TEST


def famous_split(collection: Literal["small", "large"]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return (train, test) architecture-name tuples for one famous-CNN collection."""
    if collection == "small":
        return SMALL_TRAIN, SMALL_TEST
    if collection == "large":
        return (), LARGE_TEST
    raise ValueError(f"unknown collection {collection!r}")


# -------------------- weight extraction --------------------


def keras_weights(model_name: str) -> np.ndarray:
    """Load a keras.applications model and return its concatenated float32 weights."""
    from model_xray.procedures.cover_data_procs import pretrained_model
    from model_xray.configs.enums import ModelRepos
    from model_xray.configs.models import PretrainedModelConfig

    cfg = PretrainedModelConfig(name=model_name, repo=ModelRepos.KERAS)
    model = pretrained_model(cfg)
    parts = [w.flatten().astype(np.float32) for w in model.get_weights() if w.dtype.kind == "f"]
    return np.concatenate(parts)


def torch_weights(model_name: str) -> np.ndarray:
    """Load a torchvision model with default weights and return its float32 weights."""
    import torchvision

    model = torchvision.models.get_model(model_name, weights="DEFAULT")
    parts = [w.cpu().detach().numpy().flatten()
             for w in model.state_dict().values() if w.dtype == torch.float32]
    return np.concatenate(parts)


def list_torch_classification_cnns() -> list[str]:
    """All torchvision classification models (used by ingest_ptms entry point)."""
    import torchvision

    keep = ("densenet", "convnext", "efficientnet", "mobilenet", "resnet", "vgg")
    names = torchvision.models.list_models(module=torchvision.models)
    return [m for m in names if any(k in m for k in keep)]
