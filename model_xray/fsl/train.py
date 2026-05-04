"""Few-shot training of OSL CNN / SRNet via the Siamese triplet wrapper.

The heavy lifting (architecture, triplet generator, fit loop, embedding head)
already lives in `model_xray.models.siamese.Siamese`. This module is a thin,
ZenML-free entrypoint with paper-aligned defaults so the per-experiment runners
in scripts/experiments/ stay short.

Hyperparameters follow Section sec:training_setup of the paper:
  - dist: l2 (default), lr: 6e-5
  - mode "ub" (upper-bound): up to 100 epochs with the threshold callback
  - mode "es" (early-stop): 1 epoch
  - mode "st" (standard):   5 epochs
  - batch size: 16 for OSL CNN, 32 for SRNet
  - dropout 0.5 in OSL CNN
"""

from __future__ import annotations

from typing import Literal, Optional

import numpy as np


_MODE_EPOCHS = {"ub": 100, "es": 1, "st": 5}


def _default_batch_size(model_arch: str) -> int:
    return 32 if model_arch == "srnet" else 16


def train_fsl(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    model_arch: Literal["osl_siamese_cnn", "srnet"] = "osl_siamese_cnn",
    imsize: int = 100,
    n_channels: int = 1,
    mode: Literal["ub", "es", "st"] = "ub",
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    lr: float = 6e-5,
    dist: Literal["l2", "cosine"] = "l2",
    dropout_rate: float = 0.5,
    triplets_per_class: int = 10,
    train_loss_threshold_lower: float = 0.1,
    train_loss_threshold_upper: float = 0.4,
    verbose: int = 0,
    callbacks: Optional[list] = None,
):
    """Train one FSL detector.

    Returns a `Siamese` instance whose `.test_all(X, y)` returns {'centroid': acc, 'nn': acc}.
    """
    # Lazy imports keep this module importable without TF when only the API is needed.
    from model_xray.models.siamese import MyThresholdCallback, Siamese

    if epochs is None:
        epochs = _MODE_EPOCHS.get(mode, 100)
    if batch_size is None:
        batch_size = _default_batch_size(model_arch)
    if callbacks is None:
        callbacks = [MyThresholdCallback(
            ub_mode=(mode == "ub"),
            threshold_lower=train_loss_threshold_lower,
            threshold_upper=train_loss_threshold_upper,
        )]

    model = Siamese(
        img_input_shape=(imsize, imsize, n_channels),
        dist=dist,
        lr=lr,
        dropout_rate=dropout_rate,
        model_arch=model_arch,
    )
    model.fit_and_keep_refs(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose,
        callbacks=callbacks,
        size=triplets_per_class,
    )
    return model
