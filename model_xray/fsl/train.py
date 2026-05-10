"""Few-shot training of OSL CNN / SRNet via the Siamese triplet wrapper.

The heavy lifting (architecture, triplet generator, fit loop, embedding head)
already lives in `model_xray.models.siamese.Siamese`. This module is a thin,
ZenML-free entrypoint with paper-aligned defaults so the per-experiment runners
in scripts/experiments/ stay short.

Hyperparameters follow Section 4.3 (Training Setup and Hyperparameters) of the paper:
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
    import tensorflow as tf
    # Enable memory growth: TF otherwise pre-allocates the entire GPU memory at
    # first op, which blocks SRNet at 256x256 (intermediate (n, 16, 256, 256)
    # activation tensors don't fit alongside TF's pre-allocated chunk on a
    # 9.8 GB RTX 3080). With memory_growth, TF allocates lazily and lets us
    # use the full physical memory.
    for gpu in tf.config.list_physical_devices('GPU'):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass  # Already initialized; safe to ignore.
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

    # Siamese expects a channel axis; add one for grayscale image reps that
    # ship as (n, h, w) from the data.attack_pipeline path.
    if X_train.ndim == 3:
        X_train = np.expand_dims(X_train, axis=-1)

    model = Siamese(
        img_input_shape=(imsize, imsize, n_channels),
        dist=dist,
        lr=lr,
        dropout_rate=dropout_rate,
        model_arch=model_arch,
    )
    # Pass training data as the validation triplet source — Siamese.fit_and_keep_refs
    # constructs validation triplets unconditionally and crashes on (None, None).
    # Using train as val is fine for the FSL setting (3+3 samples; eval happens via
    # test_all afterwards).
    model.fit_and_keep_refs(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose,
        callbacks=callbacks,
        size=triplets_per_class,
        test_data=(X_train, y_train),
    )
    return model
