"""High-level X-LSB-Attack-Fill -> image / phis pipeline.

This is the central function used by both the runtime/memory study (D6 synthetic
tensors) and the GHRP zoo ingestion (D5). It chains:

    raw float32 weights
      -> optional X-LSB-Attack-Fill via canonical procedures.embedding_procs
      -> either Yin-style 92-d phi feature vector ("phis")
         or a paper image representation (default "gf" = Grayscale-Fourpart)
         followed by skimage-backed resize/normalize.

Image representation backends are dispatched to model_xray.procedures.image_rep_procs.
"""

from __future__ import annotations

from typing import Literal, Optional

import numpy as np

from model_xray.baselines.b2_yin import calc_phis_all
from model_xray.configs.enums import PayloadType
from model_xray.configs.models import (
    EmbedPayloadConfig,
    EmbedPayloadMetadata,
    ImagePreprocessConfig,
    SKImagePreprocessConfig,
    XLSBAttackConfig,
    ret_na_val,
)
from model_xray.procedures.embedding_procs import MalBytes, x_lsb_attack
from model_xray.procedures.image_preprocess_procs import execute_image_preprocess
from model_xray.procedures.image_rep_procs import (
    _bitbytes,
    _bitbytes1d,
    _bits_1d,
    _grayscale_fourpart,
    _rgb,
    _straight,
)


ImageRep = Literal["gf", "rgb", "bb", "bb1d", "bits", "s", "phis", "fcm"]


def image_rep_ws(
    ws: np.ndarray,
    imsize: Optional[int] = 100,
    image_rep: Optional[ImageRep] = "gf",
) -> np.ndarray:
    """Apply image representation `image_rep` to a (n, n_weights) float32 array."""
    if image_rep == "gf":
        imgs = _grayscale_fourpart(ws)
    elif image_rep == "rgb":
        imgs = _rgb(ws)
    elif image_rep == "bb":
        imgs = _bitbytes(ws)
    elif image_rep == "bb1d":
        imgs = _bitbytes1d(ws)
    elif image_rep == "bits":
        imgs = _bits_1d(ws)
    elif image_rep == "s":
        imgs = _straight(ws)
    elif image_rep == "fcm":
        imgs = ws
    else:
        raise ValueError(f"Unknown image representation: {image_rep!r}")

    if imsize is None or imsize <= 0:
        return imgs

    pp_cfg = ImagePreprocessConfig(
        image_height=imsize,
        image_width=imsize,
        image_preprocess_config=SKImagePreprocessConfig(),
    )
    return execute_image_preprocess(imgs, pp_cfg)


def img_pp_xlsb_attack(
    ws: np.ndarray,
    imsize: int = 256,
    x: int = 0,
    payload_filepath: Optional[str] = None,
    image_rep: Optional[ImageRep] = "gf",
    attack_chunk_size: Optional[int] = None,
) -> np.ndarray:
    """Apply X-LSB-Attack-Fill at severity X (no-op if X <= 0), then represent.

    Parameters
    ----------
    ws : (n_models, n_weights) float32
    imsize : preprocess target size (ignored when image_rep == "phis")
    x : number of LSB mantissa bits to overwrite. 0 means no attack (benign).
    payload_filepath : path to malware payload bytes, or None for a uniform
        pseudo-random payload (Experiment-4 setting).
    image_rep : "phis" -> 92-dim Yin et al. feature vector; otherwise an image rep.
    attack_chunk_size : weights-per-chunk passed to procedures.x_lsb_attack.
    """
    if x > 0:
        cfg = XLSBAttackConfig(x=x)
        emb_cfg = EmbedPayloadConfig(
            embed_payload_type=(PayloadType.RANDOM if payload_filepath is None else PayloadType.BINARY_FILE),
            embed_proc_config=cfg,
            embed_payload_metadata=EmbedPayloadMetadata(
                payload_filepath=payload_filepath if payload_filepath is not None else ret_na_val(),
            ),
        )
        mal_bytes = MalBytes(embed_payload_config=emb_cfg, appended_bytes=None)
        ws = x_lsb_attack(
            ws,
            x_lsb_attack_config=cfg,
            mal_bytes_gen=mal_bytes,
            chunk_size=attack_chunk_size,
        )

    if image_rep == "phis":
        return calc_phis_all(ws)
    if image_rep is not None:
        out = image_rep_ws(ws, imsize=imsize, image_rep=image_rep)
        if out.ndim == 2:
            out = np.expand_dims(out, axis=0)
        return out
    return ws
