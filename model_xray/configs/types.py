"""Type aliases for the project's deep-learning model representations.

The heavy framework imports (`tensorflow.keras`, `transformers`, `torch`) are
deferred to first attribute access via PEP 562 module-level `__getattr__`.
This keeps lightweight users -- threshold detectors, NumPy-only feature
extractors, plot scripts -- from paying TF + transformers startup cost just
to import a downstream module that mentions one of these aliases in its
type hints.

The aliases are still used at runtime by `model_xray.configs.enums.ModelRepos.
determine_model_type` (isinstance checks) and `model_xray.utils.general_utils.
try_coerce_data` (get_args -> isinstance), so they need to evaluate to real
classes when actually accessed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Type, Union

import numpy as np
import numpy.typing as npt


if TYPE_CHECKING:
    # Static-analyzer-only imports; no runtime cost.
    from torch.nn import Module as _torchModule
    from tensorflow.keras import Model as _tfkerasModel
    from keras import Model as _kerasModel
    from transformers import PreTrainedModel as _HFPreTrainedModel
    from transformers import TFPreTrainedModel as _HFTFPreTrainedModel


_LAZY_NAMES = {
    "torchModel", "kerasModel", "tfkerasModel",
    "HFPreTrainedModel", "HFTFPreTrainedModel",
    "DL_MODEL_TYPE", "COVER_DATA_TYPE",
}
_cache: dict = {}


def _resolve(name: str):
    if name in _cache:
        return _cache[name]
    if name == "torchModel":
        from torch.nn import Module
        v = Type[Module]
    elif name == "tfkerasModel":
        from tensorflow.keras import Model
        v = Type[Model]
    elif name == "kerasModel":
        from keras import Model as KModel
        from tensorflow.keras import Model as TFKModel
        v = Union[KModel, TFKModel]
    elif name == "HFPreTrainedModel":
        from transformers import PreTrainedModel
        v = Type[PreTrainedModel]
    elif name == "HFTFPreTrainedModel":
        from transformers import TFPreTrainedModel
        v = Type[TFPreTrainedModel]
    elif name == "DL_MODEL_TYPE":
        v = Union[
            _resolve("torchModel"), _resolve("kerasModel"), _resolve("tfkerasModel"),
            _resolve("HFPreTrainedModel"), _resolve("HFTFPreTrainedModel"),
        ]
    elif name == "COVER_DATA_TYPE":
        v = Union[_resolve("DL_MODEL_TYPE"), np.ndarray]
    else:
        raise AttributeError(name)
    _cache[name] = v
    return v


def __getattr__(name: str):  # PEP 562
    if name in _LAZY_NAMES:
        return _resolve(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
