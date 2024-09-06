from typing import Union

import numpy as np
import numpy.typing as npt

from torch.nn import Module as torchModule
from tensorflow.keras import Model as tfModel
from transformers import PreTrainedModel as HFPreTrainedModel
from transformers import TFPreTrainedModel as HFTFPreTrainedModel

DL_MODEL_TYPE = Union[torchModule, tfModel, HFPreTrainedModel, HFTFPreTrainedModel]

COVER_DATA_TYPE = Union[DL_MODEL_TYPE, np.ndarray]