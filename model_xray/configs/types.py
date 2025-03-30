from typing import Type, Union

import numpy as np
import numpy.typing as npt

from torch.nn import Module as torchModel
from tensorflow.keras import Model as tfkerasModel
from keras import Model as kerasModel
from transformers import PreTrainedModel as HFPreTrainedModel
from transformers import TFPreTrainedModel as HFTFPreTrainedModel

# from tensorflow.data import Dataset as tfDataset

torchModel = Type[torchModel]
# tfModel = Type[tfModel]
HFPreTrainedModel = Type[HFPreTrainedModel]
HFTFPreTrainedModel = Type[HFTFPreTrainedModel]
kerasModel = Union[kerasModel, tfkerasModel]

# torchModel = Type["torch.nn.Module"]
# tfkerasModel = Type["tensorflow.keras.Model"]
# kerasModel = Type["keras.Model"]
# HFPreTrainedModel = Type["transformers.PreTrainedModel"]
# HFTFPreTrainedModel = Type["transformers.TFPreTrainedModel"]

DL_MODEL_TYPE = Union[torchModel, kerasModel, tfkerasModel, HFPreTrainedModel, HFTFPreTrainedModel]
# DL_MODEL_TYPE = Union["torch.nn.Module", "keras.Model", "tensorflow.keras.Model", "transformers.PreTrainedModel", "transformers.TFPreTrainedModel"]

COVER_DATA_TYPE = Union[DL_MODEL_TYPE, np.ndarray]