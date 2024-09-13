from typing import Type, Union

import numpy as np
import numpy.typing as npt

from torch.nn import Module as torchModel
from tensorflow.keras import Model as tfkerasModel
from keras import Model as kerasModel
from transformers import PreTrainedModel as HFPreTrainedModel
from transformers import TFPreTrainedModel as HFTFPreTrainedModel

from tensorflow.data import Dataset as tfDataset

# torchModule = Type[torchModule]
# tfModel = Type[tfModel
# HFPreTrainedModel = Type[HFPreTrainedModel
# HFTFPreTrainedModel = Type[HFTFPreTrainedModel
# kerasModel = Union[kerasModel, tfkerasModel]

DL_MODEL_TYPE = Union[torchModel, kerasModel, tfkerasModel, HFPreTrainedModel, HFTFPreTrainedModel]

COVER_DATA_TYPE = Union[DL_MODEL_TYPE, np.ndarray]