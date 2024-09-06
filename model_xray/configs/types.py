from typing import Union

from torch.nn import Module as torchModule
from tensorflow.keras import Model as tfModel
from transformers import PreTrainedModel as HFPreTrainedModel
from transformers import TFPreTrainedModel as HFTFPreTrainedModel

DL_MODEL_TYPE = Union[torchModule, tfModel, HFPreTrainedModel, HFTFPreTrainedModel]