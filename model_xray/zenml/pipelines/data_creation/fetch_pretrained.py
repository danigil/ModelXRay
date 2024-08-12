from zenml import pipeline, step

from typing import Union
from torch import Module as torchModule
from tensorflow.keras import Model as tfModel
from transformers import PreTrainedModel as HFPreTrainedModel
from transformers import TFPreTrainedModel as HFTFPreTrainedModel

from model_xray.zenml.pipelines.data_creation.data_classes import ModelRepos
from model_xray.utils.model_utils import ret_pretrained_model_by_name

@step
def fetch_pretrained(model_repo: ModelRepos, pretrained_model_name: str) -> Union[torchModule, tfModel, HFPreTrainedModel, HFTFPreTrainedModel]:
    model = ret_pretrained_model_by_name(model_name = pretrained_model_name, lib=model_repo.value)
    
    return model