import numpy as np
import numpy.typing as npt
from zenml import ArtifactConfig, pipeline, step, get_step_context

from typing import Union
from typing_extensions import Annotated
from torch.nn import Module as torchModule
from tensorflow.keras import Model as tfModel
# from tensorflow.keras.models import Functional as tfFunctionalModel
# from tensorflow import keras
# tfFunctionalModel = keras.src.models.functional.Functional
# tfFunctionalModel = keras.Functional
from transformers import PreTrainedModel as HFPreTrainedModel
from transformers import TFPreTrainedModel as HFTFPreTrainedModel

from model_xray.config_classes import ModelRepos, PretrainedModelConfig
from model_xray.utils.model_utils import ret_pretrained_model_by_name
from model_xray.utils.model_utils import extract_weights as extract_weights_util
from model_xray.options import model_collections

@step(enable_cache=True)
def fetch_pretrained(pretrained_model_config: PretrainedModelConfig) -> (
    Annotated[
        Union[torchModule, tfModel, HFPreTrainedModel, HFTFPreTrainedModel],
        ArtifactConfig(
            name="fetched_pretrained_model",
        ),
    ]
):
    pretrained_model_name = pretrained_model_config.name
    model_repo = pretrained_model_config.repo

    print(f"Fetching pretrained model {pretrained_model_name} from {model_repo.value}")

    step_context = get_step_context()
    step_context.add_output_metadata(
        output_name="fetched_pretrained_model",
        metadata={
            "pretrained_model_config": pretrained_model_config.to_dict(),
        }
    )
    step_context.add_output_tags(
        output_name="fetched_pretrained_model",
        tags=["pretrained_model", model_repo.value.lower()]
    )

    model = ret_pretrained_model_by_name(model_name = pretrained_model_name, lib=model_repo.value)
    
    return model

@step(enable_cache=True)
def extract_weights(
        model: Union[torchModule, tfModel, HFPreTrainedModel, HFTFPreTrainedModel],
    ) -> (
    Annotated[
        np.ndarray,
        ArtifactConfig(
            name="extracted_weights",
        ),
    ]
):
    w = extract_weights_util(model=model)

    step_context = get_step_context()
    step_context.add_output_metadata(
        output_name="extracted_weights",
        metadata={
            "weights_properties": {
                "dtype": str(w.dtype).lower(),
                "amount": len(w)
            }
        }
    )
    step_context.add_output_tags(
        output_name="extracted_weights",
        tags=["weights_extracted", "weights_dl_model"]
    )

    print(f"Extracted {len(w)} weights from model")

    return w


@pipeline(enable_cache=True)
def fetch_pretrained_model_and_extract_weights(pretrained_model_config: PretrainedModelConfig):
    model = fetch_pretrained(pretrained_model_config=pretrained_model_config)
    w = extract_weights(model=model)

    return w

if __name__ == "__main__":
    model_names = model_collections['famous_le_10m']

    for model_name in model_names:
        fetch_pretrained_model_and_extract_weights(model_repo=ModelRepos.KERAS, pretrained_model_name=model_name)