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

from model_xray.config_classes import ModelRepos
from model_xray.utils.model_utils import ret_pretrained_model_by_name
from model_xray.utils.model_utils import extract_weights as extract_weights_util
from model_xray.options import model_collections

@step(enable_cache=True)
def fetch_pretrained(model_repo: ModelRepos, pretrained_model_name: str) -> (
    Annotated[
        Union[torchModule, tfModel, HFPreTrainedModel, HFTFPreTrainedModel],
        ArtifactConfig(
            name="fetched_pretrained_model",
            # run_metadata={"metadata_key": "metadata_value"},
            # tags=["tag_name"],
        ),
    ]
):
    print(f"Fetching pretrained model {pretrained_model_name} from {model_repo.value}")

    step_context = get_step_context()
    step_context.add_output_metadata(
        output_name="fetched_pretrained_model",
        metadata={
            "pretrained_model": {
                "name":pretrained_model_name.lower(),
                "lib": model_repo.value.lower()
            }
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
        model_repo: ModelRepos
    ) -> (
    Annotated[
        npt.NDArray,
        ArtifactConfig(
            name="extracted_weights",
        ),
    ]
):
    w = extract_weights_util(model=model, lib=model_repo.value)

    step_context = get_step_context()
    step_context.add_output_metadata(
        output_name="extracted_weights",
        metadata={
            "weights_properties": {
                "attacked":False,
                "lib": model_repo.value.lower(),
                "dtype": str(w.dtype).lower(),
                "amount": len(w)
            }
        }
    )
    step_context.add_output_tags(
        output_name="extracted_weights",
        tags=["weights_extracted", "weights_dl_model"]
    )

    print(f"Extracted {len(w)} weights from {model_repo.value} model")

    return w


@pipeline(enable_cache=True)
def fetch_pretrained_model_and_extract_weights(model_repo: ModelRepos, pretrained_model_name: str):
    model = fetch_pretrained(model_repo=model_repo, pretrained_model_name=pretrained_model_name)
    w = extract_weights(model=model, model_repo=model_repo)

    return w

if __name__ == "__main__":
    model_names = model_collections['famous_le_10m']

    for model_name in model_names:
        fetch_pretrained_model_and_extract_weights(model_repo=ModelRepos.KERAS, pretrained_model_name=model_name)