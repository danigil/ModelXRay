from zenml import ArtifactConfig, pipeline, step, get_step_context

from typing import Union
from typing_extensions import Annotated
from torch.nn import Module as torchModule
from tensorflow.keras import Model as tfModel
from transformers import PreTrainedModel as HFPreTrainedModel
from transformers import TFPreTrainedModel as HFTFPreTrainedModel

from model_xray.zenml.pipelines.data_creation.data_classes import ModelRepos
from model_xray.utils.model_utils import ret_pretrained_model_by_name

@step
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
            "pretrained_model_lib": model_repo.value.lower(),
            "pretrained_model_arch": pretrained_model_name.lower()
        }
    )
    step_context.add_output_tags(
        output_name="fetched_pretrained_model",
        tags=["pretrained_model", model_repo.value.lower()]
    )

    print("Adding metadata and tags to the output artifact")

    model = ret_pretrained_model_by_name(model_name = pretrained_model_name, lib=model_repo.value)
    
    return model



@pipeline
def fetch_pretrained_models(model_repo: ModelRepos, pretrained_model_name: str):
    model = fetch_pretrained(model_repo=model_repo, pretrained_model_name=pretrained_model_name)

if __name__ == "__main__":
    fetch_pretrained_models(model_repo=ModelRepos.KERAS, pretrained_model_name="MobileNet")