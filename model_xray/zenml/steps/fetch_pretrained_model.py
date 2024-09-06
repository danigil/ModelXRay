import numpy as np
import numpy.typing as npt
from zenml import ArtifactConfig, pipeline, step, get_step_context

from typing import Union
from typing_extensions import Annotated


from model_xray.configs.models import PretrainedModelConfig
from model_xray.utils.model_utils import ret_pretrained_model_by_name
from model_xray.utils.model_utils import extract_weights as extract_weights_util
from model_xray.options import model_collections

from model_xray.configs.types import DL_MODEL_TYPE

@step(enable_cache=True)
def fetch_pretrained_model_step(pretrained_model_config: PretrainedModelConfig) -> (
    Annotated[
        DL_MODEL_TYPE,
        ArtifactConfig(
            name="fetched_pretrained_model",
        ),
    ]
):
    pretrained_model_name = pretrained_model_config.name
    model_repo = pretrained_model_config.repo

    train_dataset = pretrained_model_config.train_dataset

    print(f"Fetching pretrained model ({train_dataset}), arch: {pretrained_model_name} from {model_repo.value}")

    # step_context = get_step_context()
    # step_context.add_output_metadata(
    #     output_name=pretrained_model_config.model_dump_json(),
    #     metadata={
    #         "pretrained_model_config": pretrained_model_config,
    #     }
    # )
    # step_context.add_output_tags(
    #     output_name="fetched_pretrained_model",
    #     tags=["pretrained_model", model_repo.value.lower()]
    # )

    model = ret_pretrained_model_by_name(model_name = pretrained_model_name, lib=model_repo.value, train_dataset=train_dataset)
    
    return model

