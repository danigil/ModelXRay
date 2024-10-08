import dataclasses
from typing import Optional
import numpy as np
import numpy.typing as npt

from zenml import ArtifactConfig, Model, get_pipeline_context, get_step_context, log_artifact_metadata, step, pipeline, log_model_metadata
from zenml.client import Client
from zenml.new.pipelines.pipeline import Pipeline

from model_xray.configs.types import DL_MODEL_TYPE
from model_xray.configs.models import *
# from model_xray.zenml.pipelines.model_evaluation.eval_model import retrieve_model_weights
# from model_xray.zenml.pipelines.data_creation.model_attack import embed_payload_into_pretrained_weights_pipeline
from model_xray.procedures.image_rep_procs import execute_image_rep_proc, image_rep_map
from model_xray.options import model_collections

from typing_extensions import Annotated

from PIL import Image

@step(enable_cache=True)
def create_image_representation_step(
    data: Union[np.ndarray, DL_MODEL_TYPE],
    image_rep_config: ImageRepConfig
) -> (
    Annotated[
        np.ndarray,
        ArtifactConfig(
            name="image_representation",
        ),
    ]
):
    image_rep = execute_image_rep_proc(data, image_rep_config)

    step_context = get_step_context()
    step_context.add_output_metadata(
        output_name="image_representation",
        metadata={
            'image_rep_config': image_rep_config.model_dump(mode="json"),
        }
    )
    step_context.add_output_tags(
        output_name="image_representation",
        tags=["image_representation"]
    )

    return image_rep