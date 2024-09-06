import dataclasses
from typing import Optional
import numpy as np

from zenml import ArtifactConfig, Model, get_pipeline_context, get_step_context, log_artifact_metadata, step, pipeline, log_model_metadata
from zenml.client import Client
from zenml.new.pipelines.pipeline import Pipeline

from model_xray.procedures.image_preprocess_procs import execute_image_preprocess
from model_xray.configs.models import *
from model_xray.procedures.image_rep_procs import image_rep_map
from model_xray.options import model_collections

from typing_extensions import Annotated

from PIL import Image

@step(enable_cache=True)
def image_preprocessing(
    image: np.ndarray,
    image_preprocess_config: ImagePreprocessConfig,
) -> (
    Annotated[
        np.ndarray,
        ArtifactConfig(
            name="image_preprocessed",
        ),
    ]
):
    if image.ndim == 3:
        image = image[0]
    
    im_preprocessed = execute_image_preprocess(image, image_preprocess_config)

    log_artifact_metadata(
        artifact_name="image_preprocessed",
        metadata={
            "image_preprocess_config": image_preprocess_config.model_dump(mode="json"),
        },
    )

    return im_preprocessed