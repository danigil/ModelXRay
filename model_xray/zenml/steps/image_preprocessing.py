import dataclasses
from typing import Optional
import numpy as np
import numpy.typing as npt

from zenml import ArtifactConfig, log_artifact_metadata, step

from model_xray.procedures.image_preprocess_procs import execute_image_preprocess
from model_xray.configs.models import *
from model_xray.procedures.image_rep_procs import image_rep_map
from model_xray.options import model_collections

from typing_extensions import Annotated

from PIL import Image

@step(enable_cache=True)
def image_preprocessing_step(
    image: np.ndarray,
    image_preprocess_config: ImagePreprocessConfig,

    log_metadata: Optional[bool] = True,
) -> (
    Annotated[
        np.ndarray,
        ArtifactConfig(
            name="image_preprocessed",
        ),
    ]
):  
    im_preprocessed = execute_image_preprocess(image, image_preprocess_config)

    if log_metadata:
        log_artifact_metadata(
            artifact_name="image_preprocessed",
            metadata={
                "image_preprocess_config": image_preprocess_config.model_dump(mode="json"),
            },
        )

    return im_preprocessed