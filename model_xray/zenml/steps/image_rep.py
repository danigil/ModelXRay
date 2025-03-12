import dataclasses
from typing import Optional
import numpy as np
import numpy.typing as npt

from zenml import ArtifactConfig, get_step_context, step

from model_xray.configs.types import DL_MODEL_TYPE
from model_xray.configs.models import *
from model_xray.procedures.image_rep_procs import execute_image_rep_proc
from model_xray.options import model_collections

from typing_extensions import Annotated

@step(enable_cache=False)
def create_image_representation_step(
    data: Union[np.ndarray, DL_MODEL_TYPE],
    image_rep_config: ImageRepConfig,

    log_metadata: Optional[bool] = True,
) -> (
    Annotated[
        np.ndarray,
        ArtifactConfig(
            name="image_representation",
        ),
    ]
):
    image_rep = execute_image_rep_proc(data, image_rep_config)

    if log_metadata:
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