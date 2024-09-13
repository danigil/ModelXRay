import dataclasses
from typing import Any, Optional
import numpy as np

from zenml import ArtifactConfig, Model, get_pipeline_context, get_step_context, log_artifact_metadata, step, pipeline, log_model_metadata
from zenml.client import Client

from model_xray.procedures.cover_data_procs import get_cover_data
from model_xray.configs.models import *
from model_xray.configs.types import COVER_DATA_TYPE

@step
def fetch_cover_data_step(
    cover_data_config: CoverDataConfig,
) -> (
    Annotated[
        COVER_DATA_TYPE,
        ArtifactConfig(
            name="cover_data",
        ),
    ]
):
    cover_data = get_cover_data(cover_data_config)

    log_artifact_metadata(
        artifact_name="cover_data",
        metadata={
            "cover_data_config": cover_data_config.model_dump(mode="json"),
        },
    )

    return cover_data