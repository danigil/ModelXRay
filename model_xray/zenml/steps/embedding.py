
from concurrent.futures import ProcessPoolExecutor
import itertools
from typing_extensions import Annotated
import numpy as np
import numpy.typing as npt
from zenml import ArtifactConfig, get_step_context, log_artifact_metadata, step, pipeline

from model_xray.configs.models import *
from model_xray.procedures.embedding_procs import MalBytes, embed_type_map, execute_embedding_proc

from model_xray.options import model_collections
from model_xray.configs.types import COVER_DATA_TYPE

@step
def embed_payload_into_cover_data_step(
    cover_data: COVER_DATA_TYPE,
    embed_payload_config: EmbedPayloadConfig
) -> Annotated[
        COVER_DATA_TYPE,
        ArtifactConfig(
            name="stego_data",
        ),
    ]:

    stego_data = execute_embedding_proc(cover_data=cover_data, embed_payload_config=embed_payload_config)

    log_artifact_metadata(
        artifact_name="stego_data",
        metadata={
            "embed_payload_config": embed_payload_config.model_dump(mode="json"),
        },
    )

    return stego_data