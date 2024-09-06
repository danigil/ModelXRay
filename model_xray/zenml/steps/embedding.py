
from concurrent.futures import ProcessPoolExecutor
import itertools
from typing_extensions import Annotated
import numpy as np
import numpy.typing as npt
from zenml import ArtifactConfig, get_step_context, step, pipeline

from model_xray.configs.models import *
from model_xray.procedures.embedding_procs import MalBytes, embed_type_map, execute_embedding_proc

# from model_xray.zenml.pipelines.data_creation.fetch_pretrained import fetch_pretrained_model_and_extract_weights
from model_xray.options import model_collections

@step
def embed_payload_into_cover_data(
    cover_data,
    embed_payload_config: EmbedPayloadConfig
):
    stego_data = execute_embedding_proc(cover_data=cover_data, embed_payload_config=embed_payload_config)

    return stego_data