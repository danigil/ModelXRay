
from concurrent.futures import ProcessPoolExecutor
import itertools
from typing_extensions import Annotated
import numpy as np
import numpy.typing as npt
from zenml import ArtifactConfig, get_step_context, step, pipeline

from model_xray.config_classes import EmbedPayloadConfig, EmbedType, PayloadType, PretrainedModelConfig, XLSBAttackConfig
from model_xray.utils.mal_embedding_utils import MalBytes, embed_type_map

from model_xray.zenml.pipelines.data_creation.fetch_pretrained import fetch_pretrained_model_and_extract_weights
from model_xray.config_classes import ModelRepos
from model_xray.options import model_collections


@step(enable_cache=True)
def embed_payload_into_weights(
    weights: np.ndarray,
    embed_payload_config: EmbedPayloadConfig
) -> (
    Annotated[
        np.ndarray,
        ArtifactConfig(
            name="embedded_weights",
        ),
    ]
):
    embed_func = embed_type_map[embed_payload_config.embed_type]

    mal_bytes_gen = MalBytes(embed_payload_config=embed_payload_config, appended_bytes=None)
    weights_embedded = embed_func(weights, embed_payload_config.embed_proc_config, mal_bytes_gen=mal_bytes_gen)

    step_context = get_step_context()
    step_context.add_output_metadata(
        output_name="embedded_weights",
        metadata={
            "embed_payload_config": embed_payload_config.to_dict(),
            "weights_properties": {
                "dtype": str(weights.dtype).lower(),
                "amount": len(weights)
            }
        }
    )
    step_context.add_output_tags(
        output_name="embedded_weights",
        tags=["weights_embedded", "weights_dl_model"]
    )

    return weights_embedded

@pipeline(enable_cache=True)
def embed_payload_into_pretrained_weights_pipeline(
    pretrained_model_config: PretrainedModelConfig,

    embed_payload_config: EmbedPayloadConfig
):
    pretrained_model_name = pretrained_model_config.name

    print(f"Starting embedding {embed_payload_config.embed_payload_type} payload into {pretrained_model_name} pretrained weights using {embed_payload_config.embed_type} embedding with x={embed_payload_config.embed_proc_config.x}")

    pretrained_model_weights = fetch_pretrained_model_and_extract_weights(
        pretrained_model_config=pretrained_model_config,
    )

    weights_embedded = embed_payload_into_weights(pretrained_model_weights, embed_payload_config)

    return weights_embedded

from model_xray.zenml.pipelines.data_creation.fetch_pretrained import fetch_pretrained_model_and_extract_weights

if __name__ == "__main__":

    model_names = model_collections['famous_le_100m']

    for i, model_name in enumerate(model_names):
        for x in range(1, 24):
            embedding_config = EmbedPayloadConfig(
                embed_type=EmbedType.X_LSB_ATTACK,
                embed_proc_config=XLSBAttackConfig(
                    x=x,
                    fill=True,
                    msb=False,
                    payload_type=PayloadType.RANDOM,
                )
            )

            try:
                embed_payload_into_pretrained_weights_pipeline(model_name, ModelRepos.KERAS, embedding_config)
            except Exception as e:
                print(f"!! Error embedding {model_name} with x={x}: {e}")

        print(f"~~ Finished {i+1}/{len(model_names)}: {model_name} ~~")