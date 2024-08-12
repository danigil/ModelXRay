
from typing_extensions import Annotated
import numpy as np
import numpy.typing as npt
from zenml import ArtifactConfig, get_step_context, step, pipeline

from model_xray.config_classes import EmbedPayloadConfig, EmbedType, PayloadType, XLSBAttackConfig
from model_xray.utils.mal_embedding_utils import embed_type_map

from model_xray.zenml.pipelines.data_creation.fetch_pretrained import fetch_pretrained_model_and_extract_weights
from model_xray.zenml.pipelines.data_creation.data_classes import ModelRepos


@step
def embed_payload_into_weights(
    weights: npt.NDArray,
    
    embed_payload_config: EmbedPayloadConfig
) -> (
    Annotated[
        npt.NDArray,
        ArtifactConfig(
            name="embedded_weights",
        ),
    ]
):
    embed_func = embed_type_map[embed_payload_config.embed_type]

    weights_embedded = embed_func(weights, embed_payload_config.embed_proc_config)

    step_context = get_step_context()
    step_context.add_output_metadata(
        output_name="embedded_weights",
        metadata={
            "embedding_properties": {
                "name":embed_payload_config.embed_type,
            },
            "embedding_config": embed_payload_config.embed_proc_config.to_dict(),
            "weights_properties": {
                "attacked":True,
                "dtype": str(weights.dtype).lower(),
                "amount": len(weights)
            }
        }
    )
    step_context.add_output_tags(
        output_name="embedded_weights",
        tags=["weights_embedded", "weights_dl_model"]
    )

    print(f"Embedded {embed_payload_config.embed_proc_config.payload_type} payload into weights using {embed_payload_config.embed_type} embedding")

    return weights_embedded

@pipeline
def embed_payload_into_weights_pipeline(
    weights: npt.NDArray,
    embed_payload_config: EmbedPayloadConfig
) -> npt.NDArray:
    weights_embedded = embed_payload_into_weights(weights, embed_payload_config)

    return weights_embedded

if __name__ == "__main__":
    pretrained_model_name = "MobileNet"

    pretrained_model_weights = fetch_pretrained_model_and_extract_weights(
        model_repo=ModelRepos.KERAS,
        pretrained_model_name=pretrained_model_name
    )

    embedding_config = EmbedPayloadConfig(
        embed_type=EmbedType.X_LSB_ATTACK_FILL,
        embed_proc_config=XLSBAttackConfig(
            x=1,
            fill=True,
            msb=False,
            payload_type=PayloadType.RANDOM,
        )
    )

    embed_payload_into_weights_pipeline(weights=pretrained_model_weights, embed_payload_config=embedding_config)