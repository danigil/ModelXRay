
from concurrent.futures import ProcessPoolExecutor
import itertools
from typing_extensions import Annotated
import numpy as np
import numpy.typing as npt
from zenml import ArtifactConfig, get_step_context, step, pipeline

from model_xray.config_classes import EmbedPayloadConfig, EmbedType, PayloadType, XLSBAttackConfig
from model_xray.utils.mal_embedding_utils import embed_type_map

from model_xray.zenml.pipelines.data_creation.fetch_pretrained import fetch_pretrained_model_and_extract_weights
from ModelXRay.model_xray.config_classes import ModelRepos
from model_xray.options import model_collections


@step
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

    return weights_embedded

@pipeline
def embed_payload_into_pretrained_weights_pipeline(
    pretrained_model_name:str,
    pretrained_model_repo:ModelRepos,

    embed_payload_config: EmbedPayloadConfig
):
    print(f"Starting embedding {embed_payload_config.embed_proc_config.payload_type} payload into {pretrained_model_name} pretrained weights using {embed_payload_config.embed_type} embedding with x={embed_payload_config.embed_proc_config.x}")

    pretrained_model_weights = fetch_pretrained_model_and_extract_weights(
        model_repo=pretrained_model_repo,
        pretrained_model_name=pretrained_model_name
    )

    weights_embedded = embed_payload_into_weights(pretrained_model_weights, embed_payload_config)

    return weights_embedded

from model_xray.zenml.pipelines.data_creation.fetch_pretrained import fetch_pretrained_model_and_extract_weights

if __name__ == "__main__":

    model_names = model_collections['famous_le_100m']

    for i, model_name in enumerate(model_names):
        for x in range(1, 24):
            embedding_config = EmbedPayloadConfig(
                embed_type=EmbedType.X_LSB_ATTACK_FILL,
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

    # pretrained_model_names = [
    #     "MobileNet",
    #     "MobileNetV2",
    # ]

    # xs = [1, 2, 4, 8, 16]

    # for pretrained_model_name in pretrained_model_names:
    #     fetch_pretrained_model_and_extract_weights(pretrained_model_name=pretrained_model_name, model_repo=ModelRepos.KERAS)

    
    # executor = ProcessPoolExecutor(max_workers=5)
    # lst = list()

    # for pretrained_model_name, x in itertools.product(pretrained_model_names, xs):
    #     embedding_config = EmbedPayloadConfig(
    #         embed_type=EmbedType.X_LSB_ATTACK_FILL,
    #         embed_proc_config=XLSBAttackConfig(
    #             x=x,
    #             fill=True,
    #             msb=False,
    #             payload_type=PayloadType.RANDOM,
    #         )
    #     )

    #     lst.append(executor.submit(embed_payload_into_pretrained_weights_pipeline, pretrained_model_name, ModelRepos.KERAS, embedding_config))

    # for future in lst:
    #     future.result()
    # executor.shutdown()
    

    # pretrained_model_name = "MobileNet"
    # pretrained_model_repo = ModelRepos.KERAS

    # embedding_config = EmbedPayloadConfig(
    #     embed_type=EmbedType.X_LSB_ATTACK_FILL,
    #     embed_proc_config=XLSBAttackConfig(
    #         x=8,
    #         fill=True,
    #         msb=False,
    #         payload_type=PayloadType.RANDOM,
    #     )
    # )

    # embed_payload_into_pretrained_weights_pipeline(pretrained_model_name, pretrained_model_repo, embed_payload_config=embedding_config)