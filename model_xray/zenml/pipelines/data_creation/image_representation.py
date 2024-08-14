import numpy as np
from zenml import ArtifactConfig, get_step_context, step, pipeline
from model_xray.zenml.pipelines.data_creation.data_classes import ModelRepos
from model_xray.zenml.pipelines.data_creation.model_attack import embed_payload_into_pretrained_weights_pipeline
from model_xray.config_classes import EmbedPayloadConfig, EmbedType, GrayscaleLastMBytesConfig, ImageRepConfig, ImageType, PayloadType, XLSBAttackConfig
from model_xray.utils.image_rep_utils import image_rep_map

from typing_extensions import Annotated


@step
def create_image_representation(
    data: np.ndarray,
    image_rep_config: ImageRepConfig
) -> (
    Annotated[
        np.ndarray,
        ArtifactConfig(
            name="image_representation",
        ),
    ]
):
    image_rep_func = image_rep_map[image_rep_config.image_type]

    if data.ndim == 1:
        passed_data = data.reshape(1, -1)
    else:
        passed_data = data

    image_rep = image_rep_func(passed_data, image_rep_config)

    step_context = get_step_context()
    step_context.add_output_metadata(
        output_name="image_representation",
        metadata={
            "image_properties": {
                "name":image_rep_config.image_type,
            },
            "image_config": image_rep_config.image_rep_config.to_dict(),
            "image_shape": image_rep.shape
        }
    )
    step_context.add_output_tags(
        output_name="image_representation",
        tags=["image_represented"]
    )

    return image_rep

@pipeline
def image_representation_from_attacked_pretrained_pipeline(
    pretrained_model_name:str,
    pretrained_model_repo:ModelRepos,

    embed_payload_config: EmbedPayloadConfig,

    image_rep_config: ImageRepConfig
):
    ew = embed_payload_into_pretrained_weights_pipeline(
        pretrained_model_name=pretrained_model_name,
        pretrained_model_repo=pretrained_model_repo,
        embed_payload_config=embed_payload_config
    )

    image_rep = create_image_representation(ew, image_rep_config)

    return image_rep

if __name__ == "__main__":
    pretrained_model_name = "MobileNet"
    pretrained_model_repo = ModelRepos.KERAS

    embedding_config = EmbedPayloadConfig(
        embed_type=EmbedType.X_LSB_ATTACK_FILL,
        embed_proc_config=XLSBAttackConfig(
            x=8,
            fill=True,
            msb=False,
            payload_type=PayloadType.RANDOM,
        )
    )

    image_representation_from_attacked_pretrained_pipeline(
        pretrained_model_name=pretrained_model_name,
        pretrained_model_repo=pretrained_model_repo,
        embed_payload_config=embedding_config,
        image_rep_config=ImageRepConfig(
            image_type=ImageType.GRAYSCALE_LAST_M_BYTES,
            image_rep_config=GrayscaleLastMBytesConfig(m=4)
        )
    )