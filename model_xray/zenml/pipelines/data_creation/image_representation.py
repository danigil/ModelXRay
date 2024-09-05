import dataclasses
from typing import Optional
import numpy as np
from zenml import ArtifactConfig, Model, get_pipeline_context, get_step_context, log_artifact_metadata, step, pipeline, log_model_metadata
from zenml.client import Client
from zenml.new.pipelines.pipeline import Pipeline

from model_xray.config_classes import GrayscaleThreepartWeightedAvgConfig, ImagePreprocessConfig, ImageResamplingFilter, ModelRepos, PretrainedModelConfig
from model_xray.zenml.pipelines.model_evaluation.eval_model import retrieve_model_weights
from model_xray.zenml.pipelines.data_creation.model_attack import embed_payload_into_pretrained_weights_pipeline
from model_xray.config_classes import EmbedPayloadConfig, EmbedType, GrayscaleLastMBytesConfig, ImageRepConfig, ImageType, PayloadType, XLSBAttackConfig
from model_xray.procedures.image_rep_procs import image_rep_map
from model_xray.options import model_collections

from typing_extensions import Annotated

from PIL import Image


@step(enable_cache=True)
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
            'image_rep_config': image_rep_config.to_dict(),
        }
    )
    step_context.add_output_tags(
        output_name="image_representation",
        tags=["image_represented"]
    )

    return image_rep

def get_preprocessed_image_version_name(
    pretrained_model_name:str,
    pretrained_model_repo:ModelRepos,
    image_preprocess_config: ImagePreprocessConfig,
    image_rep_config: ImageRepConfig,
    embed_payload_config: Optional[EmbedPayloadConfig] = None
):
    return (f"name:{pretrained_model_name}_"
            f"repo:{pretrained_model_repo}_"
            f"im_type:{image_rep_config.image_type}_"
            f"im_cfg:{str(image_rep_config.image_rep_config).lower()}_" if image_rep_config.image_rep_config is not None else ""
            f"im_size:{image_preprocess_config.image_size[0]}x{image_preprocess_config.image_size[1]}_"
            f"reshape_algo:{image_preprocess_config.image_reshape_algo}_"
            f"attack_type:{embed_payload_config.embed_type}" if embed_payload_config is not None else ""
            f"attack_cfg:{str(embed_payload_config.embed_proc_config).lower()}" if embed_payload_config is not None and embed_payload_config.embed_proc_config is not None else ""
            )


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
    im = Image.fromarray(image)

    im_resized = im.resize(
        size = (image_preprocess_config.image_height, image_preprocess_config.image_width),
        resample = image_preprocess_config.image_reshape_algo.to_pil_image_resampling_filter()
    )

    log_artifact_metadata(
        artifact_name="image_preprocessed",
        metadata={
            "image_preprocess_config": image_preprocess_config.to_dict(),
        },
    )

    return np.asarray(im_resized)

@pipeline(enable_cache=True)
def image_representation_from_pretrained_pipeline(
    pretrained_model_config: PretrainedModelConfig,

    image_rep_config: ImageRepConfig,

    embed_payload_config: Optional[EmbedPayloadConfig] = None,
):
    
    weights = retrieve_model_weights(
        pretrained_model_config=pretrained_model_config,

        embed_payload_config=embed_payload_config
    )

    image_rep = create_image_representation(weights, image_rep_config)

    return image_rep

def ret_pretrained_model_version_name(
    pretrained_model_config: PretrainedModelConfig,
):
    return f"name:{pretrained_model_config.name}_repo:{pretrained_model_config.repo}"
    
def ret_pretrained_model(
    pretrained_model_config: PretrainedModelConfig,
):
    return Model(
        name="model_pretrained",
        version=ret_pretrained_model_version_name(
            pretrained_model_config=pretrained_model_config,
        )
    )

@pipeline
def _preprocessed_image_representation_from_pretrained_pipeline(
    pretrained_model_config: PretrainedModelConfig,

    image_rep_config: ImageRepConfig,
    image_preprocess_config: ImagePreprocessConfig,

    embed_payload_config: Optional[EmbedPayloadConfig] = None,
) -> (
    Annotated[
        np.ndarray,
        ArtifactConfig(
            name="image_preprocessed",
        ),
    ]
):
    image_rep = image_representation_from_pretrained_pipeline(
        pretrained_model_config=pretrained_model_config,

        image_rep_config=image_rep_config,
        embed_payload_config=embed_payload_config
    )

    image_rep_preprocessed = image_preprocessing(
        image=image_rep, image_preprocess_config=image_preprocess_config,
    )

    return image_rep_preprocessed


def _ret_pipeline_with_custom_model(
    *,
    model: Model = Model(name="model_pretrained", version="model_pretrained"),
    pipeline: Pipeline = _preprocessed_image_representation_from_pretrained_pipeline,

    **kwargs,
):
    return pipeline.with_options(model=model, **kwargs)

def ret_pipeline_with_pretrained_model(
    pipeline: Pipeline,
    **kwargs
):
    def wrap(**inner_kwargs):

        pretrained_model_config = inner_kwargs["pretrained_model_config"]

        model = ret_pretrained_model(
            pretrained_model_config=pretrained_model_config,
        )

        return _ret_pipeline_with_custom_model(
            model=model,
            pipeline=pipeline,
            **kwargs
        )(**inner_kwargs)

    return wrap

preprocessed_image_representation_from_pretrained_pipeline = ret_pipeline_with_pretrained_model(
    pipeline=_preprocessed_image_representation_from_pretrained_pipeline,
    enable_cache=True,
)

if __name__ == "__main__":


    # x= 5
    

    # im_res = preprocessed_image_representation_from_pretrained_pipeline(
    #     pretrained_model_config=PretrainedModelConfig(
    #         name='MobileNet',
    #         repo=ModelRepos.KERAS
    #     ),
    #     embed_payload_config = EmbedPayloadConfig.ret_x_lsb_attack_fill_config(x),

    #     image_preprocess_config = ImagePreprocessConfig(
    #         image_height=100,
    #         image_width=100,
    #         image_reshape_algo=ImageResamplingFilter.BICUBIC
    #     ),

    #     image_rep_config = ImageRepConfig(
    #         image_type=ImageType.GRAYSCALE_FOURPART,
    #         image_rep_config=None
    #     )
    # )

    # model_names = model_collections['famous_le_100m'].union(model_collections['famous_le_10m'])
    model_names = model_collections['famous_le_10m']
    # # model_names = ['MobileNet', 'MobileNetV2']

    for i, model_name in enumerate(model_names):
        for x in range(1, 24):
            if x == 0:
                embedding_config = None
            else:
                # embedding_config = EmbedPayloadConfig.ret_random_x_lsb_attack_fill_config(x)
                embedding_config = EmbedPayloadConfig.ret_filebytes_x_lsb_attack_fill_config(
                    x=x,
                    payload_filepath='/mnt/exdisk2/model_xray/malware_payloads/m_77e05'
                )

            try:
                im_res = preprocessed_image_representation_from_pretrained_pipeline(
                    pretrained_model_config=PretrainedModelConfig(
                        name=model_name,
                        repo=ModelRepos.KERAS
                    ),
                    embed_payload_config = embedding_config,

                    image_preprocess_config = ImagePreprocessConfig(
                        image_height=100,
                        image_width=100,
                        image_reshape_algo=ImageResamplingFilter.BICUBIC
                    ),
                    image_rep_config = ImageRepConfig(
                        image_type=ImageType.GRAYSCALE_FOURPART,
                    )
                    # image_rep_config = ImageRepConfig(
                    #     image_type=ImageType,
                    #     image_rep_config=GrayscaleThreepartWeightedAvgConfig()
                    # )
                )

            except Exception as e:
                print(f"!! Error creating img from {model_name} with x={x}: {e}")
                break


        print(f"~~ Finished {i+1}/{len(model_names)}: {model_name} ~~")