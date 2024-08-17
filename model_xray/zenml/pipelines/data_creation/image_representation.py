import dataclasses
from typing import Optional
import numpy as np
from zenml import ArtifactConfig, Model, get_pipeline_context, get_step_context, log_artifact_metadata, step, pipeline, log_model_metadata
from zenml.client import Client
from zenml.new.pipelines.pipeline import Pipeline

from model_xray.config_classes import ImagePreprocessConfig, ImageResamplingFilter, ModelRepos
from model_xray.zenml.pipelines.model_evaluation.eval_model import retrieve_model_weights
from model_xray.zenml.pipelines.data_creation.model_attack import embed_payload_into_pretrained_weights_pipeline
from model_xray.config_classes import EmbedPayloadConfig, EmbedType, GrayscaleLastMBytesConfig, ImageRepConfig, ImageType, PayloadType, XLSBAttackConfig
from model_xray.utils.image_rep_utils import image_rep_map
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
            "image_properties": {
                "name":image_rep_config.image_type,
                "shape": f"{image_rep.shape}"
            },
            "image_config": image_rep_config.image_rep_config.to_dict() if image_rep_config.image_rep_config is not None else "None",
            # "image_shape": image_rep.shape
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

    # pretrained_model_name:str,
    # pretrained_model_repo:ModelRepos,

    # image_rep_config: ImageRepConfig,

    # embed_payload_config: Optional[EmbedPayloadConfig] = None,

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
        size = image_preprocess_config.image_size,
        resample = image_preprocess_config.image_reshape_algo.to_pil_image_resampling_filter()
    )

    log_artifact_metadata(
        artifact_name="image_preprocessed",
        metadata={
            "image_preprocess_properties": {
                "shape": f"{image_preprocess_config.image_size}",
                "resample_algo": image_preprocess_config.image_reshape_algo
            }
        },
    )

    # log_artifact_metadata(
    #     artifact_name="image_preprocessed",
    #     metadata={
    #         "pretrained_model_info": {
    #             "name": pretrained_model_name,
    #             "repo": pretrained_model_repo.value
    #         },
    #         "image_preprocess_info": dataclasses.asdict(image_preprocess_config),
    #         "image_rep_info": dataclasses.asdict(image_rep_config),
    #         "embed_payload_info": dataclasses.asdict(embed_payload_config) if embed_payload_config is not None else "None",
    #     },
    #     artifact_version=get_preprocessed_image_version_name(
    #         pretrained_model_name=pretrained_model_name,
    #         pretrained_model_repo=pretrained_model_repo,
    #         image_preprocess_config=image_preprocess_config,
    #         image_rep_config=image_rep_config,
    #         embed_payload_config=embed_payload_config
    #     )
    # )

    return np.asarray(im_resized)

@pipeline(enable_cache=True)
def image_representation_from_pretrained_pipeline(
    pretrained_model_name:str,
    pretrained_model_repo:ModelRepos,

    image_rep_config: ImageRepConfig,

    embed_payload_config: Optional[EmbedPayloadConfig] = None,
):

    weights = retrieve_model_weights(
        pretrained_model_name=pretrained_model_name,
        pretrained_model_repo=pretrained_model_repo,

        embed_payload_config=embed_payload_config
    )

    image_rep = create_image_representation(weights, image_rep_config)

    return image_rep

def ret_pretrained_model_version_name(
    pretrained_model_name:str,
    pretrained_model_repo:ModelRepos,
):
    return f"name:{pretrained_model_name.lower()}_repo:{pretrained_model_repo.lower()}"
    
def ret_pretrained_model(
    pretrained_model_name:str,
    pretrained_model_repo:ModelRepos,
):
    return Model(
        name="model_pretrained",
        version=ret_pretrained_model_version_name(
            pretrained_model_name=pretrained_model_name,
            pretrained_model_repo=pretrained_model_repo
        )
    )

@pipeline
def _preprocessed_image_representation_from_pretrained_pipeline(
    pretrained_model_name:str,
    pretrained_model_repo:ModelRepos,

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
    # model_name="model_pretrained"
    # model_version=ret_pretrained_model_version_name(
    #     pretrained_model_name=pretrained_model_name,
    #     pretrained_model_repo=pretrained_model_repo
    # )
    # try:
    #     model_ver = Client().get_model_version(
    #         model_name_or_id=model_name,
    #         model_version_name_or_number_or_id=model_version,
    #     )
    #     was_created_in_this_run = False
    # except KeyError:
    #     model_ver = Client().create_model_version(
    #         model_name_or_id=model_name,
    #         name=model_version,
    #     )
    #     was_created_in_this_run = True

    # model = model_ver.to_model_class(was_created_in_this_run=was_created_in_this_run)

    
    # model = get_pipeline_context().model
    # model.log_metadata(
    #     metadata={
    #         "model_pretrained_info": {
    #             "name": pretrained_model_name,
    #             "repo": pretrained_model_repo
    #         },
    #     }
    # )
    # log_model_metadata(
    #     metadata={
    #         "model_pretrained_info": {
    #             "name": pretrained_model_name,
    #             "repo": pretrained_model_repo
    #         },
    #     }
    # )
    # model_param = model
    # model_param = Model(
    #     name=model_name,
    #     version=model_version,
    # )
    # image_representation_from_pretrained_pipeline.configure(model=model_param)
    image_rep = image_representation_from_pretrained_pipeline(
        pretrained_model_name=pretrained_model_name,
        pretrained_model_repo=pretrained_model_repo,

        image_rep_config=image_rep_config,
        embed_payload_config=embed_payload_config
    )

    # image_preprocessing.configure(model=model_param)
    image_rep_preprocessed = image_preprocessing(
        image=image_rep, image_preprocess_config=image_preprocess_config,
        # pretrained_model_name=pretrained_model_name,
        # pretrained_model_repo=pretrained_model_repo,

        # image_rep_config=image_rep_config,
        # embed_payload_config=embed_payload_config
    )

    # print(model.data_artifacts.keys())

    return image_rep_preprocessed



def _ret_pipeline_with_custom_model(
    *,
    model: Model = Model(name="model_pretrained", version="model_pretrained"),
    pipeline: Pipeline = _preprocessed_image_representation_from_pretrained_pipeline,

    **kwargs,
    # *args,
    # **kwargs
):
    return pipeline.with_options(model=model, **kwargs)

def ret_pipeline_with_pretrained_model(
    pipeline: Pipeline,
    # pretrained_model_name:str,
    # pretrained_model_repo:ModelRepos,
    **kwargs
):
    def wrap(**inner_kwargs):

        pretrained_model_name = inner_kwargs["pretrained_model_name"]
        pretrained_model_repo = inner_kwargs["pretrained_model_repo"]

        model = ret_pretrained_model(
            pretrained_model_name=pretrained_model_name,
            pretrained_model_repo=pretrained_model_repo
        )

        return _ret_pipeline_with_custom_model(
            model=model,
            pipeline=pipeline,
            # *args,
            # pretrained_model_name=pretrained_model_name,
            # pretrained_model_repo=pretrained_model_repo,
            **kwargs
        )(**inner_kwargs)

    return wrap

preprocessed_image_representation_from_pretrained_pipeline = ret_pipeline_with_pretrained_model(
    pipeline=_preprocessed_image_representation_from_pretrained_pipeline
)

if __name__ == "__main__":
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

    # im_res = image_representation_from_pretrained_pipeline(
    #     pretrained_model_name=pretrained_model_name,
    #     pretrained_model_repo=pretrained_model_repo,
    #     embed_payload_config=embedding_config,

    #     image_rep_config=ImageRepConfig(
    #         image_type=ImageType.GRAYSCALE_FOURPART,
    #         image_rep_config=None
    #     )
    # )

    model_names = model_collections['famous_le_100m'].union(model_collections['famous_le_10m'])
    # # model_names = ['MobileNet', 'MobileNetV2']

    for i, model_name in enumerate(model_names):
        for x in range(0, 24):
            if x == 0:
                embedding_config = None
            else:
                embedding_config = EmbedPayloadConfig.ret_x_lsb_attack_fill_config(x)

            try:
                im_res = preprocessed_image_representation_from_pretrained_pipeline(
                    pretrained_model_name=model_name,
                    pretrained_model_repo=ModelRepos.KERAS,
                    embed_payload_config = embedding_config,

                    image_preprocess_config = ImagePreprocessConfig(
                        image_size=(100, 100),
                        image_reshape_algo=ImageResamplingFilter.BICUBIC
                    ),

                    image_rep_config = ImageRepConfig(
                        image_type=ImageType.GRAYSCALE_FOURPART,
                        image_rep_config=None
                    )
                )
                # im = im_res.steps["create_image_representation"].output.load()

                # n, h, w = im.shape

                # if n != 1 or h!=w:
                #     print(f"!! Error creating img from {model_name} with x={x}: {im.shape}")
                    
                #     print(im.shape)

                #     exit(1)

            except Exception as e:
                print(f"!! Error creating img from {model_name} with x={x}: {e}")
                break


        print(f"~~ Finished {i+1}/{len(model_names)}: {model_name} ~~")