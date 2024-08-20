from typing import Annotated, Dict, Iterable, List, Optional, Set, Tuple, TypeVar
import numpy as np

import itertools

import pandas as pd
from zenml import ExternalArtifact, step
from zenml import ArtifactConfig, Model, get_pipeline_context, get_step_context, log_artifact_metadata, step, pipeline, log_model_metadata
from zenml.client import Client
from zenml.new.pipelines.pipeline import Pipeline
from zenml.artifacts.utils import load_artifact_from_response

from model_xray.config_classes import *
from model_xray.zenml.pipelines.data_creation.image_representation import ret_pretrained_model_version_name
from model_xray.options import model_collections
from model_xray.utils.general_utils import flatten_dict, query_df_using_dict

from typing import TypedDict

PreprocessedImagesDataDict = TypedDict('PreprocessedImagesDataDict', {
    'preprocessed_image': np.ndarray,
    'metadata': PreprocessedImageLineage,
    'artifact_uri': str,
})

PipelineRunName = TypeVar("PipelineRunName", bound=str)

def _get_preprocessed_images(
    pretrained_model_config: PretrainedModelConfig,
    preprocessed_img_cfgs: Optional[Set[PreprocessedImageLineage]] = None
) -> Dict[PipelineRunName, PreprocessedImagesDataDict]:
    accum_metadata_keys = [
        'pretrained_model_config', 'image_rep_config', 'image_preprocess_config', 'embed_payload_config',
    ]

    client = Client()

    model = client.get_model_version(
        model_name_or_id="model_pretrained",
        model_version_name_or_number_or_id=ret_pretrained_model_version_name(pretrained_model_config=pretrained_model_config)
    )

    ret = {}

    prev_metadatas = set()

    for pipeline_run_name, pipeline_run_reponse in model.pipeline_runs.items():
        if preprocessed_img_cfgs is not None and len(preprocessed_img_cfgs) == len(prev_metadatas):
            break

        curr_steps = pipeline_run_reponse.steps
        if 'image_preprocessing' not in curr_steps.keys():
            continue

        metadata_dict ={}

        for step_name, step_response in curr_steps.items():
            curr_step_metadata = step_response.output.run_metadata

            for accum_metadata_key in accum_metadata_keys:
                if accum_metadata_key in curr_step_metadata:
                    metadata_dict[accum_metadata_key] = curr_step_metadata[accum_metadata_key].value

        curr_preprocessed_image_lineage = PreprocessedImageLineage.from_dict(metadata_dict)
        if curr_preprocessed_image_lineage in prev_metadatas or (preprocessed_img_cfgs is not None and curr_preprocessed_image_lineage not in preprocessed_img_cfgs):
            continue

        curr_preprocessed_image = curr_steps['image_preprocessing'].output.load()
        if curr_preprocessed_image is None:
            continue

        ret[pipeline_run_name] = {
            'preprocessed_image': curr_preprocessed_image,
            'metadata': curr_preprocessed_image_lineage,
            'artifact_uri': curr_steps['image_preprocessing'].output.uri,
        }

        prev_metadatas.add(curr_preprocessed_image_lineage)

    if preprocessed_img_cfgs is not None and len(preprocessed_img_cfgs) != len(prev_metadatas):
        raise ValueError(f"get_preprocessed_images: Not all requested preprocessed images were found. requested: {len(preprocessed_img_cfgs)}, found: {len(prev_metadatas)}")

    return ret

def _ret_preprocessed_images(
    preprocessed_img_cfgs: Set[PreprocessedImageLineage],
):
    ret = {}

    requested_pretrained_models = set([preprocessed_img_cfg.pretrained_model_config for preprocessed_img_cfg in preprocessed_img_cfgs])
    for pretrained_model_config in requested_pretrained_models:
        preprocessed_images = _get_preprocessed_images(
            pretrained_model_config=pretrained_model_config,
            preprocessed_img_cfgs=set([preprocessed_img_cfg for preprocessed_img_cfg in  preprocessed_img_cfgs if preprocessed_img_cfg.pretrained_model_config == pretrained_model_config])
        )

        for pipeline_run_name, preprocessed_image_and_metadata_dict in preprocessed_images.items():
            ret[preprocessed_image_and_metadata_dict['metadata']] = preprocessed_image_and_metadata_dict['preprocessed_image']
    
    return ret

def _ret_preprocessed_images_from_registry(
    preprocessed_images_registry: pd.DataFrame,
    preprocessed_img_cfgs: List[PreprocessedImageLineage],
):
    ret = {}
    for preprocessed_img_cfg in preprocessed_img_cfgs:
        query_dict = preprocessed_img_cfg.to_flat_dict()

        df_query = query_df_using_dict(df=preprocessed_images_registry, query_dict=query_dict)
        if len(df_query) == 0:
            raise ValueError(f"get_preprocessed_images_from_registry: No preprocessed image found for {preprocessed_img_cfg}")

        artifact_uri = df_query['artifact_uri'].values[0]
        preprocessed_img = np.load(artifact_uri)

        ret[preprocessed_img_cfg] = preprocessed_img

    return ret

        

def _compile_preprocessed_images_dataset(
    preprocessed_img_cfgs: Set[PreprocessedImageLineage],
    preprocessed_images_registry: Optional[pd.DataFrame] = None,
) -> Tuple[
    Annotated[np.ndarray, "X"],
    Annotated[np.ndarray, "y"],
]:
    if preprocessed_images_registry is not None:
        preprocessed_images_dict = _ret_preprocessed_images_from_registry(
            preprocessed_images_registry=preprocessed_images_registry,
            preprocessed_img_cfgs=preprocessed_img_cfgs
        )
    else:
        preprocessed_images_dict = _ret_preprocessed_images(preprocessed_img_cfgs=preprocessed_img_cfgs)

    cfgs, imgs = zip(*sorted(preprocessed_images_dict.items(), key=lambda x: str(x[0].to_dict())))

    X = np.array(imgs)
    y = np.array([preprocessed_img_cfg.label() for preprocessed_img_cfg in cfgs])

    return X, y

def compile_preprocessed_images_dataset(
    pretrained_model_configs: List[PretrainedModelConfig],
    x_values: Set[Union[int, None]],
    image_preprocess_config: ImagePreprocessConfig
) -> Tuple[
    Annotated[np.ndarray, "X"],
    Annotated[np.ndarray, "y"],
]:
    preprocessed_img_cfgs = [PreprocessedImageLineage.ret_default_preprocessed_image_w_x_lsb_attack(
        pretrained_model_config=pretrained_model_config,
        x=x, image_preprocess_config=image_preprocess_config)
        for
        pretrained_model_config,x,image_preprocess_config
        in itertools.product(pretrained_model_configs, x_values, [image_preprocess_config])
    ]

    df_img_registry = Client().get_artifact_version("preprocessed_images_registry").load()

    X, y = _compile_preprocessed_images_dataset(preprocessed_img_cfgs=preprocessed_img_cfgs, preprocessed_images_registry=df_img_registry)

    return X, y

@step
def compile_preprocessed_images_registry(
    pretrained_model_configs: List[PretrainedModelConfig],
) -> Annotated[
    pd.DataFrame,
    "preprocessed_images_registry"
]:
    preprocessed_images_data_dicts:List[PreprocessedImagesDataDict] = []
    pretrained_model_entry_amounts = {}
    for pretrained_model_config in pretrained_model_configs:
        curr_preprocessed_images_data_dicts = _get_preprocessed_images(
            pretrained_model_config=pretrained_model_config
        )

        preprocessed_images_data_dicts.extend(curr_preprocessed_images_data_dicts.values())
        curr_amount = len(curr_preprocessed_images_data_dicts)
        pretrained_model_entry_amounts[str(pretrained_model_config)] = curr_amount
        print(f"\tsuccessfully compiled {curr_amount} preprocessed images from {pretrained_model_config.name} pretrained model")

    preprocessed_images_data_dicts = [{**preprocessed_images_data_dict['metadata'].to_flat_dict(), 'artifact_uri':preprocessed_images_data_dict['artifact_uri']} for preprocessed_images_data_dict in preprocessed_images_data_dicts]

    print(f"successfully compiled {len(preprocessed_images_data_dicts)} preprocessed images from {len(pretrained_model_configs)} pretrained models")

    df = pd.DataFrame(preprocessed_images_data_dicts)

    log_artifact_metadata(
        artifact_name="preprocessed_images_registry",
        metadata={
            "pretrained_model_entry_amounts": dict(sorted(pretrained_model_entry_amounts.items(), key=lambda x: x[0])),
        },
    )

    return df

@pipeline
def compile_preprocessed_images_registry_pipeline(
    pretrained_model_configs: List[PretrainedModelConfig],
):
    df = compile_preprocessed_images_registry(pretrained_model_configs=ExternalArtifact(value=sorted(pretrained_model_configs, key=lambda x: str(x))))

    return df



if __name__ == "__main__":
    # model_names = model_collections['famous_le_100m'].union(model_collections['famous_le_10m'])
    model_names = ["MobileNet", "MobileNetV2"]
    pretrained_model_configs = [PretrainedModelConfig(name=model_name, repo=ModelRepos.KERAS) for model_name in model_names]

    compile_preprocessed_images_registry_pipeline(pretrained_model_configs=pretrained_model_configs)

    