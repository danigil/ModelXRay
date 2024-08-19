from typing import Annotated, Optional, Set, Tuple
import numpy as np
from zenml import step
from zenml import ArtifactConfig, Model, get_pipeline_context, get_step_context, log_artifact_metadata, step, pipeline, log_model_metadata
from zenml.client import Client
from zenml.new.pipelines.pipeline import Pipeline

from model_xray.config_classes import ModelRepos, PreprocessedImageLineage, PretrainedModelConfig
from model_xray.zenml.pipelines.data_creation.image_representation import ret_pretrained_model_version_name

def get_preprocessed_images(
    pretrained_model_config: PretrainedModelConfig,
    preprocessed_img_cfgs: Optional[Set[PreprocessedImageLineage]] = None
):
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
            'image': curr_preprocessed_image,
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
        preprocessed_images = get_preprocessed_images(
            pretrained_model_config=pretrained_model_config,
            preprocessed_img_cfgs=set([preprocessed_img_cfg for preprocessed_img_cfg in  preprocessed_img_cfgs if preprocessed_img_cfg.pretrained_model_config == pretrained_model_config])
        )

        for pipeline_run_name, preprocessed_image_and_metadata_dict in preprocessed_images.items():
            ret[preprocessed_image_and_metadata_dict['metadata']] = preprocessed_image_and_metadata_dict['image']
    
    return ret

def compile_preprocessed_images_dataset(
    preprocessed_img_cfgs: Set[PreprocessedImageLineage],
) -> Tuple[
    Annotated[np.ndarray, "X"],
    Annotated[np.ndarray, "y"],
]:
    preprocessed_images_dict = _ret_preprocessed_images(preprocessed_img_cfgs=preprocessed_img_cfgs)

    cfgs, imgs = zip(*sorted(preprocessed_images_dict.items(), key=lambda x: str(x[0].to_dict())))

    X = np.array(imgs)
    y = np.array([preprocessed_img_cfg.label() for preprocessed_img_cfg in cfgs])

    return X, y