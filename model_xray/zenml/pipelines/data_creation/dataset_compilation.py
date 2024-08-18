from typing import Optional, Set
from zenml import step
from zenml import ArtifactConfig, Model, get_pipeline_context, get_step_context, log_artifact_metadata, step, pipeline, log_model_metadata
from zenml.client import Client
from zenml.new.pipelines.pipeline import Pipeline

from model_xray.config_classes import ModelRepos, PreprocessedImageLineage, PretrainedModelConfig
from model_xray.zenml.pipelines.data_creation.image_representation import ret_pretrained_model_version_name

def get_preprocessed_images(
    pretrained_model_name:str,
    pretrained_model_repo:ModelRepos,

    preprocessed_img_cfgs: Optional[Set[PreprocessedImageLineage]] = None
):
    accum_metadata_keys = [
        'embedding_properties', 'embedding_config', 'image_properties', 'image_config', 'image_preprocess_properties'
    ]

    client = Client()

    model = client.get_model_version(
        model_name_or_id="model_pretrained",
        model_version_name_or_number_or_id=ret_pretrained_model_version_name(pretrained_model_name, pretrained_model_repo)
    )

    ret = {}

    for pipeline_run_name, pipeline_run_reponse in model.pipeline_runs.items():
        curr_steps = pipeline_run_reponse.steps
        if 'image_preprocessing' not in curr_steps.keys():
            continue

        metadata_dict ={}

        for step_name, step_response in curr_steps.items():
            curr_step_metadata = step_response.output.run_metadata

            for accum_metadata_key in accum_metadata_keys:
                if accum_metadata_key in curr_step_metadata:
                    metadata_dict[accum_metadata_key] = curr_step_metadata[accum_metadata_key].value

        # curr_preprocessed_image = curr_steps['image_preprocessing'].output.load()

        ret[pipeline_run_name] = {
            'image': curr_preprocessed_image,
            'metadata': metadata_dict
        }

    return ret

def _compile_preprocessed_images_dataset(
    preprocessed_img_cfgs: Set[PreprocessedImageLineage],
):
    requested_pretrained_models = set([preprocessed_img_cfg.pretrained_model_cfg for preprocessed_img_cfg in preprocessed_img_cfgs])