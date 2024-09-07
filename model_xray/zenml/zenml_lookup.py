

from typing import Literal
from model_xray.zenml.runtime_zenml_models import ret_zenml_model_preprocesssed_image_lineage
from model_xray.config_classes import PreprocessedImageLineage
from model_xray.zenml.pipelines.data_creation.preprocessed_image import preprocessed_image_pipeline

from zenml.client import Client

zenml_client = Client()

class ZenMLModelNotFoundError(Exception):
    pass

class ArtifactNotFoundError(Exception):
    pass

def try_get_artifact_preprocessed_image(
    preprocessed_image_lineage: PreprocessedImageLineage,
    artifact_name: Literal['cover_data', 'stego_data', 'image_representation', 'image_preprocessed'],
):
    try:
        zenml_model = ret_zenml_model_preprocesssed_image_lineage(preprocessed_image_lineage)
        model_version = zenml_client.get_model_version(
            model_name_or_id=zenml_model.name,
            model_version_name_or_number_or_id=zenml_model.version
        )

        artifact = model_version.get_artifact(artifact_name)
        if artifact is None:
            raise(ArtifactNotFoundError, "artifact not found")
        
        return artifact.load()
    except KeyError as e:
        raise(ZenMLModelNotFoundError, "zenml model not found")

def get_artifact_preprocessed_image(
    preprocessed_image_lineage: PreprocessedImageLineage,
    artifact_name: Literal['cover_data', 'stego_data', 'image_representation', 'image_preprocessed'],
    fallback: bool = False
):
    try:
        try_ret = try_get_artifact_preprocessed_image(preprocessed_image_lineage, artifact_name)
        return try_ret
    except Exception as e:
        if fallback:
            print(f"get_artifact_preprocessed_image: fallback: executing pipeline.")
            preprocessed_image_pipeline(preprocessed_image_lineage)

            try_ret = try_get_artifact_preprocessed_image(preprocessed_image_lineage, artifact_name)

