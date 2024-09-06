
from model_xray.configs.models import PreprocessedImageLineage, PretrainedModelConfig

from zenml import Model as ZenMLModel

def ret_zenml_model_pretrained_model(
    pretrained_model_config: PretrainedModelConfig,
):
    return ZenMLModel(
        name="pretrained_model",
        version=pretrained_model_config.model_dump_json()
    )

def ret_zenml_model_preprocesssed_image_lineage(
    preprocessed_image_lineage_config: PreprocessedImageLineage,
):
    return ZenMLModel(
        name="preprocessed_image_lineage",
        version=f'preprocessed_image_lineage sha256: {preprocessed_image_lineage_config.str_hash()}'
    )