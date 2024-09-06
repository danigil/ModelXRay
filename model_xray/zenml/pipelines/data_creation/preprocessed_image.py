

from zenml import pipeline

from model_xray.zenml.steps.image_preprocessing import image_preprocessing_step
from model_xray.zenml.steps.image_rep import create_image_representation_step
from model_xray.zenml.steps.embedding import embed_payload_into_cover_data_step
from model_xray.zenml.steps.cover_data import fetch_cover_data_step
from model_xray.zenml.pipelines.pipeline_utils import ret_pipeline_with_zenml_model_pp_image_lineage
from model_xray.configs.models import *


@pipeline
def _preprocessed_image_pipeline(preprocessed_image_lineage: PreprocessedImageLineage):
    cover_data = fetch_cover_data_step(cover_data_config=preprocessed_image_lineage.cover_data_config.model_dump())

    if preprocessed_image_lineage.embed_payload_config != ret_na_val():
        stego_data = embed_payload_into_cover_data_step(cover_data=cover_data, embed_payload_config=preprocessed_image_lineage.embed_payload_config.model_dump())
        data = stego_data
    else:
        data = cover_data

    image_rep = create_image_representation_step(data=data, image_rep_config=preprocessed_image_lineage.image_rep_config.model_dump())
    preprocessed_image = image_preprocessing_step(image=image_rep, image_preprocess_config=preprocessed_image_lineage.image_preprocess_config.model_dump())

    return preprocessed_image

preprocessed_image_pipeline = ret_pipeline_with_zenml_model_pp_image_lineage(
    _preprocessed_image_pipeline
)
