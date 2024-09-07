
from zenml import Model, get_pipeline_context, pipeline

from model_xray.zenml.runtime_zenml_models import ret_zenml_model_preprocesssed_image_lineage
from model_xray.zenml.pipelines.model_evaluation.val_datasets import get_imagenet12_val_tfds_pipeline
from model_xray.zenml.steps.dl_dataset_utils import get_imagenet12_val_tfds_step, get_tfds_subset_step
from model_xray.zenml.steps.dl_model_eval import eval_model_step

from model_xray.zenml.pipelines.pipeline_utils import ret_pipeline_with_zenml_model_pp_image_lineage
from model_xray.configs.models import *

from model_xray.configs.types import kerasModel



@pipeline(enable_cache=True)
def _preprocessed_image_lineage_models_eval(
    preprocessed_image_lineage: PreprocessedImageLineage,
    ds_name:Literal['imagenet12'],
    take_subset: Optional[int] = None,
):
    zenml_model = ret_zenml_model_preprocesssed_image_lineage(preprocessed_image_lineage)

    cover_data = zenml_model.get_artifact('cover_data')
    score_cover = eval_model_step(
        model=cover_data,
        ds_name=ds_name,
        take_subset=take_subset
    )

    zenml_model.save_artifact(
        score_cover, f'score_cover_ds:{ds_name.lower()}_subset:{take_subset}'
    )

    if preprocessed_image_lineage.embed_payload_config != ret_na_val():
        stego_data = zenml_model.get_artifact('stego_data')
        score_stego = eval_model_step(
            model=stego_data,
            ds_name=ds_name,
            take_subset=take_subset
        )

        zenml_model.save_artifact(
            score_stego, f'score_stego_ds:{ds_name.lower()}_subset:{take_subset}'
        )


    


        

    