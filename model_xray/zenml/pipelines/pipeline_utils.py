from zenml import Model as ZenMLModel
from zenml.new.pipelines.pipeline import Pipeline

from model_xray.zenml.runtime_zenml_models import ret_zenml_model_pretrained_model, ret_zenml_model_preprocesssed_image_lineage
from model_xray.configs.models import PretrainedModelConfig, PreprocessedImageLineage

def _ret_pipeline_with_custom_model(
    *,
    model: ZenMLModel,
    pipeline: Pipeline,

    **kwargs,
):
    return pipeline.with_options(model=model, **kwargs)

def ret_pipeline_with_zenml_model_pretrained_model(
    pipeline: Pipeline,
    **kwargs
):
    def wrap(pretrained_model_config: PretrainedModelConfig, **inner_kwargs):
        zenml_model_pretrained_model = ret_zenml_model_pretrained_model(
            pretrained_model_config=pretrained_model_config,
        )

        return _ret_pipeline_with_custom_model(
            model=zenml_model_pretrained_model,
            pipeline=pipeline,
            **kwargs
        )(pretrained_model_config, **inner_kwargs)

    return wrap

def ret_pipeline_with_zenml_model_pp_image_lineage(
    pipeline: Pipeline,
    **kwargs
):
    def wrap(preprocessed_image_lineage_config: PreprocessedImageLineage,**inner_kwargs):
        zenml_model_preprocesssed_image_lineage = ret_zenml_model_preprocesssed_image_lineage(
            preprocessed_image_lineage_config=preprocessed_image_lineage_config,
        )

        return _ret_pipeline_with_custom_model(
            model=zenml_model_preprocesssed_image_lineage,
            pipeline=pipeline,
            **kwargs
        )(preprocessed_image_lineage_config, **inner_kwargs)

    return wrap