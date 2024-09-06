from zenml import pipeline

from model_xray.zenml.pipelines.pipeline_utils import ret_pipeline_with_zenml_model_pretrained_model
from model_xray.zenml.steps.dl_model_utils import extract_dl_model_weights_step
from model_xray.configs.models import PretrainedModelConfig

from model_xray.zenml.steps.fetch_pretrained_model import fetch_pretrained_model_step


@pipeline(enable_cache=True)
def _fetch_pretrained_model_and_extract_weights_pipeline(pretrained_model_config: PretrainedModelConfig):
    model = fetch_pretrained_model_step(pretrained_model_config=pretrained_model_config)
    w = extract_dl_model_weights_step(model=model)

    return w

fetch_pretrained_model_and_extract_weights_pipeline = ret_pipeline_with_zenml_model_pretrained_model(
    pipeline=_fetch_pretrained_model_and_extract_weights_pipeline,
    enable_cache=True
)

# if __name__ == "__main__":
#     model_names = model_collections['famous_le_10m']

#     for model_name in model_names:
#         fetch_pretrained_model_and_extract_weights(model_repo=ModelRepos.KERAS, pretrained_model_name=model_name)