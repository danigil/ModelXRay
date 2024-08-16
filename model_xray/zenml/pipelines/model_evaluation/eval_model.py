from typing import Union
from zenml import ArtifactConfig, get_step_context, step, pipeline

from ModelXRay.model_xray.config_classes import ModelRepos
from tensorflow.keras import Model as tfModel

@step
def eval_model(
    model: Union[tfModel,],
)