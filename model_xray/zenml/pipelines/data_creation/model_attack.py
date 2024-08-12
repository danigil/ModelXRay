
import numpy as np
from zenml import step

from model_xray.config_classes import EmbedPayloadConfig

@step
def embed_payload_into_weights(
    weights: np.ndarray,
    
    embed_payload_config: EmbedPayloadConfig
):
    pass