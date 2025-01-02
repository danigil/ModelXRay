from model_xray.configs.models import *
from model_xray.configs.enums import *

from model_xray.options import *
from model_xray.configs.types import COVER_DATA_TYPE, DL_MODEL_TYPE
from model_xray.procedures.image_preprocess_procs import execute_image_preprocess
from model_xray.procedures.image_rep_procs import execute_image_rep_proc
import numpy as np

def model_preprocess(model: DL_MODEL_TYPE,
                     image_rep_config: ImageRepConfig,
                     image_preprocess_config: ImagePreprocessConfig
) -> np.ndarray:
    
    image_rep = execute_image_rep_proc(model, image_rep_config)
    image_rep_preprocessed = execute_image_preprocess(image_rep, image_preprocess_config)

    del image_rep

    return image_rep_preprocessed

def model_inference(model: DL_MODEL_TYPE,
                    image_rep_preprocessed: np.ndarray
) -> np.ndarray:
    return model.predict(image_rep_preprocessed)