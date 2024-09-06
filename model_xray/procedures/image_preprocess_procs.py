

import numpy as np

from model_xray.configs.models import ImagePreprocessConfig
from model_xray.configs.enums import *

from PIL import Image

def pillow_preprocess(image: np.ndarray, image_preprocess_config: ImagePreprocessConfig) -> np.ndarray:
    if image.ndim == 3:
        image = image[0]
    im = Image.fromarray(image)

    im_resized = im.resize(
        size = (image_preprocess_config.image_height, image_preprocess_config.image_width),
        resample = image_preprocess_config.image_reshape_algo.to_pil_image_resampling_filter()
    )

    return np.asarray(im_resized)

def execute_image_preprocess(image: np.ndarray, image_preprocess_config: ImagePreprocessConfig) -> np.ndarray:
    preprocess_backend_type = image_preprocess_config.image_preprocess_backend

    preprocess_backend = preprocess_backend_type_map.get(preprocess_backend_type, None)
    if preprocess_backend is None:
        raise NotImplementedError(f'execute_image_preprocess | got {preprocess_backend_type}, not implemented')

    return preprocess_backend(image, image_preprocess_config)

preprocess_backend_type_map = {
    ImagePreprocessBackend.PILLOW: pillow_preprocess,
}