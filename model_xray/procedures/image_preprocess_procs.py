

import numpy as np
import numpy.typing as npt

from model_xray.configs.models import ImagePreprocessConfig
from model_xray.configs.enums import *

from PIL import Image
from skimage.transform import resize as skimage_resize

def pillow_preprocess(image: np.ndarray, image_preprocess_config: ImagePreprocessConfig) -> np.ndarray:
    # if image.ndim == 3:
    #     image = image[0]
    im = Image.fromarray(image)

    im_resized = im.resize(
        size = (image_preprocess_config.image_height, image_preprocess_config.image_width),
        resample = image_preprocess_config.image_reshape_algo.to_pil_image_resampling_filter()
    )

    return np.asarray(im_resized)

def numpy_preprocess(images: np.ndarray, image_preprocess_config: ImagePreprocessConfig) -> np.ndarray:
    image_height = image_preprocess_config.image_height
    image_width = image_preprocess_config.image_width
    target_size = image_height * image_width

    assert image_height == image_width, f'numpy_preprocess | image_height ({image_height}) != image_width ({image_width})'
    # assert images.ndim == 2, f'numpy_preprocess | images.ndim ({images.ndim}) != 2'

    if images.ndim == 2:
        n_m, n_w = images.shape

        chunks = np.array_split(images, target_size, axis=1)
        downsampled = np.concatenate([np.mean(c,axis=1, keepdims=True) for c in chunks]).reshape(n_m, image_height, image_width)
    elif images.ndim == 3:
        n_m, imsize_h, imsize_w = images.shape
        assert imsize_h == imsize_w, f'numpy_preprocess | imsize_h ({imsize_h}) != imsize_w ({imsize_w})'
        images = images.reshape(n_m, -1)

        chunks = np.array_split(images, target_size, axis=1)
        downsampled = np.concatenate([np.mean(c,axis=1, keepdims=True) for c in chunks]).reshape(n_m, image_height, image_width)
    elif images.ndim == 4:
        n_m, imsize_h, imsize_w, n_c = images.shape
        assert imsize_h == imsize_w, f'numpy_preprocess | imsize_h ({imsize_h}) != imsize_w ({imsize_w})'
        images = np.transpose(images, (0, 3, 1, 2)).reshape(n_m, n_c, -1)

        chunks = np.array_split(images, target_size, axis=2)
        downsampled = np.concatenate([np.mean(c,axis=2, keepdims=True) for c in chunks]).reshape(n_m, n_c, image_height, image_width)
        downsampled = np.transpose(downsampled, (0, 2, 3, 1))

    return downsampled
    
def skimage_preprocess(images: np.ndarray, image_preprocess_config: ImagePreprocessConfig) -> np.ndarray:
    imsize_h = image_preprocess_config.image_height
    imsize_w = image_preprocess_config.image_width
    if images.ndim == 4:
        images_n, images_h, images_w, images_c = images.shape  

        images_resized = skimage_resize(
            images,
            output_shape=(images_n, imsize_h, imsize_w),
            preserve_range=True,
            anti_aliasing=True,
            clip=True,
        )

    elif images.ndim == 3:
        # Treat 3D as (n_models, height, width) — single-channel grayscale stacks
        # (the GF representation produces this layout). The legacy branch that
        # interpreted axis 2 as channels and only resized height was a bug:
        # it left axis 2 at the original GF tile width.
        images_n, images_h, images_w = images.shape
        images_resized = skimage_resize(
            images,
            output_shape=(images_n, imsize_h, imsize_w),
            preserve_range=True,
            anti_aliasing=True,
            clip=True,
        )

    return images_resized

def execute_image_preprocess(image: np.ndarray, image_preprocess_config: ImagePreprocessConfig) -> np.ndarray:
    preprocess_backend_type = image_preprocess_config.image_preprocess_config.image_preprocess_backend

    preprocess_backend = preprocess_backend_type_map.get(preprocess_backend_type, None)
    if preprocess_backend is None:
        raise NotImplementedError(f'execute_image_preprocess | got {preprocess_backend_type}, not implemented')

    if image.ndim == 2:
        n_models, n_weights = image.shape
    elif image.ndim == 3:
        n_models, imsize, imsize = image.shape
    elif image.ndim == 4:
        n_models, imsize, imsize, n_channels = image.shape

    if preprocess_backend_type == ImagePreprocessBackend.PILLOW:
        if n_models == 1:
            image = image[0]

            ret = preprocess_backend(image, image_preprocess_config)
        else:
            images_preprocessed = []
            for image_curr in image:
                images_preprocessed.append(preprocess_backend(image_curr, image_preprocess_config))

            ret = np.array(images_preprocessed)
    else:
        ret = preprocess_backend(image, image_preprocess_config)
    

    return ret

preprocess_backend_type_map = {
    ImagePreprocessBackend.PILLOW: pillow_preprocess,
    ImagePreprocessBackend.NUMPY: numpy_preprocess,
    ImagePreprocessBackend.SKIMAGE: skimage_preprocess,
}