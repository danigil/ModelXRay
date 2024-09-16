from enum import IntEnum, StrEnum

"""
    Cover Data Types
"""

class CoverDataTypes(StrEnum):
    PRETRAINED_MODEL = 'pretrained_model'
    MALEFICNET_COVER_MODEL = 'maleficnet'
    GHRP_MODEL_ZOO = 'ghrp'

"""
    Pretrained Model Repositories
"""
class ModelRepos(StrEnum):
    KERAS = 'keras'
    PYTORCH = 'torch'
    HUGGINGFACE = 'huggingface'

    @classmethod
    def determine_model_type(cls, model):
        from model_xray.configs.types import kerasModel, torchModel
        if isinstance(model, torchModel):
            return cls.PYTORCH
        elif isinstance(model, kerasModel):
            return cls.KERAS
        else:
            raise NotImplementedError(f'determine_model_type | got model type {type(model)}, not implemented')

"""
    Embedding
"""

class EmbedType(StrEnum):
    X_LSB_ATTACK = 'x_lsb_attack'
    MALEFICNET = 'maleficnet'

class PayloadType(StrEnum):
    RANDOM = 'random'
    BINARY_FILE = 'binary_file'
    PYTHON_BYTES = 'python_bytes'

"""
    Image Representations
"""

class ImageType(StrEnum):
    GRAYSCALE_FOURPART = 'grayscale_fourpart'
    RGB = 'rgb'

    GRAYSCALE_LAST_M_BYTES = 'grayscale_last_m_bytes'
    GRAYSCALE_THREEPART_WEIGHTED_AVG = 'grayscale_threepart_weighted_avg'

class ImageResamplingFilter(StrEnum):
    BICUBIC = 'bicubic'
    NEAREST = 'nearest'
    BILINEAR = 'bilinear'
    HAMMING = 'hamming'
    LANCZOS = 'lanczos'
    BOX = 'box'

    def to_pil_image_resampling_filter(self):
        from PIL import Image
        if self == ImageResamplingFilter.BICUBIC:
            return Image.Resampling.BICUBIC
        elif self == ImageResamplingFilter.NEAREST:
            return Image.Resampling.NEAREST
        elif self == ImageResamplingFilter.BILINEAR:
            return Image.Resampling.BILINEAR
        elif self == ImageResamplingFilter.HAMMING:
            return Image.Resampling.HAMMING
        elif self == ImageResamplingFilter.LANCZOS:
            return Image.Resampling.LANCZOS
        elif self == ImageResamplingFilter.BOX:
            return Image.Resampling.BOX
        else:
            raise NotImplementedError(f'to_pil_image_resampling_filter | got {self}, not implemented')

class ImagePreprocessBackend(StrEnum):
    PILLOW = 'pillow'
    OPENCV = 'opencv'

class PreprocessedImageDatasetLabel(IntEnum):
    BENIGN = 0
    ATTACKED = 1
