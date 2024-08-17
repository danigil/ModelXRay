from dataclasses import dataclass
# from pydantic.dataclasses import dataclass
from enum import StrEnum
from typing import Optional, Tuple, Union

from tensorflow.keras import Model as tfModel
from torch.nn import Module as torchModel

from PIL import Image

import hashlib

class ModelRepos(StrEnum):
    KERAS = 'keras'
    PYTORCH = 'torch'
    HUGGINGFACE = 'huggingface'

    @classmethod
    def determine_model_type(cls, model):
        if isinstance(model, torchModel):
            return cls.PYTORCH
        elif isinstance(model, tfModel):
            return cls.KERAS
        else:
            raise NotImplementedError(f'determine_model_type | got model type {type(model)}, not implemented')

"""
    Attack CFGs
"""
class EmbedType(StrEnum):
    X_LSB_ATTACK_FILL = 'x_lsb_attack_fill'

class PayloadType(StrEnum):
    RANDOM = 'random'
    BINARY_FILE = 'binary_file'
    PYTHON_BYTES = 'python_bytes'

@dataclass(repr=False)
class XLSBAttackConfig:
    x: int
    fill: bool = True
    msb: bool = False
    payload_bytes: Union[None, bytes] = None
    payload_type: PayloadType = PayloadType.RANDOM
    payload_filepath: Union[None, str] = None

    def __repr__(self):
        repr_str = f'XLSBAttackConfig(x={self.x}, fill={self.fill}, msb={self.msb}, payload_type={self.payload_type}'
        if self.payload_type == PayloadType.PYTHON_BYTES:
            repr_str += f', payload_bytes_md5={hashlib.md5(self.payload_bytes).hexdigest()}'
        elif self.payload_type == PayloadType.BINARY_FILE:
            repr_str += f', payload_filepath={self.payload_filepath}'

    def to_dict(self):
        return {
            'x': self.x,
            'fill': self.fill,
            'msb': self.msb,
            'payload_bytes_md5': hashlib.md5(self.payload_bytes).hexdigest() if self.payload_type == PayloadType.PYTHON_BYTES else None,
            'payload_type': self.payload_type,
            'payload_filepath': self.payload_filepath
        }

@dataclass(repr=False)
class XLSBExtractConfig:
    x: int
    n_bytes: int = None
    fill: bool = True
    msb: bool = False
    

@dataclass
class EmbedPayloadConfig:
    embed_type: EmbedType = EmbedType.X_LSB_ATTACK_FILL
    embed_proc_config: Union[XLSBAttackConfig, None] = None

    @staticmethod
    def ret_x_lsb_attack_fill_config(x: int):
        return EmbedPayloadConfig(
            embed_type=EmbedType.X_LSB_ATTACK_FILL,
            embed_proc_config=XLSBAttackConfig(
                x=x,
                fill=True,
                msb=False,
                payload_type=PayloadType.RANDOM
            )
        )

"""
    Image Representation CFGs
"""

class ImageType(StrEnum):
    GRAYSCALE_FOURPART = 'grayscale_fourpart'
    RGB = 'rgb'

    GRAYSCALE_LAST_M_BYTES = 'grayscale_last_m_bytes'

@dataclass
class GrayscaleLastMBytesConfig:
    m: int

    def to_dict(self):
        return {
            'm': self.m
        }

    def __repr__(self):
        return f'GrayscaleLastMBytesConfig(m={self.m})'

@dataclass
class ImageRepConfig:
    image_type: ImageType = ImageType.GRAYSCALE_FOURPART
    image_rep_config: Union[GrayscaleLastMBytesConfig, None] = None

class ImageResamplingFilter(StrEnum):
    BICUBIC = 'bicubic'
    NEAREST = 'nearest'
    BILINEAR = 'bilinear'
    HAMMING = 'hamming'
    LANCZOS = 'lanczos'
    BOX = 'box'

    def to_pil_image_resampling_filter(self):
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

@dataclass
class ImagePreprocessConfig:
    image_size: Tuple[int, int] = (100,100)
    image_reshape_algo:ImageResamplingFilter = ImageResamplingFilter.BICUBIC

"""
    Eval CFGs
"""

class ClassificationMetric(StrEnum):
    TopKCategoricalAccuracy = 'TopKCategoricalAccuracy'

@dataclass(unsafe_hash=True)
class TopKCategoricalAccuracyMetricConfig:
    k: int

    def to_dict(self):
        return {
            'k': self.k
        }

@dataclass(unsafe_hash=True)
class ClassificationMetricConfig:
    metric_type: ClassificationMetric
    classification_metric_config: Optional[Union[TopKCategoricalAccuracyMetricConfig, ]] = None

    @staticmethod
    def ret_top_k_categorical_accuracy_config(k: int):
        return ClassificationMetricConfig(
            metric_type=ClassificationMetric.TopKCategoricalAccuracy,
            classification_metric_config=TopKCategoricalAccuracyMetricConfig(
                k=k
            )
        )

class DatasetType(StrEnum):
    ImageDataset = 'ImageDataset'

@dataclass
class ImageDatasetConfig:
    image_size: Tuple[int, int]

@dataclass
class DatasetPreprocessConfig:
    take: Optional[int] = None

@dataclass
class DatasetConfig:
    dataset_type: DatasetType
    dataset_config: Optional[Union[ImageDatasetConfig,]] = None
    dataset_preprocess_config: Optional[DatasetPreprocessConfig] = None

    @staticmethod
    def ret_img_ds_config(image_size: Tuple[int, int], take: Optional[int] = None):
        return DatasetConfig(
            dataset_type=DatasetType.ImageDataset,
            dataset_config=ImageDatasetConfig(
                image_size=image_size
            ),
            dataset_preprocess_config=DatasetPreprocessConfig(
                take=take
            )
        )
