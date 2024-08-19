from dataclasses import dataclass
# from pydantic.dataclasses import dataclass
from enum import IntEnum, StrEnum
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

@dataclass(frozen=True)
class PretrainedModelConfig:
    name: str = 'MobileNet'
    repo: ModelRepos = ModelRepos.KERAS

    def to_dict(self, short_version=False):
        if short_version:
            return {
                'name': self.name,
            }
        else:
            return {
                'name': self.name,
                'repo': self.repo
            }

    @staticmethod
    def from_dict(metadata_dict):
        return PretrainedModelConfig(
            name=metadata_dict['name'],
            repo=ModelRepos(metadata_dict['repo'])
        )

    def __repr__(self):
        return f'PretrainedModelConfig(name={self.name}, repo={self.repo})'

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

    def __hash__(self):
        return hash(self.__repr__())

    def __repr__(self):
        repr_str = f'XLSBAttackConfig(x={self.x}, fill={self.fill}, msb={self.msb}, payload_type={self.payload_type}'
        if self.payload_type == PayloadType.PYTHON_BYTES:
            repr_str += f', payload_bytes_md5={hashlib.md5(self.payload_bytes).hexdigest()})'
        elif self.payload_type == PayloadType.BINARY_FILE:
            repr_str += f', payload_filepath={self.payload_filepath})'

        return repr_str

    @staticmethod
    def from_dict(metadata_dict):
        return XLSBAttackConfig(
            x=metadata_dict['x'],
            fill=metadata_dict['fill'],
            msb=metadata_dict['msb'],
            payload_bytes=metadata_dict['payload_bytes'] if metadata_dict['payload_type'] == PayloadType.PYTHON_BYTES else None,
            payload_type=PayloadType(metadata_dict['payload_type']),
            payload_filepath=metadata_dict['payload_filepath'] if metadata_dict['payload_type'] == PayloadType.BINARY_FILE else None
        )

    def to_dict(self, short_version=False):
        if short_version:
            return {
                'x': self.x,
                'payload_type': self.payload_type
            }
        else:
            return {
                'x': self.x,
                'fill': self.fill,
                'msb': self.msb,
                'payload_bytes_md5': hashlib.md5(self.payload_bytes).hexdigest() if self.payload_type == PayloadType.PYTHON_BYTES else None,
                'payload_type': self.payload_type,
                'payload_filepath': self.payload_filepath
            }

@dataclass(repr=False, frozen=True)
class XLSBExtractConfig:
    x: int
    n_bytes: int = None
    fill: bool = True
    msb: bool = False
    

@dataclass(frozen=True)
class EmbedPayloadConfig:
    embed_type: EmbedType = EmbedType.X_LSB_ATTACK_FILL
    embed_proc_config: Optional[XLSBAttackConfig] = None

    @staticmethod
    def from_dict(metadata_dict):
        if metadata_dict is None or metadata_dict == {} or metadata_dict == 'None':
            return None

        embed_proc_config = None
        embed_type = EmbedType(metadata_dict['embed_type'])
        if embed_type == EmbedType.X_LSB_ATTACK_FILL:
            embed_proc_config = XLSBAttackConfig.from_dict(metadata_dict['embed_proc_config'])
        
        return EmbedPayloadConfig(
            embed_type=embed_type,
            embed_proc_config=embed_proc_config
        )

    def to_dict(self, short_version=False):
        return {
            'embed_type': self.embed_type,
            'embed_proc_config': self.embed_proc_config.to_dict(short_version=short_version) if self.embed_proc_config is not None else None
        }

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

@dataclass(frozen=True)
class GrayscaleLastMBytesConfig:
    m: int

    @staticmethod
    def from_dict(metadata_dict):
        return GrayscaleLastMBytesConfig(
            m=metadata_dict['m']
        )
        
    def to_dict(self):
        return {
            'm': self.m
        }

    def __repr__(self):
        return f'GrayscaleLastMBytesConfig(m={self.m})'

@dataclass(frozen=True)
class ImageRepConfig:
    image_type: ImageType = ImageType.GRAYSCALE_FOURPART
    image_rep_config: Optional[GrayscaleLastMBytesConfig] = None

    @staticmethod
    def from_dict(metadata_dict):
        image_type = ImageType(metadata_dict['image_type'])
        image_rep_config = None
        if image_type == ImageType.GRAYSCALE_LAST_M_BYTES:
            image_rep_config = GrayscaleLastMBytesConfig.from_dict(metadata_dict['image_rep_config'])
        
        return ImageRepConfig(
            image_type=image_type,
            image_rep_config=image_rep_config
        )

    def to_dict(self):
        return {
            'image_type': self.image_type,
            'image_rep_config': self.image_rep_config.to_dict() if self.image_rep_config is not None else None
        }

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

@dataclass(frozen=True)
class ImagePreprocessConfig:
    image_size: Tuple[int, int] = (100,100)
    image_reshape_algo:ImageResamplingFilter = ImageResamplingFilter.BICUBIC

    @staticmethod
    def from_dict(metadata_dict):
        return ImagePreprocessConfig(
            image_size=tuple(metadata_dict['image_size']),
            image_reshape_algo=ImageResamplingFilter(metadata_dict['image_reshape_algo'])
        )

    def to_dict(self):
        return {
            'image_size': self.image_size,
            'image_reshape_algo': self.image_reshape_algo
        }

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


"""
    Preprocessed image lineage
"""

class PreprocessedImageDatasetLabel(IntEnum):
    BENIGN = 0
    ATTACKED = 1

def dict_to_str(metadata_dict, indent=1):
    return dict_strs_to_str(dict_to_strs(metadata_dict), indent=indent)

def dict_to_strs(metadata_dict):
    return [f'{k.title()}: {v}' for k,v in metadata_dict.items()]

def dict_strs_to_str(dict_strs, indent=1):
    spacer_str = '\n' + '\t'*indent
    return f'{spacer_str.join(dict_strs)}\n'

@dataclass(frozen=True)
class PreprocessedImageLineage:
    pretrained_model_config: PretrainedModelConfig
    image_rep_config: ImageRepConfig
    image_preprocess_config: ImagePreprocessConfig

    embed_payload_config: Optional[EmbedPayloadConfig] = None

    @staticmethod
    def ret_default_preprocessed_image_w_x_lsb_attack(
        pretrained_model_config: Optional[PretrainedModelConfig] = None,
        x: Optional[int] = None
    ):
        return PreprocessedImageLineage(
            pretrained_model_config=PretrainedModelConfig() if pretrained_model_config is None else pretrained_model_config,
            image_rep_config=ImageRepConfig(),
            image_preprocess_config=ImagePreprocessConfig(),
            embed_payload_config=EmbedPayloadConfig.ret_x_lsb_attack_fill_config(x) if x is not None else None
        )

    def to_dict(self):
        return {
            'pretrained_model_config': self.pretrained_model_config.to_dict(),
            'image_rep_config': self.image_rep_config.to_dict(),
            'image_preprocess_config': self.image_preprocess_config.to_dict(),
            'embed_payload_config': self.embed_payload_config.to_dict() if self.embed_payload_config is not None else None
        }

    @staticmethod
    def from_dict(metadata_dict):
        return PreprocessedImageLineage(
            pretrained_model_config=PretrainedModelConfig.from_dict(metadata_dict['pretrained_model_config']),
            image_rep_config=ImageRepConfig.from_dict(metadata_dict['image_rep_config']),
            image_preprocess_config=ImagePreprocessConfig.from_dict(metadata_dict['image_preprocess_config']),
            embed_payload_config=EmbedPayloadConfig.from_dict(metadata_dict.get('embed_payload_config', None))
        )

    def __repr__(self):
        return (f'Preprocessed Image Lineage:\n'
                f'\tPretrained Model:\n'
                f'\t\t{dict_to_str(self.pretrained_model_config.to_dict(short_version=True), indent=2)}'
                f'\tAttack:\n'
                f'\t\tAttacked: {self.is_attacked()}\n'
                
                f'\t\tAttack Config:\n'
                f'\t\t\t{dict_to_str(self.embed_payload_config.to_dict(short_version=True), indent=3) if self.is_attacked() else "NA"}\n'

                # f'\n\t\tAttack Config:\n' if self.is_attacked() else f'\n'
                # f'\t\t{dict_to_str(self.embed_payload_config.to_dict(short_version=True), indent=2)}\n' if self.is_attacked() else f'\n'
                )

    def is_attacked(self) -> bool:
        return self.embed_payload_config is not None

    def label(self) -> PreprocessedImageDatasetLabel:
        return PreprocessedImageDatasetLabel.ATTACKED if self.is_attacked() else PreprocessedImageDatasetLabel.BENIGN