from dataclasses import dataclass
# from pydantic.dataclasses import dataclass
from enum import IntEnum, StrEnum
from typing import Literal, Optional, Tuple, Union

from tensorflow.keras import Model as tfModel
from torch.nn import Module as torchModel

from PIL import Image

import hashlib

from model_xray.utils.general_utils import flatten_dict

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
    X_LSB_ATTACK = 'x_lsb_attack'
    MALEFICNET = 'maleficnet'

class PayloadType(StrEnum):
    RANDOM = 'random'
    BINARY_FILE = 'binary_file'
    PYTHON_BYTES = 'python_bytes'

@dataclass(repr=False, unsafe_hash=True)
class XLSBAttackConfig:
    x: int
    fill: bool = True
    msb: bool = False

    def __repr__(self):
        repr_str = f'XLSBAttackConfig(x={self.x}, fill={self.fill}, msb={self.msb})'

        return repr_str

    @staticmethod
    def from_dict(metadata_dict):
        return XLSBAttackConfig(
            x=metadata_dict['x'],
            fill=metadata_dict['fill'],
            msb=metadata_dict['msb'],
        )

    def to_dict(self, short_version=False):
        if short_version:
            return {
                'x': self.x,
            }
        else:
            return {
                'x': self.x,
                'fill': self.fill,
                'msb': self.msb,
            }

@dataclass(repr=False, frozen=True)
class XLSBExtractConfig:
    x: int
    n_bytes: int = None
    fill: bool = True
    msb: bool = False
    
@dataclass(repr=False, unsafe_hash=True)
class MaleficnetAttackConfig:
    dataset: Literal['cifar10'] = 'cifar10'
    dim: int = 32
    num_classes: int = 10
    only_pretrained: bool = False
    fine_tuning: bool = False
    epochs: int = 60
    batch_size: int = 64
    random_seed: int = 8
    gamma: float = 0.0009

    chunk_factor: int = 6

@dataclass(frozen=False, unsafe_hash=True)
class EmbedPayloadMetadata:
    payload_bytes_md5: Optional[str] = None
    payload_filepath: Optional[str] = None

    @staticmethod
    def from_dict(metadata_dict):
        return EmbedPayloadMetadata(
            payload_bytes_md5=metadata_dict.get('payload_bytes_md5', None),
            payload_filepath=metadata_dict.get('payload_filepath', None)
        )

    def to_dict(self):
        return {
            'payload_bytes_md5': self.payload_bytes_md5,
            'payload_filepath': self.payload_filepath
        }

@dataclass(unsafe_hash=True)
class EmbedPayloadConfig:
    embed_type: EmbedType = EmbedType.X_LSB_ATTACK
    embed_payload_type: PayloadType = PayloadType.RANDOM
    embed_proc_config: Optional[Union[XLSBAttackConfig, MaleficnetAttackConfig]] = None
    embed_payload_metadata: Optional[EmbedPayloadMetadata] = None

    @staticmethod
    def from_dict(metadata_dict):
        if metadata_dict is None or metadata_dict == {} or metadata_dict == 'None':
            return None

        embed_proc_config = None
        embed_type = EmbedType(metadata_dict['embed_type'])
        if embed_type == EmbedType.X_LSB_ATTACK:
            embed_proc_config = XLSBAttackConfig.from_dict(metadata_dict['embed_proc_config'])

        embed_payload_type = PayloadType(metadata_dict['embed_payload_type'])
        
        embed_payload_metadata = None
        if 'embed_payload_metadata' in metadata_dict:
            embed_payload_metadata = EmbedPayloadMetadata(
                payload_bytes_md5=metadata_dict['embed_payload_metadata'].get('payload_bytes_md5', None),
                payload_filepath=metadata_dict['embed_payload_metadata'].get('payload_filepath', None)
            )
        
        return EmbedPayloadConfig(
            embed_type=embed_type,
            embed_payload_type=embed_payload_type,
            embed_proc_config=embed_proc_config,
            embed_payload_metadata=embed_payload_metadata
        )

    def to_dict(self, short_version=False):
        return {
            'embed_type': self.embed_type,
            'embed_payload_type': self.embed_payload_type,
            'embed_proc_config': self.embed_proc_config.to_dict(short_version=short_version) if self.embed_proc_config is not None else None,
            'embed_payload_metadata': self.embed_payload_metadata.to_dict() if self.embed_payload_metadata is not None else None
        }

    @staticmethod
    def ret_random_x_lsb_attack_fill_config(x: int):
        return EmbedPayloadConfig(
            embed_type=EmbedType.X_LSB_ATTACK,
            embed_payload_type=PayloadType.RANDOM,
            embed_proc_config=XLSBAttackConfig(
                x=x,
                fill=True,
                msb=False,
            )
        )

    @staticmethod
    def ret_filebytes_x_lsb_attack_fill_config(x: int, payload_filepath: str):
        return EmbedPayloadConfig(
            embed_type=EmbedType.X_LSB_ATTACK,
            embed_payload_type=PayloadType.BINARY_FILE,
            embed_proc_config=XLSBAttackConfig(
                x=x,
                fill=True,
                msb=False,
            ),
            embed_payload_metadata=EmbedPayloadMetadata(
                payload_filepath=payload_filepath
            )
        )

"""
    Image Representation CFGs
"""

class ImageType(StrEnum):
    GRAYSCALE_FOURPART = 'grayscale_fourpart'
    RGB = 'rgb'

    GRAYSCALE_LAST_M_BYTES = 'grayscale_last_m_bytes'
    GRAYSCALE_THREEPART_WEIGHTED_AVG = 'grayscale_threepart_weighted_avg'

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
class GrayscaleThreepartWeightedAvgConfig:
    byte_1_weight: float = 0.2
    byte_2_weight: float = 0.3
    byte_3_weight: float = 0.5

    @staticmethod
    def from_dict(metadata_dict):
        return GrayscaleThreepartWeightedAvgConfig(
            byte_1_weight=metadata_dict['byte_1_weight'],
            byte_2_weight=metadata_dict['byte_2_weight'],
            byte_3_weight=metadata_dict['byte_3_weight']
        )

    def to_dict(self):
        return {
            'byte_1_weight': self.byte_1_weight,
            'byte_2_weight': self.byte_2_weight,
            'byte_3_weight': self.byte_3_weight
        }

    def ret_byte_weights(self):
        return self.byte_1_weight, self.byte_2_weight, self.byte_3_weight

@dataclass(frozen=True)
class ImageRepConfig:
    image_type: ImageType = ImageType.GRAYSCALE_FOURPART
    image_rep_config: Optional[Union[GrayscaleLastMBytesConfig, GrayscaleThreepartWeightedAvgConfig]] = None

    @staticmethod
    def ret_image_rep_config(image_type_str: str):
        if image_type_str == ImageType.GRAYSCALE_LAST_M_BYTES:
            return ImageRepConfig(
                image_type=ImageType.GRAYSCALE_LAST_M_BYTES,
                image_rep_config=GrayscaleLastMBytesConfig()
            )
        elif image_type_str == ImageType.GRAYSCALE_THREEPART_WEIGHTED_AVG:
            return ImageRepConfig(
                image_type=ImageType.GRAYSCALE_THREEPART_WEIGHTED_AVG,
                image_rep_config=GrayscaleThreepartWeightedAvgConfig()
            )
        elif image_type_str == ImageType.GRAYSCALE_FOURPART:
            return ImageRepConfig(
                image_type=ImageType.GRAYSCALE_FOURPART,
                image_rep_config=None
            )
        else:
            raise NotImplementedError(f'ret_image_rep_config | got {image_type_str}, not implemented')

    @staticmethod
    def from_dict(metadata_dict):
        image_type = ImageType(metadata_dict['image_type'])
        image_rep_config = None
        if image_type == ImageType.GRAYSCALE_LAST_M_BYTES:
            image_rep_config = GrayscaleLastMBytesConfig.from_dict(metadata_dict['image_rep_config'])
        
        if image_type == ImageType.GRAYSCALE_THREEPART_WEIGHTED_AVG:
            image_rep_config = GrayscaleThreepartWeightedAvgConfig.from_dict(metadata_dict['image_rep_config'])
        
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
    image_height: int = 100
    image_width: int = 100 
    image_reshape_algo:ImageResamplingFilter = ImageResamplingFilter.BICUBIC

    @staticmethod
    def from_dict(metadata_dict):
        return ImagePreprocessConfig(
            image_height=metadata_dict['image_height'],
            image_width=metadata_dict['image_width'],
            image_reshape_algo=ImageResamplingFilter(metadata_dict['image_reshape_algo'])
        )

    def to_dict(self):
        return {
            'image_height': self.image_height,
            'image_width': self.image_width,
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
        image_rep_config: Optional[ImageRepConfig] = None,
        image_preprocess_config: Optional[ImagePreprocessConfig] = None,
        x: Optional[int] = None
    ):
        return PreprocessedImageLineage(
            pretrained_model_config=PretrainedModelConfig() if pretrained_model_config is None else pretrained_model_config,
            image_rep_config=ImageRepConfig() if image_rep_config is None else image_rep_config,
            image_preprocess_config=ImagePreprocessConfig() if image_preprocess_config is None else image_preprocess_config,
            embed_payload_config=EmbedPayloadConfig.ret_random_x_lsb_attack_fill_config(x) if x is not None else None
        )

    def to_dict(self):
        return {
            'pretrained_model_config': self.pretrained_model_config.to_dict(),
            'image_rep_config': self.image_rep_config.to_dict(),
            'image_preprocess_config': self.image_preprocess_config.to_dict(),
            'embed_payload_config': self.embed_payload_config.to_dict() if self.embed_payload_config is not None else None
        }

    def to_flat_dict(self, parent_key: str = 'metadata:', separator: str = '.') -> dict:
        return {f'{parent_key}{k}':v for k,v in flatten_dict(self.to_dict(), separator=separator).items()}

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