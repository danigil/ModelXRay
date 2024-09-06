

from pathlib import Path
import pickle
from typing import Annotated, Literal, Optional, Union
from pydantic import BaseModel, ConfigDict, Field, FilePath, computed_field

import hashlib

from model_xray.configs.enums import *

NA_VAL = 'NA'
NA_VAL_TYPE = Literal['NA']

def ret_na_val() -> NA_VAL_TYPE:
    return NA_VAL

"""
    Cover Data CFGs
"""

class PretrainedModelConfig(BaseModel):
    model_config = ConfigDict(from_attributes=True, frozen=True)

    cover_data_type: Literal[CoverDataTypes.PRETRAINED_MODEL] = CoverDataTypes.PRETRAINED_MODEL

    name: str = 'MobileNet'
    repo: ModelRepos = ModelRepos.KERAS
    train_dataset: Literal['imagenet12'] = 'imagenet12'

class DummyCoverDataConfig(BaseModel):
    model_config = ConfigDict(from_attributes=True, frozen=True)

    cover_data_type: Literal[CoverDataTypes.DUMMY] = CoverDataTypes.DUMMY

class CoverDataConfig(BaseModel):
    model_config = ConfigDict(from_attributes=True, frozen=True)

    cover_data_cfg: Annotated[
        Union[
            PretrainedModelConfig,
            DummyCoverDataConfig
        ],
        Field(
            default=PretrainedModelConfig(),
            discriminator='cover_data_type'
        )
    ]


"""
    Attack CFGs
"""

class XLSBAttackConfig(BaseModel):
    model_config = ConfigDict(from_attributes=True, frozen=True)

    attack_type: Literal[EmbedType.X_LSB_ATTACK] = EmbedType.X_LSB_ATTACK

    x: int = 1
    fill: bool = True
    msb: bool = False

class XLSBExtractConfig(BaseModel):
    model_config = ConfigDict(from_attributes=True, frozen=True)

    x: int
    n_bytes: int = None
    fill: bool = True
    msb: bool = False
    

class MaleficnetAttackConfig(BaseModel):
    model_config = ConfigDict(from_attributes=True, frozen=True)

    attack_type: Literal[EmbedType.MALEFICNET] = EmbedType.MALEFICNET

    dataset: Literal['cifar10'] = 'cifar10'
    dim: int = 32
    num_classes: int = 10
    only_pretrained: bool = False
    epochs: int = 60
    batch_size: int = 64
    random_seed: int = 8
    gamma: float = 0.0009

    chunk_factor: int = 6

"""
    Embedding CFGs
"""

class EmbedPayloadMetadata(BaseModel):
    model_config = ConfigDict(from_attributes=True, frozen=True)

    payload_bytes_md5: Union[str, NA_VAL_TYPE] = Field(default_factory=ret_na_val)
    payload_filepath: Union[FilePath, NA_VAL_TYPE] = Field(default_factory=ret_na_val)

    @computed_field
    @property
    def payload_filename(self) -> Union[str, NA_VAL_TYPE]:
        if self.payload_filepath == NA_VAL:
            return NA_VAL
        else:
            return Path(self.payload_filepath).name

class EmbedPayloadConfig(BaseModel):
    model_config = ConfigDict(from_attributes=True, frozen=True)

    # embed_type: EmbedType = EmbedType.X_LSB_ATTACK
    embed_payload_type: PayloadType = PayloadType.RANDOM
    embed_proc_config: Annotated[
        Union[
            XLSBAttackConfig,
            MaleficnetAttackConfig
        ],
        Field(
            default=XLSBAttackConfig(),
            discriminator='attack_type'
        )
    ]
    embed_payload_metadata: EmbedPayloadMetadata = EmbedPayloadMetadata()

    # @staticmethod
    # def ret_x_lsb_attack_fill_config(x: int, payload_filepath: Optional[str] = None):
    #     if payload_filepath is not None:
    #         return EmbedPayloadConfig.ret_filebytes_x_lsb_attack_fill_config(x, payload_filepath)
    #     else:
    #         return EmbedPayloadConfig.ret_random_x_lsb_attack_fill_config(x)

    # @staticmethod
    # def ret_random_x_lsb_attack_fill_config(x: int):
    #     return EmbedPayloadConfig(
    #         embed_type=EmbedType.X_LSB_ATTACK,
    #         embed_payload_type=PayloadType.RANDOM,
    #         embed_proc_config=XLSBAttackConfig(
    #             x=x,
    #             fill=True,
    #             msb=False,
    #         )
    #     )

    # @staticmethod
    # def ret_filebytes_x_lsb_attack_fill_config(x: int, payload_filepath: str):
    #     return EmbedPayloadConfig(
    #         embed_type=EmbedType.X_LSB_ATTACK,
    #         embed_payload_type=PayloadType.BINARY_FILE,
    #         embed_proc_config=XLSBAttackConfig(
    #             x=x,
    #             fill=True,
    #             msb=False,
    #         ),
    #         embed_payload_metadata=EmbedPayloadMetadata(
    #             payload_filepath=payload_filepath
    #         )
    #     )

"""
    Image Representation Procedure CFGs
"""

class GrayscaleFourpartConfig(BaseModel):
    model_config = ConfigDict(from_attributes=True, frozen=True)

    image_rep_type: Literal[ImageType.GRAYSCALE_FOURPART] = ImageType.GRAYSCALE_FOURPART

class GrayscaleLastMBytesConfig(BaseModel):
    model_config = ConfigDict(from_attributes=True, frozen=True)

    image_rep_type: Literal[ImageType.GRAYSCALE_LAST_M_BYTES] = ImageType.GRAYSCALE_LAST_M_BYTES

    m: int

class GrayscaleThreepartWeightedAvgConfig(BaseModel):
    model_config = ConfigDict(from_attributes=True, frozen=True)

    image_rep_type: Literal[ImageType.GRAYSCALE_THREEPART_WEIGHTED_AVG] = ImageType.GRAYSCALE_THREEPART_WEIGHTED_AVG

    byte_1_weight: float = 0.2
    byte_2_weight: float = 0.3
    byte_3_weight: float = 0.5

    def ret_byte_weights(self):
        return self.byte_1_weight, self.byte_2_weight, self.byte_3_weight

class ImageRepConfig(BaseModel):
    model_config = ConfigDict(from_attributes=True, frozen=True)

    # image_type: ImageType = ImageType.GRAYSCALE_FOURPART
    image_rep_proc_config: Annotated[
        Union[
            GrayscaleFourpartConfig,
            GrayscaleLastMBytesConfig,
            GrayscaleThreepartWeightedAvgConfig
        ],
        Field(default=GrayscaleFourpartConfig(), discriminator='image_rep_type')
    ]

    # @staticmethod
    # def ret_image_rep_config(image_type_str: str):
    #     if image_type_str == ImageType.GRAYSCALE_LAST_M_BYTES:
    #         return ImageRepConfig(
    #             image_type=ImageType.GRAYSCALE_LAST_M_BYTES,
    #             image_rep_proc_config=GrayscaleLastMBytesConfig()
    #         )
    #     elif image_type_str == ImageType.GRAYSCALE_THREEPART_WEIGHTED_AVG:
    #         return ImageRepConfig(
    #             image_type=ImageType.GRAYSCALE_THREEPART_WEIGHTED_AVG,
    #             image_rep_proc_config=GrayscaleThreepartWeightedAvgConfig()
    #         )
    #     elif image_type_str == ImageType.GRAYSCALE_FOURPART:
    #         return ImageRepConfig(
    #             image_type=ImageType.GRAYSCALE_FOURPART,
    #             image_rep_proc_config=None
    #         )
    #     else:
    #         raise NotImplementedError(f'ret_image_rep_config | got {image_type_str}, not implemented')
        
"""
    Image Preprocessing CFGs
"""

class PillowPreprocessConfig(BaseModel):
    model_config = ConfigDict(from_attributes=True, frozen=True)

    image_preprocess_backend: Literal[ImagePreprocessBackend.PILLOW] = ImagePreprocessBackend.PILLOW

class OpenCVPreprocessConfig(BaseModel):
    model_config = ConfigDict(from_attributes=True, frozen=True)

    image_preprocess_backend: Literal[ImagePreprocessBackend.OPENCV] = ImagePreprocessBackend.OPENCV

class ImagePreprocessConfig(BaseModel):
    model_config = ConfigDict(from_attributes=True, frozen=True)

    image_height: int = 100
    image_width: int = 100 
    image_reshape_algo:ImageResamplingFilter = ImageResamplingFilter.BICUBIC

    image_preprocess_config: Annotated[
        Union[
            PillowPreprocessConfig,
            OpenCVPreprocessConfig,
        ],
        Field(default=PillowPreprocessConfig(), discriminator='image_preprocess_backend')
    ]

"""
    Preprocessed Image Lineage CFG
"""

class PreprocessedImageLineage(BaseModel):
    model_config = ConfigDict(from_attributes=True, frozen=True)

    cover_data_config: CoverDataConfig

    image_rep_config: ImageRepConfig
    image_preprocess_config: ImagePreprocessConfig

    embed_payload_config: Union[EmbedPayloadConfig, NA_VAL_TYPE] = Field(default_factory=ret_na_val)

    @computed_field
    @property
    def label(self) -> PreprocessedImageDatasetLabel:
        return PreprocessedImageDatasetLabel.ATTACKED if self.is_attacked() else PreprocessedImageDatasetLabel.BENIGN

    def is_attacked(self) -> bool:
        return self.embed_payload_config != ret_na_val()
    
    def str_hash(self) -> str:
        return hashlib.sha256(pickle.dumps(self)).hexdigest()

    @staticmethod
    def ret_ppil_from_pretrained_model(
        pretrained_model_config: PretrainedModelConfig,
        image_rep_config: ImageRepConfig,
        image_preprocess_config: ImagePreprocessConfig,
        embed_payload_config: Optional[EmbedPayloadConfig] = None
    ):
        return PreprocessedImageLineage(
            cover_data_config=CoverDataConfig(cover_data_cfg=pretrained_model_config),
            image_rep_config=image_rep_config,
            image_preprocess_config=image_preprocess_config,
            embed_payload_config=embed_payload_config if embed_payload_config is not None else ret_na_val()
        )

