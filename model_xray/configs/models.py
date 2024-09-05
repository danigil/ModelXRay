

from typing import Literal, Optional, Union
from pydantic import BaseModel, Field

from enums import *

NA_VAL = 'NA'

"""
    Cover Data CFGs
"""

class PretrainedModelConfig(BaseModel):
    cover_data_type: Literal[CoverDataTypes.PRETRAINED_MODEL]

    name: str = 'MobileNet'
    repo: ModelRepos = ModelRepos.KERAS
    train_dataset: Literal['imagenet12'] = 'imagenet12'

class DummyCoverDataConfig(BaseModel):
    cover_data_type: Literal[CoverDataTypes.DUMMY]

class CoverDataConfig(BaseModel):
    cover_data_cfg: Union[PretrainedModelConfig, DummyCoverDataConfig] = Field(discriminator='cover_data_type')


"""
    Attack CFGs
"""

class XLSBAttackConfig(BaseModel):
    attack_type: Literal[EmbedType.X_LSB_ATTACK] = EmbedType.X_LSB_ATTACK

    x: int
    fill: bool = True
    msb: bool = False

class XLSBExtractConfig(BaseModel):
    x: int
    n_bytes: int = None
    fill: bool = True
    msb: bool = False
    

class MaleficnetAttackConfig(BaseModel):
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
    payload_bytes_md5: Optional[str] = None
    payload_filepath: Optional[str] = None

class EmbedPayloadConfig(BaseModel):
    embed_type: EmbedType = EmbedType.X_LSB_ATTACK
    embed_payload_type: PayloadType = PayloadType.RANDOM
    embed_proc_config: Optional[Union[XLSBAttackConfig, MaleficnetAttackConfig]] = Field(default=None ,discriminator='attack_type')
    embed_payload_metadata: Optional[EmbedPayloadMetadata] = None

    @staticmethod
    def ret_x_lsb_attack_fill_config(x: int, payload_filepath: Optional[str] = None):
        if payload_filepath is not None:
            return EmbedPayloadConfig.ret_filebytes_x_lsb_attack_fill_config(x, payload_filepath)
        else:
            return EmbedPayloadConfig.ret_random_x_lsb_attack_fill_config(x)

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
    Image Representation Procedure CFGs
"""

# class ImageRepProcConfig(BaseModel):
#     pass

class GrayscaleLastMBytesConfig(BaseModel):
    image_rep_type: Literal[ImageType.GRAYSCALE_LAST_M_BYTES] = ImageType.GRAYSCALE_LAST_M_BYTES

    m: int

class GrayscaleThreepartWeightedAvgConfig(BaseModel):
    image_rep_type: Literal[ImageType.GRAYSCALE_THREEPART_WEIGHTED_AVG] = ImageType.GRAYSCALE_THREEPART_WEIGHTED_AVG

    byte_1_weight: float = 0.2
    byte_2_weight: float = 0.3
    byte_3_weight: float = 0.5

    def ret_byte_weights(self):
        return self.byte_1_weight, self.byte_2_weight, self.byte_3_weight

class ImageRepConfig(BaseModel):
    # image_type: ImageType = ImageType.GRAYSCALE_FOURPART
    image_rep_proc_config: Optional[Union[GrayscaleLastMBytesConfig, GrayscaleThreepartWeightedAvgConfig]] = Field(default=None ,discriminator='image_rep_type')

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

class ImagePreprocessConfig(BaseModel):
    image_height: int = 100
    image_width: int = 100 
    image_reshape_algo:ImageResamplingFilter = ImageResamplingFilter.BICUBIC

"""
    Preprocessed Image Lineage CFGs
"""

class PreprocessedImageLineage(BaseModel):
    cover_data_config: CoverDataConfig

    image_rep_config: ImageRepConfig
    image_preprocess_config: ImagePreprocessConfig

    embed_payload_config: Optional[EmbedPayloadConfig] = Field(default=NA_VAL)

    # @staticmethod
    # def ret_default_preprocessed_image_w_x_lsb_attack(
    #     pretrained_model_config: Optional[PretrainedModelConfig] = None,
    #     image_rep_config: Optional[ImageRepConfig] = None,
    #     image_preprocess_config: Optional[ImagePreprocessConfig] = None,
    #     x: Optional[int] = None,
    #     payload_filepath: Optional[str] = None
    # ):
    #     return PreprocessedImageLineage(
    #         pretrained_model_config=PretrainedModelConfig() if pretrained_model_config is None else pretrained_model_config,
    #         image_rep_config=ImageRepConfig() if image_rep_config is None else image_rep_config,
    #         image_preprocess_config=ImagePreprocessConfig() if image_preprocess_config is None else image_preprocess_config,
    #         embed_payload_config=EmbedPayloadConfig.ret_x_lsb_attack_fill_config(x, payload_filepath=payload_filepath) if x is not None else None
    #     )

    def is_attacked(self) -> bool:
        return self.embed_payload_config is not None and isinstance(self.embed_payload_config.embed_proc_config, AttackConfig)

    def label(self) -> PreprocessedImageDatasetLabel:
        return PreprocessedImageDatasetLabel.ATTACKED if self.is_attacked() else PreprocessedImageDatasetLabel.BENIGN

