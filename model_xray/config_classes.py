from dataclasses import dataclass
from enum import StrEnum
from typing import Union

import hashlib

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

@dataclass
class EmbedPayloadConfig:
    embed_type: EmbedType = EmbedType.X_LSB_ATTACK_FILL
    embed_proc_config: Union[XLSBAttackConfig, None] = None

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

@dataclass
class ImageRepConfig:
    image_type: ImageType = ImageType.GRAYSCALE_FOURPART
    image_rep_config: Union[GrayscaleLastMBytesConfig, None] = None
    