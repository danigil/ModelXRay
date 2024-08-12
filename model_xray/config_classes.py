from dataclasses import dataclass
from enum import StrEnum
from typing import Union

class EmbedType(StrEnum):
    X_LSB_ATTACK_FILL = 'x_lsb_attack_fill'

class PayloadType(StrEnum):
    RANDOM = 'random'
    BINARY_FILE = 'binary_file'
    PYTHON_BYTES = 'python_bytes'

@dataclass
class XLSBAttackConfig:
    x: int
    fill: bool = True
    payload_bytes: Union[None, bytes] = None
    payload_type: PayloadType = PayloadType.RANDOM
    payload_filepath: Union[None, str] = None

@dataclass
class EmbedPayloadConfig:
    embed_type: EmbedType = EmbedType.X_LSB_ATTACK_FILL
    embed_proc_config: Union[XLSBAttackConfig] = None
