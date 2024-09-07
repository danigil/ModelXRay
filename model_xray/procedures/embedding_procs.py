from __future__ import annotations
import hashlib
# import logging
# log = logging.getLogger(__name__)
from typing import Callable, Dict, Optional, get_args, get_type_hints
import numpy as np
import numpy.typing as npt

from model_xray.configs.types import COVER_DATA_TYPE, DL_MODEL_TYPE
from model_xray.utils.general_utils import ndarray_to_bytes_arr, bytes_arr_to_ndarray, try_coerce_data
from model_xray.utils.logging_utils import request_logger
logger = request_logger(__name__)
from model_xray.config_classes import *

import math

class MalBytes:
    def __init__(self, embed_payload_config: Optional[EmbedPayloadConfig] = None, appended_bytes: Optional[bytes] = None):
        self.embed_payload_config = embed_payload_config
        self._appended_bytes = appended_bytes


    def get_bytes(self, n_bytes:Optional[int] = None) -> bytes:
        if self.embed_payload_config is not None:
            if self.embed_payload_config.embed_payload_type == PayloadType.PYTHON_BYTES:
                if self._appended_bytes is None:
                    raise ValueError("MalBytes: appended_bytes must be provided if embed_payload_type is PYTHON_BYTES")

            if self.embed_payload_config.embed_payload_type == PayloadType.BINARY_FILE:
                if self.embed_payload_config.embed_payload_metadata is None:
                    raise ValueError("MalBytes: embed_payload_metadata must be provided if embed_payload_type is BINARY_FILE")

                if self.embed_payload_config.embed_payload_metadata.payload_filepath is None:
                    raise ValueError("MalBytes: payload_filepath must be provided if embed_payload_type is BINARY_FILE")

        if self.embed_payload_config is None and self._appended_bytes is not None:
            return self._appended_bytes

        if self.embed_payload_config is None:
            self.embed_payload_config = EmbedPayloadConfig()

        if self.embed_payload_config.embed_payload_type == PayloadType.BINARY_FILE:
            with open(self.embed_payload_config.embed_payload_metadata.payload_filepath, 'rb') as f:
                ret_bytes = f.read()
        elif self.embed_payload_config.embed_payload_type == PayloadType.PYTHON_BYTES:
            ret_bytes = self._appended_bytes
        elif self.embed_payload_config.embed_payload_type == PayloadType.RANDOM:
            rng = np.random.default_rng()

            if n_bytes is None:
                raise ValueError("MalBytes.get_bytes: n_bytes must be provided if embed_payload_type is RANDOM")

            ret_bytes = rng.bytes(n_bytes)

        self.embed_payload_config.embed_payload_metadata.payload_bytes_md5 = self.ret_md5(ret_bytes)

        return ret_bytes

    def set_appended_bytes(self, appended_bytes: bytes):
        self._appended_bytes = appended_bytes
        if self.embed_payload_config is not None:
            self.embed_payload_config.embed_payload_type = PayloadType.PYTHON_BYTES


    def ret_md5(self, mal_bytes: bytes):
        return hashlib.md5(mal_bytes).hexdigest()

def x_lsb_attack(host: np.ndarray, x_lsb_attack_config: XLSBAttackConfig, mal_bytes_gen: MalBytes, inplace: bool = False,) -> np.ndarray:
    
    if x_lsb_attack_config.x % 8 != 0:
        return _x_lsb_attack_numpy_bin(host, x_lsb_attack_config, mal_bytes_gen=mal_bytes_gen, inplace=inplace)
    else:
        return _x_lsb_attack_numpy(host, x_lsb_attack_config, mal_bytes_gen=mal_bytes_gen, inplace=inplace)    

def _x_lsb_attack_numpy_bin(host: np.ndarray, x_lsb_attack_config: XLSBAttackConfig, mal_bytes_gen: MalBytes, inplace: bool = False) -> np.ndarray:
    host_bytes = ndarray_to_bytes_arr(host)

    n_w = len(host_bytes)
    capacity = x_lsb_attack_config.x*n_w
    byte_capacity = math.ceil(capacity / 8)

    n_total_bits = host.dtype.itemsize * 8
    n_unattacked_bits = n_total_bits - x_lsb_attack_config.x

    mal_bytes = mal_bytes_gen.get_bytes(n_bytes = byte_capacity)
    mal_bytes = np.frombuffer(mal_bytes, dtype=np.uint8)[:byte_capacity].reshape((byte_capacity, 1))

    if n_unattacked_bits == 0:
        return bytes_arr_to_ndarray(mal_bytes, dtype=host.dtype, shape=host.shape)

    assert n_unattacked_bits >= 0, f"_x_lsb_attack_numpy_bin: n_unattacked_bits must be greater than or equal to 0, got: {n_unattacked_bits}"

    host_bytes_unpacked = np.unpackbits(host_bytes, axis=-1, count=n_unattacked_bits, bitorder='big')

    mal_bits = np.unpackbits(mal_bytes, bitorder='big')[:capacity].reshape((n_w, x_lsb_attack_config.x))

    stacked = np.hstack((host_bytes_unpacked, mal_bits))

    host_bytes_packed = np.packbits(stacked, axis=-1, bitorder='big')

    return bytes_arr_to_ndarray(host_bytes_packed, dtype=host.dtype, shape=host.shape)

def _x_lsb_attack_numpy(host: np.ndarray, x_lsb_attack_config: XLSBAttackConfig, mal_bytes_gen: MalBytes, inplace: bool = False) -> np.ndarray:
    assert x_lsb_attack_config.x % 8 == 0, "_x_lsb_attack_numpy: x must be a multiple of 8"

    host_as_bytearr = ndarray_to_bytes_arr(host).copy()

    n_w = len(host_as_bytearr)
    n_bytes_to_change_in_each_weight = x_lsb_attack_config.x//8
    n_bytes_total = n_bytes_to_change_in_each_weight * n_w

    if n_bytes_total < 1:
        return host

    mal_bytes = mal_bytes_gen.get_bytes(n_bytes=n_bytes_total)
    mal_bytes = np.frombuffer(mal_bytes, dtype=np.uint8)[:n_bytes_total].reshape((n_w, n_bytes_to_change_in_each_weight))

    # mal_bytes = mal_bytes[:n_bytes_total].reshape((n_bytes_total, 1))

    host_as_bytearr[..., -n_bytes_to_change_in_each_weight:] = mal_bytes

    return bytes_arr_to_ndarray(host_as_bytearr, dtype=host.dtype)

def maleficnet_attack(host: np.ndarray, maleficnet_attack_config: MaleficnetAttackConfig, mal_bytes_gen: MalBytes, inplace: bool = False) -> np.ndarray:
    pass

def x_lsb_extract(host: np.ndarray, x_lsb_extract_config: XLSBExtractConfig) -> bytes:
    host_bytes = ndarray_to_bytes_arr(host)
    msb = x_lsb_extract_config.msb

    if x_lsb_extract_config.fill:
        end = None
    else:
        end = x_lsb_extract_config.n_bytes

    if x_lsb_extract_config.x % 8 == 0:

        n_bytes_to_read_in_each_weight = x_lsb_extract_config.x//8

        if msb:
            ret = host_bytes[..., :n_bytes_to_read_in_each_weight]
        else:
            ret = host_bytes[..., -n_bytes_to_read_in_each_weight:]

        return ret.tobytes()[:end]
    else:
        if msb:
            host_last_x_bits = np.unpackbits(host_bytes, axis=-1, count=None, bitorder='big')[..., :x_lsb_extract_config.x].flatten()
        else:
            host_last_x_bits = np.unpackbits(host_bytes, axis=-1, count=None, bitorder='big')[..., -x_lsb_extract_config.x:].flatten()

        if len(host_last_x_bits) % 8 != 0:
            host_last_x_bits = np.pad(host_last_x_bits, (0, 8 - len(host_last_x_bits) % 8))

        return np.packbits(host_last_x_bits).tobytes()[:end]


def execute_embedding_proc(*, cover_data: COVER_DATA_TYPE, embed_payload_config: EmbedPayloadConfig, validate_host:bool=True, try_coerce_host=True, **additional_kwargs):
    embed_type = embed_payload_config.embed_proc_config.attack_type

    mal_bytes_gen = MalBytes(embed_payload_config=embed_payload_config, appended_bytes=None)

    embed_func = embed_type_map.get(embed_type, None)
    if embed_func is None:
        raise ValueError(f"execute_embedding_proc: embed_type {embed_type} not supported")

    host_expected_type = get_type_hints(embed_func).get('host', None)

    data = cover_data
    
    if validate_host and host_expected_type is not None:
        if not isinstance(cover_data, host_expected_type):
            if not try_coerce_host:
                raise ValueError(f"execute_embedding_proc: cover_data must be of type {host_expected_type}, got: {type(cover_data)}")
            
            coerced_cover_data = try_coerce_data(cover_data, host_expected_type)
            if coerced_cover_data is None:
                raise ValueError(f"execute_embedding_proc: cover_data must be of type {host_expected_type}, got: {type(cover_data)}, and could not be coerced to {host_expected_type}")
                
            data = coerced_cover_data

    data_embedded = embed_func(data, embed_payload_config.embed_proc_config, mal_bytes_gen=mal_bytes_gen, **additional_kwargs)

    if data_embedded is None:
        raise ValueError(f"execute_embedding_proc: embed_func {embed_func} returned None")

    if isinstance(data_embedded, type(cover_data)):
        return data_embedded

    data_embedded_coerced = try_coerce_data(data_embedded, type(cover_data), reference_data=cover_data)
    if data_embedded_coerced is None:
        raise ValueError(f"execute_embedding_proc: data_embedded must be of type {type(cover_data)}, got: {type(data_embedded)}, and could not be coerced to {type(cover_data)}")
    
    return data_embedded_coerced

embed_type_map: Dict[EmbedType, Callable] = {
    EmbedType.X_LSB_ATTACK: x_lsb_attack
}
