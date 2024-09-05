import hashlib
from typing import Callable, Dict, Optional
import numpy as np

from model_xray.utils.general_utils import ndarray_to_bytes_arr, bytes_arr_to_ndarray
from model_xray.config_classes import EmbedPayloadConfig, EmbedPayloadMetadata, EmbedType, PayloadType, XLSBAttackConfig, XLSBExtractConfig

import math

class MalBytes:
    def __init__(self, embed_payload_config: Optional[EmbedPayloadConfig] = None, appended_bytes: Optional[bytes] = None):
        self.embed_payload_config = embed_payload_config
        self._appended_bytes = appended_bytes

        if self.embed_payload_config is not None:
            if self.embed_payload_config.embed_payload_type == PayloadType.PYTHON_BYTES:
                if self._appended_bytes is None:
                    raise ValueError("MalBytes: appended_bytes must be provided if embed_payload_type is PYTHON_BYTES")

            if self.embed_payload_config.embed_payload_type == PayloadType.BINARY_FILE:
                if self.embed_payload_config.embed_payload_metadata is None:
                    raise ValueError("MalBytes: embed_payload_metadata must be provided if embed_payload_type is BINARY_FILE")

                if self.embed_payload_config.embed_payload_metadata.payload_filepath is None:
                    raise ValueError("MalBytes: payload_filepath must be provided if embed_payload_type is BINARY_FILE")


    def get_bytes(self, n_bytes:Optional[int] = None) -> bytes:
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

"""
def _x_lsb_attack_bitstring(host: np.ndarray, x_lsb_attack_config: XLSBAttackConfig, inplace: bool = False) -> np.ndarray:
    if inplace:
        host_curr = host
    else:
        host_curr = np.copy(host)

    assert np.array_equal(host_curr, host), "_x_lsb_attack_bitstring: host_curr is not a copy of host"

    orig_dtype = host.dtype
    orig_shape = host.shape

    dtype_map = {
        np.uint8: 'uintle8',
        np.uint16: 'uintle16',
        np.uint32: 'uintle32',
        np.uint64: 'uintle64',

        np.int8: 'intle8',
        np.int16: 'intle16',
        np.int32: 'intle32',
        np.int64: 'intle64',

        np.float16: 'floatle16',
        np.float32: 'floatle32',
        np.float64: 'floatle64',
    }

    def dtype_to_bitstring(dtype):
        if isinstance(dtype, str):
            if "uint" in dtype:
                return dtype.replace("uint", "uintle")

            if "int" in dtype:
                return dtype.replace("int", "intle")

            if "float" in dtype:
                return dtype.replace("float", "floatle")
        else:
            return dtype_map[dtype]

   

    c = Array_w_npslice(dtype_to_bitstring(str(host.dtype).lower()))
    c.fromfile(host_curr.tobytes())
    host_curr = c

    n_w = len(host_curr)
    n_b = host_curr.itemsize

    if n_b < x_lsb_attack_config.x:
        raise ValueError(f"n_b must be greater than or equal to x, n_b: {n_b}, x: {x_lsb_attack_config.x}")

    capacity = n_w*x_lsb_attack_config.x

    # print(f'capacity: {capacity}, n_b: {n_b}')

    fill = x_lsb_attack_config.fill
    lsb = x_lsb_attack_config.x

    if x_lsb_attack_config.payload_type == PayloadType.BINARY_FILE:
        with open(x_lsb_attack_config.payload_filepath, 'rb') as malware_file:
            malware_bytes = malware_file.read()
        bits = BitArray(bytes=malware_bytes)
    elif x_lsb_attack_config.payload_type == PayloadType.PYTHON_BYTES:
        bits = BitArray(bytes=x_lsb_attack_config.payload_bytes)
    elif x_lsb_attack_config.payload_type == PayloadType.RANDOM:
        maxlen = math.floor(math.log2(sys.maxsize))
        q = capacity//maxlen
        r = capacity%maxlen
        if q>0:
            randint = getrandbits(maxlen)
            bits = BitArray(uint=randint, length=maxlen)
            for i in range(q-1):
                randint = getrandbits(maxlen)
                bits.append(BitArray(uint=randint, length=maxlen))
            if r>0:
                randint = getrandbits(r)
                bits.append(BitArray(uint=randint, length=r))
        else:
            randint = getrandbits(r)
            bits = BitArray(uint=randint, length=r) 
    
    bits = BitArray(bin=f'0b{bits.bin}')
    # print(f'payload bits: {bits.bin}, len(bits): {len(bits)}')

    if fill:
        dupe_amount = math.ceil(capacity/len(bits))
        # print(f'dupe_amount: {dupe_amount}')

        bits *= dupe_amount
        # print(f'len(bits): {len(bits)}')
    else:
        if len(bits)>capacity:
            raise ValueError(f"Malware is too large for the host, len(bits): {len(bits)}, n_b: {capacity}")

    reshaped = Array(f'bin{lsb}', bits[:capacity])
    remainder = bits[capacity:]
    if len(remainder)>0:
        remainder = Array(f'bin{len(remainder)}', remainder)

    # print(f'reshaped: {reshaped}, remainder: {remainder}')

    host_curr[:, (n_b-lsb):] = reshaped
    if not fill and len(remainder)>0:
        host_curr[:, (n_b-len(remainder)):] = remainder

    ret_arr = np.frombuffer(host_curr.tobytes(), dtype=orig_dtype).reshape(orig_shape)

    return ret_arr

def x_lsb_extract_old(host: np.ndarray, x_lsb_attack_config: XLSBAttackConfig) -> bytes:
    print(f"x_lsb_extract: msb: {x_lsb_attack_config.msb}")
    
    orig_dtype = host.dtype
    orig_shape = host.shape

    c = Array_w_npslice(str(orig_dtype).lower())
    c.fromfile(host.tobytes())
    host = c

    n_w = len(host)
    n_b = host.itemsize
    capacity = n_w*x_lsb_attack_config.x

    # print(f'capacity: {capacity}, n_b: {n_b}')

    lsb = x_lsb_attack_config.x

    if x_lsb_attack_config.msb:
        bits = host[:, :(n_b-lsb)]
    else:
        bits = host[:, (n_b-lsb):]
    # print(f'bits: {bits}')

    # bits = BitArray(f'0b{bits.bin}')

    return bits.tobytes()
"""

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





embed_type_map: Dict[EmbedType, Callable] = {
    EmbedType.X_LSB_ATTACK: x_lsb_attack
}
