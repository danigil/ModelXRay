import copy
import sys
from typing import Tuple
import numpy as np
import numpy.typing as npt

def mcwa_to_bytes_arr(mcwa: np.ndarray) -> npt.NDArray[np.uint8]:
    assert isinstance(mcwa.dtype.itemsize, int) and mcwa.dtype.itemsize>=1
    newshape = mcwa.shape + (mcwa.dtype.itemsize,)
    dtype = np.dtype('=u1') # force little-endian

    mcwa_decon = np.frombuffer(mcwa.tobytes(order='C'), dtype=dtype).reshape(newshape)
    return np.flip(mcwa_decon, axis=-1)

def bytes_arr_to_mcwa(mcwa: np.ndarray, dtype=np.uint8, shape=None):
    if shape is None:
        newshape = mcwa.shape[0:-1]
    else:
        newshape = shape

    dtype_new = np.dtype(dtype)
    dtype_new = dtype_new.newbyteorder('=')

    return np.frombuffer(np.flip(mcwa, axis=-1).tobytes(order='C'), dtype=dtype_new).reshape(newshape)

from model_xray.config_classes import PayloadType, XLSBAttackConfig

from bitstring import BitArray, Array
from random import getrandbits
import math

# def get_n_randbits(n: int) -> BitArray:
#     randint = getrandbits(n)

#     return BitArray(uint=randint, length=n)

class Array_w_npslice(Array):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def deepcopy(self):
        return Array_w_npslice(copy.deepcopy(self.dtype), BitArray(copy.deepcopy(self.data.tobytes())))
        
    def shape(self):
        return (len(self), self.itemsize)

    def _parse_slices(self, slice_0, slice_1):
        self_shape = self.shape()
        slice_0_new = [slice_0.start, slice_0.stop, slice_0.step]
        slice_1_new = [slice_1.start, slice_1.stop, slice_1.step]

        if slice_0.start is None:
            slice_0_new[0] = 0
        if slice_0.stop is None:
            slice_0_new[1] = self_shape[0]
        if slice_0.step is None:
            slice_0_new[2] = 1
        if slice_1.start is None:
            slice_1_new[0] = 0
        if slice_1.stop is None:
            slice_1_new[1] = self_shape[1]
        if slice_1.step is None:
            slice_1_new[2] = 1

        slice_0 = slice(*slice_0_new)
        slice_1 = slice(*slice_1_new)
        return slice_0, slice_1

    def __getitem__(self, key: Tuple[slice, slice]):
        slice_0, slice_1=self._parse_slices(*key)
        self_shape = self.shape()

        assert slice_0.stop - slice_0.start <= len(self)
        assert slice_1.stop - slice_1.start <= self.itemsize

        d = BitArray()
        for i in range(slice_0.start, slice_0.stop, slice_0.step):
            d.append(self.data[i*self_shape[1]+slice_1.start:i*self_shape[1]+slice_1.stop])

        ret = Array(f'bin{slice_1.stop-slice_1.start}')
        ret.data = d

        return ret

        # lindex = slice_0.start * self_shape[1]+slice_1.start
        # rindex = slice_0.stop * self_shape[1]+slice_1.stop

        # return Array(f'bin{slice_1.stop-slice_1.start}',self.data[lindex:rindex])

    def __setitem__(self, key: Tuple[slice, slice], value: Array):
        if 'bin' not in value.dtype.name:
            raise ValueError("Value must be a binary array")

        slice_0, slice_1=self._parse_slices(*key)
        self_shape = self.shape()

        if slice_0.stop - slice_0.start != len(value):
            raise ValueError("Shape dim0 mismatch")

        if slice_1.stop - slice_1.start != value.itemsize:
            raise ValueError("Shape dim1 mismatch")

        for i, offset in enumerate(range(slice_0.start, slice_0.stop)):
            lindex = offset * self_shape[1]+slice_1.start
            self.data.overwrite(f'0b{value[i]}',lindex)


def x_lsb_attack(host: np.ndarray, x_lsb_attack_config: XLSBAttackConfig, inplace: bool = False) -> np.ndarray:
    if inplace:
        host_curr = host
    else:
        host_curr = np.copy(host)

    orig_dtype = host.dtype
    orig_shape = host.shape

    c = Array_w_npslice(str(orig_dtype).lower())
    c.fromfile(host_curr.tobytes())
    host_curr = c

    n_w = len(host_curr)
    n_b = host_curr.itemsize
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

def x_lsb_extract(host: np.ndarray, x_lsb_attack_config: XLSBAttackConfig) -> bytes:
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