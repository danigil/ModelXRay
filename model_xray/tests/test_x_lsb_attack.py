import itertools
import math
import os
import numpy as np

from model_xray.config_classes import PayloadType, XLSBAttackConfig
from model_xray.utils.mal_embedding_utils import _x_lsb_attack_numpy, _x_lsb_attack_numpy_bin, x_lsb_extract

def _check_attack(arr, config, attack_func):
    x = config.x

    config.msb = False
    arr_attacked = attack_func(arr, config)

    config.msb = True

    n_b = arr.dtype.itemsize * 8
    unattacked_bits_amnt = n_b - x

    config.x = unattacked_bits_amnt

    unattacked_bytes_orig = x_lsb_extract(arr, config)
    unattacked_bytes_attacked = x_lsb_extract(arr_attacked, config)

    assert unattacked_bytes_orig == unattacked_bytes_attacked

    config.msb = False
    config.x = x
    extracted_bytes = x_lsb_extract(arr_attacked, config)

    assert extracted_bytes == config.payload_bytes

    return arr_attacked

def _test_x_lsb_attack_single(attack_func: callable, arr:np.ndarray, config: XLSBAttackConfig):
    return _check_attack(arr, config, attack_func)

rng = np.random.default_rng()

def test_x_lsb_attack_random(
    dtypes = [np.uint16, np.uint32, np.uint64, np.float32, np.float64],
    n_ws = [8, 16, 32, 64, 128]
):
    

    config = XLSBAttackConfig(
        x = 3,
        fill=True,
        payload_type=PayloadType.PYTHON_BYTES,
        payload_bytes=None
    )

    for dtype, n_w in itertools.product(dtypes, n_ws):
        dtype_str = dtype.__name__
        n_bits = np.dtype(dtype).itemsize * 8

        for x in range(1, n_bits-1):
            capacity = x*n_w
            byte_capacity = math.ceil(capacity / 8)

            randbytes = os.urandom(byte_capacity)

            config.x = x
            config.payload_bytes = randbytes

            if np.issubdtype(dtype, np.floating):
                arr = rng.random(size=(n_w, ), dtype=dtype)
            elif np.issubdtype(dtype, np.integer):
                arr = rng.integers(low=0, high = 2**n_bits, size=(n_w,), dtype=dtype)
            else:
                raise ValueError(f"Unknown dtype {dtype}")

            arr_attacked_np_bin = _test_x_lsb_attack_single(_x_lsb_attack_numpy_bin, arr, config)

            if x % 8 == 0:
                arr_attacked_numpy = _test_x_lsb_attack_single(_x_lsb_attack_numpy, arr, config)
                assert np.array_equal(arr_attacked_numpy, arr_attacked_np_bin)