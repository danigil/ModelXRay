import itertools
import math
import os
import numpy as np
import numpy.typing as npt

from model_xray.config_classes import PayloadType, XLSBAttackConfig, EmbedPayloadConfig
from model_xray.procedures.embedding_procs import _x_lsb_attack_numpy, _x_lsb_attack_numpy_bin, x_lsb_extract, MalBytes

def _check_attack(arr:np.ndarray, config: EmbedPayloadConfig, mal_bytes_gen: MalBytes, attack_func: callable,):
    x_lsb_attack_config = config.embed_proc_config
    x = x_lsb_attack_config.x

    x_lsb_attack_config.msb = False
    arr_attacked = attack_func(arr, x_lsb_attack_config, mal_bytes_gen)

    x_lsb_attack_config.msb = True

    n_b = arr.dtype.itemsize * 8
    unattacked_bits_amnt = n_b - x

    x_lsb_attack_config.x = unattacked_bits_amnt

    unattacked_bytes_orig = x_lsb_extract(arr, x_lsb_attack_config)
    unattacked_bytes_attacked = x_lsb_extract(arr_attacked, x_lsb_attack_config)

    assert unattacked_bytes_orig == unattacked_bytes_attacked

    x_lsb_attack_config.msb = False
    x_lsb_attack_config.x = x
    extracted_bytes = x_lsb_extract(arr_attacked, x_lsb_attack_config)

    assert extracted_bytes == mal_bytes_gen._appended_bytes

    return arr_attacked

def _test_x_lsb_attack_single(arr:np.ndarray, config: XLSBAttackConfig, mal_bytes_gen: MalBytes, attack_func: callable,):
    return _check_attack(arr, config, mal_bytes_gen, attack_func)

rng = np.random.default_rng()

def test_x_lsb_attack_random(
    dtypes = [np.uint16, np.uint32, np.uint64, np.float32, np.float64],
    n_ws = [8, 16, 32, 64, 128]
):
    

    config = EmbedPayloadConfig.ret_random_x_lsb_attack_fill_config(x=3)
    config.embed_payload_type = PayloadType.PYTHON_BYTES

    mal_bytes_gen = MalBytes(embed_payload_config=config, appended_bytes=None)

    for dtype, n_w in itertools.product(dtypes, n_ws):
        dtype_str = dtype.__name__
        n_bits = np.dtype(dtype).itemsize * 8

        for x in range(1, n_bits-1):
            capacity = x*n_w
            byte_capacity = math.ceil(capacity / 8)

            randbytes = os.urandom(byte_capacity)

            config.embed_proc_config.x = x
            mal_bytes_gen.set_appended_bytes(randbytes)

            if np.issubdtype(dtype, np.floating):
                arr = rng.random(size=(n_w, ), dtype=dtype)
            elif np.issubdtype(dtype, np.integer):
                arr = rng.integers(low=0, high = 2**n_bits, size=(n_w,), dtype=dtype)
            else:
                raise ValueError(f"Unknown dtype {dtype}")

            arr_attacked_np_bin = _test_x_lsb_attack_single(arr, config, mal_bytes_gen, _x_lsb_attack_numpy_bin)

            if x % 8 == 0:
                arr_attacked_numpy = _test_x_lsb_attack_single(arr, config, mal_bytes_gen, _x_lsb_attack_numpy)
                assert np.array_equal(arr_attacked_numpy, arr_attacked_np_bin)