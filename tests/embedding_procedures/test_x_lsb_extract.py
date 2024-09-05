import numpy as np

from model_xray.config_classes import XLSBExtractConfig
from model_xray.utils.mal_embedding_utils import x_lsb_extract

rng = np.random.default_rng()

def test_x_lsb_extract_bit_at_end():
    arr = np.array([
        0b11111110, 0b11111100, 0b11111000,
        0b11110000, 0b11100000, 0b11000000,
        0b10000000, 0b00000000
    ], dtype=np.uint8)
    payload = np.array([255,], dtype=np.uint8).tobytes()

    config = XLSBExtractConfig(
        x=1,
        fill=True,
        msb=False,
    )
    # arr_attacked = _x_lsb_attack_numpy_bin(arr, config)

    arr_attacked = np.array([
        0b11111111, 0b11111101, 0b11111001,
        0b11110001, 0b11100001, 0b11000001,
        0b10000001, 0b00000001
    ], dtype=np.uint8)
    # assert np.array_equal(arr_attacked, arr_attacked_expected)

    extracted_bytes = x_lsb_extract(arr_attacked, config)

    assert extracted_bytes == payload

def test_x_lsb_extract_encoded_msg(
    msg: str = "Hello, World!",
    x_end = 8
):
    msg_bytes = msg.encode()

    msg_bytes_arr = np.frombuffer(msg_bytes, dtype=np.uint8)
    msg_bits_arr = np.unpackbits(msg_bytes_arr)

    n_bits = len(msg_bits_arr)
    n_bytes = len(msg_bytes)

    for x in range(1, x_end):
        pad_to_x_amnt = x - (n_bits % x)
        if pad_to_x_amnt == x:
            pad_to_x_amnt = 0
        msg_bits_arr_copy = np.pad(msg_bits_arr.copy(), (0, pad_to_x_amnt), mode='constant', constant_values=0)

        new_n_bits = len(msg_bits_arr_copy)

        msg_bits_arr_reshaped = np.reshape(msg_bits_arr_copy, (new_n_bits // x, x))

        z_arr = np.zeros((new_n_bits // x, 8 - x), dtype=np.uint8)

        msg_bits_arr_padded = np.concatenate((z_arr, msg_bits_arr_reshaped), axis=1)

        msg_bits_arr_bytes = np.packbits(msg_bits_arr_padded, axis=1)

        config = XLSBExtractConfig(
            x=x,
            fill=False,
            msb=False,
            n_bytes=len(msg_bytes)
        )

        extracted_bytes = x_lsb_extract(msg_bits_arr_bytes, config)

        assert extracted_bytes == msg_bytes

        # msg_bits_arr_padded = np.pad(msg_bits_arr_copy, (0, (len(msg_bits_arr) - x) % 8), mode='constant', constant_values=0)
        pass

    # arr = rng.integers(0, 255, size=(len(msg),), dtype=np.uint8)

    # payload = msg.encode()

    # for x in range(1,8):
    #     config = XLSBAttackConfig(
    #         x=x,
    #         fill=False,
    #         payload_type=PayloadType.PYTHON_BYTES,
    #         payload_bytes=payload
    #     )

    #     arr_attacked = _x_lsb_attack_numpy_bin(arr, config)

    #     extracted_bytes = x_lsb_extract(arr_attacked, config)

    #     assert extracted_bytes == config.payload_bytes