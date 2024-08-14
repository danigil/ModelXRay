import numpy as np

from model_xray.utils.image_rep_utils import _grayscale_fourpart

dt_uint8_be = np.dtype('>u1')
dt_float32_be = np.dtype('>f4')

dt_float32_ne = np.dtype('=f4')

def test_gf_simple():
    expected_end_result = np.array([[
        [61, 251,],
        [231, 109,]
    ]],
    dtype=dt_uint8_be)

    a_input = np.array([0.123], dtype=dt_float32_ne)

    result = _grayscale_fourpart(a_input)

    assert np.array_equal(result, expected_end_result)

    expected_end_result = np.array([[
        [ 61,  64, 251,   8],
        [ 66,   0, 154,   0],
        [231,  81, 109, 236],
        [  0,   0,   0,   0]]], dtype=dt_uint8_be)

    a_input = np.array([0.123, 2.13, 77.0], dtype=dt_float32_ne)

    result = _grayscale_fourpart(a_input)

    assert np.array_equal(result, expected_end_result)