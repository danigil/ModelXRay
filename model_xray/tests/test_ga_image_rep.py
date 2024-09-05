
import numpy as np
from model_xray.config_classes import *
from model_xray.utils.image_rep_utils import _grayscale_threepart_weighted_avg

dt_uint8_be = np.dtype('>u1')
dt_float32_be = np.dtype('>f4')

dt_float32_ne = np.dtype('=f4')


def test_ga_simple():
    config=ImageRepConfig(
        image_type=ImageType.GRAYSCALE_THREEPART_WEIGHTED_AVG,
        image_rep_config=GrayscaleThreepartWeightedAvgConfig(),
    )

    expected_end_result = np.array([[
        [50,]
    ]],
    dtype=dt_uint8_be)

    a_input = np.array([[0.123]], dtype=dt_float32_ne)

    result = _grayscale_threepart_weighted_avg(a_input, config=config)

    assert np.array_equal(result, expected_end_result)

    expected_end_result = np.array(
        [[[148, 143],
        [  0,   0]],

       [[  5, 147],
        [  0,   0]]], dtype=dt_uint8_be)

    a_input = np.array([[0.123, 2.13],[77.0, -1.23]], dtype=dt_float32_ne)

    result = _grayscale_threepart_weighted_avg(a_input, config=config)

    assert np.array_equal(result, expected_end_result)