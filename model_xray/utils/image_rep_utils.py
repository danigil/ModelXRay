import numpy as np
import numpy.typing as npt

from model_xray.utils.general_utils import ndarray_to_bytes_arr

from model_xray.config_classes import GrayscaleThreepartWeightedAvgConfig, ImageRepConfig, ImageType, GrayscaleLastMBytesConfig

def calc_closest_square(num: int) -> int:
    return int(np.ceil(np.sqrt(num)) ** 2)

def is_square(num: int) -> bool:
    return calc_closest_square(num) == num

def ret_padded_square(data: np.ndarray) -> np.ndarray:
    n_models, n_weights = data.shape

    closest_square = calc_closest_square(n_weights)
    closest_square_sqrt = int(np.sqrt(closest_square))

    data_padded = np.pad(data, ((0, 0), (0, closest_square-n_weights),), mode='constant', constant_values=0)

    data_padded_reshaped = data_padded.reshape(data_padded.shape[:-1] + (closest_square_sqrt, closest_square_sqrt))

    return data_padded_reshaped

def _grayscale_lastmbytes(data: np.ndarray, config: ImageRepConfig) -> np.ndarray:
    """ Grayscale image representation of the last m bytes of the data, reshaped to m squares and padded to the closest square number to form a big square image.
    
        >>> arr = np.array([[0.98, 0.1, -1.25], [0.123, 0.6, 1.0]], dtype=np.float32)
        >>> config = GrayscaleLastMBytesConfig(m=4)
        >>> _grayscale_lastmbytes(arr, config)
        array([[[ 63  61 122 204]
                [191   0 160   0]
                [225 204  72 205]
                [  0   0   0   0]]

                [[ 61  63 251  25]
                [ 63   0 128   0]
                [231 153 109 154]
                [  0   0   0   0]]], dtype=uint8)

    """
    assert config.image_rep_config is not None, "grayscale_lastmbytes image rep expects a config with image_rep_config, got None"
    assert isinstance(config.image_rep_config, GrayscaleLastMBytesConfig), f"grayscale_lastmbytes image rep expects a config with GrayscaleLastMBytesConfig, got {type(config.image_rep_config)}"

    m = config.image_rep_config.m

    assert is_square(m), f"grayscale_lastmbytes image rep expects m to be a square number, got: {m}" 
    
    data_n_bytes = data.dtype.itemsize
    assert data_n_bytes >= m, f"grayscale_lastmbytes image rep expects data to have at least m={m} bytes, got {data_n_bytes}-byte long data"

    assert data.ndim == 2, f"grayscale_lastmbytes image rep expects 2D data (n_models, n_weights), got {data.ndim}D data"

    data_bytes = ndarray_to_bytes_arr(data)

    last_m_bytes = data_bytes[..., -m:]

    n_models, n_weights = data.shape

    closest_square = calc_closest_square(n_weights)
    closest_square_sqrt = int(np.sqrt(closest_square))

    last_bytes_padded = np.pad(last_m_bytes, ((0, 0), (0, closest_square-n_weights), (0, 0)), mode='constant', constant_values=0)
    last_bytes_padded = np.swapaxes(last_bytes_padded, -1, -2)

    last_bytes_padded_reshaped = last_bytes_padded.reshape(last_bytes_padded.shape[:-1] + (closest_square_sqrt, closest_square_sqrt))

    n, m, d, _ = last_bytes_padded_reshaped.shape
    s = int(np.sqrt(m))
    
    # Reshape to (n, s, s, d, d)
    reshaped = last_bytes_padded_reshaped.reshape(n, s, s, d, d)
    
    # Transpose to (n, s, d, s, d)
    transposed = reshaped.transpose(0, 1, 3, 2, 4)
    
    # Reshape to final shape (n, d*s, d*s)
    result = transposed.reshape(n, d*s, d*s)
    
    return result

def _grayscale_fourpart(data: npt.NDArray[np.float32], config: ImageRepConfig = None) -> np.ndarray:
    assert data.dtype == np.float32 or data.dtype.itemsize==4, f"grayscale_fourpart image rep expects 4-byte long data, got {data.dtype.itemsize}-byte long data"
    
    gs_l_m_cfg = ImageRepConfig(
        image_type = ImageType.GRAYSCALE_LAST_M_BYTES,
        image_rep_config= GrayscaleLastMBytesConfig(m=4)
    )

    if data.ndim == 1:
        passed_data = data.reshape(1, -1)
    else:
        passed_data = data

    return _grayscale_lastmbytes(passed_data, gs_l_m_cfg)

def _grayscale_fourpart_reverse(data: npt.NDArray[np.uint8], config: ImageRepConfig = None) -> npt.NDArray[np.float32]:
    pass

def _grayscale_threepart_weighted_avg(data: npt.NDArray[np.float32], config: ImageRepConfig = None) -> np.ndarray:
    assert config.image_rep_config is not None, "grayscale_weighted_avg image rep expects a config with image_rep_config, got None"
    assert isinstance(config.image_rep_config, GrayscaleThreepartWeightedAvgConfig), f"grayscale_weighted_avg image rep expects a config with GrayscaleWeightedAvgConfig, got {type(config.image_rep_config)}"

    byte_1_weight, byte_2_weight, byte_3_weight = config.image_rep_config.ret_byte_weights()

    assert isinstance(byte_1_weight, float) and isinstance(byte_2_weight, float) and isinstance(byte_3_weight, float), f"grayscale_weighted_avg image rep expects byte weights to be floats, got {type(byte_1_weight)}, {type(byte_2_weight)}, {type(byte_3_weight)}"
    
    is_in_range = lambda x: 0.0 <= x <= 1.0 
    
    assert is_in_range(byte_1_weight) and is_in_range(byte_2_weight) and is_in_range(byte_3_weight), f"grayscale_weighted_avg image rep expects byte weights to be in range [0.0, 1.0], got {byte_1_weight}, {byte_2_weight}, {byte_3_weight}"

    assert np.isclose(np.sum([byte_1_weight, byte_2_weight, byte_3_weight]), [1.0]), f"grayscale_weighted_avg image rep expects byte weights to sum to 1.0, got {byte_1_weight}, {byte_2_weight}, {byte_3_weight}: sum={np.sum([byte_1_weight, byte_2_weight, byte_3_weight])}"

    assert data.ndim == 2, f"grayscale_weighted_avg image rep expects 2D data (n_models, n_weights), got {data.ndim}D data"

    data_bytes = ndarray_to_bytes_arr(data)
    # print("data_bytes:\n", data_bytes)

    last_2_bytes = data_bytes[..., -2:]

    first_2_bytes = data_bytes[..., :2]
    first_2_bytes_unpacked = np.unpackbits(first_2_bytes, axis=-1, bitorder='big')
    # print("first_2_bytes_unpacked:\n", first_2_bytes_unpacked)

    sign_bits = first_2_bytes_unpacked[..., [0]]
    exponent_size = 8
    mantissa_remainder_bits = first_2_bytes_unpacked[..., 1+exponent_size:]

    concated_bits = np.concatenate((sign_bits, mantissa_remainder_bits), axis=-1)
    concated_bytes = np.packbits(concated_bits, axis=-1, bitorder='big')
    # print("concated_bytes:\n", concated_bytes)

    all_bytes = np.concatenate((concated_bytes, last_2_bytes), axis=-1)
    # print("all_bytes:\n", all_bytes)

    avereged_bytes = np.average(all_bytes, axis=-1, weights=[byte_1_weight, byte_2_weight, byte_3_weight]).astype(np.uint8)

    return ret_padded_square(avereged_bytes)

image_rep_map = {
    ImageType.GRAYSCALE_LAST_M_BYTES: _grayscale_lastmbytes,
    ImageType.GRAYSCALE_FOURPART: _grayscale_fourpart,
    ImageType.GRAYSCALE_THREEPART_WEIGHTED_AVG: _grayscale_threepart_weighted_avg,
}
