import numpy as np
import numpy.typing as npt

from model_xray.utils.general_utils import ndarray_to_bytes_arr, bytes_arr_to_ndarray

from model_xray.config_classes import ImageRepConfig, ImageType, GrayscaleLastMBytesConfig

def calc_closest_square(num: int) -> int:
    return int(np.ceil(np.sqrt(num)) ** 2)

def is_square(num: int) -> bool:
    return calc_closest_square(num) == num

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

    # return last_bytes_padded_reshaped

    n, m, d, _ = last_bytes_padded_reshaped.shape
    s = int(np.sqrt(m))
    
    # Reshape to (n, s, s, d, d)
    reshaped = last_bytes_padded_reshaped.reshape(n, s, s, d, d)
    
    # Transpose to (n, s, d, s, d)
    transposed = reshaped.transpose(0, 1, 3, 2, 4)
    
    # Reshape to final shape (n, d*s, d*s)
    result = transposed.reshape(n, d*s, d*s)
    
    return result

    # n = n_models
    # s = closest_square
    # d = closest_square_sqrt

    # last_bytes_padded_reshaped = last_bytes_padded_reshaped.reshape(n, s, s, d, d)
    
    # # Transpose to (n, s, d, s, d)
    # last_bytes_padded_reshaped_transposed = last_bytes_padded_reshaped.transpose(0, 1, 3, 2, 4)
    
    # # Reshape to final shape (n, d*s, d*s)
    # result = last_bytes_padded_reshaped_transposed.reshape(n, d*s, d*s)

    # return result

def _grayscale_fourpart(data: npt.NDArray[np.float32]):
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


image_rep_map = {
    ImageType.GRAYSCALE_LAST_M_BYTES: _grayscale_lastmbytes,
    ImageType.GRAYSCALE_FOURPART: _grayscale_fourpart
}
