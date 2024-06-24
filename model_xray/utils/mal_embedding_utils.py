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