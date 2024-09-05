import numpy as np

__all__ = ['dt_uint8_be', 'dt_float32_be', 'dt_float32_ne']

dt_uint8_be = np.dtype('>u1')
dt_float32_be = np.dtype('>f4')
dt_float32_ne = np.dtype('=f4')