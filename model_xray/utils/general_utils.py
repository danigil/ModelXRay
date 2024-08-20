from pprint import pprint
from typing import Dict
import numpy as np
import numpy.typing as npt

from collections.abc import MutableMapping

import pandas as pd

def ndarray_to_bytes_arr(mcwa: np.ndarray) -> npt.NDArray[np.uint8]:
    assert isinstance(mcwa.dtype.itemsize, int) and mcwa.dtype.itemsize>=1
    newshape = mcwa.shape + (mcwa.dtype.itemsize,)
    dtype = np.dtype('=u1') # force little-endian

    mcwa_decon = np.frombuffer(mcwa.tobytes(order='C'), dtype=dtype).reshape(newshape)
    return np.flip(mcwa_decon, axis=-1)

def bytes_arr_to_ndarray(mcwa: np.ndarray, dtype=np.uint8, shape=None):
    if shape is None:
        newshape = mcwa.shape[0:-1]
    else:
        newshape = shape

    dtype_new = np.dtype(dtype)
    dtype_new = dtype_new.newbyteorder('=')

    return np.frombuffer(np.flip(mcwa, axis=-1).tobytes(order='C'), dtype=dtype_new).reshape(newshape)


def flatten_dict_old(dictionary: Dict, parent_key='', separator='_', parent_separator='_') -> Dict:
    items = []
    for key, value in dictionary.items():
        new_key = parent_key + parent_separator + key if parent_key else key
        if isinstance(value, MutableMapping):
            items.extend(flatten_dict(dictionary=value, parent_key=new_key, separator=separator).items())
        else:
            items.append((new_key, value))
    return dict(items)

def flatten_dict(dictionary, parent_key='', separator='_') -> Dict:
    items = []
    for key, value in dictionary.items():
        new_key = parent_key + separator + key if parent_key else key
        if isinstance(value, MutableMapping):
            items.extend(flatten_dict(dictionary=value, parent_key=new_key, separator=separator).items())
        else:
            items.append((new_key, value))
    return dict(items)

def query_df_using_dict(df: pd.DataFrame, query_dict: dict,
                        ignore_cols: set[str] = set([
                            'artifact_uri',
                            'metadata:embed_payload_config.embed_payload_metadata.payload_bytes_md5',
                            'metadata:embed_payload_config.embed_payload_metadata.payload_filepath'
                            ]),
                        missing_val='None') -> pd.DataFrame:
    # print("~~ query_df_using_dict ~~")
    # print("df.columns:", df.columns)
    # print("query_dict:", query_dict)

    filtered_query_dict = {k:v for k,v in query_dict.items() if k in df.columns}

    missing_cols = set(df.columns) - set(filtered_query_dict.keys()) - ignore_cols
    missing_vals_dict = {k:missing_val for k in missing_cols}

    complete_query_dict = {**filtered_query_dict, **missing_vals_dict}
    # print("complete query dict")
    # pprint(complete_query_dict)
    pds = pd.Series(complete_query_dict).fillna(missing_val)

    df_copy = df.fillna(missing_val, inplace=False)

    columns_to_query = list(complete_query_dict.keys())
    sub_df = df_copy[columns_to_query]

    match_mask = sub_df == pds
    full_match_idxs = match_mask.all(axis=1)

    return df_copy.loc[full_match_idxs]