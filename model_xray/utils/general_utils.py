from pprint import pprint
from typing import Dict, get_args
import numpy as np
import numpy.typing as npt

from collections.abc import MutableMapping

import pandas as pd

def ndarray_to_bytes_arr(mcwa: np.ndarray) -> np.ndarray[np.uint8]:
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

def try_coerce_data(data, expected_type: type, **additional_kwargs):
    from model_xray.configs.types import DL_MODEL_TYPE
    from model_xray.utils.model_utils import extract_weights as extract_weights_util, load_weights_from_flattened_vector

    # if data is already of the expected type, return it
    if isinstance(data, expected_type):
        return data

    # data is a DL model, and expected_type is a numpy array
    # if so, extract the weights from the model
    if isinstance(data, get_args(DL_MODEL_TYPE)) and issubclass(expected_type, np.ndarray):
        return extract_weights_util(model=data)

    # data is a numpy array, and expected_type is a DL model
    # if so, load the weights into the a model based on a `reference_data` kwarg
    if isinstance(data, np.ndarray) and issubclass(expected_type, get_args(DL_MODEL_TYPE)):
        reference_data = additional_kwargs.get('reference_data', None)
        if reference_data is None:
            raise ValueError(f"try_coerce_data: reference_data must be provided if expected_type is a DL_MODEL_TYPE")
        
        return load_weights_from_flattened_vector(reference_data, data, inplace=False)

    return None

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
                            # 'metadata:embed_payload_config.embed_payload_metadata.payload_filepath'
                            ]),
                        missing_val='None',
                        relaxed_matching: bool = False) -> pd.DataFrame:

    filtered_query_dict = {k:v for k,v in query_dict.items() if k in df.columns and k not in ignore_cols}

    
    if relaxed_matching:
        complete_query_dict = {k:v for k,v in filtered_query_dict.items() if k not in ignore_cols}
    else:
        missing_cols = set(df.columns) - set(filtered_query_dict.keys()) - ignore_cols
        missing_vals_dict = {k:missing_val for k in missing_cols}
        complete_query_dict = {**filtered_query_dict, **missing_vals_dict}

    pds = pd.Series(complete_query_dict).fillna(missing_val)

    df_copy = df.fillna(missing_val, inplace=False)

    columns_to_query = list(complete_query_dict.keys())
    sub_df = df_copy[columns_to_query]

    match_mask = sub_df == pds
    full_match_idxs = match_mask.all(axis=1)

    return df_copy.loc[full_match_idxs]

import os, sys

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout