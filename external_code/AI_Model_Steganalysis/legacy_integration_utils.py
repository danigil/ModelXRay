import numpy as np

import os
import sys
module_path = os.path.abspath(os.path.join('..', '..'))
print('added path:', module_path)
if module_path not in sys.path:
    sys.path.append(module_path)

# from siamese import Siamese2

from model_xray.zenml.zenml_lookup import retreive_pp_imgs_datasets
from model_xray.utils.dataset_utils import *

def request_logger(logger_name: str = None, dump_to_sysout=True):
    if logger_name is None:
        logger_name = __name__

    import logging

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    if dump_to_sysout:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
logger = request_logger(__name__)

def img_flatten(arr):
    return arr.reshape(arr.shape[0], -1)

def get_train_test_datasets(mc: str, train_x:int, test_xs:Iterable[Union[None, int]] = range(0,24),
                            imsize:int=100, imtype:ImageType=ImageType.GRAYSCALE_FOURPART,
                            flatten:bool=True, normalize:bool=True,
                            embed_payload_type: PayloadType = PayloadType.BINARY_FILE,
                            payload_filepath: Optional[str] = None):

    def normalize_img(img):
        if 0 <= img.min() and img.max() <= 1:
            return img

        return img / 255.0

    X_train, y_train = None, None
    if train_x:
        trainset_name = get_dataset_name(
            mc=mc,
            ds_type='train',
            xs=[train_x, None],
            imsize=imsize,
            imtype=imtype,
            embed_payload_type=embed_payload_type,
            payload_filepath=payload_filepath,
        )
        
        # print(testset_names)

        ret = retreive_pp_imgs_datasets(
            dataset_names=[trainset_name]
        )

        X_train, y_train = ret[trainset_name]
        if flatten:
            X_train = img_flatten(X_train)

        if normalize:
            X_train = normalize_img(X_train)

    testset_names = {i: get_dataset_name(
        mc=mc,
        ds_type='test',
        xs=[i,],
        imsize=imsize,
        imtype=imtype,
        embed_payload_type=embed_payload_type,
        payload_filepath=payload_filepath,
    ) for i in test_xs}

    ret = retreive_pp_imgs_datasets(
        dataset_names=list(testset_names.values())
    )

    testsets = {}
    for i, testset_name in testset_names.items():
        X_test, y_test = ret[testset_name]
        if flatten:
            X_test = img_flatten(X_test)

        if normalize:
            X_test = normalize_img(X_test)

        testsets[i] = (X_test, y_test)

    return ((X_train, y_train), testsets)

def ret_imgs_dataset_preprocessed(
    zoo_name="famous_le_10m",
    data_type="grads",
    img_type="rgb",
    shape_x=100,
    train_size=1,
    reshape="pad",
    split=True,
    lsb=23,
    normalize=True
):
    payload_filepaths = {
        # 'famous_le_10m': '/mnt/exdisk2/model_xray/malware_payloads/m_77e05'
        'famous_le_10m': None,
    }
    (X_train, y_train), testsets = get_train_test_datasets(
        zoo_name,
        lsb,
        [None, lsb],
        imtype=ImageType(img_type),
        imsize=shape_x,
        flatten=False,
        normalize=normalize,

        # payload_filepath=payload_filepaths[zoo_name]
    )

    X_tests, y_tests = zip(*testsets.values())
    X_test, y_test = np.concatenate(X_tests), np.concatenate(y_tests)

    if split:
        return X_train, y_train, X_test, y_test
    else:
        return np.concatenate([X_train, X_test]), np.concatenate([y_train, y_test])

def siamese_eval(
    model,
    X_train_ref,
    y_train_ref,
    testsets,
    full_eval_mcs: Iterable[str] = ['famous_le_10m', 'famous_le_100m'],
    img_type="rgb",
    shape_x=100,
):
    data = []

    for mc in full_eval_mcs:
        testsets_curr = testsets[mc]
        # (_, _), testsets = get_train_test_datasets(
        #     mc,
        #     train_x=None,
        #     test_xs=range(0,24),
        #     imtype=ImageType(img_type),
        #     imsize=shape_x,
        #     flatten=False,
        # )

        for lsb in range(0, 24):
            X_test, y_test = testsets_curr[lsb]

            test_results = model.test_all(X_train_ref, y_train_ref, X_test, y_test, is_print=False,)

            acc_centroid = test_results['centroid']
            acc_nn = test_results['nn']

            data.append({
                'mc': mc,
                'lsb': lsb,
                'test_acc_centroid': acc_centroid,
                'test_acc_nn': acc_nn,
            })

    return data
    
