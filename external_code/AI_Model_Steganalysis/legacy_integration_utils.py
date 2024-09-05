import numpy as np

import os
import sys
module_path = os.path.abspath(os.path.join('..', '..'))
print('added path:', module_path)
if module_path not in sys.path:
    sys.path.append(module_path)

from model_xray.zenml.pipelines.data_creation.dataset_compilation import retreive_datasets
from model_xray.utils.dataset_utils import *

def img_flatten(arr):
    return arr.reshape(arr.shape[0], -1)

def get_train_test_datasets(mc: str, train_x:int, test_xs:Iterable[Union[None, int]] = range(0,24),
                            imsize:int=100, imtype:ImageType=ImageType.GRAYSCALE_FOURPART,
                            flatten:bool=True, normalize:bool=True):
    trainset_name = get_dataset_name(
        mc=mc,
        ds_type='train',
        xs=[train_x, None],
        imsize=imsize,
        imtype=imtype,
    )
    testset_names = {i: get_dataset_name(
        mc=mc,
        ds_type='test',
        xs=[i,],
        imsize=imsize,
        imtype=imtype,
    ) for i in test_xs}
    # print(testset_names)

    ret = retreive_datasets(
        dataset_names=[trainset_name] + list(testset_names.values())
    )

    X_train, y_train = ret[trainset_name]
    if flatten:
        X_train = img_flatten(X_train)

    if normalize:
        X_train = X_train / 255.0

    testsets = {}
    for i, testset_name in testset_names.items():
        X_test, y_test = ret[testset_name]
        if flatten:
            X_test = img_flatten(X_test)

        if normalize:
            X_test = X_test / 255.0

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
    (X_train, y_train), testsets = get_train_test_datasets(
        zoo_name,
        lsb,
        [None, lsb],
        imtype=ImageType(img_type),
        flatten=False,
        normalize=normalize
    )

    X_tests, y_tests = zip(*testsets.values())
    X_test, y_test = np.concatenate(X_tests), np.concatenate(y_tests)

    return X_train, y_train, X_test, y_test

