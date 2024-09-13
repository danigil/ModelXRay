import os
import pprint
import sys
from typing import Iterable, Literal, Union
import operator as op

import pandas as pd
# from options import SUPPORTED_FEATURES, SUPPORTED_IMG_TYPES, SUPPORTED_MCS

# import logging
# logging.basicConfig(level=logging.INFO)

# from data_locator import request_logger

from legacy_integration_utils import get_train_test_datasets, request_logger

logger = request_logger(__name__)

def _repeated_train(
    q,
    lsb = 1,
    zoo_name = "famous_le_10m",
    data_type = "weights",
    img_type = "grayscale_fourpart",
    x=100,
    
    mode: Literal['st', 'es' 'ub', 'none'] = 'ub',

    train_loss_threshold_lower = 0.1,
    train_loss_threshold_upper = 2,

    test_acc_threshold = 0.75,
    test_acc_op:Literal['and', 'or'] = 'or',

    try_amount = 10,
    run_num = 0,

    act_only_on_passed: bool = False,

    save_model: bool = False,
    save_only_first_model: bool = False,

    model_full_eval: bool = True,
    full_eval_mcs = ['famous_le_10m', 'famous_le_100m'],
):

    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
    from typing import Literal
    from siamese import Siamese2, MyThresholdCallback
    import numpy as np
    from sklearn.model_selection import train_test_split
    # from data_locator import ret_imgs_dataset_preprocessed
    import tensorflow as tf

    from itertools import combinations
    import numpy as np
    import tensorflow as tf

    from tensorflow.keras.callbacks import EarlyStopping
    from siamese import make_triplets
    # from siamese_eval import siamese_eval
    import gc

    from legacy_integration_utils import ret_imgs_dataset_preprocessed, get_train_test_datasets, siamese_eval

    # from data_locator import get_model_path

    import logging
    logging.basicConfig(level=logging.CRITICAL)

    epochs = 5

    pretrained = True if img_type=='rgb' else False
    n_channels = 3 if img_type=='rgb' else 1

    cb = None

    bool_op = op.and_ if test_acc_op == 'and' else op.or_


    if mode == 'st':
        epochs = 5
    elif mode == 'ub':
        epochs = 100
    elif mode == 'es':
        epochs = 1

    # print(f"mode: {mode}")
    # print(f"epochs: {epochs}")

    cb = MyThresholdCallback(ub_mode=True if mode=='ub' else False, threshold_lower=train_loss_threshold_lower, threshold_upper=train_loss_threshold_upper)

    X_train, y_train, X_test, y_test = ret_imgs_dataset_preprocessed(zoo_name, data_type, img_type, x, lsb=lsb, train_size=1, reshape="resize")
    # path = get_model_path(zoo_name, data_type, img_type, x, lsb, 1, "resize", mode=mode)

    triplets_train = make_triplets(X_train, y_train, is_shuffle=True)
    # callback = EarlyStopping(monitor='loss',patience=10, restore_best_weights=True)
    gc.collect()

    # train_info = SiameseTrainInfo(
    #     mode = mode,
    #     zoo_name = zoo_name,
    #     data_type = data_type,
    #     img_type = img_type,
    #     x = x,
    #     lsb = lsb,
    # )

    eval_datas = {}

    if model_full_eval:
        eval_datasets = {}
        for eval_mc in full_eval_mcs:
            if eval_mc in ('maleficnet_benigns', 'maleficnet_mals'):
                test_xs = []
            else:
                test_xs = range(0,24)

            (_, _), testsets = get_train_test_datasets(
                eval_mc,
                train_x=None,
                test_xs=test_xs,
                imtype=img_type,
                imsize=x,
                flatten=False,
            )

            eval_datasets[eval_mc] = testsets
    

    for i in range(try_amount):
        # print(f'\nround: {i+1}/{try_amount}')
        # device.reset()
        gc.collect()
        tf.keras.backend.clear_session()
        
        
        model = Siamese2(pretrained=pretrained, img_input_shape=(x,x,n_channels), dist="l2", lr=0.00006,)
        model.fit(triplets_train, epochs=epochs, batch_size=16, verbose=0, callbacks=[cb])
        # model.train_info = train_info

        # mean_train_loss = model.loss_tracker.result().numpy()
        train_loss = cb.last_loss
        # print("\ttrain loss:", train_loss)
        # if train_loss <= threshold_lower or train_loss >= threshold_upper:
        #     continue
        train_results = model.test_all(X_train, y_train, X_train, y_train, is_print=False,)

        acc_centroid_train = train_results['centroid']
        acc_nn_train = train_results['nn']
        
        print(f"\tTrain Centroid: {acc_centroid_train}, Train NN: {acc_nn_train}")

        test_results = model.test_all(X_train, y_train, X_test, y_test, is_print=False,)

        acc_centroid = test_results['centroid']
        acc_nn = test_results['nn']
        
        print(f"\tTest Centroid: {acc_centroid}, Test NN: {acc_nn}")

        model_passed = bool_op(acc_centroid >= test_acc_threshold, acc_nn >= test_acc_threshold)
        act = not act_only_on_passed or model_passed
        
        if save_model and act:
            print(f"~~ Model saved with acc_c:{acc_centroid},acc:k:{acc_nn} ~~")
            # model.save(f'{path.replace(".keras","")}_acc_c:{acc_centroid},acc:k:{acc_nn}.keras')
            # model.save(f'{path.replace(".keras","")}_es.keras')
            # model.save(path)
            if save_only_first_model:
                return

        if model_full_eval and act:
            eval_data = siamese_eval(model, X_train, y_train, eval_datasets, full_eval_mcs, img_type, x)
            df =  pd.DataFrame(eval_data)
            df['model_lsb'] = lsb
            eval_datas[run_num+i] = df
        #     eval_data = siamese_eval([model], full_eval_mcs, data_type, img_type, x)
        #     eval_datas[run_num+i] = pd.DataFrame(eval_data, columns=['model_lsb', 'test_zoo', 'lsb', 'centroid_accuracy_train', 'centroid_accuracy_test', 'knn_accuracy_train', 'knn_accuracy_test'])
        del model

    q.put(eval_datas)
import multiprocessing

def repeated_train(
    zoo_name = "famous_le_10m",
    data_type = "weights",
    img_type = "grayscale_fourpart",
    x=100,
    mode = 'ub',

    full_eval_mcs = ['famous_le_10m', 'famous_le_100m'],
    lsbs=range(1,24),

    total_runs = 10,
    batch_size = 5,

    timeout=240,
    retry_amount = 3,
    save_temp:bool = True,
    
):
    logger.info(f"Starting repeated train for zoo_name: {zoo_name} data_type: {data_type} img_type: {img_type} x: {x} mode: {mode}")
    loop_amount = total_runs // batch_size

    kwargs = {
        'zoo_name':zoo_name,
        'data_type':data_type,
        'img_type':img_type,
        'x':x,
        'mode':mode,
        'test_acc_threshold':0.75,
        'try_amount':batch_size,
        'act_only_on_passed': False,
        'save_model': False,
        'save_only_first_model': False,
        'model_full_eval': True,
        'full_eval_mcs':full_eval_mcs,
    }

    
    assert retry_amount > 0

    q = multiprocessing.Queue()
    dfs = []
    tmp_save_paths = []
    for i in range(loop_amount):
        logger.info(f"Run: {i*batch_size} - {i*batch_size+batch_size}")

        dfs_curr_batch = []
        for lsb in lsbs:
            logger.info(f"LSB: {lsb}")
            kwargs['lsb'] = lsb
            kwargs['run_num'] = i*batch_size

            try_counter = 0

            while(try_counter < retry_amount):
                if try_counter > 0:
                    logger.info(f"Retry num: {try_counter+1}/{retry_amount}")
                p = multiprocessing.Process(target=_repeated_train, args=(q,), kwargs=kwargs)
                p.start()
                p.join(timeout)

                if p.is_alive():
                    logger.error(f"Timeout in process {i}, retrying...")
                    p.terminate()
                    try_counter += 1
                    continue

                if p.exitcode == 0:
                    break
                else:
                    logger.error(f"Error in process {i}, retrying...")
                    try_counter += 1

            results = q.get(block=False)
            if results is None or len(results) == 0:
                continue

            df = pd.concat(results, ignore_index=False)
            df.reset_index(level=0, drop=False, inplace=True, names='run num')
            df.reset_index(level=0, drop=True, inplace=True)
            dfs_curr_batch.append(df)

        if len(dfs_curr_batch) == 0:
            continue

        df_curr_batch = pd.concat(dfs_curr_batch, ignore_index=False)
        if save_temp:
            tmp_save_path = os.path.join("results", "experiments", "tmp", f"results_siamese_{zoo_name}_{data_type}_{img_type}_{x}{f'_{mode}' if mode!='none' else ''}_tmp_batch{i}.csv")
            tmp_save_paths.append(tmp_save_path)
            df_curr_batch.to_csv(tmp_save_path, index=False)

        dfs.append(df_curr_batch)

    if len(dfs) > 0:
        df_final = pd.concat(dfs, ignore_index=False)
        df_final.to_csv(os.path.join("results", "experiments", f"results_siamese_{zoo_name}_{data_type}_{img_type}_{x}{f'_{mode}' if mode!='none' else ''}.csv"), index=False)

    if save_temp:
        for tmp_save_path in tmp_save_paths:
            os.remove(tmp_save_path)

if __name__ == "__main__":
    modes = ['es',]
    for mode in modes:
        # for zoo_name in ['llms_bert_conll03',]:
        #     repeated_train(
        #         zoo_name=zoo_name,
        #         full_eval_mcs=["llms_bert_conll03", "llms_bert"],
        #         total_runs=30,
        #         batch_size=10,
        #         img_type='grayscale_lastbyte',
        #         mode=mode,
        #         lsbs=range(1,11),
        #     )

        for zoo_name in ['famous_le_10m',]:
            repeated_train(zoo_name=zoo_name, total_runs=10, batch_size=10, mode=mode,
                           lsbs=range(1,3),
                           retry_amount=1,
                           full_eval_mcs=['maleficnet_benigns', 'maleficnet_mals'],
            )

        # for zoo_name in ['cnn_zoos',]:
        #     repeated_train(zoo_name=zoo_name, total_runs=10, batch_size=5, mode=mode, full_eval_mcs=['cnn_zoos', 'famous_le_10m', 'famous_le_100m'],)

    
    