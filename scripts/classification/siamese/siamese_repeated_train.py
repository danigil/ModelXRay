import os
import pprint
import sys
from typing import Iterable, Literal, Optional, Union

from model_xray.configs.enums import ImageType, PayloadType
from model_xray.options import RESULTS_SIAMESE_DIR
from model_xray.utils.logging_utils import request_logger

from model_xray.utils.script_utils import get_siamese_results_filename

logger = request_logger(__name__)

import pandas as pd


def _repeated_train(
    q,
    lsb: int,

    mc_name:Literal['famous_le_10m', 'famous_le_100m']="famous_le_10m",
    
    imtype:ImageType=ImageType.GRAYSCALE_FOURPART,
    imsize=100,
    embed_payload_type: PayloadType = PayloadType.BINARY_FILE,
    
    mode: Literal['st', 'es' 'ub', 'none'] = 'ub',

    model_arch:Literal['osl_siamese_cnn', 'srnet']='osl_siamese_cnn',

    test_subset:Optional[int] = 300,

    train_loss_threshold_lower = 0.1,
    train_loss_threshold_upper = 2,

    test_acc_threshold = 0.75,
    test_acc_op:Literal['and', 'or'] = 'or',

    try_amount = 10,
    run_num = 0,

    act_only_on_passed: bool = False,
    act_only_on_first_passed: bool = False,

    model_partial_eval: bool = False,
    model_full_eval: bool = True,
    full_eval_mcs = ['famous_le_10m', 'famous_le_100m'],
):
    import operator as op

    import logging
    logging.basicConfig(level=logging.CRITICAL)

    import gc
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

    from model_xray.models.siamese import Siamese, MyThresholdCallback, make_triplets

    import tensorflow as tf

    from model_xray.utils.script_utils import ret_imgs_dataset_preprocessed, get_train_test_datasets, siamese_eval

    from sklearn.model_selection import train_test_split

    # siamese model params
    dist: Literal['l2', 'cosine'] = "l2"
    lr:float =0.00006


    epochs = 5
    n_channels = 1
    cb = None
    bool_op = op.and_ if test_acc_op == 'and' else op.or_


    if mode == 'st':
        epochs = 5
    elif mode == 'ub':
        epochs = 100
    elif mode == 'es':
        epochs = 1

    ret = ret_imgs_dataset_preprocessed(
        mc_name=mc_name,

        lsb=lsb,
        
        imtype=imtype,
        imsize=imsize,
        embed_payload_type=embed_payload_type,

        normalize=True,
        split=True if mc_name != 'ghrp_stl10' else False,

        test_subset=test_subset,
    )

    if mc_name == 'ghrp_stl10':
        X, y = ret
        train_size = 3
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size*2, stratify=y)
    else:
        X_train, y_train, X_test, y_test = ret

    X_test_benign = X_test[y_test == 0]
    X_test_mal = X_test[y_test == 1]
    y_test_benign = y_test[y_test == 0]
    y_test_mal = y_test[y_test == 1]


    triplets_train = make_triplets(X_train, y_train, is_shuffle=True)
    gc.collect()


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
                imtype=imtype,
                imsize=imsize,
                flatten=False,

                test_subset=test_subset,
            )

            eval_datasets[eval_mc] = testsets
    

    for i in range(try_amount):
        gc.collect()
        tf.keras.backend.clear_session()

        print(f"\tRun: {run_num+i}")

        cb = MyThresholdCallback(ub_mode=True if mode=='ub' else False, threshold_lower=train_loss_threshold_lower, threshold_upper=train_loss_threshold_upper)
        
        
        model = Siamese(
            img_input_shape=(imsize,imsize,n_channels),
            dist=dist,
            lr=lr,
            model_arch=model_arch,
        )

        batch_size = 2 if model_arch == 'srnet' else 16

        model.fit(triplets_train, epochs=epochs, batch_size=batch_size, verbose=0, callbacks=[cb])

        train_results = model.test_all(X_train, y_train, X_train, y_train, is_print=False,)

        acc_centroid_train = train_results['centroid']
        acc_nn_train = train_results['nn']
        
        print(f"\t\tTrain Centroid: {acc_centroid_train}, Train NN: {acc_nn_train}, Train Loss: {cb.last_loss}")

        model_passed = True
        if model_partial_eval:
            test_results_benign = model.test_all(X_train, y_train, X_test_benign, y_test_benign, is_print=False,)

            acc_centroid_benign = test_results_benign['centroid']
            acc_nn_benign = test_results_benign['nn']
            
            print(f"\t\tTest (benign): Centroid: {acc_centroid_benign}, Test NN: {acc_nn_benign}")

            test_results_mal = model.test_all(X_train, y_train, X_test_mal, y_test_mal, is_print=False,)
            acc_centroid_mal = test_results_mal['centroid']
            acc_nn_mal = test_results_mal['nn']

            print(f"\t\tTest (mal): Centroid: {acc_centroid_mal}, Test NN: {acc_nn_mal}")

            acc_centroid = (acc_centroid_benign + acc_centroid_mal) / 2
            acc_nn = (acc_nn_benign + acc_nn_mal) / 2

            model_passed = bool_op(acc_centroid >= test_acc_threshold, acc_nn >= test_acc_threshold)
            
        act = not act_only_on_passed or model_passed

        if act:
            if model_full_eval:
                eval_data = siamese_eval(model, X_train, y_train, eval_datasets, full_eval_mcs)
            elif model_partial_eval:
                eval_data_benign = {
                    'mc': mc_name,
                    'lsb': 0,
                    'test_acc_centroid': acc_centroid_benign,
                    'test_acc_nn': acc_nn_benign,
                }

                eval_data_mal = {
                    'mc': mc_name,
                    'lsb': lsb,
                    'test_acc_centroid': acc_centroid_mal,
                    'test_acc_nn': acc_nn_mal,
                }

                eval_data = [eval_data_benign, eval_data_mal]

            df =  pd.DataFrame(eval_data)
            df['model_lsb'] = lsb
            df['model_arch'] = model_arch

            eval_datas[run_num+i] = df

            if model_passed and act_only_on_first_passed:
                break

        del model

    q.put(eval_datas)

import multiprocessing

def repeated_train(
    mc_name:Literal['famous_le_10m', 'famous_le_100m']="famous_le_10m",
    
    imtype:ImageType=ImageType.GRAYSCALE_FOURPART,
    imsize=100,
    embed_payload_type: PayloadType = PayloadType.BINARY_FILE,
    
    mode = 'ub',

    model_arch:Literal['osl_siamese_cnn', 'srnet']='osl_siamese_cnn',

    model_partial_eval: bool = False,

    full_eval_mcs = ['famous_le_10m', 'famous_le_100m'],
    lsbs=range(1,24),

    total_runs = 10,
    batch_size = 5,

    timeout=240,
    retry_amount = 3,
    save_temp:bool = True,

    siamese_results_dir: Optional[str] = None,
):
    if model_arch == 'srnet':
        assert imsize == 256, "SRNet only supports 256x256 images"

    if siamese_results_dir is None:
        siamese_results_dir = RESULTS_SIAMESE_DIR
    results_dir = siamese_results_dir
    os.makedirs(results_dir, exist_ok=True)

    tmp_results_dir = os.path.join(results_dir, "tmp")
    if save_temp:
        os.makedirs(tmp_results_dir, exist_ok=True)

    logger.info(f"Starting repeated train for mc_name: {mc_name}, imtype: {imtype}, imsize: {imsize}, embed_payload_type: {embed_payload_type}, mode: {mode}")
    loop_amount = total_runs // batch_size

    kwargs = {
        'mc_name':mc_name,
        'imtype':imtype,
        'imsize':imsize,
        'embed_payload_type':embed_payload_type,

        'mode':mode,

        'model_arch':model_arch,

        'test_acc_threshold':0.75,

        'try_amount':batch_size,

        'act_only_on_passed': False,
        'act_only_on_first_passed': False,

        'model_partial_eval': model_partial_eval,

        'model_full_eval': True if full_eval_mcs is not None else False,
        'full_eval_mcs':full_eval_mcs,
    }

    assert retry_amount > 0

    results_filename = get_siamese_results_filename(
        mc_name=mc_name,
        imtype=imtype,
        imsize=imsize,
        mode=mode,

        embed_payload_type=embed_payload_type,
        model_arch=model_arch,
    )

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
                    p.terminate()
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
            tmp_save_path = os.path.join(tmp_results_dir, f"{results_filename}_tmp_batch{i}.csv")
            tmp_save_paths.append(tmp_save_path)
            df_curr_batch.to_csv(tmp_save_path, index=False)

        dfs.append(df_curr_batch)

    if len(dfs) > 0:
        df_final = pd.concat(dfs, ignore_index=False)
        df_final.to_csv(os.path.join(results_dir, f"{results_filename}.csv"), index=False)

    if save_temp:
        for tmp_save_path in tmp_save_paths:
            os.remove(tmp_save_path)

        os.rmdir(tmp_results_dir)

if __name__ == "__main__":
    print("starting siamese repeated train")

    # modes = ['es','ub', 'st']
    modes=['ub',]
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

        # for mc_name in ['famous_le_10m','famous_le_100m']:
        for mc_name in ['ghrp_stl10',]:
            repeated_train(mc_name=mc_name, total_runs=30, batch_size=5, mode=mode,

                            imsize=100,
                            model_arch='osl_siamese_cnn',

                           lsbs=range(1,24),
                           retry_amount=3,
                           timeout=2400,
                        #    full_eval_mcs=['famous_le_10m','famous_le_100m', 'maleficnet_benigns', 'maleficnet_mals'],
                            # full_eval_mcs=['ghrp_stl10'],
                            full_eval_mcs=None,
                            model_partial_eval=True,
            )

        # for zoo_name in ['cnn_zoos',]:
        #     repeated_train(zoo_name=zoo_name, total_runs=10, batch_size=5, mode=mode, full_eval_mcs=['cnn_zoos', 'famous_le_10m', 'famous_le_100m'],)

    
    