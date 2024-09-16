import itertools

from model_xray.zenml.zenml_lookup import get_pp_imgs_dataset_by_name, try_get_artifact_preprocessed_image
from model_xray.utils.dataset_utils import get_dataset_name
from model_xray.zenml.pipelines.data_creation.dataset_compilation_new import compile_and_save_preprocessed_images_dataset_step, compile_and_save_preprocessed_images_dataset_pipeline, save_dataset
from model_xray.configs.models import *
from model_xray.configs.enums import *

from model_xray.utils.script_utils import normalize_img

from model_xray.options import *
import numpy as np

def get_x(pp_img_lineage, split_size, normalize=True):
    try:
        X = try_get_artifact_preprocessed_image(pp_img_lineage)
    except Exception as e:
        print(f'Failed {pp_img_lineage}')
        print(e)
        return None, None
    X_1, X_2 = np.split(X, [split_size])

    if normalize:
        X_1 = normalize_img(X_1)
        X_2 = normalize_img(X_2)

    return X_1, X_2

if __name__ == "__main__":

    mcs = ['ghrp_stl10']

    imsize = 100
    imtype = ImageType.GRAYSCALE_FOURPART
    embed_payload_type = PayloadType.BINARY_FILE
    # embed_malware_payload_filepath = '/mnt/exdisk2/model_xray/malware_payloads/m_77e05'

    train_split_size = 3
    
    image_rep_config=ImageRepConfig.ret_image_rep_config_by_type(imtype)
    image_preprocess_config=ImagePreprocessConfig(
        image_height=imsize,
        image_width=imsize,
    )

    for curr_mc in mcs:
        cover_data_config=CoverDataConfig(
            cover_data_cfg=GHRPModelZooConfig(
                mz_name=curr_mc.split('_')[1]
            )
        )
        # try:
        print(f'starting {curr_mc}')
        embed_malware_payload_filepath = get_payload_filepath(curr_mc)

        print(f'starting {curr_mc} train')
        # try:
        for train_x in range(1, 24):
            print(f'\t!! starting train_x: {train_x}')

            train_dataset_name = get_dataset_name(
                curr_mc,
                xs=[None, train_x],
                imsize=imsize,
                imtype=imtype,
                ds_type='train',
                embed_payload_type=embed_payload_type,
                payload_filepath=embed_malware_payload_filepath,
            )

            # X,y = get_pp_imgs_dataset_by_name(train_dataset_name)
            # if X is None or y is None:
            #     print(f"\t%% ds {train_dataset_name} not found, compiling")
                
            pp_img_lineage_benign = PreprocessedImageLineage(
                cover_data_config=cover_data_config,
                image_rep_config=image_rep_config,
                image_preprocess_config=image_preprocess_config,
                embed_payload_config=ret_na_val(),
            )

            pp_img_lineage_mal = PreprocessedImageLineage(
                cover_data_config=cover_data_config,
                image_rep_config=image_rep_config,
                image_preprocess_config=image_preprocess_config,
                embed_payload_config=EmbedPayloadConfig(
                    embed_payload_type=embed_payload_type,
                    embed_proc_config=XLSBAttackConfig(x=train_x),
                    embed_payload_metadata=EmbedPayloadMetadata(
                        payload_filepath=embed_malware_payload_filepath
                    )
                ),
            )

            X_benign_train, _ = get_x(pp_img_lineage_benign, train_split_size)
            X_mal_train, _ = get_x(pp_img_lineage_mal, train_split_size)

            X_train = np.concatenate([X_benign_train, X_mal_train])
            y_train = np.concatenate([np.full(len(X_benign_train), PreprocessedImageDatasetLabel.BENIGN), np.full(len(X_mal_train), PreprocessedImageDatasetLabel.ATTACKED)])

            save_dataset(X_train, y_train, train_dataset_name)

            # else:
            #     print(f"\t^^ ds {train_dataset_name} found, skipping")

            print(f'\t~~ finished train_x: {train_x}')


        print(f'starting {curr_mc} test')
        for test_x in range(0, 24):
            print(f'\t!! starting test_x: {test_x}')

            test_dataset_name = get_dataset_name(
                curr_mc,
                xs=[test_x,],
                imsize=imsize,
                imtype=imtype,
                ds_type='test',
                embed_payload_type=embed_payload_type,
                payload_filepath=embed_malware_payload_filepath,
            )

            # X,y = get_pp_imgs_dataset_by_name(test_dataset_name)
            # if X is None or y is None:
            #     print(f"\t%% ds {test_dataset_name} not found, compiling")

            pp_img_lineage = PreprocessedImageLineage(
                cover_data_config=cover_data_config,
                image_rep_config=image_rep_config,
                image_preprocess_config=image_preprocess_config,
                embed_payload_config=EmbedPayloadConfig(
                    embed_payload_type=embed_payload_type,
                    embed_proc_config=XLSBAttackConfig(x=test_x),
                    embed_payload_metadata=EmbedPayloadMetadata(
                        payload_filepath=embed_malware_payload_filepath
                    )
                ) if test_x != 0 else ret_na_val(),
            )

            _, X_test = get_x(pp_img_lineage, train_split_size)
            label = PreprocessedImageDatasetLabel.BENIGN if test_x == 0 else PreprocessedImageDatasetLabel.ATTACKED

            y_test = np.full(len(X_test), label)

            save_dataset(X_test, y_test, test_dataset_name)

            # else:
            #     print(f"\t^^ ds {test_dataset_name} found, skipping")

            print(f'\t~~ finished test_x: {test_x}')
