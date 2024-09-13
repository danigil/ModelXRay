import itertools

from model_xray.zenml.zenml_lookup import get_pp_imgs_dataset_by_name
from model_xray.utils.dataset_utils import get_dataset_name
from model_xray.zenml.pipelines.data_creation.dataset_compilation_new import compile_and_save_preprocessed_images_dataset_step, compile_and_save_preprocessed_images_dataset_pipeline
from model_xray.configs.models import *
from model_xray.configs.enums import *

from model_xray.options import *

if __name__ == "__main__":

    mcs = ['famous_le_10m', 'famous_le_100m']

    imsize = 100
    imtype = ImageType.GRAYSCALE_FOURPART
    embed_payload_type = PayloadType.BINARY_FILE
    # embed_malware_payload_filepath = '/mnt/exdisk2/model_xray/malware_payloads/m_77e05'

    for curr_mc in mcs:
        # try:
        print(f'starting {curr_mc}')
        embed_malware_payload_filepath = get_payload_filepath(curr_mc)

        curr_split = dataset_split[curr_mc]
        train_mzs, test_mzs = curr_split

        print(f'starting {curr_mc} train')
        # try:
        for train_x in range(1, 24):
            print(f'\t!! starting train_x: {train_x}')
            train_pp_img_lineages = set()
            for x_curr in [None, train_x]:
                for model_name in train_mzs:
                    pp_img_lineage_curr = PreprocessedImageLineage.ret_ppil(
                        model_name=model_name,
                        im_type=imtype,
                        im_size=imsize,
                        x=x_curr,
                        embed_payload_type=embed_payload_type,
                        embed_payload_filepath=embed_malware_payload_filepath,
                    )

                    train_pp_img_lineages.add(pp_img_lineage_curr)

            train_dataset_name = get_dataset_name(
                curr_mc,
                xs=[None, train_x],
                imsize=imsize,
                imtype=imtype,
                ds_type='train',
                embed_payload_type=embed_payload_type,
                payload_filepath=embed_malware_payload_filepath,
            )
            X,y = get_pp_imgs_dataset_by_name(train_dataset_name)
            if X is None or y is None:
                print(f"\t%% ds {train_dataset_name} not found, compiling")
                compile_and_save_preprocessed_images_dataset_pipeline(
                    preprocessed_img_lineages=train_pp_img_lineages,
                    dataset_name=train_dataset_name,
                    fallback=False,
                )
            else:
                print(f"\t^^ ds {train_dataset_name} found, skipping")

            print(f'\t~~ finished train_x: {train_x}')
        # except Exception as e:
            # print(f'Failed {curr_mc} train')
            # print(e)
            # continue

        print(f'starting {curr_mc} test')
        # try:
        for test_x in range(0, 24):
            print(f'\t!! starting test_x: {test_x}')
            test_pp_img_lineages = set()

            for model_name in test_mzs:
                pp_img_lineage_curr = PreprocessedImageLineage.ret_ppil(
                    model_name=model_name,
                    im_type=imtype,
                    im_size=imsize,
                    x=test_x if test_x != 0 else None,
                    embed_payload_type=embed_payload_type,
                    embed_payload_filepath=embed_malware_payload_filepath,
                )

                test_pp_img_lineages.add(pp_img_lineage_curr)

            test_dataset_name = get_dataset_name(
                curr_mc,
                xs=[test_x,],
                imsize=imsize,
                imtype=imtype,
                ds_type='test',
                embed_payload_type=embed_payload_type,
                payload_filepath=embed_malware_payload_filepath,
            )

            X,y = get_pp_imgs_dataset_by_name(test_dataset_name)
            if X is None or y is None:
                print(f"\t%% ds {test_dataset_name} not found, compiling")
                compile_and_save_preprocessed_images_dataset_pipeline(
                    preprocessed_img_lineages=test_pp_img_lineages,
                    dataset_name=test_dataset_name,
                    fallback=False,
                )
            else:
                print(f"\t^^ ds {test_dataset_name} found, skipping")

            print(f'\t~~ finished test_x: {test_x}')
        # except Exception as e:
        #     print(f'Failed {curr_mc} test')
        #     print(e)
        #     continue
        print(f'Finished {curr_mc}')
        # except Exception as e:
        #     print(f'Failed {curr_mc}')
        #     print(e)
        #     continue