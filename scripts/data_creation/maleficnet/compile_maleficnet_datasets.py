import itertools

import os
import sys

# def add_module_path():
#     module_path = os.path.abspath(os.path.join(os.pardir))
#     print('added path:', module_path)
#     if module_path not in sys.path:
#         sys.path.append(module_path)

# from script_utils import add_module_path
# add_module_path()

from model_xray.zenml.zenml_lookup import get_pp_imgs_dataset_by_name
from model_xray.utils.dataset_utils import get_dataset_name
from model_xray.zenml.pipelines.data_creation.dataset_compilation_new import compile_and_save_preprocessed_images_dataset_step, compile_and_save_preprocessed_images_dataset_pipeline
from model_xray.configs.models import *
from model_xray.configs.enums import *

from model_xray.options import *



if __name__ == "__main__":

    imsize = 100
    imtype = ImageType.GRAYSCALE_FOURPART

    benign_pp_img_lineages = set()
    mal_pp_img_lineages = set()

    for model_name in maleficnet_cover_model_names:
        # print(f'!! starting model_name: {model_name}')
        curr_mal_options = maleficnet_mal_options_map[model_name]

        for mal_name \
        in curr_mal_options + ['NA',]:
            
            curr_embed_payload_config = ret_na_val()

            if mal_name != 'NA':
                payload_filepath = get_maleficnet_payload_filepath(mal_name)

                curr_embed_payload_config = EmbedPayloadConfig(
                    embed_payload_type=PayloadType.BINARY_FILE,
                    embed_proc_config=MaleficnetAttackConfig(
                        malware_path_str=payload_filepath
                    ),
                    embed_payload_metadata=EmbedPayloadMetadata(
                        payload_filepath=payload_filepath
                    )
                )

            pp_img_lineage = PreprocessedImageLineage(
                cover_data_config=CoverDataConfig(
                    cover_data_cfg=MaleficnetCoverModelConfig(
                        name=model_name,
                    )
                ),
                image_rep_config=ImageRepConfig.ret_image_rep_config_by_type(imtype),
                image_preprocess_config=ImagePreprocessConfig(
                    image_height=imsize,
                    image_width=imsize,
                ),
                embed_payload_config=curr_embed_payload_config
            )

            if mal_name == 'NA':
                benign_pp_img_lineages.add(pp_img_lineage)
            else:
                mal_pp_img_lineages.add(pp_img_lineage)

    # benign_dataset_name = "maleficnet_benigns"
    benign_dataset_name = get_dataset_name(
        mc="maleficnet_benigns",
        xs=[],
        imsize=imsize,
        imtype=imtype,
        ds_type='test',
    )

    X,y = get_pp_imgs_dataset_by_name(benign_dataset_name)
    if X is None or y is None:
        print(f"\t%% ds {benign_dataset_name} not found, compiling")
        compile_and_save_preprocessed_images_dataset_pipeline(
            preprocessed_img_lineages=benign_pp_img_lineages,
            dataset_name=benign_dataset_name,
            fallback=False,
        )

    mal_dataset_name = get_dataset_name(
        mc="maleficnet_mals",
        xs=[],
        imsize=imsize,
        imtype=imtype,
        ds_type='test',
    )
    X,y = get_pp_imgs_dataset_by_name(mal_dataset_name)
    if X is None or y is None:
        print(f"\t%% ds {mal_dataset_name} not found, compiling")
        compile_and_save_preprocessed_images_dataset_pipeline(
            preprocessed_img_lineages=mal_pp_img_lineages,
            dataset_name=mal_dataset_name,
            fallback=False,
        )