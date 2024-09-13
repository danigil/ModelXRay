import itertools

import numpy as np
from model_xray.zenml.zenml_lookup import try_get_artifact_preprocessed_image
from model_xray.configs.models import *
from model_xray.configs.enums import *

from model_xray.options import *

from model_xray.zenml.pipelines.data_creation.preprocessed_image import preprocessed_image_pipeline

if __name__ == "__main__":
    famous_le_10m_model_names = model_collections['famous_le_10m']
    # famous_le_10m_model_names = ["MobileNetV3Small", "MobileNetV3Large"]
    famous_le_100m_model_names = model_collections['famous_le_100m']

    # famous_le_100m_model_names = ["ResNet50",]

    im_types = [
        ImageRepConfig(
            image_rep_proc_config=GrayscaleFourpartConfig()
        ),
        # ImageRepConfig(
        #     image_rep_proc_config=GrayscaleThreepartWeightedAvgConfig()
        # ),
    ]

    im_preprocesses = [
        ImagePreprocessConfig(
            image_height=100,
            image_width=100,
        ),
        ImagePreprocessConfig(
            image_height=256,
            image_width=256,
        ),
    ]

    na_embed_payload_configs = [ret_na_val()]
    random_embed_payload_configs = [
        EmbedPayloadConfig(
            embed_payload_type=PayloadType.RANDOM,
            embed_proc_config=XLSBAttackConfig(x=x),
        )
        for x in range(1,24)
    ]
    # random_embed_payload_configs=[]
    file_embed_payload_configs_famous_le_10m = [
        EmbedPayloadConfig(
            embed_payload_type=PayloadType.BINARY_FILE,
            embed_proc_config=XLSBAttackConfig(x=x),
            embed_payload_metadata=EmbedPayloadMetadata(
                payload_filepath="/mnt/exdisk2/model_xray/malware_payloads/m_77e05"
            )
        )
        for x in range(1,24)
    ]
    # file_embed_payload_configs_famous_le_10m = []
    file_embed_payload_configs_famous_le_100m = [
        EmbedPayloadConfig(
            embed_payload_type=PayloadType.BINARY_FILE,
            embed_proc_config=XLSBAttackConfig(x=x),
            embed_payload_metadata=EmbedPayloadMetadata(
                payload_filepath="/mnt/exdisk2/model_xray/malware_payloads/m_b3ed9"
            )
        )
        for x in range(1,24)
    ]
    
    famous_le_10m_embed_payload_configs = na_embed_payload_configs + random_embed_payload_configs + file_embed_payload_configs_famous_le_10m
    famous_le_10m_total_product_amount = np.prod([
        len(iterable) for iterable in [famous_le_10m_model_names,
                                       im_types, im_preprocesses,
                                       famous_le_10m_embed_payload_configs]])
    """
    print('starting famous_le_10m')
    for i, (model_name, im_type, im_preprocess, embed_payload) in enumerate(itertools.product(
        famous_le_10m_model_names,
        im_types,
        im_preprocesses,
        famous_le_10m_embed_payload_configs
    )):
        # if i > 10:
        #     break
        print(f'\t!! starting {i+1}/{famous_le_10m_total_product_amount}')
        # print(f'\t\tmodel_name: {model_name}\n\t\tim_type: {im_type.model_dump(mode="json")}\n\t\tim_preprocess: {im_preprocess.model_dump(mode="json")}\n\t\tembed_payload: {embed_payload}')
        pp_img_lineage = PreprocessedImageLineage(
            cover_data_config=CoverDataConfig(
                cover_data_cfg=PretrainedModelConfig(
                    name=model_name
                )
            ),
            image_rep_config=im_type,
            image_preprocess_config=im_preprocess,
            embed_payload_config=embed_payload
        )
        print(f'\t\tcurr pp_img_lineage:\n@@@@@@@@@@@\n{pp_img_lineage.model_dump(mode="json")}\n@@@@@@@@@@@')

        try:
            preprocessed_image_pipeline(pp_img_lineage)
            print(f'\t~~ finished {i+1}/{famous_le_10m_total_product_amount}')
        except Exception as e:
            print(f'\t~~ failed {i+1}/{famous_le_10m_total_product_amount}')
            print(f'\t\t{e}')
    """
    
    famous_le_100m_embed_payload_configs = na_embed_payload_configs + random_embed_payload_configs + file_embed_payload_configs_famous_le_100m
    famous_le_100m_total_product_amount = np.prod([
        len(iterable) for iterable in [famous_le_100m_model_names,
                                       im_types, im_preprocesses,
                                       famous_le_100m_embed_payload_configs]])
    
    print('starting famous_le_100m')
    for i, (model_name, im_type, im_preprocess, embed_payload) in enumerate(itertools.product(
        famous_le_100m_model_names,
        im_types,
        im_preprocesses,
        famous_le_100m_embed_payload_configs
    )):
        # if i > 10:
        #     break

        print(f'\t!! starting {i+1}/{famous_le_100m_total_product_amount}')
        print(f'\t\tmodel_name: {model_name}\n\t\tim_type: {im_type.model_dump(mode="json")}\n\t\tim_preprocess: {im_preprocess.model_dump(mode="json")}\n\t\tembed_payload: {embed_payload}')
        pp_img_lineage = PreprocessedImageLineage(
            cover_data_config=CoverDataConfig(
                cover_data_cfg=PretrainedModelConfig(
                    name=model_name
                )
            ),
            image_rep_config=im_type,
            image_preprocess_config=im_preprocess,
            embed_payload_config=embed_payload
        )

        try:
            artifact_lookup = try_get_artifact_preprocessed_image(pp_img_lineage)
            print(f'\t\t## found artifact, skipping pipeline execution')
        except Exception as e:
            print(f'\t\t%% didn\'t find artifact')
            preprocessed_image_pipeline(pp_img_lineage)
            
        print(f'\t~~ finished {i+1}/{famous_le_100m_total_product_amount}')
    