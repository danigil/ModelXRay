import itertools

import numpy as np
from model_xray.zenml.zenml_lookup import try_get_artifact_preprocessed_image
from model_xray.configs.models import *
from model_xray.configs.enums import *

from model_xray.options import *

from model_xray.zenml.pipelines.data_creation.preprocessed_image import preprocessed_image_pipeline, preprocessed_image_pipeline_wo_cache

if __name__ == "__main__":
    ghrp_model_names = [
        'stl10',
    ]

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
        # ImagePreprocessConfig(
        #     image_height=256,
        #     image_width=256,
        # ),
    ]

    na_embed_payload_configs = [ret_na_val()]
    random_embed_payload_configs = [
        EmbedPayloadConfig(
            embed_payload_type=PayloadType.RANDOM,
            embed_proc_config=XLSBAttackConfig(x=x),
        )
        for x in range(1,24)
    ]
    random_embed_payload_configs=[]
    file_embed_payload_configs_ghrp = [
        EmbedPayloadConfig(
            embed_payload_type=PayloadType.BINARY_FILE,
            embed_proc_config=XLSBAttackConfig(x=x),
            embed_payload_metadata=EmbedPayloadMetadata(
                payload_filepath="/mnt/exdisk2/model_xray/malware_payloads/m_6054f"
            )
        )
        for x in range(1,24)
    ]
    
    ghrp_embed_payload_configs = na_embed_payload_configs + random_embed_payload_configs + file_embed_payload_configs_ghrp
    ghrp_total_product_amount = np.prod([
        len(iterable) for iterable in [ghrp_model_names,
                                       im_types, im_preprocesses,
                                       ghrp_embed_payload_configs]])
    
    
    for i, (mz_name, im_type, im_preprocess, embed_payload) in enumerate(itertools.product(
        ghrp_model_names,
        im_types,
        im_preprocesses,
        ghrp_embed_payload_configs
    )):
        # if i > 0:
        #     break

        print(f'\t!! starting {i+1}/{ghrp_total_product_amount}')
        
        pp_img_lineage = PreprocessedImageLineage(
            cover_data_config=CoverDataConfig(
                cover_data_cfg=GHRPModelZooConfig(
                    mz_name=mz_name
                )
            ),
            image_rep_config=im_type,
            image_preprocess_config=im_preprocess,
            embed_payload_config=embed_payload
        )

        # try:
        #     artifact_lookup = try_get_artifact_preprocessed_image(pp_img_lineage)
        #     print(f'\t\t## found artifact, skipping pipeline execution')
        # except Exception as e:
        #     print(f'\t\t%% didn\'t find artifact')
        #     preprocessed_image_pipeline(pp_img_lineage)
        preprocessed_image_pipeline_wo_cache(pp_img_lineage)
            
        print(f'\t~~ finished {i+1}/{ghrp_total_product_amount}')
    