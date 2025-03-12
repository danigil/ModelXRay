import itertools
from typing import List

import numpy as np
from model_xray.zenml.zenml_lookup import try_get_artifact_preprocessed_image
from model_xray.configs.models import *
from model_xray.configs.enums import *

from model_xray.options import *

from model_xray.zenml.pipelines.data_creation.preprocessed_image import preprocessed_image_pipeline

_mc_options_inner = {
    'famous_le_10m': {
        'imsizes': [100, 256],
        'imtypes': [ImageType.GRAYSCALE_FOURPART],
        'payloadtypes': ['NA', 'RANDOM', 'FILE'],
    },

    'famous_le_100m': {
        'imsizes': [100, 256],
        'imtypes': [ImageType.GRAYSCALE_FOURPART],
        'payloadtypes': ['NA', 'RANDOM', 'FILE'],
    },

    'slms_100m_1b': {
        'imsizes': [256],
        'imtypes': [ImageType.GRAYSCALE_LAST_M_BYTES],
        'payloadtypes': ['NA', 'RANDOM',],
    },

    'slms_1b_2b': {
        'imsizes': [256],
        'imtypes': [ImageType.GRAYSCALE_LAST_M_BYTES],
        'payloadtypes': ['NA', 'RANDOM',],
    },
}

def get_mc_options(
    mc_name:str,

    mc_opts:Optional[Dict]=None,

    imsizes = [100, 256],
    imtypes = [ImageType.GRAYSCALE_FOURPART],

    payloadtypes: List[Literal['NA' 'RANDOM' 'FILE']] = ['NA'],
):
    if mc_opts:
        return mc_opts[mc_name]

    model_names = model_collections[mc_name]
    xs_amount = xs_amounts[mc_name]

    im_types = [
        ImageRepConfig.ret_image_rep_config_by_type(imtype)
        for imtype in imtypes
    ]
    im_preprocesses = [
        ImagePreprocessConfig(
            image_height=ims,
            image_width=ims,
        )
        for ims in imsizes
    ]
    embed_payload_configs = []

    if 'NA' in payloadtypes:
        embed_payload_configs.append(ret_na_val())

    if 'RANDOM' in payloadtypes:
        embed_payload_configs += [
            EmbedPayloadConfig(
                embed_payload_type=PayloadType.RANDOM,
                embed_proc_config=XLSBAttackConfig(x=x),
            )
            for x in range(1,xs_amount+1)
        ]

    if 'FILE' in payloadtypes:
        payload_filepath = get_payload_filepath(mc_name)
        embed_payload_configs += [
            EmbedPayloadConfig(
                embed_payload_type=PayloadType.BINARY_FILE,
                embed_proc_config=XLSBAttackConfig(x=x),
                embed_payload_metadata=EmbedPayloadMetadata(
                    payload_filepath=payload_filepath
                )
            )
            for x in range(1,xs_amount+1)
        ]

    return {
        'model_names': model_names,
        'im_types': im_types,
        'im_preprocesses': im_preprocesses,
        'embed_payload_configs': embed_payload_configs
    }

mc_options = {
    mc_name: get_mc_options(mc_name, **mc_opts)
    for mc_name, mc_opts in _mc_options_inner.items()
}

get_mc_options = partial(get_mc_options, mc_opts=mc_options)

def create_pp_imgs_by_mc_name(
    mc_name:str,

    force:Optional[bool]=False,
):
    mc_opts = get_mc_options(mc_name)
    return create_pp_imgs_loop(**mc_opts, mc_name=mc_name, force=force)



def create_pp_imgs_loop(
    model_names:List[str],
    im_types:List[ImageRepConfig],
    im_preprocesses:List[ImagePreprocessConfig],
    embed_payload_configs:List[EmbedPayloadConfig],

    mc_name:Optional[str]=None,
    force:Optional[bool]=False,

    repo: Optional[ModelRepos]=ModelRepos.PYTORCH,
):
    def try_create_pp_img():
        try:
            preprocessed_image_pipeline(pp_img_lineage)
        except Exception as e:
            print(f'\t\t!! failed: {e}')
            return
        
        print(f'\t~~ finished {i+1}/{total_product_amount}')
        return

    print(f'starting mc: {mc_name}')

    total_product_amount = np.prod([
        len(iterable) for iterable in [model_names, im_types, im_preprocesses, embed_payload_configs]
    ])

    for i, (model_name, im_type, im_preprocess, embed_payload) in enumerate(itertools.product(
        model_names,
        im_types,
        im_preprocesses,
        embed_payload_configs
    )):

        print(f'\t!! starting {i+1}/{total_product_amount}')

        pp_img_lineage = PreprocessedImageLineage(
            cover_data_config=CoverDataConfig(
                cover_data_cfg=PretrainedModelConfig.ret_ptm_config_by_name(
                    model_name=model_name,
                    repo=repo,
                )
            ),
            image_rep_config=im_type,
            image_preprocess_config=im_preprocess,
            embed_payload_config=embed_payload
        )
        print(f'\t\tcurr pp_img_lineage:\n@@@@@@@@@@@\n{pp_img_lineage.model_dump(mode="json")}\n@@@@@@@@@@@')

        if force:
            try_create_pp_img()
            continue

        try:
            artifact_lookup = try_get_artifact_preprocessed_image(pp_img_lineage)
            print(f'\t\t## found artifact, skipping pipeline execution')
        except Exception as e:
            print(f'\t\t%% didn\'t find artifact')
            try_create_pp_img()
