import itertools
from typing import List, Tuple

import numpy as np
from model_xray.configs.models import *
from model_xray.configs.enums import *

from model_xray.options import *

im_types = [
    ImageRepConfig(
        image_rep_proc_config=GrayscaleFourpartConfig()
    ),
    ImageRepConfig(
        image_rep_proc_config=GrayscaleThreepartWeightedAvgConfig()
    ),
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
famous_le_100m_embed_payload_configs = na_embed_payload_configs + random_embed_payload_configs + file_embed_payload_configs_famous_le_100m

def get_options_for_mc(mc_name:str):
    model_names = model_collections.get(mc_name)
    if model_names is None:
        return None

    if mc_name == 'famous_le_10m':
        gen = (
            PreprocessedImageLineage(
                cover_data_config=CoverDataConfig(
                    cover_data_cfg=PretrainedModelConfig(
                        name=model_name
                    )
                ),
                image_rep_config=im_type,
                image_preprocess_config=im_preprocess,
                embed_payload_config=embed_payload
            )
            for
            model_name, im_type, im_preprocess, embed_payload in \
            itertools.product(model_names, im_types, im_preprocesses, famous_le_10m_embed_payload_configs)
        )
    else:
        raise ValueError(f'Unknown model collection name: {mc_name}')
    return gen

def get_options_custom(
    model_names: List[str] = ['MobileNet'],
    im_types: List[ImageType] = [ImageType.GRAYSCALE_FOURPART],
    im_sizes: List[int,] = [100,],

    embed_cfgs: List[Tuple[PayloadType, Optional[str]]] = [(PayloadType.BINARY_FILE, "/mnt/exdisk2/model_xray/malware_payloads/m_77e05")],
    xs: List[Union[None, int]] = list(range(1, 24)),
):
    for model_name, im_type, im_size, (embed_type, payload_path), x in itertools.product(
        model_names, im_types, im_sizes, embed_cfgs, xs
    ):
        pp_img_lineage = PreprocessedImageLineage.ret_ppil(
            model_name=model_name,
            im_type=im_type,
            im_size=im_size,
            is_attacked=False if x is None else True,
            embed_payload_type=embed_type,
            embed_payload_filepath=payload_path,
            x=x
        )
        yield pp_img_lineage
    