import itertools

import numpy as np
from model_xray.zenml.zenml_lookup import try_get_artifact_preprocessed_image
from model_xray.configs.models import *
from model_xray.configs.enums import *

from model_xray.options import *
from scripts.data_creation.pretrained_models.create_ptms_util import create_pp_imgs_loop, compile_pp_imgs_loop
from model_xray.zenml.pipelines.data_creation.preprocessed_image import preprocessed_image_pipeline

import torch
import torchvision.models

if __name__ == "__main__":
    model_names_classification = torchvision.models.list_models(module=torchvision.models)

    im_types = [
        ImageRepConfig(
            image_rep_proc_config=GrayscaleFourpartConfig()
        ),
        # ImageRepConfig(
        #     image_rep_proc_config=GrayscaleThreepartWeightedAvgConfig()
        # ),
    ]

    im_preprocesses = [
        # ImagePreprocessConfig(
        #     image_height=100,
        #     image_width=100,
        # ),
        ImagePreprocessConfig(
            image_height=256,
            image_width=256,
        ),
    ]

    embed_payload_configs = [
        ret_na_val(),
    ]
    xs_amount = 23
    embed_payload_configs += [
        EmbedPayloadConfig(
            embed_payload_type=PayloadType.RANDOM,
            embed_proc_config=XLSBAttackConfig(x=x),
        )
        for x in range(1,xs_amount+1)
    ]

    # create_pp_imgs_loop(
    #     model_names=model_names_classification,
    #     im_types=im_types,
    #     im_preprocesses=im_preprocesses,
    #     embed_payload_configs=embed_payload_configs,
    #     mc_name='torch_pretrained_models',
    # )

    compile_pp_imgs_loop(
        model_names=model_names_classification,
        im_types=im_types,
        im_preprocesses=im_preprocesses,
        embed_payload_configs=embed_payload_configs,
        mc_name='torch_pretrained_models',
    )
