import itertools

import numpy as np
from model_xray.zenml.zenml_lookup import try_get_artifact_preprocessed_image
from model_xray.configs.models import *
from model_xray.configs.enums import *

from model_xray.options import *

from model_xray.zenml.pipelines.data_creation.preprocessed_image import preprocessed_image_pipeline

import torch
import torchvision.models

if __name__ == "__main__":
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

    classification_models_product_amnt = np.prod([
        len(torchvision.models.list_models(module=torchvision.models)),
        len(im_types),
        len(im_preprocesses),
    ])

    model_names_classification = torchvision.models.list_models(module=torchvision.models)
    for i, (model_name, im_type, im_preprocess,) in enumerate(itertools.product(
        model_names_classification,
        im_types,
        im_preprocesses,
    )):
        print(f'\t!! starting {i+1}/{classification_models_product_amnt}')
        pp_img_lineage = PreprocessedImageLineage(
            cover_data_config=CoverDataConfig(
                cover_data_cfg=PretrainedModelConfig(
                    name=model_name,
                    repo=ModelRepos.PYTORCH,
                )
            ),
            image_rep_config=im_type,
            image_preprocess_config=im_preprocess,
            embed_payload_config=ret_na_val(),
        )

        try:
            artifact_lookup = try_get_artifact_preprocessed_image(pp_img_lineage)
            print(f'\t\t## found artifact, skipping pipeline execution')
        except Exception as e:
            print(f'\t\t%% didn\'t find artifact')
            preprocessed_image_pipeline(pp_img_lineage)
            
        print(f'\t~~ finished {i+1}/{classification_models_product_amnt}')