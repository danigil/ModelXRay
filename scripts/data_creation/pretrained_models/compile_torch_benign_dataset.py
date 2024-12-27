import itertools

from model_xray.zenml.zenml_lookup import get_pp_imgs_dataset_by_name
from model_xray.utils.dataset_utils import get_dataset_name
from model_xray.zenml.pipelines.data_creation.dataset_compilation_new import compile_and_save_preprocessed_images_dataset_step, compile_and_save_preprocessed_images_dataset_pipeline
from model_xray.configs.models import *
from model_xray.configs.enums import *

from model_xray.options import *
import torchvision

if __name__ == "__main__":
    imsize = 256
    imtype = ImageType.GRAYSCALE_FOURPART
    im_preprocess = ImagePreprocessConfig(
        image_height=imsize,
        image_width=imsize,
    )
    pp_img_lineages = set()

    model_names_classification = torchvision.models.list_models(module=torchvision.models)
    for model_name in model_names_classification:
        pp_img_lineage = PreprocessedImageLineage(
            cover_data_config=CoverDataConfig(
                cover_data_cfg=PretrainedModelConfig(
                    name=model_name,
                    repo=ModelRepos.PYTORCH,
                )
            ),
            image_rep_config=imtype,
            image_preprocess_config=im_preprocess,
            embed_payload_config=ret_na_val(),
        )

        pp_img_lineages.add(pp_img_lineage)
    
    test_dataset_name = get_dataset_name(
        "torch_pretrained_classification",
        xs=[0,],
        imsize=imsize,
        imtype=imtype,
        ds_type='test',
    )

    X,y = get_pp_imgs_dataset_by_name(test_dataset_name)
    if X is None or y is None:
        print(f"\t%% ds {test_dataset_name} not found, compiling")
        compile_and_save_preprocessed_images_dataset_pipeline(
            preprocessed_img_lineages=pp_img_lineages,
            dataset_name=test_dataset_name,
            fallback=False,
        )
    else:
        print(f"\t^^ ds {test_dataset_name} found, skipping")