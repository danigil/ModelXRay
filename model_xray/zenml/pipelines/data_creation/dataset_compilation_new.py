

from typing import Annotated, Set, Tuple

import numpy as np
from zenml import pipeline, save_artifact, step
from model_xray.zenml.zenml_lookup import get_artifact_preprocessed_image
from model_xray.configs.models import PreprocessedImageLineage


def _ret_preprocessed_images(
    preprocessed_img_lineages: Set[PreprocessedImageLineage],
    fallback=False,
):
    ret = {}

    for preprocessed_img_lineage in preprocessed_img_lineages:
        curr_image = get_artifact_preprocessed_image(
            preprocessed_img_lineage,
            artifact_name='image_preprocessed',
            fallback=fallback
        )
        
        ret[preprocessed_img_lineage.str_hash()] = curr_image

    return ret

def compile_preprocessed_images(
    preprocessed_img_lineages: Set[PreprocessedImageLineage],
    fallback=False,
) -> Tuple[
    Annotated[np.ndarray, "X"],
    Annotated[np.ndarray, "y"],
]:
    first_pp_img = next(iter(preprocessed_img_lineages))
    assert all([pp_img.image_rep_config == first_pp_img.image_rep_config for pp_img in preprocessed_img_lineages]), "compile_and_save_preprocessed_images_dataset: all preprocessed image lineages must have the same image representation configuration"

    assert all([pp_img.image_preprocess_config == first_pp_img.image_preprocess_config for pp_img in preprocessed_img_lineages]), "compile_and_save_preprocessed_images_dataset: all preprocessed image lineages must have the same image preprocess configuration"

    imgs_dict = _ret_preprocessed_images(preprocessed_img_lineages, fallback=fallback)

    xs = []
    ys = []

    for preprocessed_img_lineage in preprocessed_img_lineages:
        curr_hash = preprocessed_img_lineage.str_hash()
        
        x = imgs_dict[curr_hash]
        y = preprocessed_img_lineage.label

        xs.append(x)
        ys.append(y)

    X = np.array(xs)
    y = np.array(ys)

    return X, y

@step
def compile_and_save_preprocessed_images_dataset_step(
    preprocessed_img_lineages: Set[PreprocessedImageLineage],
    dataset_name: str,
    fallback=False,
):
    artifact_name_x = f"{dataset_name}_x"
    artifact_name_y = f"{dataset_name}_y"

    X, y = compile_preprocessed_images(preprocessed_img_lineages, fallback=fallback)

    save_artifact(
        data=X,
        name=artifact_name_x,
    )

    save_artifact(
        data=y,
        name=artifact_name_y,
    )

@pipeline
def compile_and_save_preprocessed_images_dataset_pipeline(
    preprocessed_img_lineages: Set[PreprocessedImageLineage],
    dataset_name: str,
    fallback=False,
):
    compile_and_save_preprocessed_images_dataset_step(
        preprocessed_img_lineages=preprocessed_img_lineages,
        dataset_name=dataset_name,
        fallback=fallback,
    )