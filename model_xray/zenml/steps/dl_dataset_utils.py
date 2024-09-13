

from typing import Annotated

from zenml import ArtifactConfig, log_artifact_metadata, step
from model_xray.val_datasets.imagenet12.imagenet12_util import ret_imagnet12_tf_ds
from model_xray.configs.types import tfDataset

@step
def get_imagenet12_val_tfds_step(
    image_height:int,
    image_width:int,
    interpolation: str = 'bilinear',
) -> Annotated[
    tfDataset,
    ArtifactConfig(
        name="imagenet12_val_tfds",
    ),
]:
    ds = ret_imagnet12_tf_ds(
        image_size=(image_height, image_width),
        interpolation=interpolation
    )

    log_artifact_metadata(
        artifact_name="imagenet12_val_tfds",
        metadata={
            "getter_func_args":{
                "image_height": image_height,
                "image_width": image_width,
                "interpolation": interpolation
            }
        },
    )

    return ds

@step
def get_tfds_subset_step(
    dataset: tfDataset,
    subset_size: int,
) -> Annotated[
    tfDataset,
    ArtifactConfig(
        name="tfds_subset",
    ),
]:
    subset = dataset.take(subset_size)

    log_artifact_metadata(
        artifact_name="tfds_subset",
        metadata={
            "subset_size": subset_size,
        },
    )

    return subset