from typing import Annotated, Optional

from zenml import ArtifactConfig, Model, log_artifact_metadata, pipeline, step
from model_xray.zenml.steps.dl_dataset_utils import get_imagenet12_val_tfds_step, get_tfds_subset_step
from model_xray.val_datasets.imagenet12.imagenet12_util import ret_imagnet12_tf_ds
from model_xray.configs.types import tfDataset

@pipeline(
    name="imagenet12_val_tfds_pipeline",
    model=Model(
        name="val_datasets",
        version="1"
    )
)
def get_imagenet12_val_tfds_pipeline(
    image_height=224,
    image_width=224,
    interpolation: str = 'bilinear',
    take_subset: Optional[int] = None,
) -> tfDataset:
    print(f"get_imagenet12_val_tfds_pipeline: image_height={image_height}, image_width={image_width}, interpolation={interpolation}, take_subset={take_subset}")

    ds = get_imagenet12_val_tfds_step(
        image_height=image_height,
        image_width=image_width,
        interpolation=interpolation
    )

    if take_subset:
        ds = get_tfds_subset_step(ds, take_subset)

    return ds

if __name__ == "__main__":
    get_imagenet12_val_tfds_pipeline()
