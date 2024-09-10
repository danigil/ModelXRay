

from typing import Iterable, Literal, Optional, Union
from model_xray.configs.enums import *
from model_xray.utils.dataset_utils import get_dataset_name
from model_xray.zenml.runtime_zenml_models import ret_zenml_model_preprocesssed_image_lineage
from model_xray.configs.models import PreprocessedImageLineage
from model_xray.zenml.pipelines.data_creation.preprocessed_image import preprocessed_image_pipeline

from zenml.client import Client

zenml_client = Client()

class ZenMLModelNotFoundError(Exception):
    pass

class ArtifactNotFoundError(Exception):
    pass

def try_get_artifact_preprocessed_image(
    preprocessed_image_lineage: PreprocessedImageLineage,
    artifact_name: Literal['cover_data', 'stego_data', 'image_representation', 'image_preprocessed'] = 'image_preprocessed',
):
    try:
        zenml_model = ret_zenml_model_preprocesssed_image_lineage(preprocessed_image_lineage)
        model_version = zenml_client.get_model_version(
            model_name_or_id=zenml_model.name,
            model_version_name_or_number_or_id=zenml_model.version
        )

        artifact = model_version.get_artifact(artifact_name)
        if artifact is None:
            raise(ArtifactNotFoundError, "artifact not found")
        
        return artifact.load()
    except KeyError as e:
        raise(ZenMLModelNotFoundError, "zenml model not found")

def get_artifact_preprocessed_image(
    preprocessed_image_lineage: PreprocessedImageLineage,
    artifact_name: Literal['cover_data', 'stego_data', 'image_representation', 'image_preprocessed'] = 'image_preprocessed',
    fallback: bool = False
):
    try:
        try_ret = try_get_artifact_preprocessed_image(preprocessed_image_lineage, artifact_name)
        return try_ret
    except Exception as e:
        if fallback:
            print(f"get_artifact_preprocessed_image: fallback: executing pipeline.")
            preprocessed_image_pipeline(preprocessed_image_lineage)

            try_ret = try_get_artifact_preprocessed_image(preprocessed_image_lineage, artifact_name)


def get_pp_imgs_dataset_by_name(
    dataset_name:str
):
    x_name = f'{dataset_name}_x'
    y_name = f'{dataset_name}_y'

    try:
        x = zenml_client.get_artifact_version(
            x_name
        )

        y = zenml_client.get_artifact_version(
            y_name
        )
    except Exception as e:
        print(f'get_pp_imgs_dataset_by_name: dataset {dataset_name} not found')
        return None, None
    
    return x.load(), y.load()

def get_pp_imgs_dataset_by_params(
    mc: str,
    xs: Iterable[Union[None, int]],
    ds_type:Literal['train', 'test'],
    imsize: int=100,
    imtype: ImageType= ImageType.GRAYSCALE_FOURPART,
    embed_payload_type: PayloadType = PayloadType.RANDOM,
    payload_filepath: Optional[str] = None,
):
    dataset_name = get_dataset_name(
        mc=mc,
        xs=xs,
        imsize=imsize,
        imtype=imtype,
        ds_type=ds_type,
        embed_payload_type=embed_payload_type,
        payload_filepath=payload_filepath,
    )

    return get_pp_imgs_dataset_by_name(dataset_name)
