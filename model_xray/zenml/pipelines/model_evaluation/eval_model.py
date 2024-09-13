from typing import Dict, List, Literal, Optional, Union
import numpy as np
import numpy.typing as npt
from zenml import ArtifactConfig, get_step_context, step, pipeline

from model_xray.zenml.pipelines.data_creation.model_attack import embed_payload_into_pretrained_weights_pipeline, embed_payload_into_weights
from model_xray.config_classes import ClassificationMetric, ClassificationMetricConfig, DatasetConfig, DatasetType, EmbedPayloadConfig, ImageDatasetConfig, ModelRepos, PretrainedModelConfig
from tensorflow.keras import Model as tfModel
from tensorflow.data import Dataset as tfDataset

from tensorflow.keras.utils import to_categorical as keras_to_categorical
# from tensorflow import Dataset as tfDataset

from model_xray.utils.eval_utils import calc_metric

from datasets import Dataset as HFDataset

from typing_extensions import Annotated
from typing import Type

from zenml.artifacts.unmaterialized_artifact import UnmaterializedArtifact

from pydantic import BaseModel, SkipValidation

from zenml.integrations.tensorflow.materializers.tf_dataset_materializer import TensorflowDatasetMaterializer

@step
def get_val_ds(
    ds_name: Literal['imagenet12'],
    lib: ModelRepos,
    ds_config: DatasetConfig
    ) -> (
    Annotated[
        Union[tfDataset,],
        ArtifactConfig(
            name="val_dataset",
        ),
    ]
):
    
    if lib == ModelRepos.KERAS:
        if ds_name == 'imagenet12':
            assert ds_config.dataset_type == DatasetType.ImageDataset, f'get_val_ds | ds_config should be an ImageDataset, got: {ds_config.dataset_type}'
            ds = ret_imagnet12_tf_ds(
                image_size=ds_config.dataset_config.image_size,
            )
        else:
            raise NotImplementedError(f'get_val_ds | ds_name {ds_name} not implemented')

        if ds_config.dataset_preprocess_config is not None:
            if ds_config.dataset_preprocess_config.take is not None:
                ds = ds.take(ds_config.dataset_preprocess_config.take)
    else:
        raise NotImplementedError(f'get_val_ds | lib {lib} not implemented')

    return ds

@step
def apply_keras_app_preprocessing(
    ds: tfDataset,
    model_name: str
) -> (
    Annotated[
        tfDataset,
        ArtifactConfig(
            name="preprocessed_dataset",
        ),
    ]
):
    preprocessing_func = ret_model_preprocessing_by_name(model_name=model_name, lib=ModelRepos.KERAS)
    ds = ds.map(lambda x, y: (preprocessing_func(x), y))

    return ds

@step
def extract_y_true(
    ds: Union[tfDataset,],
    ) -> (
    Annotated[
        np.ndarray,
        ArtifactConfig(
            name="y_true",
        ),
    ]
):

    if isinstance(ds, tfDataset):
        y_true = np.concatenate([y for x, y in ds], axis=0)
    else:
        raise NotImplementedError(f'extract_y_true | ds type {type(ds)} not implemented')

    return y_true

@step
def eval_model(
    model: Union[tfModel,],
    model_name: str,

    ds: Union[tfDataset,]
    ) -> (
    Annotated[
        np.ndarray,
        ArtifactConfig(
            name="model_preds",
        ),
    ]
):
    
    model_repo = ModelRepos.determine_model_type(model)

    if model_repo == ModelRepos.KERAS:
        import tensorflow as tf


        with tf.device('/gpu:0'):
            preprocessing_func = ret_model_preprocessing_by_name(model_name=model_name, lib=ModelRepos.KERAS)
            ds = ds.map(lambda x, y: (preprocessing_func(x), y))
            # ds = ds.take(100).map(lambda x, y: (preprocessing_func(x), y))

            y_pred = model.predict(ds)
    else:
        raise NotImplementedError(f'eval_model | model_repo {model_repo} not implemented')

    step_context = get_step_context()
    step_context.add_output_metadata(
        output_name="model_preds",
        metadata={}
    )
    step_context.add_output_tags(
        output_name="model_preds",
        tags=["model_preds", model_repo.value.lower()]
    )

    return y_pred

@step
def calc_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,

    requested_metrics: List[ClassificationMetricConfig]
) -> (
    Annotated[
        Dict[ClassificationMetricConfig, float],
        ArtifactConfig(
            name="classification_metrics",
        ),
    ]
):
    assert y_pred.ndim == 2, f"y_pred should be one-hot-encoded, got shape {y_pred.shape}"
    n_samples_pred, n_classes_pred = y_pred.shape
    if y_true.ndim == 1:
        y_true = keras_to_categorical(y_true, n_classes_pred)

    n_samples_true, n_classes_true = y_true.shape
    assert n_samples_true == n_samples_pred, f"y_true and y_pred should have the same number of samples, got {n_samples_true} and {n_samples_pred}"
    assert n_classes_true == n_classes_pred, f"y_true and y_pred should have the same number of classes, got {n_classes_true} and {n_classes_pred}"

    metrics = {
        current_metric_cfg:calc_metric(y_true, y_pred, current_metric_cfg) for current_metric_cfg in requested_metrics
    }

    step_context = get_step_context()
    step_context.add_output_metadata(
        output_name="classification_metrics",
        metadata={}
    )
    step_context.add_output_tags(
        output_name="classification_metrics",
        tags=["classification_metrics"]
    )

    return metrics

from model_xray.zenml.pipelines.data_creation.fetch_pretrained import fetch_pretrained as fetch_pretrained_step, fetch_pretrained_model_and_extract_weights

from model_xray.utils.model_utils import extract_weights, load_weights_from_flattened_vector, ret_model_preprocessing_by_name
from model_xray.config_classes import TopKCategoricalAccuracyMetricConfig

@step(enable_cache=False)
def load_weights_into_model(
    model: Union[tfModel, ],
    model_weights: np.ndarray
) -> Union[tfModel, ]:
    load_weights_from_flattened_vector(model=model, model_weights=model_weights)

    return model
    
@pipeline(enable_cache=True)
def retrieve_model_weights(
    pretrained_model_config: PretrainedModelConfig,

    embed_payload_config: Optional[EmbedPayloadConfig] = None
) -> np.ndarray:
    if embed_payload_config is None:
        weights = fetch_pretrained_model_and_extract_weights(
            pretrained_model_config=pretrained_model_config,
        )
    else:
        weights = embed_payload_into_pretrained_weights_pipeline(
            pretrained_model_config=pretrained_model_config,

            embed_payload_config=embed_payload_config
        )

    return weights

@pipeline
def evaluate_pretrained_model(
    model_repo: ModelRepos, pretrained_model_name: str,

    ds_name: Literal['imagenet12'],
    ds_config: DatasetConfig,

    requested_metrics: list[ClassificationMetricConfig],

    embed_payload_config: Optional[EmbedPayloadConfig] = None
):

    model = fetch_pretrained_step(model_repo=model_repo, pretrained_model_name=pretrained_model_name)

    if embed_payload_config is not None:
        weights = retrieve_model_weights(
            pretrained_model_name=pretrained_model_name,
            pretrained_model_repo=model_repo,

            embed_payload_config=embed_payload_config
        )

        model = load_weights_into_model(model=model, model_weights=weights)

    ds = get_val_ds(ds_name=ds_name, lib=model_repo, ds_config=ds_config)

    y_true = extract_y_true(ds=ds)
    y_pred = eval_model(model=model, model_name=pretrained_model_name, ds=ds)
        
    metrics = calc_metrics(y_true=y_true, y_pred=y_pred, requested_metrics=requested_metrics)

    return metrics

from model_xray.val_datasets.imagenet12.imagenet12_util import ret_imagnet12_tf_ds

import datasets
from datasets import load_dataset, Image
from pathlib import Path
datasets.config.DOWNLOADED_DATASETS_PATH = Path('/mnt/exdisk1/danigil/datasets/huggingface_datasets')
datasets.config.HF_DATASETS_CACHE = Path('/mnt/exdisk1/danigil/datasets/huggingface_datasets_cache')

if __name__ == "__main__":
    for x in range(0, 24, 3):
        evaluate_pretrained_model(
            model_repo=ModelRepos.KERAS,
            pretrained_model_name='MobileNet',

            ds_name='imagenet12',
            ds_config=DatasetConfig.ret_img_ds_config(image_size=(224, 224), take=100),

            requested_metrics=[
                ClassificationMetricConfig.ret_top_k_categorical_accuracy_config(k=1),
            ],
            embed_payload_config=EmbedPayloadConfig.ret_random_x_lsb_attack_fill_config(x=x) if x > 0 else None
        )
    