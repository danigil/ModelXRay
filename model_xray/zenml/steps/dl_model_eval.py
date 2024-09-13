
from typing import Annotated, Literal, Optional
from zenml import ArtifactConfig, step
from zenml.client import Client
from zenml.config.pipeline_run_configuration import PipelineRunConfiguration

from model_xray.zenml.steps.dl_dataset_utils import get_imagenet12_val_tfds_step, get_tfds_subset_step
from model_xray.configs.enums import ModelRepos
from model_xray.utils.model_utils import ret_model_preprocessing_by_name
from model_xray.zenml.pipelines.model_evaluation.val_datasets import get_imagenet12_val_tfds_pipeline
from model_xray.configs.types import kerasModel, tfkerasModel, tfDataset, DL_MODEL_TYPE

@step
def eval_model_step(
    model: DL_MODEL_TYPE,
    ds_name:Literal['imagenet12']= 'imagenet12',
    take_subset: Optional[int] = None,
    model_name: Optional[str] = None,
) -> Annotated[
    float,
    ArtifactConfig(
        name="model_eval_score"
    )
]:
    eval_func = lookup_eval(model, ds_name)
    if eval_func is None:
        raise NotImplementedError(f"Eval function for model_type: {type(model)}, ds: {ds_name} not found.")

    score = eval_func(
        model,
        # ds_name=ds_name,
        take_subset=take_subset,
        model_name=model_name,
    )

    return score

def eval_keras_model(
    model: kerasModel,
    ds: tfDataset,
):
    score = model.evaluate(ds, verbose=0)

    return score

def eval_keras_model_on_imagenet12(
    model: kerasModel,
    take_subset: Optional[int] = None,
    model_name: Optional[str] = None,
):
    def apply_keras_app_preprocessing(
        ds: tfDataset,
        model_name: str
    ) -> tfDataset:
        preprocessing_func = ret_model_preprocessing_by_name(model_name=model_name, lib=ModelRepos.KERAS)
        ds = ds.map(lambda x, y: (preprocessing_func(x), y))

        return ds

    def get_model_input_shape(model: kerasModel, model_name: Optional[str] = None):
        fallback_dict = {
            "MobileNet": (1, 224, 224, 3),
            "MobileNetV2": (1, 224, 224, 3),
            "MobileNetV3Small": (1, 224, 224, 3),
            "MobileNetV3Large": (1, 224, 224, 3),
            "NASNetMobile": (1, 224, 224, 3),
            "DenseNet121": (1, 224, 224, 3),
            "EfficientNetV2B0": (1, 224, 224, 3),
            "EfficientNetV2B1": (1, 240, 240, 3),
        }

        last_resort_shape = (1, 224, 224, 3)

        model_input_shape = model.input_shape
        _, height, width, _ = model_input_shape

        if height is None or width is None:
            if model_name:
                _, height, width, _ = fallback_dict.get(model_name, (None,)*4)
            
            if height is None or width is None:
                _, height, width, _ = last_resort_shape

        return height, width
            


    from tensorflow import keras
    # model_input_shape = model.input_shape
    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(),
        optimizer=keras.optimizers.Adam(0.001),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )

    # run = get_imagenet12_val_tfds_pipeline(
    #     image_height=model_input_shape[1],
    #     image_width=model_input_shape[2],
    #     take_subset=take_subset
    # )
    # run_config = PipelineRunConfiguration(
    #     steps={
    #         "get_imagenet12_val_tfds_step":{
    #             "parameters":{
    #                 "image_height": model_input_shape[1],
    #                 "image_width": model_input_shape[2],
    #             }
    #             # "interpolation": "bilinear"
    #         },
    #         # "get_tfds_subset_step":{
    #         #     "dataset":
    #         #     "take_subset": take_subset
    #         # }
    #     }
    # )
    # run = Client().trigger_pipeline(
    #     pipeline_name_or_id='imagenet12_val_tfds_pipeline',
    #     run_configuration=run_config
    # )

    # # Client().get_pipeline_run(run.id).run()

    # if take_subset:
    #     ds = run.steps['get_tfds_subset_step'].output.load()
    # else:
    #     ds = run.steps['get_imagenet12_val_tfds_step'].output.load()

    image_height, image_width = get_model_input_shape(model, model_name)

    ds = get_imagenet12_val_tfds_step(
        image_height=image_height,
        image_width=image_width,
        # interpolation=interpolation
    )

    if take_subset:
        ds = get_tfds_subset_step(ds, take_subset)

    if model_name:
        ds = apply_keras_app_preprocessing(ds, model_name)

    score = eval_keras_model(model, ds)[1]

    return score

def lookup_eval(model, ds_name):
    model_type = type(model)

    if isinstance(model, (kerasModel, tfkerasModel)):
        return keras_eval_steps_map.get(ds_name, None)
    else:
        raise NotImplementedError(f"Model type {model_type} not supported.")

keras_eval_steps_map = {
    'imagenet12': eval_keras_model_on_imagenet12,
    'imagenet': eval_keras_model_on_imagenet12,
}