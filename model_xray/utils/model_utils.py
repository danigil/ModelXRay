from pathlib import Path
from typing import Callable, Dict, Literal, Union
import json

import numpy as np

from tensorflow.keras import Model as tfModel
from torch.nn import Module as torchModel

from model_xray.config_classes import ModelRepos

def model_processing_func(func):
    def wrap(model:Union[tfModel, torchModel] ,*args, **kwargs):
        model_repo = ModelRepos.determine_model_type(model)
        if 'model_repo' in kwargs:
            del kwargs['model_repo']
        return func(model, *args, model_repo=model_repo, **kwargs)
        
    return wrap

@model_processing_func
def load_weights_from_flattened_vector(model: Union[tfModel, torchModel], model_weights: np.ndarray, model_repo: ModelRepos = None):
    def load_weights_from_flattened_vector_torch(model: torchModel, model_weights: np.ndarray):
        import torch

        state_dict = model.state_dict()
        torch.nn.utils.vector_to_parameters(model_weights, state_dict.values())

        model.load_state_dict(state_dict)

    def load_weights_from_flattened_vector_keras(model: tfModel, model_weights: np.ndarray):
        shapes = [w.shape for w in model.get_weights()]
        splt = np.split(model_weights, np.cumsum([np.prod(s) for s in shapes])[:-1])
        weights_to_load = [arr.reshape(shapes[i]) for i,arr in enumerate(splt)]

        model.set_weights(weights_to_load)

    func_map: Dict[ModelRepos, Callable] = {
        ModelRepos.PYTORCH: load_weights_from_flattened_vector_torch,
        ModelRepos.KERAS: load_weights_from_flattened_vector_keras
    }

    return func_map[model_repo](model, model_weights)

@model_processing_func
def extract_weights(model: Union[tfModel, torchModel], model_repo: ModelRepos = None) -> np.ndarray:
    def extract_weights_pytorch(model: torchModel) -> np.ndarray:
        ws = [w.cpu().detach().numpy().flatten() for w in model.parameters()]
        w = np.concatenate(ws)

        return w

    def extract_weights_keras(model: tfModel) -> np.ndarray:
        weights = [w.ravel() for w in model.get_weights()]
        weights_filtered = [w for w in weights if w is not None and len(w) > 0]

        return np.concatenate(weights_filtered)

    func_map: Dict[ModelRepos, Callable] = {
        ModelRepos.PYTORCH: extract_weights_pytorch,
        ModelRepos.KERAS: extract_weights_keras
    }

    return func_map[model_repo](model)

def ret_pretrained_model_by_name(
    model_name,
    lib:ModelRepos,

    train_dataset:Literal['imagenet12'] = 'imagenet12'
):
    def ret_keras_model_by_name(model_name):
        import tensorflow.keras.applications
        import tensorflow.keras

        train_dataset_map = {
            'imagenet12': 'imagenet',
        }

        ret_class = getattr(tensorflow.keras.applications, model_name, None)
        if not ret_class:
            raise Exception(f"ret_keras_model_by_name | model_name {model_name} not found")

        
        model = ret_class(weights=train_dataset_map.get(train_dataset, 'imagenet'))
        return model

    
    if lib == ModelRepos.KERAS:
        return ret_keras_model_by_name(model_name)
    else:
        raise NotImplementedError(f'ret_model_by_name | lib {lib} not implemented')


def ret_model_preprocessing_by_name(model_name, lib:ModelRepos):
    if lib == ModelRepos.PYTORCH:
        return
    
    elif lib == ModelRepos.KERAS:
        keras_models_map = {
                "MobileNet": 'mobilenet', 
                "MobileNetV2": 'mobilenet_v2',
                "MobileNetV3Small": 'mobilenet_v3',
                "MobileNetV3Large": 'mobilenet_v3',
                "NASNetMobile": 'nasnet',
                "DenseNet121": 'densenet',
                "EfficientNetV2B0": 'efficientnet_v2',
                "EfficientNetV2B1": 'efficientnet_v2',

                "Xception": 'xception',
                "ResNet50": 'resnet',
                "ResNet50V2": 'resnet_v2',
                "ResNet101": 'resnet',
                "ResNet101V2": 'resnet_v2',
                "ResNet152": 'resnet',
                "ResNet152V2": 'resnet_v2',

                "InceptionV3": 'inception_v3',
                "InceptionResNetV2": 'inception_resnet_v2',
                "DenseNet169": 'densenet',
                "DenseNet201": 'densenet',
                "NASNetLarge": 'nasnet',

                "EfficientNetV2B2": 'efficientnet_v2',
                "EfficientNetV2B3": 'efficientnet_v2',
                "EfficientNetV2S": 'efficientnet_v2',
                "EfficientNetV2M": 'efficientnet_v2',

                "ConvNeXtTiny": 'convnext',
                "ConvNeXtSmall": 'convnext',
                "ConvNeXtBase": 'convnext',
        }  

        import tensorflow.keras.applications
        ret_class = getattr(tensorflow.keras.applications, keras_models_map.get(model_name, None), None)
        if not ret_class:
            raise Exception(f"ret_keras_model_by_name | model_name {model_name} not found")

        return ret_class.preprocess_input