from pathlib import Path
from typing import Literal
import json

import numpy as np
import torch

from model_xray.path_manager import pm
from model_xray.options import small_cnn_zoos

from model_xray.config_classes import ModelRepos

def load_weights_from_flattened_vector(model, model_weights: np.ndarray, lib: ModelRepos):
    def load_weights_from_flattened_vector_torch(): 
        state_dict = model.state_dict()
        torch.nn.utils.vector_to_parameters(model_weights, state_dict.values())

        model.load_state_dict(state_dict)

    def load_weights_from_flattened_vector_keras():
        shapes = [w.shape for w in model.get_weights()]
        splt = np.split(model_weights, np.cumsum([np.prod(s) for s in shapes])[:-1])
        weights_to_load = [arr.reshape(shapes[i]) for i,arr in enumerate(splt)]

        model.set_weights(weights_to_load)

    if lib == ModelRepos.PYTORCH:
        load_weights_from_flattened_vector_torch()
    elif lib == ModelRepos.KERAS: # keras
        load_weights_from_flattened_vector_keras()
    else:
        raise NotImplementedError(f'load_weights_from_flattened_vector | lib {lib} not implemented')

def extract_weights(model, lib:ModelRepos) -> np.ndarray:
    def extract_weights_pytorch(model):
        ws = [w.cpu().detach().numpy().flatten() for w in model.parameters()]
        w = np.concatenate(ws)

        return w

    def extract_weights_keras(model):
        weights = [w.ravel() for w in model.get_weights()]
        weights_filtered = [w for w in weights if w is not None and len(w) > 0]

        return np.concatenate(weights_filtered)

    if lib == ModelRepos.PYTORCH:
        return extract_weights_pytorch(model)
    elif lib == ModelRepos.KERAS:
        return extract_weights_keras(model)
    else:
        raise NotImplementedError(f'extract_weights | lib {lib} not implemented')

def ret_pretrained_model_by_name(model_name, lib:ModelRepos):
    def ret_keras_model_by_name(model_name):
        import tensorflow.keras.applications
        import tensorflow.keras

        ret_class = getattr(tensorflow.keras.applications, model_name, None)
        if not ret_class:
            raise Exception(f"ret_keras_model_by_name | model_name {model_name} not found")

        
        model = ret_class()
        return model

    def ret_torch_model_by_name(model_name):
        if model_name in small_cnn_zoos:
            from model_xray.external_code.ghrp.model_definitions.def_net import NNmodule
            zoo_dir = pm.get_small_cnn_zoo_dir_path(model_name)
            PATH_ROOT = Path(zoo_dir)
            config_model_path = PATH_ROOT.joinpath('config_zoo.json')
            config_model = json.load(config_model_path.open('r'))
            model = NNmodule(config_model)

            return model
        else:
            raise NotImplementedError(f'ret_torch_model_by_name | model_name {model_name} not implemented')

    
    if lib == ModelRepos.PYTORCH:
        return ret_torch_model_by_name(model_name)
    elif lib == ModelRepos.KERAS:
        return ret_keras_model_by_name(model_name)
    else:
        raise NotImplementedError(f'ret_model_by_name | lib {lib} not implemented')

