from pathlib import Path
from typing import Literal
import json

import numpy as np

from model_xray.path_manager import pm
from model_xray.options import small_cnn_zoos

def extract_weights(model, lib:Literal['torch', 'keras']) -> np.ndarray:
    def extract_weights_pytorch(model):
        ws = [w.cpu().detach().numpy().flatten() for w in model.parameters()]
        w = np.concatenate(ws)

        return w

    def extract_weights_keras(model):
        weights = [w.ravel() for w in model.get_weights()]
        weights_filtered = [w for w in weights if w is not None and len(w) > 0]

        return np.concatenate(weights_filtered)

    if lib == 'torch':
        return extract_weights_pytorch(model)
    elif lib == 'keras':
        return extract_weights_keras(model)
    else:
        raise NotImplementedError(f'extract_weights | lib {lib} not implemented')

def ret_pretrained_model_by_name(model_name, lib:Literal['torch', 'keras']):
    def ret_keras_model_by_name(model_name):
        import tensorflow.keras.applications
        ret_class = getattr(tensorflow.keras.applications, model_name, None)
        if not ret_class:
            raise Exception(f"ret_keras_model_by_name | model_name {model_name} not found")

        return ret_class()

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

    if lib == "torch":
        return ret_torch_model_by_name(model_name)
    elif lib == "keras":
        return ret_keras_model_by_name(model_name)
    else:
        raise NotImplementedError(f'ret_model_by_name | lib {lib} not implemented')

