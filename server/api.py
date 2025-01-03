import copy
import os, glob
from typing import Union
from fastapi import FastAPI

import numpy as np
from pydantic import BaseModel
import tensorflow as tf

class InferenceRequest(BaseModel):
    model_names: list[str]
    inputs: list[list[list[Union[float, int]]]]


MODELS_DIR = os.path.join('/', 'mnt', 'exdisk1', 'model_xray', 'models_subset')

app = FastAPI()

def _get_models_data():
    models_data = {}
    for model_path in glob.glob(os.path.join(MODELS_DIR, '*.keras')):
        model_name = os.path.basename(model_path)

        model = tf.keras.models.load_model(model_path)
        train_data = model.train_data
        train_data.pop('x')
        train_data.pop('y')

        train_data['image_rep_config'] = train_data['image_rep_config']
        train_data['image_preprocess_config'] = train_data['image_preprocess_config']

        models_data[model_name] = copy.deepcopy(train_data)
        del model

    return models_data

@app.get("/list_models")
def list_models():
    models_data = _get_models_data()

    return models_data


def _inference_single_model(model_name: str, inputs: np.ndarray):
    model_path = os.path.join(MODELS_DIR, model_name)
    model = tf.keras.models.load_model(model_path)

    y_pred = model.inference_centroid(inputs)

    del model

    return y_pred.tolist()


@app.post("/inference")
def inference(inference_request: InferenceRequest):
    response = {}
    img_shape = (100,100)

    inputs = np.array([np.array(input).reshape(img_shape) for input in inference_request.inputs])

    for model_name in inference_request.model_names:
        y_pred = _inference_single_model(model_name, inputs)
        response[model_name] = y_pred

    return response