import os
from typing import Union
from fastapi import FastAPI

import numpy as np
from pydantic import BaseModel
import tensorflow as tf

class InferenceRequest(BaseModel):
    model_names: list[str]
    inputs: list[list[list[Union[float, int]]]]


MODELS_DIR = os.path.join('/', 'mnt', 'exdisk1', 'model_xray', 'models')

app = FastAPI()

@app.get("/list_models")
def list_models():
    return {"Hello": "World"}


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