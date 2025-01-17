import copy
import os, glob
from typing import Union
from fastapi import FastAPI

import numpy as np
from pydantic import BaseModel
import tensorflow as tf

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InferenceRequest(BaseModel):
    model_names: list[str]
    inputs: list[list[list[Union[float, int]]]]


MODELS_DIR_DEFAULT = os.path.join('/', 'usr', 'local', 'app', 'ModelXRay', 'server', 'models')
MODELS_DIR = os.environ.get('MODELS_DIR', MODELS_DIR_DEFAULT)

logger.info(f'MODELS_DIR: {MODELS_DIR}')


def _get_models_data():
    models_data = {}
    for model_path in glob.glob(os.path.join(MODELS_DIR, '*.keras')):
        model_name = os.path.basename(model_path)

        model = tf.keras.models.load_model(model_path)
        train_data = model.train_data
        train_data.pop('x')
        train_data.pop('y')

        train_metadata = train_data.pop('train_metadata')

        # train_data['image_rep_config'] = train_metadata['image_rep_config']
        # train_data['image_preprocess_config'] = train_metadata['image_preprocess_config']
        train_data.update(train_metadata)


        models_data[model_name] = copy.deepcopy(train_data)
        del model

    return models_data

models_data = _get_models_data()

app = FastAPI()

@app.get("/list_models")
def list_models():
    
    return models_data


def _inference_single_model(model_name: str, inputs: np.ndarray):
    model_path = os.path.join(MODELS_DIR, model_name)
    model = tf.keras.models.load_model(model_path)

    y_pred = model.inference_centroid(inputs)

    del model

    return y_pred.tolist()

pred_mapper = {0: 'BENIGN', 1: 'MALICIOUS'}

@app.post("/inference")
def inference(inference_request: InferenceRequest):
    response = {}
    # img_shape = (100,100)

    inputs = np.array([np.array(input) for input in inference_request.inputs])

    for model_name in inference_request.model_names:
        y_pred = _inference_single_model(model_name, inputs)
        y_pred = [pred_mapper[pred] for pred in y_pred]
        response[model_name] = y_pred

    return response

@app.get("/health")
def health_check():
    # Add any additional checks you need
    return {"status": "healthy"}