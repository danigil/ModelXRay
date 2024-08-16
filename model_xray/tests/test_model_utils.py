from model_xray.utils.model_utils import *

import pytest

def _ret_simple_keras_model_sequential():
    import keras

    model = keras.Sequential(
        [
            keras.layers.Dense(2, activation="relu", name="layer1"),
            keras.layers.Dense(3, activation="relu", name="layer2"),
            keras.layers.Dense(4, name="layer3"),
        ]
    )

    return model

def _ret_simple_keras_model_functional():
    import keras

    inputs = keras.Input(shape=(784,))
    dense = keras.layers.Dense(64, activation="relu")
    x = dense(inputs)

    x = keras.layers.Dense(64, activation="relu")(x)
    outputs = keras.layers.Dense(10)(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="mnist_model")

    return model

def _ret_simple_torch_model():
    import torch
    from torch import nn

    class NeuralNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.flatten = nn.Flatten()
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(28*28, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 10),
            )

        def forward(self, x):
            x = self.flatten(x)
            logits = self.linear_relu_stack(x)
            return logits

    model = NeuralNetwork()
    return model

def test_determine_model_type_keras():
    import keras
    from tensorflow.keras import Model

    keras_model_seq = _ret_simple_keras_model_sequential()

    assert determine_model_type(keras_model_seq) == ModelRepos.KERAS
    del keras_model_seq

    keras_model_func = _ret_simple_keras_model_functional()

    assert determine_model_type(keras_model_func) == ModelRepos.KERAS
    del keras_model_func

def test_determine_model_type_torch():
    torch_model = _ret_simple_torch_model()

    assert determine_model_type(torch_model) == ModelRepos.PYTORCH
    del torch_model

def test_determine_model_type_unknown():
    not_a_model = "not a model"

    with pytest.raises(NotImplementedError):
        determine_model_type(not_a_model)

    