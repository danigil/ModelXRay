from ..context import model_xray
from model_xray.utils.model_utils import *

import pytest

def _ret_simple_keras_model_sequential(input_shape=(1,2) ,dense_sizes=[2,3,4], kernel_inits=None, bias_inits=None):
    import keras

    if kernel_inits is None:
        kernel_inits_curr = [
            "glorot_uniform" for _ in range(len(dense_sizes))
        ]
    else:
        kernel_inits_curr = kernel_inits

    if bias_inits is None:
        bias_inits_curr = [
            "zeros" for _ in range(len(dense_sizes))
        ]
    else:
        bias_inits_curr = bias_inits

    model = keras.Sequential(
        [
            keras.layers.Dense(dense_size, kernel_initializer=kernel_inits_curr[i], bias_initializer=bias_inits_curr[i])
            for i, dense_size in enumerate(dense_sizes)
        ]
    )

    model.build(input_shape)

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

    assert ModelRepos.determine_model_type(keras_model_seq) == ModelRepos.KERAS
    del keras_model_seq

    keras_model_func = _ret_simple_keras_model_functional()

    assert ModelRepos.determine_model_type(keras_model_func) == ModelRepos.KERAS
    del keras_model_func

def test_determine_model_type_torch():
    torch_model = _ret_simple_torch_model()

    assert ModelRepos.determine_model_type(torch_model) == ModelRepos.PYTORCH
    del torch_model

def test_determine_model_type_unknown():
    not_a_model = "not a model"

    with pytest.raises(NotImplementedError):
        ModelRepos.determine_model_type(not_a_model)

def test_extract_weights_keras():
    import tensorflow as tf
    import keras

    def custom_weight_init(shape, dtype=tf.float32):
        return tf.reshape((tf.range(1, np.prod(shape)+1, dtype=dtype) * 0.111 ), shape=shape)

    model_input_shape = (1,2)
    model_dense_sizes = [2,3,4]
    model_kernel_inits = [custom_weight_init]*3
    model_bias_inits = [custom_weight_init]*3

    model = _ret_simple_keras_model_sequential(
        input_shape=model_input_shape,
        dense_sizes=model_dense_sizes,
        kernel_inits=model_kernel_inits,
        bias_inits=model_bias_inits
    )

    input_kernel_weights = model_kernel_inits[0]((model_input_shape[1], model_dense_sizes[0]))
    input_bias_weights = model_bias_inits[0]((model_dense_sizes[0],))
    
    input_weights = [input_kernel_weights, input_bias_weights]

    hidden_kernel_weights = [
        model_kernel_inits[i]((model_dense_sizes[i], model_dense_sizes[i+1]))
        for i in range(len(model_dense_sizes)-1)
    ]

    hidden_bias_weights = [
        model_bias_inits[i]((model_dense_sizes[i+1],))
        for i in range(len(model_dense_sizes)-1)
    ]

    from itertools import chain
    hidden_weights = list(chain.from_iterable(zip(hidden_kernel_weights, hidden_bias_weights)))

    expected_weights = input_weights + hidden_weights
    expected_weights_flattened = np.concatenate([w.numpy().flatten() for w in expected_weights])

    weights = extract_weights(model)

    assert np.allclose(weights, expected_weights_flattened)


def test_load_weights_keras():
    model = _ret_simple_keras_model_sequential(
    )