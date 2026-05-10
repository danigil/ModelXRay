# Code from https://github.com/hlamba28/One-Shot-Learning-with-Siamese-Networks/blob/master/Siamese%20on%20Omniglot%20Dataset.ipynb

import copy
from typing import Literal, Optional
from model_xray.configs.models import ImagePreprocessConfig, ImageRepConfig
from model_xray.models.srnet import SRNet
import numpy as np
from numpy import linalg as LA
from sklearn.neighbors import KNeighborsClassifier
seed = 122
# np.random.seed(seed)
from itertools import combinations, product
import math

import time

import operator
import gc
import random
# random.seed(seed)

from random import choice, choices, sample
from sklearn.utils import shuffle

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
# tf.random.set_seed(seed)

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, Input

from tensorflow.keras.layers import MaxPooling2D, Lambda, Flatten, Dense, Dropout

from tensorflow.keras.regularizers import l2



from keras import backend as K

from sklearn.utils import shuffle
# from data_locator import request_logger
from sklearn.metrics import accuracy_score

from tqdm import tqdm

# logger = request_logger(__name__, dump_to_sysout=False)
tf.keras.saving.get_custom_objects().clear()


def make_triplets(x,y, size=None, is_shuffle=False, return_tfds:bool=True):
    # def preprocess_sample(sample):
    #     sample = np.expand_dims(sample, axis=(-1))
    #     return sample
    triplets = []

    benign_idxs = np.where(y == 0)[0]
    mal_idxs = np.where(y == 1)[0]

    if size is not None:
        benign_idxs = sample(list(benign_idxs), min(size, len(benign_idxs)))
        mal_idxs = sample(list(mal_idxs), min(size, len(mal_idxs)))

    for anchor_idx, positive_idx in combinations(benign_idxs, 2):
        anchor = x[anchor_idx, ...]
        positive = x[positive_idx, ...]

        for negative_idx in mal_idxs:
            negative = x[negative_idx, ...]
            triplets.append([anchor, positive, negative])

    for anchor_idx, positive_idx in combinations(mal_idxs, 2):
        anchor = x[anchor_idx, ...]
        positive = x[positive_idx, ...]

        for negative_idx in benign_idxs:
            negative = x[negative_idx, ...]
            triplets.append([anchor, positive, negative])

    anchors, positives, negatives = zip(*triplets)
    if is_shuffle:
        anchors, positives, negatives = shuffle(anchors, positives, negatives)

    anchors, positives, negatives = np.array(anchors), np.array(positives), np.array(negatives)

    if return_tfds:
        # return tf.data.Dataset.from_tensor_slices((anchors, positives, negatives))
        anchor_dataset = tf.data.Dataset.from_tensor_slices(anchors)
        positive_dataset = tf.data.Dataset.from_tensor_slices(positives)
        negative_dataset = tf.data.Dataset.from_tensor_slices(negatives)

        dataset = tf.data.Dataset.zip((anchor_dataset, positive_dataset, negative_dataset))
        dataset = dataset.batch(32, drop_remainder=False)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset
    else:
        return [anchors, positives, negatives]
    # return [np.array(anchors), np.array(positives), np.array(negatives)]

def make_triplets_v2(x,y, size=None, is_shuffle=False, return_tfds:bool=True):
    triplets = []

    benign_idxs = np.where(y == 0)[0]
    mal_idxs = np.where(y == 1)[0]

class MyThresholdCallback(tf.keras.callbacks.Callback):
    def __init__(self, ub_mode=True ,threshold_upper=0.5, threshold_lower=0.5):
        super(MyThresholdCallback, self).__init__()
        self.threshold_upper = threshold_upper
        self.threshold_lower = threshold_lower
        self.last_loss = None
        self.ub_mode = ub_mode

    def on_epoch_end(self, epoch, logs=None):
        if self.ub_mode:
            train_loss = logs["loss"]
            val_loss = logs.get("val_loss")
            # print(f"train loss: {train_loss}")
            if self.threshold_lower <= train_loss <= self.threshold_upper:
                self.model.stop_training = True
                self.last_loss = train_loss
    
    def on_train_end(self, logs=None):
        self.last_loss = logs["loss"]

@tf.function(reduce_retracing=True)
def pairwise_euclidean_distance(X: tf.Tensor, Y: Optional[tf.Tensor]=None, sqrt:bool=False) -> tf.Tensor:
    """
    Given an (n, m) tensor X of n vectors (each of dimension m),
    returns an (n, n) matrix of Euclidean distances between each pair of vectors.
    """

    if Y is None:
        Y = X

    # # 1) Compute the squared L2 norm of each row vector: shape (n, 1).
    # X_sq = tf.reduce_sum(tf.square(X), axis=1, keepdims=True)  # (n, 1)

    # # 2) Use the formula: dist^2 = X_sq + X_sq^T - 2 * X * X^T
    # #    Here we use matmul with transpose_b=True to get X * X^T
    # dist_sq = X_sq + tf.transpose(X_sq) - 2.0 * tf.matmul(X, X, transpose_b=True)
    # dist_sq = tf.maximum(dist_sq, 0.0)
    
    # distances = tf.sqrt(dist_sq)

    # 1) Compute the squared L2 norms of each row in X and Y.
    #    => X_sq: (n, 1), Y_sq: (p, 1)
    X_sq = tf.reduce_sum(tf.square(X), axis=1, keepdims=True)  # shape: (n, 1)
    Y_sq = tf.reduce_sum(tf.square(Y), axis=1, keepdims=True)  # shape: (p, 1)
    
    # 2) Compute pairwise squared distances.
    #    The result should be (n, p).
    #    dist^2(X[i], Y[j]) = ||X[i]||^2 + ||Y[j]||^2 - 2 * X[i] dot Y[j]
    dist_sq = X_sq + tf.transpose(Y_sq) - 2.0 * tf.matmul(X, Y, transpose_b=True)
    
    # 3) Clamp negative values to 0 for numerical stability, then sqrt.
    distances = tf.maximum(dist_sq, 0.0)
    if sqrt:
        distances = tf.sqrt(distances)

    # distances = tf.linalg.set_diag(distances, tf.zeros(tf.shape(distances)[0], distances.dtype))

    return distances

@tf.function(reduce_retracing=True)
def compute_C(A: tf.Tensor, B: tf.Tensor) -> tf.Tensor:
    """
    Compute tensor C of shape (n, q, p) such that C[i, j, k] = B[i, k] - A[i, j].

    Parameters:
    A (tf.Tensor): Tensor of shape (n, q)
    B (tf.Tensor): Tensor of shape (n, p)

    Returns:
    tf.Tensor: Tensor C of shape (n, q, p)
    """
    if A.ndim != 2:
        A = tf.expand_dims(A, axis=0)  # Add batch dimension if A is not 2D
    
    if B.ndim != 2:
        B = tf.expand_dims(B, axis=0)

    # Expand dimensions to enable broadcasting
    A_expanded = tf.expand_dims(A, axis=2)  # Now A_expanded has shape (n, q, 1)
    B_expanded = tf.expand_dims(B, axis=1)  # Now B_expanded has shape (n, 1, p)
    
    # Use broadcasting to compute the result
    C = B_expanded - A_expanded  # Resulting shape is (n, q, p)
    return C

@tf.function
def calc_dist(a,b, dist:Literal["l2", "cosine"]="l2"):
    def dist_cosine(a,b):
        # a = tf.math.l2_normalize(a, axis=-1)
        # b = tf.math.l2_normalize(b, axis=-1)
        # return tf.reduce_mean(a * b, axis=-1)

        return 1 + tf.losses.cosine_similarity(tf.nn.l2_normalize(a, 0), tf.nn.l2_normalize(b, 0), -1)

    @tf.function
    def dist_l2(a,b):
        return tf.reduce_sum(tf.square(a - b), -1)

    if dist == "l2":
        return dist_l2(a,b)
    elif dist == "cosine":
        return dist_cosine(a,b)
    else:
        raise ValueError(f"dist must be l2 or cosine, not {dist}")

@tf.keras.saving.register_keras_serializable(package="MyLayers")
class DistanceLayer(tf.keras.layers.Layer):
    """
    This layer is responsible for computing the distance between the anchor
    embedding and the positive embedding, and the anchor embedding and the
    negative embedding.
    """
    def __init__(self, dist:Literal["l2", "cosine"]="l2", **kwargs):
        super().__init__(**kwargs)
        self.dist=dist

    def call(self, anchor, positive, negative):
        ap_distance = calc_dist(anchor, positive, dist=self.dist)
        an_distance = calc_dist(anchor, negative, dist=self.dist)
        return (ap_distance, an_distance)


def reset_weights(model):
  for layer in model.layers: 
    if isinstance(layer, tf.keras.Model):
      reset_weights(layer)
      continue
    for k, initializer in layer.__dict__.items():
      if "initializer" not in k:
        continue
      # find the corresponding variable
      var = getattr(layer, k.replace("_initializer", ""))
      var.assign(initializer(var.shape, var.dtype))

def ret_initializer_weights_rand():
    return tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01)


def ret_initializer_bias_rand():
    return tf.keras.initializers.RandomNormal(mean=0.5, stddev=0.01)

def create_embedding_model(input_shape=(50, 50, 3), embedding_dim=128, spatial_dropout_rate=0.00,dropout_rate=0.00):
    inputs = Input(shape=input_shape)
    
    # Block 1
    x = tf.keras.layers.Conv2D(32, (3,3), padding="same", activation="relu",kernel_initializer=ret_initializer_weights_rand(), kernel_regularizer=l2(2e-4))(inputs)
    if spatial_dropout_rate > 0:
        x = tf.keras.layers.SpatialDropout2D(spatial_dropout_rate)(x)
    x = tf.keras.layers.MaxPooling2D((2,2))(x)  # 25x25x32
    
    # Block 2
    x = tf.keras.layers.Conv2D(64, (3,3), padding="same", activation="relu",
                               kernel_initializer=ret_initializer_weights_rand(),
                                bias_initializer=ret_initializer_bias_rand(), kernel_regularizer=l2(2e-4))(x)
    if spatial_dropout_rate > 0:
        x = tf.keras.layers.SpatialDropout2D(spatial_dropout_rate)(x)
    x = tf.keras.layers.MaxPooling2D((2,2))(x)  # 12x12x64
    
    # Block 3
    x = tf.keras.layers.Conv2D(128, (3,3), padding="same", activation="relu",kernel_initializer=ret_initializer_weights_rand(),
                                bias_initializer=ret_initializer_bias_rand(), kernel_regularizer=l2(2e-4))(x)
    if spatial_dropout_rate > 0:
        x = tf.keras.layers.SpatialDropout2D(spatial_dropout_rate)(x)
    x = tf.keras.layers.MaxPooling2D((2,2))(x)  # 6x6x128
    
    # Flatten and Fully Connected Layer to produce embedding
    x = tf.keras.layers.Flatten()(x)
    embeddings = tf.keras.layers.Dense(embedding_dim,kernel_regularizer=l2(1e-3),
                    kernel_initializer=ret_initializer_bias_rand(),bias_initializer=ret_initializer_bias_rand(),activation=None)(x)

    # Optional dropout layer
    if dropout_rate > 0:
        embeddings = tf.keras.layers.Dropout(dropout_rate)(embeddings)
    
    # Optional L2-normalization (common in metric learning)
    # embeddings = tf.keras.layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=1))(embeddings)
    
    # embeddings = tf.keras.layers.BatchNormalization()(embeddings)

    model = tf.keras.models.Model(inputs, embeddings)
    return model

def calc_reverse_index_mapping(a):
    # vals, idxs = np.unique(a, return_inverse=True)
    mapping = {val: idx for idx, val in enumerate(a)}

    return mapping

def bulk_indexof(a,b):
    sorter = np.argsort(b)
    ret = sorter[np.searchsorted(b, a, sorter=sorter)]

    return ret

def coarse_label_map(y):
    return [0 if i == 0 else 1 for i in y]

@tf.keras.saving.register_keras_serializable(package="MyModels")
class Siamese(Model):
    # partially based on code from https://keras.io/examples/vision/siamese_network/
    def __init__(self,
                 margin=0.5,
                #  pretrained=False,
                 
                 dist:Literal["l2", "cosine"]="l2",
                 img_input_shape=(100,100,1),
                 lr=0.0001,
                 
                #  weights_init = "random", 
                 dropout_rate=0.5,
                 model=None,
                 model_arch:Literal['osl_siamese_cnn', 'srnet']='srnet',
                 optimizer=None,

                 train_data=None,
                 ):
        super().__init__()
        
        if model is None:
            if model_arch == 'osl_siamese_cnn':
                model = Sequential()
                model.add(Conv2D(64, (10,10), activation='relu', input_shape=img_input_shape,
                            kernel_initializer=ret_initializer_weights_rand(), kernel_regularizer=l2(2e-4)))
                model.add(MaxPooling2D())
                if dropout_rate > 0:
                    model.add(Dropout(dropout_rate))
                
                model.add(Conv2D(128, (7,7), activation='relu',
                                kernel_initializer=ret_initializer_weights_rand(),
                                bias_initializer=ret_initializer_bias_rand(), kernel_regularizer=l2(2e-4)))
                model.add(MaxPooling2D())
                if dropout_rate > 0:
                    model.add(Dropout(dropout_rate))
                
                model.add(Conv2D(128, (4,4), activation='relu', kernel_initializer=ret_initializer_weights_rand(),
                                bias_initializer=ret_initializer_bias_rand(), kernel_regularizer=l2(2e-4)))
                model.add(MaxPooling2D())
                if dropout_rate > 0:
                    model.add(Dropout(dropout_rate))
                
                model.add(Conv2D(256, (4,4), activation='relu', kernel_initializer=ret_initializer_weights_rand(),
                                bias_initializer=ret_initializer_bias_rand(), kernel_regularizer=l2(2e-4)))
                if dropout_rate > 0:
                    model.add(Dropout(dropout_rate))
                
                model.add(Flatten())
                # Note: kernel_initializer was previously set to ret_initializer_bias_rand
                # (mean=0.5, stddev=0.01), which made every kernel weight ~0.5 and
                # collapsed the Dense output to a near-constant vector for any input.
                # After L2-normalize on the embedding head this gave triplet loss
                # = margin (0.5) for the entire 100-epoch run. Use the weights
                # initializer for the kernel.
                model.add(Dense(4096, activation=None,
                    kernel_regularizer=l2(1e-3),
                    kernel_initializer=ret_initializer_weights_rand(),
                    bias_initializer=ret_initializer_bias_rand()))
            elif model_arch == 'srnet':
                # assert img_input_shape == (256,256,1), "srnet only supports 256x256x1 images"
                model = SRNet(include_top=False)

            else:
                raise ValueError(f"Unknown model_arch={model_arch!r}; supported: osl_siamese_cnn, srnet")
        
        # embedding = tf.keras.layers.BatchNormalization()(model)
        # embedding = 
        # embedding = model
        embedding = Sequential([
            model,
            tf.keras.layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=1)),
        ])

        anchor_input = tf.keras.layers.Input(name="anchor", shape=img_input_shape)
        positive_input = tf.keras.layers.Input(name="positive", shape=img_input_shape)
        negative_input = tf.keras.layers.Input(name="negative", shape=img_input_shape)

        distances = DistanceLayer(dist=dist)(
            embedding(anchor_input),
            embedding(positive_input),
            embedding(negative_input),
        )

        self.siamese_network = Model(
            inputs=[anchor_input, positive_input, negative_input], outputs=distances
        )
        
        self.margin = margin
        self.dist = dist
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

        if optimizer is None:
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        self.compile(loss=None ,optimizer=optimizer, weighted_metrics=[tf.keras.losses.categorical_crossentropy])

        self.img_input_shape = img_input_shape
        self.embedding = embedding

        self.train_data = train_data
        # self.pretrained=pretrained

    def fit_and_keep_refs(self, x_train, y_train,
                          epochs=10, batch_size=16, verbose=1, is_shuffle=True, callbacks = [],
                          train_metadata: Optional[dict]=None,
                          size:Optional[int]=None,
                          online_train:bool=False,
                          test_data:Optional[tuple]=None,
                          skip_hard_negatives:bool=False,):
        triplets_test=None
        if test_data is not None:
            x_test, y_test = test_data
            if x_test.ndim == 3:
                x_test = np.expand_dims(x_test, axis=-1)
        else:
            x_test = None
            y_test = None

        if x_train.ndim == 3:
            x_train = np.expand_dims(x_train, axis=-1)

        fit_ret=None

        if x_train.dtype == np.uint8:
            x_train = x_train.astype(np.float32) / 255.0

        if x_test is not None and x_test.dtype == np.uint8:
            x_test = x_test.astype(np.float32) / 255.0

        layer = tf.keras.layers.Normalization(axis=-1)
        layer.adapt(x_train)
        self.normalization_layer = layer

        x_train = layer(x_train)
        if x_test is not None:
            x_test = layer(x_test)
        
        if online_train:
            n=5

            x_train_ds = None
            # x_train_ds = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(x_train)).batch(32)

            # triplets_train = self.online_triplet_mine(x_train, y_train, return_tfds=True, x_train_ds=x_train_ds)

            rounds = math.ceil(epochs / n)

            curr_skip_hard_negatives = False

            for i in range(rounds):
                
                triplets_train = self.online_triplet_mine_generalized(x_train, y_train, return_tfds=True, x_train_ds=x_train_ds, skip_hard_negatives=curr_skip_hard_negatives,k=3, only_benign_triplets=True,)

                curr_skip_hard_negatives = skip_hard_negatives

                if test_data is not None:
                    x_test_ds = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(x_test)).batch(32)
                    triplets_test = self.online_triplet_mine_generalized(x_test, y_test, return_tfds=True, x_train_ds=x_test_ds,skip_hard_negatives=False, only_benign_triplets=False,)

                if len(triplets_train) == 0:
                    print("No triplets found, skipping round.")
                    continue
                fit_ret = self.fit(triplets_train, epochs=min(n, epochs), batch_size=batch_size, verbose=verbose, callbacks=callbacks, validation_data=triplets_test)
                loss = fit_ret.history['loss'][-1]

                print(f"round {i+1}/{rounds}, loss: {loss}")
                
                # if loss < 0.1:
                #     break
        else:
            triplets_test = make_triplets(x_test, y_test, is_shuffle=False, size=size)
            triplets_train = make_triplets(x_train, y_train, is_shuffle=is_shuffle, size=size)
            fit_ret = self.fit(triplets_train, epochs=epochs, batch_size=batch_size, verbose=verbose, callbacks=callbacks, validation_data=triplets_test)

        train_data = {
            'x': x_train,
            'y': y_train,
            'train_metadata': train_metadata,
        }

        self.train_data = train_data

        # self.calc_centroids(x_train, y_train)
        self.calc_train_embeddings(x_train, y_train)

        return fit_ret

    def online_triplet_mine_generalized(   
        self,
        x, y,
        return_tfds:bool=True,
        x_train_ds:Optional[tf.data.Dataset]=None,
        k=1,
        sample_fractions:Optional[dict]={0: 1.0},
        skip_hard_negatives:bool=False, only_benign_triplets:bool=True
    ):
        unique_labels = np.unique(y)
        benign_label = 0
        mal_labels = unique_labels[unique_labels != benign_label]

        specific_label_idxs = {
            label: np.where(y == label)[0] for label in unique_labels
        }

        specific_label_sample_sizes = {
            label: int(len(idxs) * sample_fractions.get(label, 0.5)) for label, idxs in specific_label_idxs.items()
        }

        specific_label_idxs_sampled = {
            label: np.random.choice(idxs, specific_label_sample_sizes[label], replace=False) for label, idxs in specific_label_idxs.items()
        }

        benign_idxs = specific_label_idxs_sampled[benign_label]
        # benign_idxs_reverse_mapping = calc_reverse_index_mapping(benign_idxs)
        # benign_idxs_reverse = benign_idxs.copy()[benign_idxs]
        mal_idxs = np.concatenate([specific_label_idxs_sampled[label] for label in mal_labels])
        # mal_idxs_reverse_mapping = calc_reverse_index_mapping(mal_idxs)
        # mal_idxs_relative 
        # ma

        all_idxs = np.concatenate([benign_idxs, mal_idxs])
        benign_idxs_relative_to_all = bulk_indexof(benign_idxs, all_idxs)
        mal_idxs_relative_to_all = bulk_indexof(mal_idxs, all_idxs)


        x_subset = tf.gather(x, indices=all_idxs, axis=0)
        x_subset_emb = x_emb = self.embedding.predict(x_subset, batch_size=32, verbose=0)

        # x_emb = tf.convert_to_tensor(self.embedding.predict(x_train_ds, batch_size=32, verbose=0))
        # x_emb_specific = {
        #     label: tf.gather(x_emb, indices=idxs) for label, idxs in specific_label_idxs_sampled.items()
        # }
        # x_specific = {
        #     label: tf.gather(x, indices=idxs) for label, idxs in specific_label_idxs_sampled.items()
        # }

        gc.collect()

        specific_label_amnts = {
            label: len(idxs) for label, idxs in specific_label_idxs_sampled.items()
        }

        benign_amnt = specific_label_amnts[benign_label]

        all_distances = pairwise_euclidean_distance(x_emb)

        benign_distances = tf.gather(all_distances, indices=benign_idxs_relative_to_all, axis=0)
        benign_distances = tf.gather(benign_distances, indices=benign_idxs_relative_to_all, axis=1)
        # benign_distances = pairwise_euclidean_distance(x_emb_specific[benign_label])
        benign_hard_positives = tf.math.top_k(benign_distances, k=min(k, benign_amnt)).indices

        benign_mal_distances = tf.gather(all_distances, indices=benign_idxs_relative_to_all, axis=0)
        benign_mal_distances = tf.gather(benign_mal_distances, indices=mal_idxs_relative_to_all, axis=1)

        # mal_distances = tf.gather(all_distances, indices=mal_idxs, axis=0)
        # mal_distances = tf.gather(mal_distances, indices=mal_idxs, axis=1)
        # mal_hard_positives = tf.math.top_k(mal_distances, k=min(k, len(mal_idxs))).indices

        # benign_mal_distances = pairwise_euclidean_distance(x_emb_specific[benign_label], x_emb_specific[mal_labels[0]])
        # benign_non_benign_distances = {i: pairwise_euclidean_distance(x_emb_specific[benign_label], x_emb_specific[i])  for i in mal_labels}

        triplets = []
        triplets_idxs = []

        compute_c_batch_size = 32
        
        benign_idxs_split = np.array_split(range(benign_amnt), math.ceil(benign_amnt / compute_c_batch_size))
        mal_idxs_split = np.array_split(range(len(mal_idxs)), math.ceil(len(mal_idxs) / compute_c_batch_size))

        for split_idx, benign_anchor_idxs_relative in enumerate(benign_idxs_split):
            # benign_anchor_idxs_absolute = benign_idxs[benign_anchor_idxs_relative]
            benign_distances_curr = tf.gather(benign_distances, benign_anchor_idxs_relative, axis=0)
            benign_mal_distances_curr = tf.gather(benign_mal_distances, benign_anchor_idxs_relative, axis=0)
            
            benign_pos_minus_neg = compute_C(benign_distances_curr, benign_mal_distances_curr)
            # benign_pos_minus_neg_top = tf.math.top_k(benign_pos_minus_neg, k=min(k, benign_amnt)).indices.numpy()

            for local_idx, benign_anchor_idx_relative in enumerate(benign_anchor_idxs_relative):
                # anchor = x_benign[benign_anchor_idx, ...]
                benign_anchor_idx = benign_idxs[benign_anchor_idx_relative]
                
                for benign_positive_idx_relative in benign_hard_positives[benign_anchor_idx_relative]:
                    benign_positive_idx = benign_idxs[benign_positive_idx_relative]

                    # print(f'benign_anchor_idx_relative: {benign_anchor_idx_relative}, benign_positive_idx_relative: {benign_positive_idx_relative}, benign_positive_idx: {benign_positive_idx}')
                    # positive = x_benign[benign_positive_idx, ...]

                    for mal_label in mal_labels:
                        curr_mal_idxs_absolute = specific_label_idxs_sampled[mal_label]
                        curr_mal_idxs_relative = bulk_indexof(curr_mal_idxs_absolute, mal_idxs)

                        benign_pos_minus_neg_curr = tf.gather(benign_pos_minus_neg, curr_mal_idxs_relative, axis=2)
                        benign_pos_minus_neg_top = tf.math.top_k(benign_pos_minus_neg_curr, k=min(k, specific_label_amnts[mal_label])).indices
                        # print(f'benign_pos_minus_neg_top: {benign_pos_minus_neg_top.shape}')

                        # print(f'anchor: {benign_anchor_idx_relative}, positive: {benign_positive_idx_relative}, negative: {benign_pos_minus_neg_top.shape}')

                        for benign_negative_idx_relative in benign_pos_minus_neg_top[local_idx, benign_positive_idx_relative]:
                            curr_diff = benign_pos_minus_neg[local_idx, benign_positive_idx_relative, benign_negative_idx_relative]

                            if curr_diff <= 0:
                                if skip_hard_negatives:
                                    break

                            benign_negative_idx = mal_idxs[benign_negative_idx_relative]
                            
                            # print(f'benign_anchor_idx: {benign_anchor_idx}, benign_positive_idx: {benign_positive_idx}, benign_negative_idx: {benign_negative_idx}')
                            # negative = x_mal[benign_negative_idx, ...]
                            # triplets.append([anchor, positive, negative])
                            triplets_idxs.append([benign_anchor_idx, benign_positive_idx, benign_negative_idx])

        if not only_benign_triplets:
            # for split_idx, mal_anchor_idxs_relative in enumerate(mal_idxs_split):
            for mal_label in mal_labels:
                # mal_anchor_idxs_absolute = mal_idxs[mal_anchor_idxs_relative]

                mal_anchor_idxs_absolute = curr_mal_idxs_absolute = specific_label_idxs_sampled[mal_label]
                mal_anchor_idxs_relative = curr_mal_idxs_relative = bulk_indexof(curr_mal_idxs_absolute, mal_idxs)

                # print(f'mal_anchor_idxs_relative: {mal_anchor_idxs_relative.shape}, mal_anchor_idxs_absolute: {mal_anchor_idxs_absolute.shape}')

                mal_distances_curr = tf.gather(all_distances, indices=mal_anchor_idxs_absolute, axis=0)
                mal_distances_curr = tf.gather(mal_distances_curr, indices=mal_anchor_idxs_absolute, axis=1)
                mal_hard_positives_curr = tf.math.top_k(mal_distances_curr, k=min(k, specific_label_amnts[mal_label])).indices.numpy()
                
                mal_benign_distances_curr = tf.gather(all_distances, indices=benign_idxs, axis=0)
                mal_benign_distances_curr = tf.gather(mal_benign_distances_curr, indices=mal_anchor_idxs_absolute, axis=1)
                mal_benign_distances_curr = tf.transpose(mal_benign_distances_curr)
                
                mal_pos_minus_neg = compute_C(mal_distances_curr, mal_benign_distances_curr)

                for local_idx, mal_anchor_idx_relative in enumerate(mal_anchor_idxs_relative):
                    # anchor = x_mal[mal_anchor_idx, ...]
                    mal_anchor_idx = mal_idxs[mal_anchor_idx_relative]

                    for mal_positive_idx_relative in mal_hard_positives_curr[local_idx]:
                        # positive = x_mal[mal_positive_idx, ...]

                        # mal_positive_idx = mal_idxs[mal_positive_idx_relative]
                        mal_positive_idx = mal_anchor_idxs_absolute[mal_positive_idx_relative]
                        # print(f'mal_positive_idx_relative: {mal_positive_idx_relative}, mal_positive_idx: {mal_positive_idx}')

                        # for benign_label in benign_labels:
                            # curr_benign_idxs_absolute = specific_label_idxs_sampled[benign_label]
                            # curr_benign_idxs_relative = bulk_indexof(curr_benign_idxs_absolute, benign_idxs)


                        # mal_pos_minus_neg_curr = tf.gather(mal_pos_minus_neg, curr_benign_idxs_relative, axis=2)
                        mal_pos_minus_neg_top = tf.math.top_k(mal_pos_minus_neg, k=min(k, specific_label_amnts[benign_label])).indices.numpy()

                        for mal_negative_idx_relative in mal_pos_minus_neg_top[local_idx, mal_positive_idx_relative]:
                            curr_diff = mal_pos_minus_neg[local_idx, mal_positive_idx_relative, mal_negative_idx_relative]

                            if curr_diff <= 0:
                                if skip_hard_negatives:
                                    break

                            mal_negative_idx = benign_idxs[mal_negative_idx_relative]
                            
                            # negative = x_benign[mal_negative_idx, ...]
                            # triplets.append([anchor, positive, negative])
                            triplets_idxs.append([mal_anchor_idx, mal_positive_idx, mal_negative_idx])

        if len(triplets_idxs) == 0:
            print("No triplets found, returning empty dataset.")
            return []

        # counts = {}

        for anchor_idx, positive_idx, negative_idx in triplets_idxs:
            y_anchor, y_positive, y_negative = y[anchor_idx], y[positive_idx], y[negative_idx]

            assert y_anchor == y_positive, f"anchor and positive labels do not match: {y_anchor} != {y_positive}"
            assert y_anchor != y_negative, f"anchor and negative labels match: {y_anchor} == {y_negative}"
            # counts[y_anchor] = {counts[]}


        anchor_idxs, positive_idxs, negative_idxs = zip(*triplets_idxs)

        # anchors, positives, negatives = zip(*triplets)
        # anchors, positives, negatives = np.array(anchors), np.array(positives), np.array(negatives)

        tf.keras.backend.clear_session()

        if return_tfds:
            anchor_idxs_tf, positive_idxs_tf, negative_idxs_tf = [tf.convert_to_tensor(idx, dtype=tf.int32) for idx in [anchor_idxs, positive_idxs, negative_idxs]]
            triplet_idx_dataset = tf.data.Dataset.from_tensor_slices(
                (anchor_idxs_tf, positive_idxs_tf, negative_idxs_tf)
            )

            # 5. Map index tuples to actual feature triplets
            def gather_triplet(a_idx, p_idx, n_idx):
                # assert y[a_idx] == y[p_idx], f"anchor and positive labels do not match: {y[a_idx]} != {y[p_idx]}"
                # assert y[a_idx] != y[n_idx], f"anchor and negative labels match: {y[a_idx]} == {y[n_idx]}"

                anchor = tf.gather(x, a_idx)
                positive = tf.gather(x, p_idx)
                negative = tf.gather(x, n_idx)
                return (anchor, positive, negative)

            dataset = triplet_idx_dataset.map(gather_triplet)

            # dataset_anchors = tf.data.Dataset.from_tensor_slices(tf.gather(x, anchor_idxs_tf))
            # dataset_positives = tf.data.Dataset.from_tensor_slices(tf.gather(x, positive_idxs_tf))
            # dataset_negatives = tf.data.Dataset.from_tensor_slices(tf.gather(x, negative_idxs_tf))

            # dataset = tf.data.Dataset.zip((dataset_anchors, dataset_positives, dataset_negatives))
            
            # anchor_dataset = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(anchors))
            # positive_dataset = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(positives))
            # negative_dataset = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(negatives))

            # dataset = tf.data.Dataset.zip((anchor_dataset, positive_dataset, negative_dataset))
            dataset = dataset.batch(32, drop_remainder=False)
            dataset = dataset.prefetch(tf.data.AUTOTUNE)
            dataset = dataset.shuffle(buffer_size=1000, reshuffle_each_iteration=True)

            return dataset
        else:
            return

    def online_triplet_mine(self, x, y, return_tfds:bool=True, x_train_ds:Optional[tf.data.Dataset]=None, k=1, benign_sample_fraction:float=1.0, mal_sample_fraction:float=0.1, skip_hard_negatives:bool=False, only_benign_triplets:bool=True):
        benign_idxs = np.where(y == 0)[0]
        mal_idxs = np.where(y == 1)[0]

        # # x_emb_benign = self.embedding.predict(x[benign_idxs, ...], batch_size=4)
        # # x_emb_mal = self.embedding.predict(x[mal_idxs, ...], batch_size=4)

        # if x.ndim == 3:
        #     x = np.expand_dims(x, axis=-1)

        # print(x.shape)

        # x_emb_benign = self.embedding(x[benign_idxs, ...])
        # x_emb_mal = self.embedding(x[mal_idxs, ...])

        # x_emb = self.embedding.predict(x_train_ds, batch_size=32, verbose=1)/
        x_emb = tf.convert_to_tensor(self.embedding.predict(x_train_ds, batch_size=32, verbose=1))
        # print(f'x: {x.shape}')
        # print(f'x_emb: {x_emb.shape}')

        benign_sample_size = int(len(benign_idxs) * benign_sample_fraction)
        mal_sample_size = int(len(mal_idxs) * mal_sample_fraction)
        benign_sample_size = min(benign_sample_size, len(benign_idxs))
        mal_sample_size = min(mal_sample_size, len(mal_idxs))

        benign_idxs = np.random.choice(benign_idxs, benign_sample_size, replace=False)
        mal_idxs = np.random.choice(mal_idxs, mal_sample_size, replace=False)

        # print(f'max(benign_idxs): {benign_idxs.max()}, min(benign_idxs): {benign_idxs.min()}')
        # print(f'max(mal_idxs): {mal_idxs.max()}, min(mal_idxs): {mal_idxs.min()}')

        
        # x_emb_benign = x_emb[benign_idxs, ...]
        # x_emb_mal = x_emb[mal_idxs, ...]
        x_emb_benign = tf.gather(x_emb, indices=benign_idxs)
        x_emb_mal = tf.gather(x_emb, indices=mal_idxs)
        # print(f'x_emb_benign: {x_emb_benign.shape}')
        # print(f'x_emb_mal: {x_emb_mal.shape}')
        
        gc.collect()

        # x_benign = x[benign_idxs, ...]
        # x_mal = x[mal_idxs, ...]
        x_benign = tf.gather(x, indices=benign_idxs)
        x_mal = tf.gather(x, indices=mal_idxs)

        benign_amnt = len(benign_idxs)
        mal_amnt = len(mal_idxs)


        benign_distances = pairwise_euclidean_distance(x_emb_benign)
        benign_mal_distances = pairwise_euclidean_distance(x_emb_benign, x_emb_mal)
        # print(f'benign_distances: {benign_distances.shape},')
        # print(f'benign_mal_distances: {benign_mal_distances.shape}')
        # print(f'benign_mal_distances: {benign_mal_distances.shape}, dtype: {benign_mal_distances.dtype}, type: {type(benign_mal_distances)}')
        # print(f'benign_distances: {benign_distances.shape}')
        # benign_hard_positives = tf.argmax(benign_distances, axis=1)
        benign_hard_positives = tf.math.top_k(benign_distances, k=min(k, benign_amnt)).indices.numpy()
        # print(f'benign_hard_positives: {benign_hard_positives.shape}')
        # benign_hard_negatives = tf.argmin(benign_mal_distances, axis=1)
        # benign_hard_negatives = tf.math.top_k(-benign_mal_distances, k=k).indices.numpy()
        # benign_pos_minus_neg = compute_C(benign_distances, benign_mal_distances)
        # benign_pos_minus_neg_top = tf.math.top_k(benign_pos_minus_neg, k=k).indices.numpy()
        # print(benign_pos_minus_neg_top.shape)

        mal_distances = pairwise_euclidean_distance(x_emb_mal)
        mal_benign_distances = tf.transpose(benign_mal_distances)
        mal_hard_positives = tf.math.top_k(mal_distances, k=min(k, mal_amnt)).indices.numpy()
        # mal_hard_negatives = tf.argmin(tf.transpose(benign_mal_distances), axis=1)
        # mal_hard_negatives = tf.math.top_k(-tf.transpose(benign_mal_distances), k=k).indices.numpy()
        # mal_pos_minus_neg = compute_C(mal_distances, mal_benign_distances)
        # mal_pos_minus_neg_top = tf.math.top_k(mal_pos_minus_neg, k=k).indices.numpy()

        # print(f'starting triplet mining, benign sample size: {benign_sample_size}, mal sample size: {mal_sample_size}')

        triplets = []
        triplets_idxs = []

        compute_c_batch_size = 32
        benign_idxs_split = np.array_split(range(len(benign_idxs)), math.ceil(len(benign_idxs) / compute_c_batch_size))
        # print(f'benign_idxs_split: {benign_idxs_split}')
        mal_idxs_split = np.array_split(range(len(mal_idxs)), math.ceil(len(mal_idxs) / compute_c_batch_size))

        for split_idx, benign_anchor_idxs in enumerate(benign_idxs_split):
            benign_pos_minus_neg = compute_C(tf.gather(benign_distances, benign_anchor_idxs), tf.gather(benign_mal_distances, benign_anchor_idxs))
            # print(f'benign_pos_minus_neg: {benign_pos_minus_neg.shape}')
            benign_pos_minus_neg_top = tf.math.top_k(benign_pos_minus_neg, k=min(k, benign_amnt)).indices.numpy()

            # print(f'benign_pos_minus_neg: {benign_pos_minus_neg.shape}')
            # print(f'benign_pos_minus_neg_top: {benign_pos_minus_neg_top.shape}')

            for local_idx, benign_anchor_idx in enumerate(benign_anchor_idxs):
                # print(f'benign_anchor_idx: {benign_anchor_idx}')
                anchor = x_benign[benign_anchor_idx, ...]
        # for benign_anchor_idx, anchor in enumerate(x_benign):

            # anchor = x[benign_anchor_idx, ...]

            # idx_tensor = tf.convert_to_tensor([benign_anchor_idx,], dtype=tf.int32)

                # benign_pos_minus_neg = compute_C(benign_distances[benign_anchor_idx,...], benign_mal_distances[benign_anchor_idx,...])
                # # print(f'benign_pos_minus_neg: {benign_pos_minus_neg.shape}')
                # benign_pos_minus_neg_top = tf.math.top_k(benign_pos_minus_neg, k=k).indices.numpy()
                
                for benign_positive_idx in benign_hard_positives[benign_anchor_idx]:
                    positive = x_benign[benign_positive_idx, ...]

                    # print(f'benign_positive_idx: {benign_positive_idx}')
                    # print(f'benign_hard_positives: {benign_hard_positives[benign_anchor_idx]}')

                    # for benign_negative_idx in benign_hard_negatives[benign_anchor_idx]:
                    # for benign_negative_idx in choices(range(len(mal_idxs)), k=k):
                    # for benign_negative_idx in benign_pos_minus_neg_top[benign_anchor_idx, benign_positive_idx]:
                    for benign_negative_idx in benign_pos_minus_neg_top[local_idx, benign_positive_idx]:
                        curr_diff = benign_pos_minus_neg[local_idx, benign_positive_idx, benign_negative_idx]

                        if curr_diff <= 0:
                            # benign_negative_idx = choice(range(len(mal_idxs)))
                            if skip_hard_negatives:
                                continue
                        
                        # print(f'f(b,m) - f(b,b): {benign_pos_minus_neg[0, benign_positive_idx, benign_negative_idx]}')
                        negative = x_mal[benign_negative_idx, ...]
                        # triplets.append([anchor, positive, negative])
                        triplets_idxs.append([benign_anchor_idx, benign_positive_idx, benign_negative_idx])

        n_benign_triplets = len(triplets)
        # print(f'benign triplets: {n_benign_triplets}')
        # print(f'benign_triplets_idxs: {triplets_idxs}')
        if not only_benign_triplets:
        # for mal_anchor_idx in mal_idxs:
        #     anchor = x[mal_anchor_idx, ...]
        # for mal_anchor_idx, anchor in enumerate(x_mal):
            for split_idx, mal_anchor_idxs in enumerate(mal_idxs_split):

                mal_pos_minus_neg = compute_C(tf.gather(mal_distances, mal_anchor_idxs), tf.gather(mal_benign_distances, mal_anchor_idxs))
                mal_pos_minus_neg_top = tf.math.top_k(mal_pos_minus_neg, k=min(k, mal_amnt)).indices.numpy()

                for local_idx, mal_anchor_idx in enumerate(mal_anchor_idxs):
                    anchor = x[mal_anchor_idx, ...]
                    for mal_positive_idx in mal_hard_positives[mal_anchor_idx]:
                        positive = x_mal[mal_positive_idx, ...]

                        # for mal_negative_idx in mal_hard_negatives[mal_anchor_idx]:
                        # for mal_negative_idx in choices(range(len(benign_idxs)), k=k):
                        for mal_negative_idx in mal_pos_minus_neg_top[local_idx, mal_positive_idx]:
                            curr_diff = mal_pos_minus_neg[local_idx, mal_positive_idx, mal_negative_idx]

                            if curr_diff <= 0:
                                # mal_negative_idx = choice(range(len(benign_idxs)))
                                if skip_hard_negatives:
                                    continue
                            # print(f'f(m,b) - f(m,m): {mal_pos_minus_neg[0, mal_positive_idx, mal_negative_idx]}')
                            negative = x_benign[mal_negative_idx, ...]
                            # triplets.append([anchor, positive, negative])
                            triplets_idxs.append([mal_anchor_idx, mal_positive_idx, mal_negative_idx])



        # print(f'mal triplets: {len(triplets) - n_benign_triplets}')

        if len(triplets) == 0:
            print("No triplets found, returning empty dataset.")
            return []

        anchors, positives, negatives = zip(*triplets)
        anchors, positives, negatives = np.array(anchors), np.array(positives), np.array(negatives)

        # tf.reset_default_graph()
        tf.keras.backend.clear_session()

        if return_tfds:
            # return tf.data.Dataset.from_tensor_slices((anchors, positives, negatives))
            anchor_dataset = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(anchors))
            positive_dataset = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(positives))
            negative_dataset = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(negatives))

            dataset = tf.data.Dataset.zip((anchor_dataset, positive_dataset, negative_dataset))
            dataset = dataset.batch(32, drop_remainder=False)
            dataset = dataset.prefetch(tf.data.AUTOTUNE)

            return dataset
        else:
            return [anchors, positives, negatives]
        # benign_mal_distances = pairwise_euclidean_distance(x_emb_benign, x_emb_mal)

        # for anchor_idx, positive_idx in combinations(benign_idxs, 2):
        #     anchor = x[anchor_idx, ...]
        #     positive = x[positive_idx, ...]

        #     for negative_idx in mal_idxs:
        #         negative = x[negative_idx, ...]
        #         triplets.append([anchor, positive, negative])

    def online_triplet_mine_efficient(self, x, y, return_tfds:bool=True, x_train_ds:Optional[tf.data.Dataset]=None, k=1, benign_sample_fraction:float=1.0, mal_sample_fraction:float=0.1, skip_hard_negatives:bool=False, only_benign_triplets:bool=True):
        benign_idxs = np.where(y == 0)[0]
        mal_idxs = np.where(y == 1)[0]

        x_emb = tf.convert_to_tensor(self.embedding.predict(x_train_ds, batch_size=32, verbose=1))

        benign_sample_size = int(len(benign_idxs) * benign_sample_fraction)
        mal_sample_size = int(len(mal_idxs) * mal_sample_fraction)
        benign_sample_size = min(benign_sample_size, len(benign_idxs))
        mal_sample_size = min(mal_sample_size, len(mal_idxs))

        benign_idxs = np.random.choice(benign_idxs, benign_sample_size, replace=False)
        mal_idxs = np.random.choice(mal_idxs, mal_sample_size, replace=False)

        x_emb_benign = tf.gather(x_emb, indices=benign_idxs)
        x_emb_mal = tf.gather(x_emb, indices=mal_idxs)

        
        gc.collect()

        x_benign = tf.gather(x, indices=benign_idxs)
        x_mal = tf.gather(x, indices=mal_idxs)

        benign_amnt = len(benign_idxs)
        mal_amnt = len(mal_idxs)


        benign_distances = pairwise_euclidean_distance(x_emb_benign)
        benign_mal_distances = pairwise_euclidean_distance(x_emb_benign, x_emb_mal)

        benign_hard_positives = tf.math.top_k(benign_distances, k=min(k, benign_amnt)).indices.numpy()

        mal_distances = pairwise_euclidean_distance(x_emb_mal)
        mal_benign_distances = tf.transpose(benign_mal_distances)
        mal_hard_positives = tf.math.top_k(mal_distances, k=min(k, mal_amnt)).indices.numpy()

        triplets = []
        triplets_idxs = []

        compute_c_batch_size = 32
        benign_idxs_split = np.array_split(range(len(benign_idxs)), math.ceil(len(benign_idxs) / compute_c_batch_size))
        mal_idxs_split = np.array_split(range(len(mal_idxs)), math.ceil(len(mal_idxs) / compute_c_batch_size))

        for split_idx, benign_anchor_idxs in enumerate(benign_idxs_split):
            benign_pos_minus_neg = compute_C(tf.gather(benign_distances, benign_anchor_idxs), tf.gather(benign_mal_distances, benign_anchor_idxs))
            benign_pos_minus_neg_top = tf.math.top_k(benign_pos_minus_neg, k=min(k, benign_amnt)).indices.numpy()

            for local_idx, benign_anchor_idx in enumerate(benign_anchor_idxs):
                # anchor = x_benign[benign_anchor_idx, ...]
                
                for benign_positive_idx in benign_hard_positives[benign_anchor_idx]:
                    # positive = x_benign[benign_positive_idx, ...]

                    for benign_negative_idx in benign_pos_minus_neg_top[local_idx, benign_positive_idx]:
                        curr_diff = benign_pos_minus_neg[local_idx, benign_positive_idx, benign_negative_idx]

                        if curr_diff <= 0:
                            if skip_hard_negatives:
                                continue
                        # negative = x_mal[benign_negative_idx, ...]
                        triplets_idxs.append([benign_anchor_idx, benign_positive_idx, benign_negative_idx])

        # n_benign_triplets = len(triplets)
        # if not only_benign_triplets:
        #     for split_idx, mal_anchor_idxs in enumerate(mal_idxs_split):

        #         mal_pos_minus_neg = compute_C(tf.gather(mal_distances, mal_anchor_idxs), tf.gather(mal_benign_distances, mal_anchor_idxs))
        #         mal_pos_minus_neg_top = tf.math.top_k(mal_pos_minus_neg, k=min(k, mal_amnt)).indices.numpy()

        #         for local_idx, mal_anchor_idx in enumerate(mal_anchor_idxs):
        #             # anchor = x[mal_anchor_idx, ...]
        #             for mal_positive_idx in mal_hard_positives[mal_anchor_idx]:
        #                 # positive = x_mal[mal_positive_idx, ...]
        #                 for mal_negative_idx in mal_pos_minus_neg_top[local_idx, mal_positive_idx]:
        #                     curr_diff = mal_pos_minus_neg[local_idx, mal_positive_idx, mal_negative_idx]

        #                     if curr_diff <= 0:
        #                         if skip_hard_negatives:
        #                             continue
        #                     # negative = x_benign[mal_negative_idx, ...]
        #                     triplets_idxs.append([mal_anchor_idx, mal_positive_idx, mal_negative_idx])


        if len(triplets_idxs) == 0:
            print("No triplets found, returning empty dataset.")
            return []

        anchor_idxs, positive_idxs, negative_idxs = zip(*triplets_idxs)

        # anchors, positives, negatives = zip(*triplets)
        # anchors, positives, negatives = np.array(anchors), np.array(positives), np.array(negatives)

        tf.keras.backend.clear_session()

        if return_tfds:
            anchor_idxs_tf, positive_idxs_tf, negative_idxs_tf = [tf.convert_to_tensor(idx, dtype=tf.int32) for idx in [anchor_idxs, positive_idxs, negative_idxs]]
            triplet_idx_dataset = tf.data.Dataset.from_tensor_slices(
                (anchor_idxs_tf, positive_idxs_tf, negative_idxs_tf)
            )

            # 5. Map index tuples to actual feature triplets
            def gather_triplet(a_idx, p_idx, n_idx):
                anchor = tf.gather(x_benign, a_idx)
                positive = tf.gather(x_benign, p_idx)
                negative = tf.gather(x_mal, n_idx)
                return (anchor, positive, negative)

            dataset = triplet_idx_dataset.map(gather_triplet)
            
            # anchor_dataset = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(anchors))
            # positive_dataset = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(positives))
            # negative_dataset = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(negatives))

            # dataset = tf.data.Dataset.zip((anchor_dataset, positive_dataset, negative_dataset))
            dataset = dataset.batch(32, drop_remainder=False)
            dataset = dataset.prefetch(tf.data.AUTOTUNE)
            dataset = dataset.shuffle(buffer_size=1000, reshuffle_each_iteration=True)

            return dataset
        else:
            return
            # return [anchors, positives, negatives]

    def call(self, inputs, training=False):
        return self.siamese_network(inputs, training=training)

    def test(self, x,y, verbose=None, ref_idxs=None):
        triplets_test = make_triplets(x,y, size=None, is_shuffle=False)
        ap_distance, an_distance =  self.siamese_network(triplets_test)
        loss = ap_distance - an_distance
        print(tf.math.count_nonzero(loss<0)/len(ap_distance))
        return loss

    def test_actual(self, x_train, y_train, x_test, y_test, threshold=1, is_print=True):
        assert 0 <= threshold <= 1

        op = operator.lt

        benign_idxs_test = np.where(y_test==0)[0]
        mal_idxs_test = np.where(y_test==1)[0]

        batch_size = 16
        
        split_size = math.ceil(len(x_test) / batch_size)

        x_test_splits = np.array_split(x_test, split_size)
        x_test_preds = [self.embedding(curr) for curr in x_test_splits]
        x_test_embeddings = np.vstack(x_test_preds)
        
        x_test_embeddings_benign = tf.gather(x_test_embeddings, indices=benign_idxs_test)
        x_test_embeddings_mal = tf.gather(x_test_embeddings, indices=mal_idxs_test)

        benign_results = []
        mal_results = []

        def test_single(benign_sample_embedding, mal_sample_embedding):
            benign_sample_stacked = np.broadcast_to(benign_sample_embedding, (len(benign_idxs_test), *(benign_sample_embedding.shape)))
            mal_sample_stacked = np.broadcast_to(mal_sample_embedding, (len(mal_idxs_test), *(mal_sample_embedding.shape)))

            # dist_benign_benign = dist(benign_sample_stacked, x_test_embeddings_benign)
            # dist_benign_mal = dist(benign_sample_stacked, x_test_embeddings_mal)
            
            dist_benign_benign = calc_dist(benign_sample_stacked, x_test_embeddings_benign, self.dist)
            dist_benign_mal = calc_dist(mal_sample_stacked, x_test_embeddings_benign, self.dist)

            benign_result = op(dist_benign_benign, dist_benign_mal)

            # dist_mal_mal = dist(mal_sample_stacked, x_test_embeddings_mal)
            # dist_mal_benign = dist(mal_sample_stacked, x_test_embeddings_benign)
            
            dist_mal_mal = calc_dist(mal_sample_stacked, x_test_embeddings_mal, self.dist)
            dist_mal_benign = calc_dist(benign_sample_stacked, x_test_embeddings_mal, self.dist)

            mal_result =  op(dist_mal_mal, dist_mal_benign)

            benign_results.append(benign_result)
            mal_results.append(mal_result)

        benign_idxs_train = np.where(y_train==0)[0]
        mal_idxs_train = np.where(y_train==1)[0]

        x_train_embeddings = self.embedding.predict(x_train)

        for benign_idx, mal_idx in product(benign_idxs_train, mal_idxs_train):
            test_single(x_train_embeddings[benign_idx], x_train_embeddings[mal_idx])

        benign_results = np.array(benign_results)
        mal_results = np.array(mal_results)

        # print(benign_results.shape, mal_results.shape)
        # print(benign_results)
        # print(np.count_nonzero(benign_results, axis=0))
        # print(mal_results)

        benign_passed = (np.count_nonzero(benign_results, axis=0) / len(benign_results)) >= threshold
        benign_success = np.count_nonzero(benign_passed) / len(benign_passed)

        mal_passed = (np.count_nonzero(mal_results, axis=0) / len(mal_results)) >= threshold
        mal_success = np.count_nonzero(mal_passed) / len(mal_passed)

        if is_print:
            print(f'benign success: {benign_success}')
            print(f'mal success: {mal_success}')

        return benign_success, mal_success

    def calc_train_embeddings(self, x_train, y_train):
        if x_train is None or y_train is None:
            x_train = self.train_data['x']
            y_train = self.train_data['y']

        y_vals = np.unique(y_train)
        # benign_label = 0
        # mal_labels = y_vals[y_vals != benign_label]

        specific_idxs_train = {
            label: np.where(y_train == label)[0] for label in y_vals
        }

        x_train_embeddings = self.embedding.predict(x_train, batch_size=32, verbose=0)
        x_train_embeddings_specific = {
            label: tf.gather(x_train_embeddings, indices=specific_idxs_train[label]) for label in y_vals
        }

        if self.train_data is not None:
            self.train_data['x_train_embeddings'] = x_train_embeddings
            # self.train_data['x_train_embeddings_benign'] = x_train_embeddings_benign
            # self.train_data['x_train_embeddings_mal'] = x_train_embeddings_mal
            self.train_data['x_train_embeddings_specific'] = x_train_embeddings_specific

        return x_train_embeddings_specific

    def calc_centroids(self, x_train, y_train, apply_transforms:Literal['NA', 'L2', 'CL2']='NA'):
        if self.train_data is not None and 'x_train_embeddings_specific' in self.train_data:
            # x_train_embeddings_benign = self.train_data['x_train_embeddings_benign']
            # x_train_embeddings_mal = self.train_data['x_train_embeddings_mal']
            x_train_embeddings_specific = self.train_data['x_train_embeddings_specific']
        else:
            # x_train_embeddings_benign, x_train_embeddings_mal = self.calc_train_embeddings(x_train, y_train)
            x_train_embeddings_specific = self.calc_train_embeddings(x_train, y_train)

        # centroid_benign = copy.deepcopy(x_train_embeddings_benign)
        # centroid_mal = copy.deepcopy(x_train_embeddings_mal)
        # centroid_specific = x_train_embeddings_specific

        # print(f'centroid_benign: {centroid_benign.shape}')
        # print(f'centroid_mal: {centroid_mal.shape}')
        # print(f'benign norm: {tf.norm(centroid_benign, ord=2, axis=-1)}')

        centroid_specific = {label: tf.reduce_mean(embeddings, axis=0) for label, embeddings in x_train_embeddings_specific.items()}

        if apply_transforms != 'NA':
            if apply_transforms == 'CL2':
                x_train_embeddings = self.train_data['x_train_embeddings']
                mean_train = tf.reduce_mean(x_train_embeddings, axis=0)
                
                for label, centroid in centroid_specific.items():
                    # centroid = tf.reduce_mean(embeddings, axis=0)
                    centroid -= mean_train
                    # centroid /= LA.norm(centroid, 2)
                    centroid /= tf.norm(centroid, ord=2, axis=0)

                    centroid_specific[label] = centroid

                self.train_data['mean_train'] = mean_train
            # x_train_embeddings /= LA.norm(x_train_embeddings, 2, 1)[:, None]
            elif apply_transforms == 'L2':
                for label, centroid in centroid_specific.items():
                    # centroid = tf.reduce_mean(embeddings, axis=0)
                    centroid /= tf.norm(centroid, ord=2, axis=0)

                    centroid_specific[label] = centroid

        

        # centroid_benign = tf.reduce_mean(centroid_benign, axis=0).numpy()
        # centroid_mal = tf.reduce_mean(centroid_mal, axis=0).numpy()

        # print(f'centroid_benign: {centroid_benign.shape}')
        # print(f'centroid_mal: {centroid_mal.shape}')

        centroid_knn = KNeighborsClassifier(n_neighbors=1)
        centroid_knn.fit(np.array(list(centroid_specific.values())), list(centroid_specific.keys()))
        # centroid_knn.fit(np.array([centroid_benign, centroid_mal]), [0,1])

        if self.train_data is not None:
            # self.train_data['x_train_embeddings_benign'] = x_train_embeddings_benign
            # self.train_data['x_train_embeddings_mal'] = x_train_embeddings_mal
            # self.train_data['centroid_benign'] = centroid_benign
            # self.train_data['centroid_mal'] = centroid_mal
            self.train_data['centroid_specific'] = centroid_specific
            self.train_data[f'centroid_knn_{apply_transforms}'] = centroid_knn

        return centroid_specific, centroid_knn

    def inference_centroid(self, x_test, x_train=None, y_train=None, apply_transforms:Literal['NA', 'L2', 'CL2']='NA', coarse_label: bool=False, x_test_embeddings=None):
        if self.train_data is not None and 'centroid_knn' in self.train_data:
            # centroid_benign = self.train_data['centroid_benign']
            # centroid_mal = self.train_data['centroid_mal']
            centroid_knn = self.train_data[f'centroid_knn_{apply_transforms}']
        else:
            # centroid_benign, centroid_mal = self.calc_centroids(x_train, y_train, apply_transforms=apply_transforms)
            centroid_knn = self.calc_centroids(x_train, y_train, apply_transforms=apply_transforms)[1]

        if x_test_embeddings is None:
            batch_size = 16
            split_size = math.ceil(len(x_test) / batch_size)

            x_test_splits = np.array_split(x_test, split_size)
            x_test_preds = [self.embedding(curr) for curr in x_test_splits]
            x_test_embeddings = np.vstack(x_test_preds)
        else:
            x_test_embeddings = copy.deepcopy(x_test_embeddings)

        if apply_transforms != 'NA':
            if apply_transforms == 'CL2':
                if self.train_data is not None and 'mean_train' in self.train_data:
                    mean_train = self.train_data['mean_train']
                else:
                    mean_train = tf.reduce_mean(self.embedding.predict(x_train), axis=0)
                x_test_embeddings -= mean_train
                x_test_embeddings /= tf.norm(x_test_embeddings, ord=2, axis=0)
            elif apply_transforms == 'L2':
                x_test_embeddings /= tf.norm(x_test_embeddings, ord=2, axis=0)


            # print(f'x_test_embeddings: {x_test_embeddings.shape}')
            # x_test_embeddings /= tf.norm(x_test_embeddings, ord=2, axis=0)


        # if apply_transforms:
        #     x_train_embeddings = self.train_data['x_train_embeddings']
        #     mean_train = tf.reduce_mean(x_train_embeddings, axis=0)
        #     x_test_embeddings -= mean_train
        #     # x_test_embeddings /= LA.norm(x_test_embeddings, 2, 1)[:, None]
        #     x_test_embeddings /= tf.norm(x_test_embeddings, ord=2,)

            # centroid_benign -= mean_train
            # centroid_mal -= mean_train
            # centroid_benign /= LA.norm(centroid_benign, 2)
            # centroid_mal /= LA.norm(centroid_mal, 2)

        

        y_pred = centroid_knn.predict(x_test_embeddings)

        if coarse_label:
            y_pred = coarse_label_map(y_pred)

        return y_pred
            
    def test_centroid(self, x_test, y_test, x_train=None, y_train=None, is_print=True, apply_transforms:Literal['NA', 'L2', 'CL2']='NA', return_acc=True, coarse_label: bool=False,x_test_embeddings=None):

        y_pred = self.inference_centroid(x_test, x_train=x_train, y_train=y_train, apply_transforms=apply_transforms, coarse_label=coarse_label, x_test_embeddings=x_test_embeddings)
        if return_acc:
            if coarse_label:
                y_test = coarse_label_map(y_test)

            acc = accuracy_score(y_test, y_pred)
            if is_print:
                print(f'knn (centroid) accuracy: {acc}')
            return acc
        else:
            return y_pred

    def inference_nn(self, x_test, x_train=None, y_train=None, k=1, metric:Literal['cosine', 'euclidean', 'cityblock']='euclidean',coarse_label: bool=False,x_test_embeddings=None):
        if self.train_data is not None:
            x_train = self.train_data['x']
            y_train = self.train_data['y']
        else:
            assert x_train is not None and y_train is not None, "train data must be provided"

        if x_test_embeddings is None:
            batch_size = 16
            
            split_size = math.ceil(len(x_test) / batch_size)

            x_test_splits = np.array_split(x_test, split_size)
            x_test_preds = [self.embedding(curr) for curr in x_test_splits]
            x_test_embeddings = np.vstack(x_test_preds)

        if self.train_data is not None and 'x_train_embeddings' in self.train_data:
            x_train_embeddings = self.train_data['x_train_embeddings']
        else:
            x_train_embeddings = self.embedding.predict(x_train)

        knn = KNeighborsClassifier(n_neighbors=k, metric=metric)
        knn.fit(x_train_embeddings, y_train)

        y_pred = knn.predict(x_test_embeddings)
        if coarse_label:
            y_pred = coarse_label_map(y_pred)

        return y_pred
    
    def test_nn(self, x_test, y_test, x_train=None, y_train=None, k=1, metric:Literal['cosine', 'euclidean', 'cityblock']='euclidean', is_print=True, return_acc=True, coarse_label: bool=False,x_test_embeddings=None):
        y_pred = self.inference_nn(x_test, x_train=x_train, y_train=y_train, k=k, metric=metric, coarse_label=coarse_label,x_test_embeddings=x_test_embeddings)
        
        if return_acc:
            if coarse_label:
                y_test = coarse_label_map(y_test)

            acc = accuracy_score(y_test, y_pred)
            if is_print:
                print(f'accuracy: {acc}')
            return acc
        else:
            return y_pred
    
    def test_all(self, x_test, y_test, x_train=None, y_train=None, is_print=True, k=1, metric:Literal['cosine', 'euclidean', 'cityblock']='euclidean', return_acc=True, centroid_apply_transforms:Literal['NA', 'L2', 'CL2']='NA', coarse_label: bool=False, include_time: bool=False):
        time_embeddings_start = time.time()
        x_test_embeddings = self.get_embeddings(x_test)
        time_embeddings = time.time() - time_embeddings_start

        time_centroid_start = time.time()

        ret_centroid = self.test_centroid(x_test, y_test, x_train=x_train, y_train=y_train, is_print=is_print, apply_transforms=centroid_apply_transforms, return_acc=return_acc, coarse_label=coarse_label, x_test_embeddings=x_test_embeddings)

        time_centroid = time.time() - time_centroid_start

        time_nn_start = time.time()

        ret_nn = self.test_nn(x_test, y_test, x_train=x_train, y_train=y_train, k=k, metric=metric, is_print=is_print, return_acc=return_acc, coarse_label=coarse_label, x_test_embeddings=x_test_embeddings)

        time_nn = time.time() - time_nn_start

        if include_time:
            times = {
                'embeddings': time_embeddings,
                'centroid': time_centroid,
                'nn': time_nn,
            }
            return {'centroid': ret_centroid, 'nn': ret_nn, 'times': times}
        else:
            return {'centroid': ret_centroid, 'nn': ret_nn}


    def train_step(self, data):
        # GradientTape is a context manager that records every operation that
        # you do inside. We are using it here to compute the loss so we can get
        # the gradients and apply them using the optimizer specified in
        # `compile()`.
        with tf.GradientTape() as tape:
            loss = self._compute_loss(data, training=True)

        # Storing the gradients of the loss function with respect to the
        # weights/parameters.
        gradients = tape.gradient(loss, self.siamese_network.trainable_weights)

        # Applying the gradients on the model using the specified optimizer
        self.optimizer.apply_gradients(
            zip(gradients, self.siamese_network.trainable_weights)
        )

        # Let's update and return the training loss metric.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        loss = self._compute_loss(data, training=False)

        # Let's update and return the loss metric.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def _compute_loss(self, data, training=False):
        # The output of the network is a tuple containing the distances
        # between the anchor and the positive example, and the anchor and
        # the negative example.
        ap_distance, an_distance = self.siamese_network(data, training=training)

        # Computing the Triplet Loss by subtracting both distances and
        # making sure we don't get a negative value.
        loss = ap_distance - an_distance
        loss = tf.maximum(loss + self.margin, 0.0)
        return loss

    def get_embeddings(self, x):
        if x.ndim == 3:
            x = np.expand_dims(x, axis=-1)

        if x.dtype == np.uint8:
            x = x.astype(np.float32) / 255.0

        # Apply the same Keras Normalization layer that fit_and_keep_refs adapted on
        # the training data. Without this, test embeddings live in a different
        # input distribution than training and centroid/NN classification collapses
        # to chance even when the model trained correctly.
        if hasattr(self, 'normalization_layer') and self.normalization_layer is not None:
            x = self.normalization_layer(x)
        return self.embedding.predict(x)

    # def visualize_learned_embedding(self, )

    @property
    def metrics(self):
        # We need to list our metrics here so the `reset_states()` can be
        # called automatically.
        return [self.loss_tracker]

    def get_config(self):
        def transform_pydantic_dict_member_to_dict(d, key):
            if key in d and not isinstance(d[key], dict):
                d[key] = d[key].dict()
            return

        base_config = super().get_config()
        config = {
            "embedding": tf.keras.saving.serialize_keras_object(self.embedding),
            "optimizer": tf.keras.saving.serialize_keras_object(self.optimizer),
            # "pretrained": self.pretrained,
            "img_input_shape": self.img_input_shape,
            "dist": self.dist,

            "train_data": None,
        }

        if hasattr(self, 'train_data'):
            if 'train_metadata' in self.train_data:
                train_metadata_dict = self.train_data['train_metadata']

                transform_pydantic_dict_member_to_dict(train_metadata_dict, 'image_rep_config')
                transform_pydantic_dict_member_to_dict(train_metadata_dict, 'image_preprocess_config')
                
            config["train_data"] = self.train_data

        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):

        embedding_config = config.pop("embedding")
        optimizer_config = config.pop("optimizer")

        train_data = config.pop("train_data")
        if train_data is not None:
            train_metadata_dict = train_data['train_metadata']

            train_metadata_dict['image_rep_config'] = ImageRepConfig.model_validate(train_metadata_dict['image_rep_config'])
            train_metadata_dict['image_preprocess_config'] = ImagePreprocessConfig.model_validate(train_metadata_dict['image_preprocess_config'])

            train_data['x'] = tf.keras.saving.deserialize_keras_object(train_data['x'])
            train_data['y'] = tf.keras.saving.deserialize_keras_object(train_data['y'])

        embedding = tf.keras.saving.deserialize_keras_object(embedding_config)
        optimizer = tf.keras.saving.deserialize_keras_object(optimizer_config)
        return cls(model=embedding, optimizer=optimizer, train_data=train_data, **config)

    @staticmethod
    def _is_triplets(inputs):
        if not isinstance(inputs, list) or len(inputs) != 3:
            return False

        anchors, positives, negatives = inputs
        if not (isinstance(anchors, np.ndarray) and isinstance(positives, np.ndarray) and isinstance(negatives, np.ndarray)):
            return False
        
        if not (anchors.shape == positives.shape == negatives.shape):
            return False

        return True