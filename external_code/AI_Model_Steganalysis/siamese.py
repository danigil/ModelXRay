# Code from https://github.com/hlamba28/One-Shot-Learning-with-Siamese-Networks/blob/master/Siamese%20on%20Omniglot%20Dataset.ipynb

from typing import Literal
import numpy as np
from numpy import linalg as LA
from sklearn.neighbors import KNeighborsClassifier
seed = 122
# np.random.seed(seed)
from itertools import combinations, product
import math

import operator

import random
# random.seed(seed)

from random import sample
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

# def store_imgs_dataset(zoo_name="famous_le_10m", data_type="grads", img_type="rgb", shape_x=100, lsb=23, train_size=1, reshape="pad"):
#     X_train, y_train, X_test, y_test = ret_imgs_dataset_preprocessed(zoo_name, data_type, img_type, shape_x, lsb=lsb, train_size=train_size, reshape=reshape)

#     save_path = f"./data/datasets/zoo_name:{zoo_name},data_type:{data_type},img_type:{img_type},shape_x:{shape_x},lsb:{lsb},train_size:{train_size},reshape:{reshape}.npz"
#     # print(X_train.dtype, y_train.dtype, X_test.dtype, y_test.dtype)

#     np.savez_compressed(save_path, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

def initialize_weights(shape, name=None, dtype=None, dummy=False):
    """
        The paper, http://www.cs.utoronto.ca/~gkoch/files/msc-thesis.pdf
        suggests to initialize CNN layer weights with mean as 0.0 and standard deviation of 0.01
    """
    if dummy:
        return np.zeros(shape)
    return np.random.normal(loc = 0.0, scale = 1e-2, size = shape)

def initialize_bias(shape, name=None, dtype=None, dummy=False):
    """
        The paper, http://www.cs.utoronto.ca/~gkoch/files/msc-thesis.pdf
        suggests to initialize CNN layer bias with mean as 0.5 and standard deviation of 0.01
    """
    if dummy:
        return np.zeros(shape)
    return np.random.normal(loc = 0.5, scale = 1e-2, size = shape)

def get_siamese_model(input_shape, model=None, dist="l1", sigmoid=True, weights_init="random"):

    def dist_l1(vects):
        x,y = vects
        return K.sum( K.abs(x-y),axis=1,keepdims=True)

    def dist_l2(vects):
        x, y = vects
        sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
        return K.sqrt(K.maximum(sum_square, K.epsilon()))
        # return K.sqrt(K.maximum(sum_square, K.epsilon()))

    def dist_cosine(vects):
        x, y = vects
        # return CosineSimilarity()(x,y)
        x = K.l2_normalize(x, axis=-1)
        y = K.l2_normalize(y, axis=-1)
        return -K.mean(x * y, axis=-1, keepdims=True)

    def dist_euc_tripltet(vects):
        x, y = vects
        return tf.reduce_sum(tf.square(x - y), -1)

    # Define the tensors for the two input images
    left_input = Input(input_shape)
    right_input = Input(input_shape)
    
    if model is None:
        if weights_init == "random":
            initialize_weights = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01)
            initialize_bias = tf.keras.initializers.RandomNormal(mean=0.5, stddev=0.01)
        else:
            initialize_weights = tf.keras.initializers.Zeros()
            initialize_bias = tf.keras.initializers.Zeros()

    # Convolutional Neural Network
        model = Sequential()
        model.add(Conv2D(64, (10,10), activation='relu', input_shape=input_shape,
                    kernel_initializer=initialize_weights, kernel_regularizer=l2(2e-4)))
        model.add(MaxPooling2D())
        model.add(Conv2D(128, (7,7), activation='relu',
                        kernel_initializer=initialize_weights,
                        bias_initializer=initialize_bias, kernel_regularizer=l2(2e-4)))
        model.add(MaxPooling2D())
        model.add(Conv2D(128, (4,4), activation='relu', kernel_initializer=initialize_weights,
                        bias_initializer=initialize_bias, kernel_regularizer=l2(2e-4)))
        model.add(MaxPooling2D())
        model.add(Conv2D(256, (4,4), activation='relu', kernel_initializer=initialize_weights,
                        bias_initializer=initialize_bias, kernel_regularizer=l2(2e-4)))
        model.add(Flatten())
        model.add(Dense(4096, activation=None,
                    kernel_regularizer=l2(1e-3),
                    kernel_initializer=initialize_weights,bias_initializer=initialize_bias))
    
    encoded_l = model(left_input)
    encoded_r = model(right_input)

    if dist == "l1":
        dist_layer = Lambda(dist_l1)
    elif dist == "l2":
        dist_layer = Lambda(dist_l2)
    elif dist == "cosine":
        dist_layer = Lambda(dist_cosine)
    elif dist == "euc_triplet":
        dist_layer = Lambda(dist_euc_tripltet)
    
    distance = dist_layer([encoded_l, encoded_r])
    
    # Add a dense layer with a sigmoid unit to generate the similarity score
    if sigmoid:
        prediction = Dense(1,activation='sigmoid',bias_initializer=initialize_bias)(distance)
    else:
        prediction = distance
    
    # Connect the inputs with the outputs
    siamese_net = Model(inputs=[left_input,right_input],outputs=prediction)
    
    # return the model
    return siamese_net



# make pairs
def make_pairs_random(x, y):
    num_classes = max(y) + 1
    digit_indices = [np.where(y == i)[0] for i in range(num_classes)]

    pairs = []
    labels = []

    for idx1 in range(len(x)):
        # add a matching example
        x1 = x[idx1]
        label1 = y[idx1]
        idx2 = random.choice(digit_indices[label1])
        x2 = x[idx2]
        
        pairs += [[x1, x2]]
        labels += [1]
    
        label2 = 0 if label1==1 else 1
        # add a not matching example
        # label2 = random.randint(0, num_classes-1)
        # while label2 == label1:
        #     label2 = random.randint(0, num_classes-1)

        idx2 = random.choice(digit_indices[label2])
        x2 = x[idx2]
        
        pairs += [[x1, x2]]
        labels += [0]

    return np.array(pairs), np.array(labels)

def make_pairs_all(x,y, is_shuffle=False):
    benign_idxs = np.where(y==0)[0]
    mal_idxs = np.where(y==1)[0]

    pairs = []
    labels = []

    for x_ben_1_idx, x_ben_2_idx in combinations(benign_idxs, 2):
        x_ben_1 = x[x_ben_1_idx]
        x_ben_2 = x[x_ben_2_idx]
        pairs.append((x_ben_1, x_ben_2))
        labels.append(1)

    for x_mal_1_idx, x_mal_2_idx in combinations(mal_idxs, 2):
        x_mal_1 = x[x_mal_1_idx]
        x_mal_2 = x[x_mal_2_idx]
        pairs.append((x_mal_1, x_mal_2))
        labels.append(1)

    for x_ben_idx, x_mal_idx in product(benign_idxs, mal_idxs):
        x_ben = x[x_ben_idx]
        x_mal = x[x_mal_idx]
        pairs.append((x_ben, x_mal))
        labels.append(0)

    if is_shuffle:
        pairs, labels = shuffle(pairs, labels)

    return np.array(pairs), np.array(labels)

def make_pairs_custom(x,y):
    pass

def identity(x,y):
    return x,y

def make_triplets(x,y, size=None, is_shuffle=False):
    def preprocess_sample(sample):
        sample = np.expand_dims(sample, axis=(-1))
        return sample
    triplets = []

    benign_idxs = np.where(y == 0)[0]
    mal_idxs = np.where(y == 1)[0]

    if size is not None:
        benign_idxs = sample(list(benign_idxs), min(size, len(benign_idxs)))
        mal_idxs = sample(list(mal_idxs), min(size, len(mal_idxs)))

    for anchor_idx, positive_idx in combinations(benign_idxs, 2):
        anchor = preprocess_sample(x[anchor_idx])
        positive = preprocess_sample(x[positive_idx])

        for negative_idx in mal_idxs:
            negative = preprocess_sample(x[negative_idx])
            triplets.append([anchor, positive, negative])

    for anchor_idx, positive_idx in combinations(mal_idxs, 2):
        anchor = preprocess_sample(x[anchor_idx])
        positive = preprocess_sample(x[positive_idx])

        for negative_idx in benign_idxs:
            negative = preprocess_sample(x[negative_idx])
            triplets.append([anchor, positive, negative])

    anchors, positives, negatives = zip(*triplets)
    if is_shuffle:
        anchors, positives, negatives = shuffle(anchors, positives, negatives)
    return [np.array(anchors), np.array(positives), np.array(negatives)]

pairs_func_map = {
    "random": make_pairs_random,
    "custom": make_pairs_custom,
    "all": make_pairs_all,
    "none": identity,
}

f = lambda x: 1 if x>0.5 else 0
vf = np.vectorize(f)
class Siamese:
    def __init__(self, input_shape=(100,100,1), model=None, dist="l1", pairs_function="random", lr=0.00006, sigmoid=True) -> None:
        
        assert dist in ("l1", "l2", "cosine", "euc_triplet")
        assert pairs_function in ("random", "custom", "none", "all")

        self.sigmoid = sigmoid

        self.model = get_siamese_model(input_shape, model=model, dist=dist, sigmoid=sigmoid)
        optimizer = Adam(learning_rate = lr)
        self.model.compile(loss="binary_crossentropy",optimizer=optimizer)

        self.pairs_fun = pairs_func_map[pairs_function]

    def fit(self, x,y, batch_size=16, epochs=10, verbose=None):
        pairs_train, labels_train = self.pairs_fun(x, y)
        # print(f"pairs_train all shape: {pairs_train.shape} pairs_train left shape: {pairs_train[:,0].shape}")
        # return
        # print(f"pairs_train shape: {pairs_train[:,0].shape}")
        self.model.fit([pairs_train[:,0], pairs_train[:,1]], labels_train[:], batch_size=batch_size, epochs=epochs, validation_split=0.1,verbose=verbose)

    def test(self, x,y, verbose=1, ref_idxs=[0]):

        if self.sigmoid:
            op = operator.gt
        else:
            op = operator.lt

        benign_idxs = np.where(y==0)[0]
        mal_idxs = np.where(y==1)[0]

        def test_single(ref_idx=0):
            benign_sample = np.array([x[benign_idxs[ref_idx]]])
            mal_sample = np.array([x[mal_idxs[ref_idx]]])

            benign_left = np.broadcast_to(benign_sample, (len(benign_idxs)-1, *(benign_sample.shape[1:])))
            benign_right = np.vstack([x[benign_idxs[0:ref_idx]], x[benign_idxs[ref_idx+1:]]])

            mal_left = np.broadcast_to(mal_sample, (len(mal_idxs)-1, *(mal_sample.shape[1:])))
            mal_right = np.vstack([x[mal_idxs[0:ref_idx]], x[mal_idxs[ref_idx+1:]]])

            y_pred_benign_benign = self.model.predict([benign_left, benign_right] ,batch_size=16, verbose=verbose)
            y_pred_benign_mal = self.model.predict([mal_left, benign_right] ,batch_size=16,verbose=verbose)
            benign_success = np.count_nonzero(op(y_pred_benign_benign,y_pred_benign_mal))

            y_pred_mal_benign = self.model.predict([benign_left, mal_right] ,batch_size=16,verbose=verbose)
            y_pred_mal_mal = self.model.predict([mal_left, mal_right] ,batch_size=16,verbose=verbose)

            mal_success = np.count_nonzero(op(y_pred_mal_mal,y_pred_mal_benign))

            print(f'\t\t\tbenign success: {benign_success/(len(benign_idxs)-1)}')
            print(f'\t\t\tmal success: {mal_success/(len(mal_idxs)-1)}')

        assert ref_idxs is None or isinstance(ref_idxs, (int, list, str))

        if ref_idxs is None:
            return test_single(0)
        elif isinstance(ref_idxs, int):
            return test_single(ref_idxs)
        elif isinstance(ref_idxs, list):
            for ref_idx in ref_idxs:
                print(f"test with ref_idx: {ref_idx}")
                return test_single(ref_idx)
        elif isinstance(ref_idxs, str):
            if ref_idxs == "all":
                for ref_idx in range(len(benign_idxs)):
                    print(f"test with ref_idx: {ref_idx}")
                    return test_single(ref_idx)
            else:
                raise ValueError(f"ref_idxs must be all, random or None, not {ref_idxs}")

        return

    def test_actual(self, x_train, y_train, x_test, y_test, verbose=1):

        if self.sigmoid:
            op = operator.gt
        else:
            op = operator.lt

        benign_idxs_test = np.where(y_test==0)[0]
        mal_idxs_test = np.where(y_test==1)[0]

        benign_right = x_test[benign_idxs_test]
        mal_right = x_test[mal_idxs_test]

        benign_results = []
        mal_results = []

        def test_single(benign_sample, mal_sample):
            benign_left = np.broadcast_to(benign_sample, (len(benign_idxs_test), *(benign_sample.shape[1:])))
            
            mal_left = np.broadcast_to(mal_sample, (len(mal_idxs_test), *(mal_sample.shape[1:])))

            y_pred_benign_benign = self.model.predict([benign_left, benign_right] ,batch_size=16, verbose=verbose)
            y_pred_benign_mal = self.model.predict([mal_left, benign_right] ,batch_size=16,verbose=verbose)
            benign_result = op(y_pred_benign_benign,y_pred_benign_mal)

            y_pred_mal_benign = self.model.predict([benign_left, mal_right] ,batch_size=16,verbose=verbose)
            y_pred_mal_mal = self.model.predict([mal_left, mal_right] ,batch_size=16,verbose=verbose)

            mal_result =  op(y_pred_mal_mal,y_pred_mal_benign)

            benign_results.append(benign_result)
            mal_results.append(mal_result)

        benign_idxs_train = np.where(y_train==0)[0]
        mal_idxs_train = np.where(y_train==1)[0]

        for benign_idx, mal_idx in tqdm(product(benign_idxs_train, mal_idxs_train)):
            # print(f"test with benign_idx: {benign_idx}, mal_idx: {mal_idx}")
            test_single(x_train[benign_idx], x_train[mal_idx])

        benign_results = np.array(benign_results)
        mal_results = np.array(mal_results)

        return benign_results, mal_results


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
            # print(f"train loss: {train_loss}")
            if self.threshold_lower <= train_loss <= self.threshold_upper:
                self.model.stop_training = True
                self.last_loss = train_loss
    
    def on_train_end(self, logs=None):
        self.last_loss = logs["loss"]

def calc_dist(a,b, dist:Literal["l2", "cosine"]="l2"):
    def dist_cosine(a,b):
        # a = tf.math.l2_normalize(a, axis=-1)
        # b = tf.math.l2_normalize(b, axis=-1)
        # return tf.reduce_mean(a * b, axis=-1)

        return 1 + tf.losses.cosine_similarity(tf.nn.l2_normalize(a, 0), tf.nn.l2_normalize(b, 0), -1)

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
            # ap_distance = tf.reduce_sum(tf.square(anchor - positive), -1)
            # an_distance = tf.reduce_sum(tf.square(anchor - negative), -1)
            # return (ap_distance, an_distance)

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

@tf.keras.saving.register_keras_serializable(package="MyModels")
class Siamese2(Model):
    def __init__(self,dropout_rate=0.5, pretrained=False, model=None, optimizer=None, dist:Literal["l2", "cosine"]="l2", img_input_shape=(100,100,1), margin=0.5, weights_init = "random", lr=0.0001):
        super().__init__()
        
        if model is None:

            if weights_init == "random":
                initialize_weights = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01)
                initialize_bias = tf.keras.initializers.RandomNormal(mean=0.5, stddev=0.01)
            elif weights_init == "zeros":
                initialize_weights = tf.keras.initializers.Zeros()
                initialize_bias = tf.keras.initializers.Zeros()
            elif weights_init == 'ones':
                initialize_weights = tf.keras.initializers.Ones()
                initialize_bias = tf.keras.initializers.Ones()
            else:
                raise ValueError(f"weights_init must be random, zeros or ones, not {weights_init}")

            if pretrained:
                # base_cnn = tf.keras.applications.densenet.DenseNet121(
                #     weights='imagenet', input_shape=img_input_shape, include_top=False
                # )
                # base_cnn = tf.keras.applications.resnet.ResNet50(
                #     weights='imagenet', input_shape=img_input_shape, include_top=False
                # )
                
                base_cnn = tf.keras.applications.MobileNetV3Small(
                    weights='imagenet', input_shape=img_input_shape, include_top=False, include_preprocessing=False
                )

                # base_cnn = tf.keras.applications.resnet_v2.ResNet101V2(
                #     weights=None, input_shape=img_input_shape, include_top=False
                # )
                
                # w_old = extract_weights_keras(base_cnn)
                # base_cnn = tf.keras.applications.vgg19.VGG19(
                #     weights=None, input_shape=img_input_shape, include_top=False
                # )

                flatten = tf.keras.layers.Flatten()(base_cnn.output)
                # dense1 = tf.keras.layers.Dense(512, activation="relu")(flatten)
                # # dense1 = tf.keras.layers.BatchNormalization()(dense1)
                # dense2 = tf.keras.layers.Dense(256, activation="relu")(dense1)
                # # dense2 = tf.keras.layers.BatchNormalization()(dense2)
                # output = tf.keras.layers.Dense(256)(dense2)
                # dense = tf.keras.layers.Dense(4096, activation=None,
                #     kernel_regularizer=l2(1e-3),
                #     kernel_initializer=initialize_weights,bias_initializer=initialize_bias)(flatten)
                # output=dense
                output = flatten
                
                # trainable = True
                # for layer in base_cnn.layers:
                #     if hasattr(layer, 'kernel_initializer'):
                #         # print(f"layer: {layer.name} has kernel_initializer")
                #         layer.kernel_initializer = initialize_weights
                #     if hasattr(layer, 'bias_initializer'):
                #         # print(f"layer: {layer.name} has bias_initializer")
                #         layer.bias_initializer = initialize_bias
                #     if hasattr(layer, 'kernel_regularizer'):
                #         layer.kernel_regularizer = l2(2e-4)

                    # if layer.name == "conv5_block1_out":
                    #     trainable = True
                    # layer.trainable = trainable

                # reset_weights(base_cnn)

                model = Model(base_cnn.input, output, name="Embedding")

                
            else:
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
                model.add(Dense(4096, activation=None,
                    kernel_regularizer=l2(1e-3),
                    kernel_initializer=ret_initializer_bias_rand(),bias_initializer=ret_initializer_bias_rand()))
        
        embedding = model

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
        # self.siamese_network.compile(loss=None, optimizer=optimizer, weighted_metrics=[tf.keras.losses.categorical_crossentropy])
        self.compile(loss=None ,optimizer=optimizer, weighted_metrics=[tf.keras.losses.categorical_crossentropy])

        # self.model = Siamese(input_shape=img_input_shape, model=embedding, sigmoid=False, dist=dist)

        self.img_input_shape = img_input_shape
        self.embedding = embedding
        self.pretrained=pretrained
        
        # w_new = extract_weights_keras(self.embedding)
        
        # print(f"weights are equal: {np.array_equal(w_old, w_new)}")
        
        # self.model_inference = get_siamese_model(input_shape, model=embedding, dist="l1")

    def call(self, inputs):
        return self.siamese_network(inputs)

    def test(self, x,y, verbose=None, ref_idxs=None):
        # return self.model.test(x,y, verbose=verbose, ref_idxs=ref_idxs)

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

        x_train_embeddings = self.embedding(x_train)

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

    def test_centroid(self, x_train, y_train, x_test, y_test, is_print=True, apply_transforms=False, vanilla=False, knn=False, return_acc=True):
        benign_idxs_train = np.where(y_train==0)[0]
        mal_idxs_train = np.where(y_train==1)[0]

        x_train_embeddings = self.embedding(x_train)
        x_train_embeddings_benign = tf.gather(x_train_embeddings, indices=benign_idxs_train)
        x_train_embeddings_mal = tf.gather(x_train_embeddings, indices=mal_idxs_train)

        # print(x_train_embeddings.shape)

        

        centroid_benign = tf.reduce_mean(x_train_embeddings_benign, axis=0)
        centroid_mal = tf.reduce_mean(x_train_embeddings_mal, axis=0)

        benign_idxs_test = np.where(y_test==0)[0]
        mal_idxs_test = np.where(y_test==1)[0]

        # print(benign_idxs_test, mal_idxs_test)

        batch_size = 16
        split_size = math.ceil(len(x_test) / batch_size)

        # print(x_test.shape)

        x_test_splits = np.array_split(x_test, split_size)
        x_test_preds = [self.embedding(curr) for curr in x_test_splits]
        x_test_embeddings = np.vstack(x_test_preds)



        if apply_transforms:
            # print(x_train_embeddings.shape)
            mean_train = tf.reduce_mean(x_train_embeddings, axis=0)
            # print(mean_train.shape)
            # print(x_test_embeddings.shape)
            x_test_embeddings = x_test_embeddings - mean_train
            # print(x_test_embeddings.shape)
            x_test_embeddings /= LA.norm(x_test_embeddings, 2, 1)[:, None]
            # print(x_test_embeddings.shape)
            # x_test_embeddings = tf.math.l2_normalize(x_test_embeddings, axis=0)
            # print(x_test_embeddings.shape)

        # print(x_test_embeddings.shape)
        if vanilla:
            x_test_embeddings_benign = tf.gather(x_test_embeddings, indices=benign_idxs_test)
            x_test_embeddings_mal = tf.gather(x_test_embeddings, indices=mal_idxs_test)

            # print(x_test_embeddings_benign.shape, x_test_embeddings_mal.shape)

            dist_benign_benign = calc_dist(x_test_embeddings_benign, centroid_benign, self.dist)
            dist_benign_mal = calc_dist(x_test_embeddings_benign, centroid_mal, self.dist)

            benign_results = operator.lt(dist_benign_benign, dist_benign_mal)
            
            benign_success = np.count_nonzero(benign_results) / len(benign_results)

            dist_mal_mal = calc_dist(x_test_embeddings_mal, centroid_mal, self.dist)
            dist_mal_benign = calc_dist(x_test_embeddings_mal, centroid_benign, self.dist)

            mal_results = operator.lt(dist_mal_mal, dist_mal_benign)
            mal_success = np.count_nonzero(mal_results) / len(mal_results)

            if is_print:
                print(f'benign success: {benign_success}')
                print(f'mal success: {mal_success}')

        if knn:
            # knn = KNeighborsClassifier(n_neighbors=1)
            # knn.fit(x_train_embeddings, y_train)

            # y_pred = knn.predict(x_test_embeddings)
            # acc = accuracy_score(y_test, y_pred)
            # if is_print:
            #     print(f'knn accuracy: {acc}')

            knn = KNeighborsClassifier(n_neighbors=1)
            knn.fit([centroid_benign, centroid_mal], [0,1])

            y_pred = knn.predict(x_test_embeddings)
            if return_acc:
                acc = accuracy_score(y_test, y_pred)
                if is_print:
                    print(f'knn (centroid) accuracy: {acc}')
                return acc
            else:
                return y_pred
            # return acc

        # print(centroid_benign.shape)

    def test_nn(self, x_train, y_train, x_test, y_test, k=1, metric:Literal['cosine', 'euclidean', 'cityblock']='euclidean', is_print=True, return_acc=True):
        # assert 0 <= threshold <= 1

        # op = operator.lt

        # benign_idxs_test = np.where(y_test==0)[0]
        # mal_idxs_test = np.where(y_test==1)[0]

        batch_size = 16
        
        split_size = math.ceil(len(x_test) / batch_size)

        x_test_splits = np.array_split(x_test, split_size)
        x_test_preds = [self.embedding(curr) for curr in x_test_splits]
        x_test_embeddings = np.vstack(x_test_preds)
        
        # x_test_embeddings_benign = tf.gather(x_test_embeddings, indices=benign_idxs_test)
        # x_test_embeddings_mal = tf.gather(x_test_embeddings, indices=mal_idxs_test)

        x_train_embeddings = self.embedding(x_train)

        knn = KNeighborsClassifier(n_neighbors=k, metric=metric)
        knn.fit(x_train_embeddings, y_train)

        y_pred = knn.predict(x_test_embeddings)
        if return_acc:
            acc = accuracy_score(y_test, y_pred)
            if is_print:
                print(f'accuracy: {acc}')
            return acc
        else:
            return y_pred
    
    def test_all(self, x_train, y_train, x_test, y_test, is_print=True, k=1, metric:Literal['cosine', 'euclidean', 'cityblock']='euclidean', return_acc=True):
        ret_centroid = self.test_centroid(x_train, y_train, x_test, y_test, is_print=is_print, apply_transforms=False, vanilla=False, knn=True, return_acc=return_acc)
        ret_nn = self.test_nn(x_train, y_train, x_test, y_test, k=k, metric=metric, is_print=is_print, return_acc=return_acc)

        return {'centroid': ret_centroid, 'nn': ret_nn}


    def train_step(self, data):
        # GradientTape is a context manager that records every operation that
        # you do inside. We are using it here to compute the loss so we can get
        # the gradients and apply them using the optimizer specified in
        # `compile()`.
        with tf.GradientTape() as tape:
            loss = self._compute_loss(data)

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
        loss = self._compute_loss(data)

        # Let's update and return the loss metric.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def _compute_loss(self, data):
        # The output of the network is a tuple containing the distances
        # between the anchor and the positive example, and the anchor and
        # the negative example.
        ap_distance, an_distance = self.siamese_network(data)

        # Computing the Triplet Loss by subtracting both distances and
        # making sure we don't get a negative value.
        loss = ap_distance - an_distance
        loss = tf.maximum(loss + self.margin, 0.0)
        return loss

    @property
    def metrics(self):
        # We need to list our metrics here so the `reset_states()` can be
        # called automatically.
        return [self.loss_tracker]

    def get_config(self):
        base_config = super().get_config()
        config = {
            # "siamese_network": tf.keras.saving.serialize_keras_object(self.siamese_network),
            # "loss_tracker": tf.keras.saving.serialize_keras_object(self.loss_tracker),
            "embedding": tf.keras.saving.serialize_keras_object(self.embedding),
            "optimizer": tf.keras.saving.serialize_keras_object(self.optimizer),
            "pretrained": self.pretrained,
            "img_input_shape": self.img_input_shape,
            "dist": self.dist
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        # print(cls)
        embedding_config = config.pop("embedding")
        optimizer_config = config.pop("optimizer")
        # from pprint import pprint
        # pprint(config)
        embedding = tf.keras.saving.deserialize_keras_object(embedding_config)
        optimizer = tf.keras.saving.deserialize_keras_object(optimizer_config)
        return cls(model=embedding, optimizer=optimizer, **config)