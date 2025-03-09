# -*- coding: utf-8 -*-

#libraries

import numpy as np
from scipy import misc, ndimage, signal
import time
import time as tm
import random 
import ntpath
import os
import pandas as pd
#import cv2
import sys
import glob

#libreria para Visualizar datos
import matplotlib.pyplot as plt


#libreria para dise�ar los Modelos de deep learning
from tensorflow.keras.layers import Concatenate, Lambda
from tensorflow.keras import backend
from tensorflow.keras.layers import Lambda, Layer
from tensorflow.keras.models import Sequential
from tensorflow.keras import optimizers 
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, LSTM, SpatialDropout2D, Concatenate, Lambda
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D, UpSampling2D, BatchNormalization, ReLU
import tensorflow as tf
# from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.base import BaseEstimator
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import load_model

# Libreria para obtener metricas
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import hamming_loss
from sklearn.metrics import log_loss
from sklearn.metrics import zero_one_loss
from sklearn.metrics import matthews_corrcoef

# libreria para realizar pre procesamientos
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

## libreria para binarizar datos
from sklearn.preprocessing import label_binarize

# libreria para partir los datos en entrenamiento y test
from sklearn.model_selection import train_test_split

# librerias para K-folds
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold

# graficar modelo creado
from tensorflow.keras.utils import plot_model

# time
import datetime

#segmentar
from scipy import misc
from scipy import ndimage
import copy

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
# import tensorflow_addons as tfa

from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

import math

from scipy import misc, ndimage, signal
from sklearn.model_selection  import train_test_split
import numpy
import numpy as np
import random
import ntpath
import os
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from keras import optimizers 
from keras import regularizers
import tensorflow as tf
# import cv2
from keras import backend as K
from time import time
import time as tm
import datetime
from operator import itemgetter
import glob
# from skimage.util.shape import view_as_blocks
# from keras.utils import np_utils
#from keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, LSTM, SpatialDropout2D, Concatenate, Lambda
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D, UpSampling2D, BatchNormalization, ReLU

from scipy import misc, ndimage, signal
from sklearn.model_selection  import train_test_split
import numpy
import numpy as np
import random
import ntpath
import os
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from keras import optimizers 
from keras import regularizers
import tensorflow as tf
# import cv2
from time import time
import time as tm
import datetime
from operator import itemgetter
import glob
# from skimage.util.shape import view_as_blocks
# from keras.utils import np_utils
# from tensorflow.keras.utils import to_categorical

########################################

## Load databases

# X_train = np.load("/home/rtabares/experimentos_jp/dataset_transformer_0.2/WOW/X_train.npy")
# y_train = np.load('/home/rtabares/experimentos_jp/dataset_transformer_0.2/WOW/y_train.npy')

# X_test = np.load('/home/rtabares/experimentos_jp/dataset_transformer_0.2/WOW/X_test.npy')
# y_test = np.load('/home/rtabares/experimentos_jp/dataset_transformer_0.2/WOW/y_test.npy')

# X_valid = np.load('/home/rtabares/experimentos_jp/dataset_transformer_0.2/WOW/X_valid.npy')
# y_valid = np.load('/home/rtabares/experimentos_jp/dataset_transformer_0.2/WOW/y_valid.npy')


"""
X_ = np.concatenate([X_train, X_valid, X_test],axis=0)
Xt_ = np.concatenate([y_train, y_valid, y_test],axis=0)

X_train = np.concatenate([X_[0:24000]],axis=0)
X_valid = np.concatenate([X_[24000:32000]],axis=0)
X_test  = np.concatenate([X_[32000:40000]],axis=0)

y_train = np.concatenate([Xt_[0:24000]],axis=0)
y_valid = np.concatenate([Xt_[24000:32000]],axis=0)
y_test  = np.concatenate([Xt_[32000:40000]],axis=0)
"""


# print("datos de entrenamiento: ", X_train.shape)
# print("etiquetas de entrenamiento: ", y_train.shape)
# print("datos de validacion: ", X_valid.shape)
# print("etiquetas de validacion: ", y_valid.shape)
# print("datos de test: ", X_test.shape)
# print("etiquetas de test: ", y_test.shape)


################################################## 30 SRM FILTERS
srm_weights_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'SRM_Kernels1.npy')
srm_weights = np.load(srm_weights_path) 
biasSRM=np.ones(30)
print (srm_weights.shape)
################################################## TLU ACTIVATION FUNCTION
T3 = 3
def Tanh3(x):
    tanh3 = K.tanh(x)*T3
    return tanh3
##################################################

########################################

################################## Functions ################################################################

# Squeeze-and Excitation

def squeeze_excitation_layer(input_layer, out_dim, ratio, conv):
  squeeze = tf.keras.layers.GlobalAveragePooling2D()(input_layer)
  excitation = tf.keras.layers.Dense(units=out_dim / ratio, activation='relu')(squeeze)
  excitation = tf.keras.layers.Dense(out_dim,activation='sigmoid')(excitation)
  excitation = tf.reshape(excitation, [-1,1,1,out_dim])
  scale = tf.keras.layers.multiply([input_layer, excitation])
  if conv:
    shortcut = tf.keras.layers.Conv2D(out_dim,kernel_size=1,strides=1,
                                      padding='same',kernel_initializer='he_normal')(input_layer)
    shortcut = tf.keras.layers.BatchNormalization()(shortcut)
  else:
    shortcut = input_layer
  out = tf.keras.layers.add([shortcut, scale])
  return out



def sreLu (input):
  return ReLU(negative_slope=0.1, threshold=0)(input)

def sConv(input,parameters,size,nstrides):
  return Conv2D(parameters, (size,size), strides=(nstrides,nstrides),padding="same", kernel_initializer='glorot_normal', kernel_regularizer=tf.keras.regularizers.l2(0.0001),bias_regularizer=tf.keras.regularizers.l2(0.0001))(input)

def sBN (input):
  return tf.keras.layers.BatchNormalization(momentum=0.2, epsilon=0.001, center=True, scale=True, trainable=True, fused=None, renorm=False, renorm_clipping=None, renorm_momentum=0.4, adjustment=None)(input)

def sGlobal_Avg_Pooling (input):
  return tf.keras.layers.GlobalAveragePooling2D()(input)

def sDense (input, n_units, activate_c):
  return tf.keras.layers.Dense(n_units,activation=activate_c)(input)

def smultiply (input_1, input_2):
  return tf.keras.layers.multiply([input_1, input_2])

def sadd (input_1, input_2):
  return tf.keras.layers.add([input_1, input_2])
  
  
  
def Block_1 (input, parameter):
  output = sConv(input, parameter, 3, 1)
  output = sBN(output)
  output = sreLu(output)
  return output
  


def SE_Block(input, out_dim, ratio):
  output = sGlobal_Avg_Pooling(input)
  output = sDense(output, out_dim/ratio, 'relu')
  output = sDense(output, out_dim, 'sigmoid')
  return output
  
  
  
def Block_2 (input, parameter):
  output = Block_1(input, parameter)
  output = sConv(output, parameter, 3, 1)
  output = sBN(output)
  multiplier = SE_Block(output,  parameter, parameter)
  # output = smultiply(output, output)
  output = smultiply(multiplier, output)
  output = sadd(output, input)
  return output
  
  
from tensorflow.python.ops.gen_array_ops import shape

def Block_3 (input, parameter):
  addition = sConv(input, parameter, 1, 2)
  addition = sBN(addition)
  output = sConv(input, parameter, 3, 2)
  output = sBN(output)
  output = sreLu(output)
  output = sConv(output, parameter, 3, 1)
  output = sBN(output)
  multiplier = SE_Block(output,  parameter, parameter)
  output = smultiply(multiplier, output)
  output = sadd(output, addition)
  return output  
  
def Block_4 (input, parameter):
  output = Block_1(input, parameter)
  output = sConv(input, parameter, 3, 1)
  output = sBN(output)
  
  return output  
  
  
# To use the transformation, a parameter called parameter: was added to the transformer_2 function to set the properties of the SE_BLock that goes inside it.

  
# ViT ARCHITECTURE
#Hyperparameters 1 tRANSFORMER
# ViT ARCHITECTURE
LAYER_NORM_EPS_1 = 1e-6
PROJECTION_DIM_1 = 16
NUM_HEADS_1 = 4
NUM_LAYERS_1 = 4
MLP_UNITS_1 = [
    PROJECTION_DIM_1 * 2,
    PROJECTION_DIM_1,
]
# OPTIMIZER
LEARNING_RATE_2 = 1e-3
WEIGHT_DECAY_2 = 1e-4

IMAGE_SIZE_2 =  16# We will resize input images to this size.
PATCH_SIZE_2 = 4  # Size of the patches to be extracted from the input images.
NUM_PATCHES_2 = (IMAGE_SIZE_2 // PATCH_SIZE_2) ** 2
print(NUM_PATCHES_2)
# ViT ARCHITECTURE
LAYER_NORM_EPS_2 = 1e-6
PROJECTION_DIM_2 = 128
NUM_HEADS_2 = 4
NUM_LAYERS_2 = 4
MLP_UNITS_2 = [
    PROJECTION_DIM_2 * 2,
    PROJECTION_DIM_2
]


def position_embedding(projected_patches, num_patches=NUM_PATCHES_2, projection_dim=PROJECTION_DIM_2):
    # Build the positions.

    positions = tf.range(start=0, limit=num_patches, delta=1)

    # Encode the positions with an Embedding layer.
    encoded_positions = layers.Embedding(
        input_dim=num_patches, output_dim=projection_dim
    )(positions)

    # Add encoded positions to the projected patches.
    return projected_patches + encoded_positions

def mlp(x, dropout_rate, hidden_units):
    # Iterate over the hidden units and
    # add Dense => Dropout.
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

def transformer_1(encoded_patches):
    # Layer normalization 1.
    x1 = layers.LayerNormalization(epsilon=LAYER_NORM_EPS_1)(encoded_patches)
    
    # Multi Head Self Attention layer 1.
    attention_output = layers.MultiHeadAttention(
        num_heads=NUM_HEADS_1, key_dim=PROJECTION_DIM_1, dropout=0.1
    )(x1, x1)

    # Skip connection 1.
    x2 = layers.Add()([attention_output, encoded_patches])

    # Layer normalization 2.
    x3 = layers.LayerNormalization(epsilon=LAYER_NORM_EPS_1)(x2)

    # MLP layer 1.
    x4 = mlp(x3, hidden_units=MLP_UNITS_1, dropout_rate=0.1)

    # Skip connection 2.
    encoded_patches = layers.Add()([x4, x2])
    return encoded_patches

def Transform_sh_1(inputs):
    projected_patches = layers.Conv2D(
        filters=512,
        kernel_size=(16,16),
        strides=(1,1),
        padding="same",
    )(inputs)
    _, h, w, c = projected_patches.shape
    print(c)
    projected_patches = layers.Reshape((h * w, c))(
        projected_patches
    )  # (B, number_patches, projection_dim)

    # Iterate over the number of layers and stack up blocks of
    # Transformer.
    for i in range(NUM_LAYERS_1):
        # Add a Transformer block.
        encoded_patches = transformer_1(projected_patches)
        # Add TokenLearner layer in the middle of the
        # architecture. The paper suggests that anywhere
        # between 1/2 or 3/4 will work well.
        _, hh, c = encoded_patches.shape
        h = int(math.sqrt(hh))
        encoded_patches = layers.Reshape((h, h, c))(encoded_patches)
    print(encoded_patches.shape)
        #print(encoded_patches.shape)
    return encoded_patches

def transformer_2(encoded_patches):

    x1 = layers.LayerNormalization(epsilon=LAYER_NORM_EPS_2)(encoded_patches)

    # Multi Head Self Attention layer 1.
    attention_output = layers.MultiHeadAttention(
        num_heads=NUM_HEADS_2, key_dim=PROJECTION_DIM_2, dropout=0.1
    )(x1, x1)

    # Skip connection 1.
    x2 = layers.Add()([attention_output, encoded_patches])

    # Layer normalization 2.
    x3 = layers.LayerNormalization(epsilon=LAYER_NORM_EPS_2)(x2)

    # MLP layer 1.
    x4 = mlp(x3, hidden_units=MLP_UNITS_2, dropout_rate=0.1)

    # Skip connection 2.
    encoded_patches = layers.Add()([x4, x2])
    return encoded_patches

def Transform_sh_2(inputs):
    inputs1 = squeeze_excitation_layer(inputs, out_dim=512, ratio=32.0, conv=False)
    print(inputs1.shape)
    projected_patches = layers.Conv2D(
          filters=PROJECTION_DIM_2,
          kernel_size=(PATCH_SIZE_2, PATCH_SIZE_2),
          strides=(PATCH_SIZE_2, PATCH_SIZE_2),
          padding="VALID",
      )(inputs1)
    _, h, w, c = projected_patches.shape
    projected_patches = layers.Reshape((h * w, c))(
          projected_patches
      )  # (B, number_patches, projection_dim)
      # Add positional embeddings to the projected patches.
    encoded_patches = position_embedding(
          projected_patches
      )  # (B, number_patches, projection_dim)
    
    encoded_patches = layers.Dropout(0.1)(encoded_patches)

      # Iterate over the number of layers and stack up blocks of
      # Transformer.
    for i in range(NUM_LAYERS_2):
          # Add a Transformer block.
        encoded_patches = transformer_2(encoded_patches)

    return encoded_patches
    
    
      
########################################

################################## Model  ################################################################
 
 
def cvtstego(include_top: bool = False, compile: bool = True):
  tf.keras.backend.clear_session()
  inputs = tf.keras.Input(shape=(256,256,1), name="input_1")
  #Layer 1
  layers_ty = tf.keras.layers.Conv2D(30, (5,5), weights=[srm_weights,biasSRM], strides=(1,1), padding='same', trainable=False, activation=Tanh3, use_bias=True)(inputs)
  layers_tn = tf.keras.layers.Conv2D(30, (5,5), weights=[srm_weights,biasSRM], strides=(1,1), padding='same', trainable=True, activation=Tanh3, use_bias=True)(inputs)

  layers = tf.keras.layers.add([layers_ty, layers_tn])
  layers1 = tf.keras.layers.BatchNormalization(momentum=0.2, epsilon=0.001, center=True, scale=False, trainable=True, fused=None, renorm=False, renorm_clipping=None, renorm_momentum=0.4, adjustment=None)(layers)
  #Layer 2
  
  # L1
  layers = Block_1(layers1,64)

  # L2
  layers = Block_1(layers,64)

  # L3 - L7
  for i in range(5):
    layers = Block_2(layers,64)

  # L8 - L11
  for i in [64, 64, 128, 256]:
    layers = Block_3(layers,i)

  # L12
  layers = Block_4(layers,512)
  #CVT=Transform_sh_1(layers)
  #CVT_2=Transform_sh_1(CVT)
  CVT1=Transform_sh_2(layers)

  representation = tf.keras.layers.LayerNormalization(epsilon=LAYER_NORM_EPS_2)(CVT1)
  representation = tf.keras.layers.GlobalAvgPool1D()(representation)
  
  if include_top:
    #---------------------------------------------------Fin de Transformer 2------------------------------------------------------------------------#
    # Classify outputs.
        #FC
    layers = Dense(128,kernel_initializer='glorot_normal',kernel_regularizer=tf.keras.regularizers.l2(0.0001),bias_regularizer=tf.keras.regularizers.l2(0.0001))(representation)
    layers = ReLU(negative_slope=0.1, threshold=0)(layers)
    layers = Dense(64,kernel_initializer='glorot_normal',kernel_regularizer=tf.keras.regularizers.l2(0.0001),bias_regularizer=tf.keras.regularizers.l2(0.0001))(layers)
    layers = ReLU(negative_slope=0.1, threshold=0)(layers)
    layers = Dense(32,kernel_initializer='glorot_normal',kernel_regularizer=tf.keras.regularizers.l2(0.0001),bias_regularizer=tf.keras.regularizers.l2(0.0001))(layers)
    layers = ReLU(negative_slope=0.1, threshold=0)(layers)

    #Softmax
    outputs = Dense(2, activation="softmax", name="output_1",kernel_regularizer=tf.keras.regularizers.l2(0.0001),bias_regularizer=tf.keras.regularizers.l2(0.0001))(layers)
  else:
    outputs = representation

  
  model =tf.keras.Model(inputs = inputs, outputs=outputs)
  #Compile
  optimizer = tf.keras.optimizers.SGD(learning_rate=0.005, momentum=0.95)

  if compile:
      model.compile(optimizer= optimizer,
                    loss='binary_crossentropy',
                    metrics=['accuracy'])
      
      print ("Transformer_create")

  return model


# model2 = new_arch()  
  
  
  
# path_log_base = '/home/rtabares/Transformer_modelos_finales/logs'
# path_img_base = '/home/rtabares/Transformer_modelos_finales/images'

# if not os.path.exists(path_log_base):
#     os.makedirs(path_log_base)
# if not os.path.exists(path_img_base):
#     os.makedirs(path_img_base)  
  
  
def train(model, X_train, y_train, X_valid, y_valid, X_test, y_test, batch_size, epochs, model_name=""):
    start_time = tm.time()
    log_dir=path_log_base+"/"+model_name+"_"+str(datetime.datetime.now().isoformat()[:19].replace("T", "_").replace(":","-"))
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir, histogram_freq=1)
    filepath = log_dir+"/saved-model-{epoch:03d}-{val_accuracy:.4f}.hdf5"
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_accuracy', save_best_only=False, mode='max')
    model.reset_states()
    
    global lossTEST
    global accuracyTEST
    global lossTRAIN
    global accuracyTRAIN
    global lossVALID
    global accuracyVALID
    lossTEST,accuracyTEST   = model.evaluate(X_test, y_test,verbose=None)
    lossTRAIN,accuracyTRAIN = model.evaluate(X_train, y_train,verbose=None)
    lossVALID,accuracyVALID = model.evaluate(X_valid, y_valid,verbose=None)

    global history
    global model_Name
    global log_Dir
    model_Name = model_name
    log_Dir = log_dir
    print("Starting the training...")
    history=model.fit(X_train, y_train, epochs=epochs, 
                      callbacks=[tensorboard,checkpoint], 
                      batch_size=batch_size,validation_data=(X_valid, y_valid),verbose=2)
    
    metrics = model.evaluate(X_test, y_test, verbose=0)
     
    TIME = tm.time() - start_time
    print("Time "+model_name+" = %s [seconds]" % TIME)
    
    print("\n")
    print(log_dir)
    Final_Results_Test(log_dir) 

    return {k:v for k,v in zip (model.metrics_names, metrics)}
  
  
def Final_Results_Test(PATH_trained_models):
    global AccTest
    global LossTest
    AccTest = []
    LossTest= [] 
    B_accuracy = 0 #B --> Best
    for filename in sorted(os.listdir(PATH_trained_models)):
        if filename != ('train') and filename != ('validation'):
            print(filename)
            model = tf.keras.models.load_model(PATH_trained_models+'/'+filename, custom_objects={'Tanh3':Tanh3})
            loss,accuracy = model.evaluate(X_test, y_test,verbose=0)
            print(f'Loss={loss:.4f} y Accuracy={accuracy:0.4f}'+'\n')
            BandAccTest  = accuracy
            BandLossTest = loss
            AccTest.append(BandAccTest)    
            LossTest.append(BandLossTest)  
            
            if accuracy > B_accuracy:
                B_accuracy = accuracy
                B_loss = loss
                B_name = filename
    
    print("\n\nBest")
    print(B_name)
    print(f'Loss={B_loss:.4f} y Accuracy={B_accuracy:0.4f}'+'\n')
  
  
  
## "_______________________________________________Train__________________________________________________________________________")


#model2.load_weights("/home/rtabares/Transformer_modelos_finales/model_Transformer_WOW_04bpp_Prueba1/logs/model_Transformer_WOW_04bpp_validation_por_test_2023-02-03_12-45-46/saved-model-240-0.8823.hdf5")

#model2.summary()
 
# train(model2, X_train, y_train, X_valid, y_valid, X_test, y_test, batch_size=32, epochs=300, model_name="model_Transformer_WOW_02bpp_validation_por_test")   
  


  
  
  
  
  
  
  
  
  
  
  
  

