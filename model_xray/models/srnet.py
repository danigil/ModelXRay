import tensorflow as tf
from functools import partial
# from tensorflow.keras import layers
# from tensorflow.contrib.framework import arg_scope

class ConvBn(tf.keras.Model):
    def __init__(self, num_outputs, conv_kernel_size=3, conv_strides=1):
        super().__init__()

        self.conv = tf.keras.layers.Conv2D(
            num_outputs,
            kernel_size=conv_kernel_size,
            strides=conv_strides,
            padding='same',
            # padding='same',
            use_bias=False
            )

        self.bn = tf.keras.layers.BatchNormalization()
        

    def call(self, inputs, training=False):
        return self.bn(self.conv(inputs), training=training)

class Type1(tf.keras.Model):
    def __init__(self, num_outputs):
        super().__init__()

        self.conv_bn = ConvBn(num_outputs)
        self.relu = tf.keras.layers.ReLU()
        

    def call(self, inputs, training=False):
        return self.relu(self.conv_bn(inputs, training=training))

class Type2(tf.keras.Model):
    def __init__(self, num_outputs):
        super().__init__()

        self.type_1 = Type1(num_outputs)
        self.conv_bn = ConvBn(num_outputs)
        

    def call(self, inputs, training=False):
        x = self.type_1(inputs, training=training)
        x = self.conv_bn(x, training=training)

        x += inputs
        return x

class Type3(tf.keras.Model):
    def __init__(self, num_outputs):
        super().__init__()

        self.type_1_a = Type1(num_outputs)
        self.conv_bn_a = ConvBn(num_outputs)

        self.avg_pool_a = tf.keras.layers.AveragePooling2D(
            pool_size=(3,3),
            strides=2,
            padding='same',
        )

        self.conv_bn_b = ConvBn(num_outputs, conv_kernel_size=1, conv_strides=2)
        

    def call(self, inputs, training=False):
        x_a = self.type_1_a(inputs, training=training)
        x_a = self.conv_bn_a(x_a, training=training)
        x_a = self.avg_pool_a(x_a)

        x_b = self.conv_bn_b(inputs, training=training)

        x = x_a + x_b
        return x

class Type4(tf.keras.Model):
    def __init__(self, num_outputs):
        super().__init__()

        self.type_1 = Type1(num_outputs)
        self.conv_bn = ConvBn(num_outputs)
        self.global_avg_pool = tf.keras.layers.GlobalAveragePooling2D()
        
        

    def call(self, inputs, training=False):
        x = self.type_1(inputs, training=training)
        x = self.conv_bn(x, training=training)
        x = self.global_avg_pool(x)
        return x

class SRNet(tf.keras.Model):
    def __init__(self, include_top=False, num_classes=2):
        super().__init__()

        self.type_1s = tf.keras.Sequential([
            Type1(64),
            Type1(16),
        ])

        self.type_2s = tf.keras.Sequential([
            Type2(16),
            Type2(16),
            Type2(16),
            Type2(16),
            Type2(16),
        ])

        self.type_3s = tf.keras.Sequential([
            Type3(16),
            Type3(64),
            Type3(128),
            Type3(256),
        ])


        self.type_4 = Type4(512)

        self.include_top = include_top
        self.num_classes = num_classes

        if include_top:
            self.flatten = tf.keras.layers.Flatten()
            self.dense = tf.keras.layers.Dense(num_classes, activation='softmax')
        

    def call(self, inputs, training=False):
        x = self.type_1s(inputs, training=training)
        x = self.type_2s(x, training=training)
        x = self.type_3s(x, training=training)
        x = self.type_4(x, training=training)

        if self.include_top:
            x = self.flatten(x)
            x = self.dense(x)

        return x

