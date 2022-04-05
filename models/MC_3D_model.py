import tensorflow as tf
from tensorflow.keras import layers
import math
import numpy as np


class MCDropout(tf.keras.layers.AlphaDropout):
    """ Wrapper class to enable dropout at test time for dense layers"""
    def call(self, inputs):
        return super().call(inputs, training=True)


class MCSpatialDropout(tf.keras.layers.SpatialDropout3D):
    """ Wrapper class to enable dropout at test time for convolutional layers"""
    def call(self, inputs):
        return super().call(inputs, training=True)


def get_mc_model(input_shape, dropout=0.1, l2_weight: float = 0.0, is_mnist=False):
    """ Builds a 3D MC Dropout CNN (VGG style).
        Args:
            input_shape: input dimensionality (tuple)
            dropout: (MC) dropout rate to apply
            l2_weight: L2 regularisation parameter
            is_mnist: boolean that determines the number of Max-Pooling layers to apply
            """

    inputs = tf.keras.Input(input_shape)

    # conv_block 1
    x = tf.keras.layers.Conv3D(
        filters=64, kernel_size=3, activation="relu", padding="same", kernel_regularizer=tf.keras.regularizers.l2(l2_weight)
    )(inputs)
    x = tf.keras.layers.LayerNormalization()(x)
    x = MCSpatialDropout(dropout)(x)
    x = tf.keras.layers.MaxPool3D(pool_size=(2, 2, 1))(x)

    # conv_block_2
    x = tf.keras.layers.Conv3D(
        filters=64, kernel_size=3, activation="relu", padding="same", kernel_regularizer=tf.keras.regularizers.l2(l2_weight)
    )(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = MCSpatialDropout(dropout)(x)
    x = tf.keras.layers.Conv3D(
        filters=64, kernel_size=3, activation="relu", padding="same", kernel_regularizer=tf.keras.regularizers.l2(l2_weight)
    )(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = MCSpatialDropout(dropout)(x)
    x = tf.keras.layers.MaxPool3D(pool_size=(2, 2, 2))(x)

    # conv_block_3
    x = tf.keras.layers.Conv3D(
        filters=128, kernel_size=3, activation="relu", padding="same", kernel_regularizer=tf.keras.regularizers.l2(l2_weight)
    )(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = MCSpatialDropout(dropout)(x)
    x = tf.keras.layers.Conv3D(
        filters=128, kernel_size=3, activation="relu", padding="same", kernel_regularizer=tf.keras.regularizers.l2(l2_weight)
    )(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = MCSpatialDropout(dropout)(x)
    x = tf.keras.layers.MaxPool3D(pool_size=(2, 2, 1))(x) if not is_mnist else x

    # conv_block_4
    x = tf.keras.layers.Conv3D(
        filters=128, kernel_size=3, activation="relu", padding="same", kernel_regularizer=tf.keras.regularizers.l2(l2_weight)
    )(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = MCSpatialDropout(dropout)(x)
    x = tf.keras.layers.Conv3D(
        filters=128, kernel_size=3, activation="relu", padding="same", kernel_regularizer=tf.keras.regularizers.l2(l2_weight)
    )(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = MCSpatialDropout(dropout)(x)
    x = tf.keras.layers.MaxPool3D(pool_size=(2, 2, 1))(x) if not is_mnist else x

    # conv_block_5
    x = tf.keras.layers.Conv3D(
        filters=256, kernel_size=3, activation="relu", padding="same", kernel_regularizer=tf.keras.regularizers.l2(l2_weight)
    )(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = MCSpatialDropout(dropout)(x)
    x = tf.keras.layers.Conv3D(
        filters=256, kernel_size=3, activation="relu", padding="same", kernel_regularizer=tf.keras.regularizers.l2(l2_weight)
    )(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = MCSpatialDropout(dropout)(x)
    x = tf.keras.layers.MaxPool3D(pool_size=(2, 2, 1))(x) if not is_mnist else x

    # conv_block_6
    x = tf.keras.layers.Conv3D(
        filters=256, kernel_size=3, activation="relu", padding="same", kernel_regularizer=tf.keras.regularizers.l2(l2_weight)
    )(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = MCSpatialDropout(dropout)(x)
    x = tf.keras.layers.Conv3D(
        filters=256, kernel_size=3, activation="relu", padding="same", kernel_regularizer=tf.keras.regularizers.l2(l2_weight)
    )(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = MCSpatialDropout(dropout)(x)
    x = tf.keras.layers.MaxPool3D(pool_size=(2, 2, 2))(x)

    # conv_block_7
    x = tf.keras.layers.Conv3D(
        filters=256, kernel_size=3, activation="relu", padding="same", kernel_regularizer=tf.keras.regularizers.l2(l2_weight)
    )(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = MCSpatialDropout(dropout)(x)
    x = tf.keras.layers.Conv3D(
        filters=256, kernel_size=3, activation="relu", padding="same", kernel_regularizer=tf.keras.regularizers.l2(l2_weight)
    )(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = MCSpatialDropout(dropout)(x)
    x = tf.keras.layers.MaxPool3D(pool_size=(2, 2, 2))(x)

    # dense
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(units=128, activation="relu")(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = MCDropout(dropout)(x)
    x = tf.keras.layers.Dense(units=128, activation="relu")(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = MCDropout(dropout)(x)

    outputs = tf.keras.layers.Dense(units=1, activation="sigmoid")(x)

    # Define the model.
    model = tf.keras.Model(inputs, outputs, name="3DCNN")
    return model

