__author__ = "Jakub Sochor"
__copyright__ = "Copyright 2018, Jakub Sochor"
__license__ = "MIT"


import tensorflow as tf
from tensorflow.keras.layers import Layer

class L2Normalization(Layer):
    """
    Normalizes features to have unit length
    """
    def call(self, x):
        return tf.nn.l2_normalize(x, dim=1)
