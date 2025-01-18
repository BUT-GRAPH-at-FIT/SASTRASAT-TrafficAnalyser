__author__ = "Jakub Sochor"
__copyright__ = "Copyright 2018, Jakub Sochor"
__license__ = "MIT"


import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow import keras
from tensorflow.keras import backend as K
import numpy as np

class BaseDistanceLayer(Layer):
    def build(self, input_shapes):
        assert len(input_shapes) == 2
        assert input_shapes[0][-1] == input_shapes[1][-1]
        super().build(input_shapes)
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], 1)


class EuclideanDistance(BaseDistanceLayer):
    def call(self, inputs):
        a, b = inputs
        diff = a - b
        diff_sq = tf.square(diff)
        return tf.sqrt(tf.reduce_sum(diff_sq, axis=1, keep_dims=True))

    
class WeightedEuclideanDistance(BaseDistanceLayer):
    def build(self, input_shapes):
        batch_size, input_dim = input_shapes[0]
        self.w = self.add_weight(name="w",
                                 shape=(1, input_dim),
                                 initializer=keras.initializers.RandomNormal(mean=1.0, stddev=0.1),
                                 constraint=keras.constraints.non_neg(),
                                 trainable=True)
        super().build(input_shapes)

    def call(self, inputs):
        a, b = inputs
        diff = a - b
        diff_sq = tf.square(diff)
        return tf.sqrt(tf.reduce_sum(self.w * diff_sq, axis=1, keep_dims=True))
    
    
    
class _MahalanobisDistanceRegularizer:
    def __init__(self, regularization, dim):
        self.regularization = regularization
        self.identity = tf.constant(np.eye(dim), dtype=tf.float32)

    def __call__(self, w):
        optimize = tf.matmul(w, w, transpose_b=True) - self.identity
        return self.regularization/2 * tf.square(tf.norm(optimize, ord="fro", axis=(0,1)))


def _mahalanobis_distance_initializer(shape, dtype=None, **kwargs):
    assert shape[0] == shape[1]
    return tf.diag(K.random_normal((shape[0], ), mean=1, stddev=0.1, dtype=dtype))


class MahalanobisDistance(BaseDistanceLayer):
    """
    Implemeted as described by Shi et al, ECCV 2016
    arXiv link: https://arxiv.org/pdf/1611.00137.pdf
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shapes):
        batch_size, input_dim = input_shapes[0]
        self.W = self.add_weight(name="W",
                                 shape=(input_dim, input_dim),
                                 initializer=_mahalanobis_distance_initializer,
                                 regularizer=_MahalanobisDistanceRegularizer(0.01, input_dim),
                                 trainable=True)
        super().build(input_shapes)

    def call(self, inputs, **kwargs):
        a,b = inputs
        diff = a-b
        M = tf.matmul(self.W, self.W, transpose_b=True)
        return tf.sqrt(tf.reduce_sum(tf.matmul(diff, M) * diff, axis=1, keep_dims=True))