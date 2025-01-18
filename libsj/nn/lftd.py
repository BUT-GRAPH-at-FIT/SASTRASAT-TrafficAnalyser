__author__ = "Jakub Sochor"
__copyright__ = "Copyright 2018, Jakub Sochor"
__license__ = "MIT"


import tensorflow as tf
from tensorflow import keras

class LFTD(keras.layers.Layer):
    """
    Learning Features in Temporal Domain
    Layer for aggregation of features in temporal domain
    Accepts inputs of shape batch_size x time_samples x feature_dimensionality
    Produces outputs of shape batch_size x output_dim
    See paper Learning Feature Aggregation in Temporal Domain for Re-Identification for more details
    """
    
    def __init__(self, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dim
        
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

    def build(self, input_shape):
        assert len(input_shape) == 3
        batch_size, time_samples, input_dim = input_shape 
        self.compression_kernel = self.add_weight(name="compression_kernel",
                                                   shape=(input_dim, self.output_dim),
                                                   initializer=keras.initializers.glorot_uniform(),
                                                   trainable=True)
        self.compression_bias = self.add_weight(name="compression_bias", 
                                                shape=(self.output_dim, ),
                                                initializer=keras.initializers.Zeros(),
                                                trainable=True)
        self.weights_gen_kernel = self.add_weight(name="weights_gen_kernel",
                                                  shape=(2*self.output_dim, self.output_dim),
                                                  initializer=keras.initializers.glorot_uniform(),
                                                  trainable=True)
        super().build(input_shape)
        
    def _matmul_in_time(self, x, kernel):
        input_shape = tf.shape(x)
        x = tf.reshape(x, [-1, input_shape[2]])
        x = tf.matmul(x, kernel)
        x = tf.reshape(x, [-1, input_shape[1], tf.shape(x)[1]])
        return x
        
        
    def call(self, x):
        y = tf.tanh(self._matmul_in_time(x, self.compression_kernel) + self.compression_bias)
        avg = tf.tile(tf.reduce_mean(y, axis=1, keepdims=True), [1, tf.shape(x)[1], 1])
        y_avg = tf.concat((y, avg), axis=2)
        weights = self._matmul_in_time(y_avg, self.weights_gen_kernel)
        weights = tf.nn.softmax(weights, axis=1)
        features = tf.reduce_sum(y*weights, axis=1)
        features = tf.nn.l2_normalize(features, axis=1)
        return features
    
    def get_config(self):
        base_config = super().get_config()
        config = dict(output_dim=self.output_dim)
        return dict(list(base_config.items()) + list(config.items()))


        
    


