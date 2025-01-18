__author__ = "Jakub Sochor"
__copyright__ = "Copyright 2018, Jakub Sochor"
__license__ = "MIT"


import tensorflow as tf

#%%
class ContrastiveLoss(object):
    """
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """
    def __init__(self, margin = 1):
        self.margin = margin
        self.__name__ = "ContrastiveLoss"

    def __call__(self, y_true, y_pred):
        return tf.reduce_mean(y_true * tf.square(y_pred) + (1 - y_true) * tf.square(tf.maximum(self.margin - y_pred, 0)))
