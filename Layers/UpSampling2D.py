import math
import numpy as np
import copy

from Vanilla_Layer import Layer


class UpSampling2D(Layer):
    def __init__(self, size=(2,2), input_shape=None):
        self.prev_shape = None
        self.trainable = True
        self.size = size
        self.input_shape = input_shape

    def forward_pass(self, X, training=True):
        self.prev_shape = X.shape
        X_new = X.repeat(self.size[0], axis=2).repeat(self.size[1], axis=3)
        return X_new

    def backward_pass(self, accum_grad):
        accum_grad = accum_grad[:, :, ::self.size[0], ::self.size[1]]
        return accum_grad

    def output_shape(self):
        channels, height, width = self.input_shape
        return channels, self.size[0] * height, self.size[1] * width
