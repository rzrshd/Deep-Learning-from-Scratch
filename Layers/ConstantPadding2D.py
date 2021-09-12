import math
import numpy as np
import copy

from Vanilla_Layer import Layer


class ConstantPadding2D(Layer):
    def __init__(self, padding, padding_value=0):
        self.padding = padding
        self.trainable = True
        if not isinstance(padding[0], tuple):
            self.padding = ((padding[0], padding[0]), padding[1])

        if not isinstance(padding[1], tuple):
            self.padding = (self.padding[0], (padding[1], padding[1]))

        self.padding_value = padding_value

    def forward_pass(self, X, training=True):
        output = np.pad(X, pad_width=((0, 0), (0, 0), self.padding[0], self.padding[1]), mode='constant', constant_values=self.padding_value)
        return output

    def backward_pass(self, accum_grad):
        pad_top, pad_left = self.padding[0][0], self.padding[1][0]
        height, width = self.input_shape[1], self.input_shape[2]
        accum_grad = accum_grad[:, :, pad_top:pad_top + height, pad_left:pad_left + width]
        return accum_grad

    def output_shape(self):
        new_height = self.input_shape[1] + np.sum(self.padding[0])
        new_width = self.input_shape[2] + np.sum(self.padding[1])
        return (self.input_shape[0], new_height, new_width)
