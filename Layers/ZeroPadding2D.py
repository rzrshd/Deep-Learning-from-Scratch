import math
import numpy as np
import copy

from ConstantPadding2D import ConstantPadding2D


class ZeroPadding2D(ConstantPadding2D):
    def __init__(self, padding):
        self.padding = padding
        if isinstance(padding[0], int):
            self.padding = ((padding[0], padding[0]), padding[1])

        if isinstance(padding[1], int):
            self.padding = (self.padding[0], (padding[1], padding[1]))

        self.padding_value = 0
