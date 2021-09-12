import numpy as np


class NesterovAcceleratedGradient:
    def __init__(self, learning_rate=0.001, momentum=0.4):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.w_updt = np.array([])

    def update(self, w, grad_func):
        approx_future_grad = np.clip(grad_func(w - self.momentum * self.w_updt), -1, 1)
        if not self.w_updt.any():
            self.w_updt = np.zeros(np.shape(w))

        self.w_updt = self.momentum * self.w_updt + self.learning_rate * approx_future_grad
        return w - self.w_updt
