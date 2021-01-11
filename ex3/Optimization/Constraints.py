import numpy as np
import math




class L2_Regularizer():
    def __init__(self, alpha):
        self.alpha = alpha
        self.regularizer = None

    def calculate_gradient(self, weights):
        return self.alpha * weights

    def norm(self, weights):
        return math.pow(np.linalg.norm(weights), 2) * self.alpha


class L1_Regularizer():
    def __init__(self, alpha):
        self.alpha = alpha

    def calculate_gradient(self, weights):
        return self.alpha * np.sign(weights)

    def norm(self, weights):
        return np.sum(np.abs(weights)) * self.alpha
