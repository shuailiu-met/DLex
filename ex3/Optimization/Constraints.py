import numpy as np 

class L2_Regularizer():
    def __init__(self,alpha):
        self.alpha = alpha

    def calculate_gradient(self,weights):
        adjust = self.alpha*weights
        return adjust

    def norm(self,weights):
        norml2 = self.alpha*np.square(np.sqrt(np.sum(np.power(np.abs(weights),2)))) ### Lhat = L + square of L2(w)
        return norml2


class L1_Regularizer():
    def __init__(self,alpha):
        self.alpha = alpha

    def calculate_gradient(self,weights):
        adjust = self.alpha*np.sign(weights)
        return adjust

    def norm(self,weights):
        norml1 = self.alpha*np.sum(np.abs(weights))
        return norml1