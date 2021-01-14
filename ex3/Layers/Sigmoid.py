import numpy as np

class Sigmoid:
    def __init__(self):
        self.activation = None

    def forward(self,input_tensor):
        y = 1/(1+np.exp((-1)*input_tensor))
        self.activation = y
        return y

    def backward(self,error_tensor):
        gradient_to_x = error_tensor*self.activation*(1-self.activation)
        return gradient_to_x