import numpy as np

class TanH:
    def __init__(self):
        self.activation = None

    def forward(self,input_tensor):
        y = np.tanh(input_tensor)
        self.activation = y
        return y

    def backward(self,error_tensor):
        gradient_to_x = error_tensor*(1-np.power(self.activation,2))
        return gradient_to_x