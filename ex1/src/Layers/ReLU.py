import numpy as np

class ReLU():
    def __init__(self):
        self.input_tensor = None#store input for further use just like in FullyConnected

    def forward(self,input_tensor):
        self.input_tensor = input_tensor
        forward = np.where(input_tensor<=0,0,input_tensor)
        return forward

    def backward(self,error_tensor):
        back_result = np.where(self.input_tensor<=0,0,error_tensor)
        return back_result
