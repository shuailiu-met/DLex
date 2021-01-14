import numpy as np
from .Base import BaseLayer

# import non-linear 
class ReLU(BaseLayer):
    def __init__(self):
        BaseLayer.__init__(self)
        self.input_tensor = None

    def forward(self,input_tensor):
        self.input_tensor = input_tensor
        forward = np.maximum(input_tensor,np.zeros((np.shape(input_tensor))))#f = max(0,x)
        return forward

    def backward(self,error_tensor):
        back_result = np.where(self.input_tensor<=0,0,error_tensor)#need to store the corresponding input tensor of this layer
        return back_result
