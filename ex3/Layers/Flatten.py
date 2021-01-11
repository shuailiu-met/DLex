import numpy as np
from Layers import Base


#forward turn each batch's m-d into 1-d
class Flatten(Base.BaseLayer):
    def __init__(self):
        self.shape = None

    def forward(self,input_tensor): #each batch's data is formed as 1-d
        shape = np.shape(input_tensor)
        self.shape = shape
        batch_num = shape[0]  #batch num = 9,as first dim of input
        one_d_size = np.prod(shape)/batch_num
        one_d_size = int(one_d_size)
        re_input = input_tensor.reshape((batch_num, one_d_size))
        return re_input

#turn 1-d back to m-d
    def backward(self,error_tensor):#error tensor is reshaped back to the size of initial input
        re_err = error_tensor.reshape(self.shape) # should be (9,3,4,11),the shape of initial input         (9,3,4,11) (9,132),can be found in test code
        return re_err