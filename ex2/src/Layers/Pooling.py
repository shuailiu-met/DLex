import numpy as np

class Pooling:
    def __init__(self,stride_shape,pooling_shape):
        self.stride_shape = stride_shape
        self.pooling_shape = pooling_shape
        self.max_loc = None

    def forward(self,input_tensor):
        if(len(np.shape(input_tensor))==3):#2d
            b,c,y = np.shape(input_tensor)
            stride_step = self.stride_shape[0]
            num_of_out = int(np.ceil())


    def backward(self,error_tensor):
        pass