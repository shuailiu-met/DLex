import numpy as np
from .Base import BaseLayer

class Dropout(BaseLayer):
    def __init__(self,probability):
        BaseLayer.__init__(self)
        self.prob = probability
        self.mask = None

    def forward(self,input_tensor):
        if self.testing_phase is True:
            out = input_tensor
        else: # set to 0 with the prob of 1-p
            input_shape = np.shape(input_tensor)
            ini = np.random.uniform(0,1,input_shape)
            self.mask = np.ones(input_shape)# if num >p, then set to 0, store pos in self.mask
            self.mask = np.where(ini>self.prob,0.,1.) # 0 -> dropout 1-> kept
            out = self.mask*input_tensor*(1/self.prob)   # for training, Y = X*mask/p, so that a mult by p in test is not needed any more
        return out

    def backward(self,error_tensor):
        if self.testing_phase is True:
            out = error_tensor
        else:
            #in forward, Y = X*mask/p, so in backward, we want delta Y-X, which is mask/p
            out = error_tensor*(self.mask/self.prob) 
        return out