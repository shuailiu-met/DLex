import numpy as np
from .Helpers import compute_bn_gradients
from .Base import BaseLayer
import copy as copy

class BatchNormalization(BaseLayer):
    def __init__(self,channels):
        BaseLayer.__init__(self)
        self.channels = channels
        self.input_tensor = None
        self.batch_mean = None
        self.batch_std = None
        self.accumulated_mean = None
        self.accumulated_std = None
        self.alpha = 0.8
        self.weights = None
        self.bias = None
        self.normalized_input = None
        self.unformated_input = None
        self.gradient_weights = None
        self.gradient_bias = None
        self.optimizer = None
        self.weights_initializier = None
        self.bias_initializer = None
        self.initialize(self.weights_initializier,self.bias_initializer) # import original weights and bias,or there will be Nonetype, initialized when creating object or it will be same for every iter
        


    def initialize(self,weights_initializer,bias_initializer):
        self.weights = np.ones(self.channels)
        self.bias = np.zeros(self.channels)

    def forward(self,input_tensor):# input b*c*y*x
        eps = np.finfo(float).eps
        
        self.unformated_input = input_tensor # used in reformat

        input_tensor = self.reformat(input_tensor) #conv reformat
        self.input_tensor = input_tensor # now all shaped in one format

        if self.testing_phase is True: # testing case ,use accumulated mean/std to replace batch mean/std
            self.normalized_input = (input_tensor-self.accumulated_mean)/np.sqrt(np.power(self.accumulated_std,2)+eps) # accumulated values are calculated during training but used only in test
        else: # training phase
            self.batch_mean = np.mean(input_tensor,axis = 0) #mean over b*datasize for each channel, axis = 0 -> b*datasize dim
            self.batch_std = np.std(input_tensor,axis = 0)
            self.normalized_input = (input_tensor-self.batch_mean)/np.sqrt(np.power(self.batch_std,2)+eps) # X_hat
            
            # accumulated mean and std is moving averaged during training but applied only during testing
            if (len(np.shape(self.accumulated_mean))==0): # first accumulated mean and std
                self.accumulated_mean = self.batch_mean
                self.accumulated_std = self.batch_std
            else:
                self.accumulated_mean = self.alpha*self.accumulated_mean+(1-self.alpha)*self.batch_mean
                self.accumulated_std = self.alpha*self.accumulated_std+(1-self.alpha)*self.batch_std

        Y_hat = self.weights*self.normalized_input + self.bias
        Y_hat = self.reformat(Y_hat)
        return Y_hat
        #when output, also reformat needed

    def reformat(self,tensor):
        if(len(np.shape(tensor))==4):# image to vector
            B,H,M,N = np.shape(tensor)  # in img2vec test, no forward, so no unformated input
            flatten_1 = tensor.reshape((B,H,M*N)) # first flatten
            transpose_1 = np.transpose(flatten_1,(0,2,1)) # transpose on axis 1/2
            flatten_2 = transpose_1.reshape((B*M*N,H)) # flatten agian
            out = flatten_2

        elif((len(np.shape(tensor))==2) and (len(np.shape(self.unformated_input))==4)) : # vector to image, self.input_tensor stores the tensor after reformat, ensure that the tensor is reformated conv tensor
            B,H,M,N = np.shape(self.unformated_input) # only conv input 4-d, formed into 2-d, other 2-d cases are originally 2-d
            recover_1 = tensor.reshape((B,M*N,H)) # recovering 2d -> 3d
            transpose_1 = np.transpose(recover_1,(0,2,1)) # transpose on axis 1/2
            recover_2 = transpose_1.reshape((B,H,M,N)) # fully recovering 3d->4d
            out = recover_2

        else: # no conv case
            out = tensor
        return out


    def backward(self,error_tensor):# training
        error_tensor_reformat = self.reformat(error_tensor) # non-conv remains the shape, conv is reformated
        self.gradient_weights = np.sum(error_tensor_reformat*self.normalized_input,axis = 0)
        self.gradient_bias = np.sum(error_tensor_reformat,axis = 0)

        error_front = compute_bn_gradients(error_tensor_reformat,self.input_tensor,self.weights,self.batch_mean,np.power(self.batch_std,2),np.finfo(float).eps) # during training, batch mean/std
        #in conv,4d->2d
        if self.optimizer is not None:
            optimizer_bias = copy.deepcopy(self.optimizer)
            optimizer_weights = copy.deepcopy(self.optimizer)
            self.weights = optimizer_weights.calculate_update(self.weights,self.gradient_weights)
            self.bias = optimizer_bias.calculate_update(self.weights,self.gradient_bias)
        
        error_front = self.reformat(error_front) #recover the format
        return error_front
        
