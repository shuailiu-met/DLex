import numpy as np
from .Base import BaseLayer


class FullyConnected(BaseLayer):
    def __init__(self,input_size,output_size): # input = (x1,x2....,xn,1) output = (y1,y2,...,ym)
        BaseLayer.__init__(self)
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.rand(input_size+1,output_size)
        self.input_tensor = None
        self._optimizer = None   #optimizer as type optimizer/property
        self._gradient_weights = None

    #return input tensor for next layer
    def forward(self,input_tensor):
        column_num = np.shape(input_tensor)[0]
        bias = np.ones((column_num,1))
        new_input = np.concatenate((input_tensor,bias),axis = 1)# add homogeneous part,inputsize as columns,so add a column of 1 on the right end
        #attention of the double transpose of W
        y = np.dot(new_input,self.weights)
        self.input_tensor = new_input
        return y


    def initialize(self,weights_initializer,bias_initializer):#weight shape inputsize*outputsize , bias size 1*outputsize,initializer is object from initializer.py
        #fan_in input_dim/size    fan_out output_dim/size
        weights = weights_initializer.initialize((self.input_size,self.output_size),self.input_size,self.output_size)
        bias = bias_initializer.initialize((1,self.output_size),self.input_size,self.output_size)
        self.weights = np.concatenate((weights,bias),axis = 0)
        

    @property
    def optimizer(self):
        return self._optimizer
    
    @optimizer.setter
    def optimizer(self,_optimizer_obj):#_optimizer is a object of type sgd
        self._optimizer = _optimizer_obj


    #returns gradient for front,first error tensor is from activation's backward, get errortensor(grandient) for each fronter layer,and update weights
    def backward(self,error_tensor):
        weight_n_rows = self.weights[0:np.shape(self.weights)[0]-1,:]# why n rows-------> answer on StudOn forum
        gradient_x = np.dot(error_tensor,np.transpose(weight_n_rows))#gradient with respect to x, 

        temp_gradient_weights = np.dot(np.transpose(self.input_tensor),error_tensor) #need to store input tensor in forward step,input tensor is always with homogenous dim
        self.gradient_weights = temp_gradient_weights #setter used 
        #update weights
        if self.optimizer is not None:
            self.weights = self._optimizer.calculate_update(self.weights,self.gradient_weights)#update weights and errors,new weights is updated for the forward,so bias need
        return gradient_x

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self,temp_gradient):
        self._gradient_weights = temp_gradient
