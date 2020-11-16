import numpy as np


class FullyConnected():
    def __init__(self,input_size,output_size): # input = (x1,x2....,xn,1) output = (y1,y2,...,ym)
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.rand(input_size+1,output_size)
        self.input_tensor = None
        self._optimizer = None   #optimizer as type optimizer/property
        self._gradient_weights = None

    def forward(self,input_tensor):
        column_num = np.shape(input_tensor)[0]
        bias = np.ones((column_num,1))
        new_input = np.concatenate((input_tensor,bias),axis = 1)
        
        #in formula,W is already tranposed, but in memory layout,transposed again,so weights is what we need
        #input should be (n+1,b), here we have (b,n+1)size as input,since we need a tranpose according to memory layout, so the (b,n+1) input is what we need

        y = np.dot(new_input,self.weights)
        self.input_tensor = new_input
        return y

    @property
    def optimizer(self):
        return self._optimizer
    
    @optimizer.setter
    def optimizer(self,_optimizer_obj):#_optimizer is a object of type sgd
        self._optimizer = _optimizer_obj
    
    def backward(self,error_tensor):
        weight_n = self.weights[0:np.shape(self.weights)[0]-1,:]#weight should remove the bias part here
        gradient_x = np.dot(error_tensor,np.transpose(weight_n))#memory layout

        temp_gradient_weights = np.dot(np.transpose(self.input_tensor),error_tensor) #need to store inputtensor in forward step,input tensor is always with homogenous dim
        self.gradient_weights = temp_gradient_weights #setter used 
        #update weights
        if self.optimizer is not None:
            self.weights = self._optimizer.calculate_update(self.weights,self.gradient_weights)
        return gradient_x

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self,temp_gradient):
        self._gradient_weights = temp_gradient
