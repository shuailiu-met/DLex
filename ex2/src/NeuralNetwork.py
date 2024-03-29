import numpy as np
import copy as copy

class NeuralNetwork():
    
    def __init__(self,optimizer,weights_initializer,bias_initializer):
        self._optimizer = optimizer
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer
        self.loss = []
        self.layers = []#layers could be simply fully connected layer or activation layer
        self.data_layer = None
        self.loss_layer = None #loss_layer is the layer for final loss computing


    def forward(self):
        input_tensor,label_tensor = self.data_layer.next()#calling next() on data_layer to get input
        self.label_tensor = label_tensor
        for each_layer in self.layers:# run over all layers ,fullyconnected layer,the last one is activation layer
            input_tensor_next = each_layer.forward(input_tensor)
            input_tensor = input_tensor_next

        if self.loss_layer is not None:
            loss_total = self.loss_layer.forward(input_tensor,label_tensor) #compute cross entropy loss by the output of activation layer
            return loss_total

    def backward(self):#use label_tensor from current input which is stored
        error_tensor = self.loss_layer.backward(self.label_tensor)#entropyloss 
        for i in range(len(self.layers)-1,-1,-1):
            error_tensor_front = self.layers[i].backward(error_tensor)
            error_tensor = error_tensor_front #for front layer(error/gradients),update weight for next iteration/trial


    def append_trainable_layer(self,layer):#assign optimizer to each layer and apply initializer on each layer(fully connected layer)
        opt = copy.deepcopy(self._optimizer)
        layer.optimizer = opt 
        weights_ini = self.weights_initializer
        bias_ini = self.bias_initializer
        layer.initialize(weights_ini,bias_ini)  #so that the weights and bias of layer is initialized
        self.layers.append(layer)

    def train(self,iterations):
        #print(np.shape(iterations))
        #print(type(iterations))    iterations is a int number
        idx = np.arange(iterations)
        for i in idx:
            loss_of_this_iteration = self.forward()
            if loss_of_this_iteration is not None:
                self.loss.append(loss_of_this_iteration)
                #forward path,compute the loss at last layer for the error computing
            self.backward()#back propagation,computes gradients and update weights for forward path,so that next iteration can work better,in next iter,forward can run with new weights
        #print(np.shape(self.loss))

    def test(self, input_tensor):
        #print(self.layers) #/////4 layers in order of fullyconnected,ReLU,fullyconnected,Softmax
        for layer in self.layers:
            input_tensor_next = layer.forward(input_tensor)
            input_tensor = input_tensor_next
        return input_tensor #the last layer is Softmax,so the last forward result is the prediction probability
        


