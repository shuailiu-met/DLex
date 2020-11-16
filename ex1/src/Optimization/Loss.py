import numpy as np

class CrossEntropyLoss():
    def __init__(self):
        self.input_tensor = None #hint as before

    def forward(self,input_tensor,label_tensor):#here input = predicted y//////label tensor is composed of 0 and 1
        self.input_tensor = input_tensor
        eps = np.finfo(float).eps
        loss_elementwise = -np.log(input_tensor+eps)*label_tensor
        loss_all = np.sum(loss_elementwise)
        return loss_all

    def backward(self,label_tensor):
        return (-np.true_divide(label_tensor,self.input_tensor))