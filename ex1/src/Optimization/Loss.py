import numpy as np

class CrossEntropyLoss():
    def __init__(self):
        self.input_tensor = None #hint as before

    def forward(self,input_tensor,label_tensor):#here input = predicted y//////label tensor is composed of 0 and 1
        self.input_tensor = input_tensor
        eps = np.finfo(float).eps
        loss_elementwise = -np.log(input_tensor+eps)*label_tensor#elementwisely calculate log,it's conditioned with yk = 1,so elementwisely multiiply label_tensor
        loss_all = np.sum(loss_elementwise)
        return loss_all

    def backward(self,label_tensor):
        return (-np.true_divide(label_tensor,self.input_tensor))#En = - y/y^ ,y^is between 0 1, if directly divide, only int part,we need float part too