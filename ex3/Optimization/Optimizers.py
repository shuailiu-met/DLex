import numpy as np

class Optimizer():
    def __init__(self):
        self.regularizer = None

    def add_regularizer(self,regularizer):
        self.regularizer = regularizer

class Sgd(Optimizer):
    def __init__(self,learning_rate):
        Optimizer.__init__(self)
        self.learning_rate = learning_rate
        #print(self.learning_rate)

    def calculate_update(self,weight_tensor,gradient_tensor):

        updated_weight = weight_tensor - self.learning_rate*gradient_tensor #in practice, gradient_tensor is the gradient of weights,which will update the weight
        if self.regularizer is not None:
            updated_weight -= self.learning_rate*self.regularizer.calculate_gradient(weight_tensor)   # adjust for regularization
        return updated_weight



class SgdWithMomentum(Optimizer):
    def __init__(self,learning_rate,momentum_rate):
        Optimizer.__init__(self)
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.vk = 0#vk starts with 0

    def calculate_update(self,weight_tensor,gradient_tensor):# Vk needs to be updated and stored for the next use
        vk_now = self.momentum_rate*self.vk - self.learning_rate*gradient_tensor
        self.vk = vk_now
        updated_weight = weight_tensor+vk_now
        if self.regularizer is not None:
            updated_weight -= self.learning_rate*self.regularizer.calculate_gradient(weight_tensor) # adjust for regularization
        return updated_weight



class Adam(Optimizer):
    def __init__(self,learning_rate,mu,rho):
        Optimizer.__init__(self)
        self.learning_rate = learning_rate
        self.mu = mu
        self.rho = rho
        self.vk = 0
        self.rk = 0   #we need vk / rk to be stored for further usage
        self.k = 1    #the int number for power part mu^k and rho^k,because we have k-1 cases, k starts with 1
        # vk,rk is usually initialized with 0

    def calculate_update(self,weight_tensor,gradient_tensor):
        gk = gradient_tensor
        vk = self.mu*self.vk + (1-self.mu)*gk
        self.vk = vk
        rk = self.rho*self.rk + (1-self.rho)*(gk*gk)  #circle dot ------> elementwise MULT
        self.rk = rk

        #bias correction
        vk_hat = vk/(1-np.power(self.mu,self.k))
        rk_hat = rk/(1-np.power(self.rho,self.k))

        self.k = self.k+1  # update k idx

        eps = np.finfo(float).eps
        updated_weight = weight_tensor - self.learning_rate*(vk_hat/(np.sqrt(rk_hat)+eps))
        if self.regularizer is not None:
            updated_weight -= self.learning_rate*self.regularizer.calculate_gradient(weight_tensor)  # adjust for regularization
        return updated_weight


