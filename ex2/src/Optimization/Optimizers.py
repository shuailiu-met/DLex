import numpy as np

class Sgd():
    def __init__(self,learning_rate):
        self.learning_rate = learning_rate
        #print(self.learning_rate)

    def calculate_update(self,weight_tensor,gradient_tensor):
        return weight_tensor - self.learning_rate*gradient_tensor #in practice, gradient_tensor is the gradient of weights,which will update the weight



class SgdWithMomentum():
    def __init__(self,learning_rate,momentum_rate):
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.vk = 0#vk starts with 0

    def calculate_update(self,weight_tensor,gradient_tensor):# Vk needs to be updated and stored for the next use
        vk_now = self.momentum_rate*self.vk - self.learning_rate*gradient_tensor
        self.vk = vk_now
        return weight_tensor+vk_now



class Adam():
    def __init__(self,learning_rate,mu,rho):
        self.learning_rate = learning_rate
        self.mu = mu
        self.rho = rho
        self.vk = 0
        self.rk = 0   #we need vk / rk to be stored for further usage
        self.k = 1    #the int number for power part mu^k and rho^k,because we have k-1 cases, k starts with 1

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
        weights_next = weight_tensor - self.learning_rate*(vk_hat/(np.sqrt(rk_hat)+eps))
        return weights_next

