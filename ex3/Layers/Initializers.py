import numpy as np 

class Constant():
    def __init__(self,const_value=0.1):
        self.const_value = const_value
    
    def initialize(self,weights_shape,fan_in,fan_out):#return a initial tensor with desired shape
        ini = np.ones(weights_shape)*self.const_value
        return ini


class UniformRandom():
    def __init__(self):
        pass

    def initialize(self,weights_shape,fan_in,fan_out):
        ini = np.random.uniform(0,1,weights_shape)
        return ini


class Xavier():
    def __init__(self):
        pass

    def initialize(self,weights_shape,fan_in,fan_out):
        mu = 0.
        sigma = np.sqrt(2./(fan_in+fan_out))
        ini = np.random.normal(mu,sigma,weights_shape)
        return ini


class He():
    def __init__(self):
        pass

    def initialize(self,weights_shape,fan_in,fan_out):
        mu = 0.
        sigma = np.sqrt(2./fan_in)
        ini = np.random.normal(mu,sigma,weights_shape)
        return ini