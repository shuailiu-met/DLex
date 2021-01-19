import numpy as np
import copy as copy
from .TanH import TanH
from .Base import BaseLayer
from .Sigmoid import Sigmoid
from .FullyConnected import FullyConnected


class RNN(BaseLayer):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        #self.ht0 = np.zeros((self.hidden_size))
        self.hidden_state = np.zeros((self.hidden_size))

        self.F1 = FullyConnected(self.input_size+self.hidden_size, self.hidden_size)
        self.F2 = FullyConnected(self.hidden_size, self.output_size)
        
        self.TanH_store = [] # a list store all TanH for each t so that the activation is stored
        self.Sigmoid_store = [] #a list store Sigmoid for each t so that the activation is stored

        self._memorize = False
        self._gradient_weights = None
        self._weights = np.random.uniform(0,1,(self.hidden_size + self.input_size+1, self.hidden_size))
        self.weights_y = np.ones((self.hidden_size+1, self.output_size))

        self._optimizer = None

    def initialize(self, weights_initializer, bias_initializer):  # weight shape inputsize*outputsize , bias size 1*outputsize,initializer is object from initializer.py
        # fan_in input_dim/size    fan_out output_dim/size
        weights = weights_initializer.initialize((self.input_size+self.hidden_size, self.hidden_size), self.input_size+self.hidden_size, self.hidden_size)
        bias = bias_initializer.initialize((1, self.hidden_size), self.input_size, self.hidden_size)
       # weights_y = weights_initializer.initialize((self.hidden_size, self.output_size), self.hidden_size, self.output_size)
        #bias_y = bias_initializer.initialize((1, self.output_size), self.hidden_size, self.output_size)

        self.weights = np.concatenate((weights, bias), axis=0)
        #self.weights_y = np.concatenate((weights_y, bias_y), axis=0)

    @property
    def memorize(self):
        return self._memorize

    @memorize.setter
    def memorize(self, value):
        self._memorize = value

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, temp_gradient):
        self._gradient_weights = temp_gradient

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, temp_weights):
        self._weights = temp_weights

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, _optimizer_obj):  # _optimizer is a object of type sgd
        self._optimizer = copy.deepcopy(_optimizer_obj)

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        self.time_steps = np.shape(input_tensor)[0]

        self.F1.weights = self.weights
        self.F2.weights = self.weights_y

        #self.Wh = np.ones((self.time_steps, self.hidden_size + self.input_size + 1, self.hidden_size))
        #self.Wy = np.ones((self.time_steps, self.hidden_size + 1, self.output_size))

        self.ht = np.zeros((self.time_steps, self.hidden_size))
        self.xt_ = np.ones((self.time_steps, self.hidden_size+self.input_size))

        if self.memorize == False:
            self.hidden_state = np.zeros((self.hidden_size))

        self.output_tensor = np.zeros((self.time_steps, self.output_size))

        for t in range(self.time_steps):
            TanH_t = TanH()
            Sigmoid_t = Sigmoid()
            if t == 0:
                self.xt_[t] = np.concatenate((self.hidden_state, self.input_tensor[t]), axis=0)
            else:
                self.xt_[t] = np.concatenate((self.ht[t - 1], self.input_tensor[t]), axis=0)


            self.ht[t] = TanH_t.forward(self.F1.forward(self.xt_[t].reshape(1, -1)))

            self.output_tensor[t] = Sigmoid_t.forward(self.F2.forward(self.ht[t].reshape(1, -1)))
            self.TanH_store.append(TanH_t)
            self.Sigmoid_store.append(Sigmoid_t)


        self.hidden_state = self.ht[self.time_steps - 1]

        return self.output_tensor


    def backward(self, error_tensor):

        dh = np.zeros((self.time_steps, self.hidden_size))
        dx = np.zeros((self.time_steps, self.input_size))
        dhx = np.zeros((self.time_steps, self.hidden_size+self.input_size))
        dWh = np.zeros((self.hidden_size+self.input_size+1, self.hidden_size))
        dWy = np.zeros((self.hidden_size+1, self.output_size))
        for t in range(self.time_steps-1, -1, -1):
            TanH = self.TanH_store[t]
            Sigmoid = self.Sigmoid_store[t]
            self.F2.input_tensor = np.concatenate((self.ht[t].reshape(1, -1), [[1]]), axis=1)
            h_gradient = self.F2.backward(Sigmoid.backward(error_tensor[t]))
            dWy += self.F2.gradient_weights
            if t == self.time_steps-1:
                h_gradient_sum = h_gradient
            else:
                h_gradient_sum = dh[t+1] + h_gradient

            self.F1.input_tensor = np.concatenate((self.xt_[t].reshape(1, -1), [[1]]), axis=1)
            dhx[t] = self.F1.backward(TanH.backward(h_gradient_sum))
            #print(np.shape(self.F1.gradient_weights))
            dWh += self.F1.gradient_weights
            dh[t] = dhx[t, 0:self.hidden_size]
            dx[t] = dhx[t, self.hidden_size:]

        self.gradient_weights = dWh
        if self.optimizer is not None:
            self.weights = self._optimizer.calculate_update(self.weights, self.gradient_weights)
            self.weights_y = self._optimizer.calculate_update(self.weights_y, dWy)

        return dx

    def calculate_regularization_loss(self):
        if self._optimizer.regularizer is not None:
            return self._optimizer.regularizer.norm(self.weights)