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

        self.TanH = TanH()
        self.Sigmoid = Sigmoid()
        self.F1 = FullyConnected(self.input_size+self.hidden_size, self.hidden_size)
        self.F2 = FullyConnected(self.hidden_size, self.output_size)

        self._memorize = False
        self._gradient_weights = None
        self._weights = np.ones((self.hidden_size + self.input_size+1, self.hidden_size))
        self.weights_y = np.ones((self.hidden_size+1, self.output_size))

        self._optimizer = None

    def initialize(self, weights_initializer, bias_initializer):  # weight shape inputsize*outputsize , bias size 1*outputsize,initializer is object from initializer.py
        # fan_in input_dim/size    fan_out output_dim/size
        weights = weights_initializer.initialize((self.input_size+self.hidden_size, self.hidden_size), self.input_size+self.hidden_size, self.hidden_size)
        bias = bias_initializer.initialize((1, self.hidden_size), self.input_size, self.hidden_size)
        weights_y = weights_initializer.initialize((self.hidden_size, self.output_size), self.hidden_size, self.output_size)
        bias_y = bias_initializer.initialize((1, self.output_size), self.hidden_size, self.output_size)

        self.weights = np.concatenate((weights, bias), axis=0)
        self.weights_y = np.concatenate((weights_y, bias_y), axis=0)

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
            if t == 0:
                self.xt_[t] = np.concatenate((self.hidden_state, self.input_tensor[t]), axis=0)
            else:
                self.xt_[t] = np.concatenate((self.ht[t - 1], self.input_tensor[t]), axis=0)


            self.ht[t] = self.TanH.forward(self.F1.forward(self.xt_[t].reshape(1, -1)))

            self.output_tensor[t] = self.Sigmoid.forward(self.F2.forward(self.ht[t].reshape(1, -1)))


        self.hidden_state = self.ht[self.time_steps - 1]

        return self.output_tensor


    def backward(self, error_tensor):

        dh = np.zeros((self.time_steps, self.hidden_size))
        dx = np.zeros((self.time_steps, self.input_size))
        dhx = np.zeros((self.time_steps, self.hidden_size+self.input_size))
        for t in range(self.time_steps-1, -1, -1):
            self.F2.input_tensor = self.ht[t].reshape(1, -1)
            dh[t] = self.F2.backward(self.Sigmoid.backward(error_tensor[t]))

            if t == self.time_steps-1:
                pass
            else:
                dh[t] += dh[t+1]

            self.F1.input_tensor = self.xt_[t].reshape(1, -1)
            dhx[t] = self.F1.backward(self.TanH.backward(dh[t]))
            dh[t] = dhx[t, 0:self.hidden_size]
            dx[t] = dhx[t, self.hidden_size:]

        self.gradient_weights = np.sum(dhx, axis=0)
        if self.optimizer is not None:
            self.weights = self._optimizer.calculate_update(self.weights, self.gradient_weights.reshape)

        return dx

    def calculate_regularization_loss(self):
        if self._optimizer.regularizer is not None:
            return self._optimizer.regularizer.norm(self.weights)




