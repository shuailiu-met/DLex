import numpy as np
import copy as copy
from .TanH import TanH
from .Base import BaseLayer
from .Sigmoid import Sigmoid


class RNN(BaseLayer):
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.ht0 = np.zeros((self.hidden_size))

        self.TanH = TanH()
        self.Sigmoid = Sigmoid()

        self._memorize = False
        self._gradient_weights = None
        self._weights = None

        self._optimizer = None
        self._optimizer_forbias = None

    @property
    def memorize(self):
        return self._memorize

    @memorize.setter
    def memorize(self, value):
        self._memorize = value

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, _optimizer_obj):  # _optimizer is a object of type sgd
        self._optimizer = copy.deepcopy(_optimizer_obj)
        self._optimizer_forbias = copy.deepcopy(_optimizer_obj)

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        #print(np.shape(input_tensor))
        self.time_steps = np.shape(input_tensor)[0]

        self.Wh = np.ones((self.time_steps, self.hidden_size, self.hidden_size + self.input_size + 1))
        self.Wy = np.ones((self.time_steps, self.output_size, self.hidden_size +1))
        self.hidden_state = np.zeros((self.time_steps, self.hidden_size))

        self.xt_ = np.ones((self.time_steps, self.hidden_size+self.input_size+1))

        if self.memorize == False:
            self.ht0 = np.zeros((self.hidden_size))

        self.output_tensor = np.zeros((self.time_steps, self.output_size))

        for t in range(self.time_steps):
            if t == 0:
                self.xt_[t] = np.concatenate((self.ht0, self.input_tensor[t], [1]),axis=0)
            else:
                self.xt_[t] = np.concatenate((self.hidden_state[t - 1], self.input_tensor[t], [1]),axis=0)


            self.hidden_state[t] = self.TanH.forward(np.dot(self.xt_[t], self.Wh[t].T))

            ht_ = np.concatenate((self.hidden_state[t], [1]), axis=0)
            #print(np.shape(ht_))
            #print(np.shape(self.Wy[t].T))


            self.output_tensor[t] = self.Sigmoid.forward(np.dot(ht_, self.Wy[t].T))
           # print(np.shape(np.dot(ht_, self.Wy[t].T)))


        self.ht0 = self.hidden_state[self.time_steps - 1]
        print(np.shape(self.output_tensor))

        return self.output_tensor

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

    def backward(self, error_tensor):
        print(np.shape(error_tensor))
        #print(np.shape(self.hidden_state))
        dh_next = np.zeros(self.hidden_size)
        dWh, dWy = np.ones((self.time_steps,  self.hidden_size + self.input_size, self.hidden_size,)), np.ones((self.time_steps, self.output_size, self.hidden_size))
        for t in range(self.time_steps-1, -1, -1):
            dy = self.Sigmoid.backward(error_tensor[t])
            dWy[t] = np.dot(dy.reshape(-1,1), self.hidden_state[t].reshape(1,-1))
            #print(np.shape(dy.reshape(1,-1)))
           # print(np.shape(self.Wy[t, :, 0:self.hidden_size].T))
            dh = np.dot(self.Wy[t, :, 0:self.hidden_size].T, dy) + dh_next

            dh *= self.TanH.backward(self.hidden_state[t])
            dWh[t] = np.dot(self.xt_[t, :self.hidden_size+self.input_size].reshape(-1,1), dh.reshape(1,-1))

           # print(np.shape(self.Wh[t, :, 0:self.hidden_size].T))
           # print(np.shape(dh.reshape(1,-1)))
            dh_next = np.dot(self.Wh[t, :, 0:self.hidden_size].T, dh)


        self. gradient_weights = dWh
        if self.optimizer is not None:
            self.Wh = self._optimizer.calculate_update(self.Wh, self.gradient_weights)
            self.Wy = self._optimizer.calculate_update(self.Wy, dWy)

        self. weights = self.Wh[:, :, self.hidden_size::self.input_size]
        print(np.shape(self.gradient_weights[:,self.hidden_size:self.hidden_size+self.input_size, :]))

        return self.gradient_weights[:,self.hidden_size:self.hidden_size+self.input_size, :]




