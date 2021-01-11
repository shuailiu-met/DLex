import numpy as np
from scipy import signal
import copy as copy
from Layers import Base


class Conv(Base.BaseLayer):
    def __init__(self, stride_shape, convolution_shape, num_kernels):
        self.stride_shape = stride_shape
        self.convolution_shape = convolution_shape
        self.num_kernels = num_kernels

        weights_shape = ((num_kernels, ) + convolution_shape)
        # for each kernel,we have convolution_shape weights,works for both 1d and 2d case
        self.weights = np.random.uniform(0, 1, weights_shape)
        self.bias = np.random.rand(num_kernels)  # Bias is an element-wise addition of a scalar value for each kernel.//bias added on each output layer with respect to kernel(one Kernel,one output layer)
        self._gradient_weights = None
        self._gradient_bias = None
        self._optimizer = None
        self._optimizer_forbias = None
        self.input_tensor = None

    # weights and bias separately
    @property
    def gradient_weights(self):
        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, gw_value):
        self._gradient_weights = gw_value

    @property
    def gradient_bias(self):
        return self._gradient_bias

    @gradient_bias.setter
    def gradient_bias(self, gb_value):
        self._gradient_bias = gb_value

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, _optimizer_obj):  # _optimizer is a object of type sgd
        #print(type(_optimizer_obj))
        self._optimizer = copy.deepcopy(_optimizer_obj)
        self._optimizer_forbias = copy.deepcopy(_optimizer_obj)

    # do corr.then stride,and put it in output array
    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        if (len(np.shape(input_tensor)) == 3):
            b, c, y = np.shape(input_tensor)
            # num_stride = int(np.ceil(y/self.stride_shape))
            temp_output = np.zeros((b, self.num_kernels, c, y))  # b k c y
            output_tensor = np.zeros((b, self.num_kernels, y))
            for batch in range(b):
                for kernel in range(self.num_kernels):
                    for channel in range(c):
                        data = input_tensor[batch, channel, :]
                        kernel_weights = self.weights[kernel, channel, :]
                        corr_channel = signal.correlate(data, kernel_weights, 'same')
                        temp_output[batch, kernel, channel, :] = corr_channel
                    sum_out = np.sum(temp_output, axis=2)  # sum along channels
                    output_tensor[batch, kernel, :] = sum_out[batch, kernel, :] + self.bias[kernel]

            stride_out = output_tensor[:, :, ::self.stride_shape[0]]

        elif (len(np.shape(input_tensor)) == 4):
            b, c, y, x = np.shape(input_tensor)
            # num_stride_y = int(np.ceil(y/self.stride_shape[0]))
            # num_stride_x = int(np.ceil(x/self.stride_shape[1]))
            temp_output = np.zeros((b, self.num_kernels, c, y, x))  # b k c y x
            output_tensor = np.zeros((b, self.num_kernels, y, x))
            for batch in range(b):
                for kernel in range(self.num_kernels):
                    for channel in range(c):
                        data = input_tensor[batch, channel, :, :]
                        kernel_weights = self.weights[kernel, channel, :, :]
                        corr_channel = signal.correlate(data, kernel_weights, 'same')
                        temp_output[batch, kernel, channel, :, :] = corr_channel
                    sum_out = np.sum(temp_output, axis=2)  # sum along channels
                    output_tensor[batch, kernel, :, :] = sum_out[batch, kernel, :, :] + self.bias[kernel]  # bias is only added once elementwise for each kernel

            stride_out = output_tensor[:, :, ::self.stride_shape[0], ::self.stride_shape[1]]
        return stride_out

    def backward(self, error_tensor):
        # in forward weight:k c n m shape
        # in backward: recombined as c k n m shape
        # we will have En-1 as shape of b c n m
        # update weights,bias and return the error of front
        # error tensor could be len 3 or 4
        # for gradient of weights, input needs to be padded

        #error_front = np.zeros(np.shape(self.input_tensor))

        #####upscaling error tensor to the shape without stride
        if (len(self.input_tensor.shape) == 4):
            stride_y, stride_x = self.stride_shape
            b, c, y, x = np.shape(self.input_tensor)
            upscaled_error = np.zeros((b, self.num_kernels, y, x))
            upscaled_error[:, :, ::stride_y, ::stride_x] = error_tensor
        else:
            stride_step = self.stride_shape[0]
            b, c, y = np.shape(self.input_tensor)
            upscaled_error = np.zeros((b, self.num_kernels, y))
            upscaled_error[:, :, ::stride_step] = error_tensor

        # flip over channel dim if 3d
        weight_for_err = self.weights
        if (self.input_tensor.shape == 4):
            weight_for_err = np.flip(weight_for_err, axis=1)

        # error front size b c y/ b c y x
        temp_error = np.zeros((np.shape(self.input_tensor)[0], np.shape(self.input_tensor)[1], self.num_kernels,
                               *np.shape(self.input_tensor)[2:]))
        for batch in range(b):
            for channel in range(c):
                for kernel in range(self.num_kernels):
                    err = upscaled_error[batch, kernel, :]  # actually convolve error with reordered channel_num kernels,   Tips:   :can also mean all dims after thies
                    weight_conv = weight_for_err[kernel, channel, :]
                    temp_error[batch, channel, kernel, :] = signal.convolve(err, weight_conv,'same')  # convolve error and weight to get front error (sum along kernel dim(which is channels of error))
                    # former kernel dim is now the channels of err
        error_front = np.sum(temp_error, axis=2)  # sum along kernel dim(error's channel dim)

        # pad input for deltaW
        if (len(self.input_tensor.shape) == 4):  # 3d
            k, c, m, n = np.shape(self.weights)
            b, c, y, x = np.shape(self.input_tensor)
            pad_input = np.zeros((b, c, y + m - 1, x + n - 1))
            for batch in range(b):
                for channel in range(c):
                    pad_input[batch, channel, int(m / 2):(y + int(m / 2)), int(n / 2):(x + int(n / 2))] = self.input_tensor[batch, channel, :, :]  # the first m/2 elemnts and the last m/2 will keep 0
        else:  # 2d
            k, c, n = np.shape(self.weights)
            b, c, y = np.shape(self.input_tensor)
            pad_input = np.zeros((b, c, y + n - 1))  # n is the kernel wid
            for batch in range(b):
                for channel in range(c):
                    pad_input[batch, channel, int(n / 2):(y + int(n / 2))] = self.input_tensor[batch, channel, :]

        # correlate padded X and En to get gradient of W
        # calculate gradient of weights
        self.gradient_weights = np.zeros(np.shape(self.weights))  # k c n/ k c n m
        temp_gradient = np.zeros((b, *np.shape(self.weights)))  # b k c n/
        for batch in range(b):
            for kernel in range(k):
                for channel in range(c):
                    Xs = pad_input[batch, channel, :]
                    E_hn = upscaled_error[batch, kernel, :]
                    temp_gradient[batch, kernel, channel, :] = signal.correlate(Xs, E_hn,mode='valid')  # valid here, due to padding, delta needs to sum along batch dim
        self.gradient_weights = np.sum(temp_gradient, axis=0)  # sum along batch dim

        # calculate gradient of bias
        self.gradient_bias = np.zeros(self.num_kernels)
        b = np.shape(error_tensor)[0]
        k = np.shape(error_tensor)[1]  # kernal num
        for batch in range(b):
            for kernel in range(k):
                self.gradient_bias[kernel] += np.sum(error_tensor[batch, kernel, :])  # sum of elements error

        # calculate weights and bias
        if self.optimizer is not None:
            self.weights = self._optimizer.calculate_update(self.weights, self.gradient_weights)
            self.bias = self._optimizer_forbias.calculate_update(self.bias, self.gradient_bias)

        return error_front

    # initialization in conv needs to change fan_in and fan__out
    def initialize(self, weights_initializer, bias_initializer):
        # fan_in = # input channels × kernel height × kernel width
        # conv shape c m n
        input_channels = self.convolution_shape[0]
        kernel_height = self.convolution_shape[1]
        kernel_width = self.convolution_shape[2]
        fan_in = input_channels * kernel_height * kernel_width
        # fan_out = # output channels × kernel height × kernel width
        output_channels = self.num_kernels
        fan_out = output_channels * kernel_height * kernel_width

        self.weights = weights_initializer.initialize(np.shape(self.weights), fan_in, fan_out)
        self.bias = bias_initializer.initialize(np.shape(self.bias), fan_in, fan_out)
