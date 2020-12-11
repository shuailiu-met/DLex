import numpy as np
from scipy import signal



class Conv:
    def __init__(self,stride_shape,convolution_shape,num_kernels):
        self.stride_shape = stride_shape
        self.convolution_shape = convolution_shape 
        self.num_kernels = num_kernels

        weights_shape = ((num_kernels,)+convolution_shape)
        #for each kernel,we have convolution_shape weights,works for both 1d and 2d case
        self.weights = np.random.uniform(0,1,weights_shape)
        self.bias = np.random.rand(num_kernels)   #Bias is an element-wise addition of a scalar value for each kernel.//bias added on each output layer with respect to kernel(one Kernel,one output layer) 
        self._gradient_weights = np.zeros(weights_shape)
        self._gradient_bias = np.zeros(num_kernels)
        self._optimizer = None
        self.input_tensor = None


    #weights and bias separately
    @property
    def gradient_weights(self):
        return self._gradient_weights
    
    @gradient_weights.setter
    def gradient_weights(self,gw_value):
        self._gradient_weights = gw_value

    @property
    def gradient_bias(self):
        return self._gradient_bias
    
    @gradient_bias.setter
    def gradient_vuas(self,gb_value):
        self._gradient_bias = gb_value   

    @property
    def optimizer(self):
        return self._optimizer
    
    @optimizer.setter
    def optimizer(self,_optimizer_obj):#_optimizer is a object of type sgd
        self._optimizer = _optimizer_obj


    #do corr.then stride,and put it in output array
    def forward(self,input_tensor):
        self.input_tensor = input_tensor
        if(len(np.shape(input_tensor))==3):
            b,c,y = np.shape(input_tensor)
           # num_stride = int(np.ceil(y/self.stride_shape))
            temp_output = np.zeros((b,self.num_kernels,c,y))# b k c y
            output_tensor = np.zeros((b,self.num_kernels,y))
            for batch in range(b):
                for kernel in range(self.num_kernels):
                    for channel in range(c):
                        data = input_tensor[batch,channel,:]
                        kernel_weights = self.weights[kernel,channel,:]
                        corr_channel = signal.correlate(data,kernel_weights,'same')
                        temp_output[batch,kernel,channel,:] = corr_channel
                    sum_out = np.sum(temp_output,axis = 2)             #sum over channels
                    output_tensor[batch,kernel,:] = sum_out[batch,kernel,:]+ self.bias[kernel]                
            stride_out = output_tensor[:,:,::self.stride_shape[0]]

        elif(len(np.shape(input_tensor))==4):
            b,c,y,x = np.shape(input_tensor)
            #num_stride_y = int(np.ceil(y/self.stride_shape[0]))
            #num_stride_x = int(np.ceil(x/self.stride_shape[1]))
            temp_output = np.zeros((b,self.num_kernels,c,y,x))# b k c y x
            output_tensor = np.zeros((b,self.num_kernels,y,x))
            for batch in range(b):
                for kernel in range(self.num_kernels):
                    for channel in range(c):
                        data = input_tensor[batch,channel,:,:]
                        kernel_weights = self.weights[kernel,channel,:,:]
                        corr_channel = signal.correlate(data,kernel_weights,'same')
                        temp_output[batch,kernel,channel,:,:] = corr_channel
                    sum_out = np.sum(temp_output,axis = 2)            #sum over channels
                    output_tensor[batch,kernel,:,:] = sum_out[batch,kernel,:,:]+ self.bias[kernel] #bias is only added once elementwise for each kernel
            stride_out = output_tensor[:,:,::self.stride_shape[0],::self.stride_shape[1]]
        return stride_out

    def backward(self,error_tensor):
        #in forward weight:k c n m shape
        #in backward: recombined as c k n m shape
        #we will have En-1 as shape of b c n m  
        #update weights,bias and return the error of front
        #errortensor could be len 3 or 4
        #for gradient of weights, input needs to be padded


        #######error tensor of the fronter
        weights_back = np.swapaxes(self.weights,0,1)# rearrange the weights to channel_num kernels , c k n/c k n m

        if(len(np.shape(error_tensor))==3):#2d case     b k n     c kernels  k is the channel number of E
            num_channels = np.shape(self.input_tensor)[1]
            b,k,n = np.shape(error_tensor)
            #upscale first 
            upscaled_error = np.zeros((b,k,np.shape(self.input_tensor)[2]))
            stride_y = self.stride_shape[0]
            upscaled_error[:,:,::stride_y] = error_tensor # b k y , the rest is 0
            error_front = np.zeros((b,num_channels,np.shape(self.input_tensor)[2])) #should be b c y
            temp_error = np.zeros((b,num_channels,k,np.shape(self.input_tensor)[2]))# b c k y sum over k

            for batch in range(b):
                for new_k in range(num_channels):
                    for new_c in range(k):
                        error_channel = upscaled_error[batch,new_c,:]
                        conv_kernel = weights_back[new_k,new_c,:]
                        conv_channel = signal.convolve(error_channel,conv_kernel,'same')
                        temp_error[batch,new_k,new_c,:] = conv_channel
            error_front = np.sum(temp_error,axis=2)


        elif(len(np.shape(error_tensor))==4):# 3d case b k m n
            num_channels = np.shape(self.input_tensor)[1]
            b,k,n,m = np.shape(error_tensor)
            #upscale first 
            upscaled_error = np.zeros((b,k,np.shape(self.input_tensor)[2],np.shape(self.input_tensor)[3]))
            stride_y = self.stride_shape[0]
            stride_x = self.stride_shape[1]
            upscaled_error[:,:,::stride_y,::stride_x] = error_tensor # b k y x
            #flip once on channel dim
            upscaled_error = np.flip(upscaled_error,axis = 1)

            error_front = np.zeros((b,num_channels,np.shape(self.input_tensor)[2],np.shape(self.input_tensor)[3])) #should be b c y
            temp_error = np.zeros((b,num_channels,k,np.shape(self.input_tensor)[2],np.shape(self.input_tensor)[3]))# b c k y sum over k

            for batch in range(b):
                for new_k in range(num_channels):
                    for new_c in range(k):
                        error_channel = upscaled_error[batch,new_c,:,:]
                        conv_kernel = weights_back[new_k,new_c,:,:]
                        conv_channel = signal.convolve(error_channel,conv_kernel,'same')
                        temp_error[batch,new_k,new_c,:,:] = conv_channel
            error_front = np.sum(temp_error,axis=2)


        ############ update weights and bias
        # first, pad input_tensor



          
        return error_front




    #initialization in conv needs to change fan_in and fan__out
    def initialize(self,weights_initializer,bias_initializer):
        #fan_in = # input channels × kernel height × kernel width
        # conv shape c m n
        input_channels = self.convolution_shape[0]
        kernel_height = self.convolution_shape[1]
        kernel_width = self.convolution_shape[2]
        fan_in = input_channels*kernel_height*kernel_width
        #fan_out = # output channels × kernel height × kernel width
        output_channels = self.num_kernels
        fan_out = output_channels*kernel_height*kernel_width

        self.weights = weights_initializer.initialize(np.shape(self.weights),fan_in,fan_out)
        self.bias = bias_initializer.initialize(np.shape(self.bias),fan_in,fan_out)