import numpy as np

class Pooling:
    def __init__(self,stride_shape,pooling_shape):
        self.stride_shape = stride_shape # 2 number
        self.pooling_shape = pooling_shape # 2 number
        self.input_tensor = None
        self.coord_x = None
        self.coord_y = None

    def forward(self,input_tensor):
        # stride a b
        # pool size n m
        # input tensor b c y x
        self.input_tensor = input_tensor
        b,c,y,x = np.shape(input_tensor)
        pooling_y = self.pooling_shape[0]
        pooling_x = self.pooling_shape[1]
        out_y = int((y-y%pooling_y)/self.stride_shape[0]) #remove the part which is not large enough for one more pooling
        out_x = int((x-x%pooling_x)/self.stride_shape[1]) #remove the part which is not large enough for one more pooling
        output = np.zeros((b,c,out_y,out_x))
        self.coord_y = np.zeros((b,c,out_y,out_x))
        self.coord_x = np.zeros((b,c,out_y,out_x))


        for batch in range(b):
            for channel in range(c):
                for out_y_idx in range(out_y):
                    for out_x_idx in range(out_x):
                        pool_origin_y = self.stride_shape[0]*out_y_idx
                        pool_origin_x = self.stride_shape[1]*out_x_idx
                        temp_array = input_tensor[batch,channel,pool_origin_y:(pool_origin_y+pooling_y),pool_origin_x:(pool_origin_x+pooling_x)]
                        max_value = np.max(temp_array)
                        output[batch,channel,out_y_idx,out_x_idx] = max_value

                        temp_coord_y,temp_coord_x = np.where(temp_array==max_value) #take the 0 index(there might be some idx with same value)

                        real_coord_y = pool_origin_y+temp_coord_y[0]
                        real_coord_x = pool_origin_x+temp_coord_x[0]
                        self.coord_y[batch,channel,out_y_idx,out_x_idx] = real_coord_y
                        self.coord_x[batch,channel,out_y_idx,out_x_idx] = real_coord_x
        return output
            


    def backward(self,error_tensor):
        #return a tensor with former shape and with values on corresponding postition and 0
        output_error = np.zeros(np.shape(self.input_tensor)) # b c y x
        b,c,out_y,out_x = np.shape(self.coord_x)
        for batch in range(b):
            for channel in range(c):
                for out_y_idx in range(out_y):
                    for out_x_idx in range(out_x):
                        co_y = int(self.coord_y[batch,channel,out_y_idx,out_x_idx])
                        co_x = int(self.coord_x[batch,channel,out_y_idx,out_x_idx])
                        value = error_tensor[batch,channel,out_y_idx,out_x_idx]
                        output_error[batch,channel,co_y,co_x] += value   #some overlaping points have done contribution to 2 lapping,so the error tensor value is sumed for this point

        return output_error