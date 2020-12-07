import numpy as np
#softmax activation

class SoftMax():
    def __init__(self):
        #self.probability_tensor = None
        pass

    def forward(self,input_tensor):#xk has to be shifted,input is 2d tensor,batch size as row number
        #print(np.shape(input_tensor)) #(9*4)
        #formula shows prediction for each sample in the batch,now we have row_numbers samples,each sample has a 1d array
        #actually estimate a prob_vector for each row(each sample)

        max_of_each_element = np.expand_dims(np.max(input_tensor,axis = 1),axis=1)
        #compute max for each row(each element),we get a (9,) array, expand on axis 1,so that we have a (9,1) array for further opt
        input_hat = input_tensor - max_of_each_element#on each row,x-max(each row),stablized input
        y_hat = np.exp(input_hat)/np.expand_dims(np.sum(np.exp(input_hat),axis = 1),axis = 1)#np.sum returns a array of (9,),we expand dim to (9,1) so it can be divided by(9,4)
        #print(np.shape(y_hat))
        self.probability_tensor = y_hat
        #print(np.sum(y_hat)) sum = 9,for each sample we do a softmax 
        return self.probability_tensor



    def backward(self,error_tensor):#error tensor here is from loss.py
        #print(np.shape(error_tensor))#(9*4)
        #yhat(9*4) scalar product
        scalar_product = error_tensor*self.probability_tensor
        sum_scalar_product = np.expand_dims(np.sum(scalar_product,axis = 1),axis = 1) #(9x1)array
        En_1 = self.probability_tensor*(error_tensor-sum_scalar_product)#(9,4)array - (9,1)array means all elements of each row - corresponding number
        return En_1
