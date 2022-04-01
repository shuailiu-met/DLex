import numpy as np

x1 = np.zeros((2,3,2))
print(x1.shape())
x2 = x1[:,:,1]
print(x2.shape())