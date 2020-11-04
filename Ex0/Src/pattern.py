import numpy as np
import matplotlib.pyplot as plt

class Checker():
    def __init__(self,resolution,tile_size):
        self.resolution = resolution
        self.tile_size = tile_size
        self.output = np.zeros((resolution,resolution))



    def draw(self):
        if self.resolution%(2*self.tile_size) != 0 :
            print ("tile size not right")
        else :
            times = int(self.resolution/(2*self.tile_size))
            print (times)
            pattern0 = np.zeros((self.tile_size,self.tile_size))
            pattern1 = np.ones((self.tile_size,self.tile_size))
            pattern01 = np.concatenate((pattern0,pattern1),axis = 0)
            pattern10 = np.concatenate((pattern1,pattern0),axis = 0)
            pattern_single = np.concatenate((pattern01,pattern10),axis = 1)
            pattern_total = np.tile(pattern_single,(times,times))
            self.output = pattern_total
            return np.copy(self.output)

    def show(self):
        plt.imshow(self.output,cmap='gray')
        plt.show()



