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
            pattern0 = np.zeros((self.tile_size,self.tile_size))#black tile
            pattern1 = np.ones((self.tile_size,self.tile_size))#white tile
            pattern01 = np.concatenate((pattern0,pattern1),axis = 0)#black + white
            pattern10 = np.concatenate((pattern1,pattern0),axis = 0)#white + black
            pattern_single = np.concatenate((pattern01,pattern10),axis = 1)#combine
            pattern_total = np.tile(pattern_single,(times,times))#copy
            self.output = pattern_total
            return np.copy(self.output)

    def show(self):
        plt.figure()
        plt.imshow(self.output,cmap='gray')
        plt.title("checkerboard")
        plt.show()




class Circle():
    def __init__(self,resolution,radius,position=(1,1)):
        self.resolution = resolution
        self.radius = radius
        self.position = position
        self.output = np.zeros((resolution,resolution))

    def draw(self):
        pos = self.position
        x = np.arange(0,self.resolution)
        y = np.arange(0,self.resolution)
        nx,ny = np.meshgrid(x,y)#form a meshgrid by x and y
        cir = np.sqrt((nx-pos[0])**2+(ny-pos[1])**2)#cir is a matrix which stores the distance to center at each grid point,size same as the grid size,according to nx ny
        out = self.output
        out[cir <= self.radius] = 1#for the position which cir meets condition, out = 1
        self.output = out
        return np.copy(out) 

    def show(self):
        plt.figure(1)
        plt.imshow(self.output,cmap='gray')
        plt.title("circle")
        plt.show()



class Spectrum():
    def __init__(self,resolution):
        self.resolution = resolution
        self.output = np.zeros((resolution,resolution,3))#3d by RGB,3rd dimension-channel,12dimension- pos,array of r*r*3
    

    def draw(self):
        R = np.arange(0,1.0,1/self.resolution)
        G = np.transpose(R)
        B = np.flip(R)
     

        out = self.output
        out[:,:,0] = np.tile(R,(self.resolution,1)) #make every row's R channel to be like this,when channel-dim = 0(R),we calculate a matrix which gives R values for each point
        out[:,:,1] = np.transpose(out[:,:,0])
        out[:,:,2] = np.tile(B,(self.resolution,1))

        self.output = out
        return np.copy(self.output)

    def show(self):
        plt.figure(1)
        plt.imshow(self.output)
        plt.title("spectrum")
        plt.show()





         





