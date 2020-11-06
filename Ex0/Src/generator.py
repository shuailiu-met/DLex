import os.path
import json
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt
import random

# In this exercise task you will implement an image generator. Generator objects in python are defined as having a next function.
# This next function returns the next generated object. In our case it returns the input of a neural network each time it gets called.
# This input consists of a batch of images and its corresponding labels.
class ImageGenerator:
    def __init__(self, file_path, label_path, batch_size, image_size, rotation=False, mirroring=False, shuffle=False):
        # Define all members of your generator class object as global members here.
        # These need to include:
        # the batch size
        # the image size
        # flags for different augmentations and whether the data should be shuffled for each epoch
        # Also depending on the size of your data-set you can consider loading all images into memory here already.
        # The labels are stored in json format and can be directly loaded as dictionary.
        # Note that the file names correspond to the dicts of the label dictionary.
        self.batch_size = batch_size
        self.image_size = image_size
        self.rotation = rotation
        self.mirroring = mirroring
        self.shuffle = shuffle

        self.file_path = file_path
        self.imagedirs = os.listdir(self.file_path)

        #for file in self.imagedirs:
         # self.imagedata = np.load(self.file_path + file)

        self.label_path = label_path
        with open(self.label_path) as js_file:
          self.labels = json.load(js_file)

        self.batch_taken = 0

        self.class_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog',
                           7: 'horse', 8: 'ship', 9: 'truck'}
        #TODO: implement constructor

    def next(self):
        # This function creates a batch of images and corresponding labels and returns them.
        # In this context a "batch" of images just means a bunch, say 10 images that are forwarded at once.
        # Note that your amount of total data might not be divisible without remainder with the batch_size.
        # Think about how to handle such cases
        #TODO: implement next method
        num = len(self.imagedirs)
        arr = np.arange(0,num)

        if(self.shuffle):
            arr = np.random.shuffle(arr)#randomly distribute 100, then we can get different batch and use the number to get the file since files are named as number.npy
        
        if(self.batch_taken+self.batch_size>num):#make sure that length of each batch is the same
            index1 = arr[self.batch_taken:num]
            num_before = num- self.batch_taken
            num_after = self.batch_size-num_before
            index2 = arr[0:num_after]
            index_batch = np.concatenate((index1,index2))
        else:
            index_batch = arr[self.batch_taken : self.batch_taken+self.batch_size]
            self.batch_taken += self.batch_size


        imgfile = []
        a = random.random()
        for i in index_batch:
          img = np.load(self.file_path+str(i)+".npy")
          img = np.resize(img,self.image_size)

          if(self.rotation):
              if(a<=0.25):
                  img = np.rot90(img,3)
              elif(0.25<a<=0.5):
                  img = np.rot90(img,2)
              elif(0.5<a<=0.75):
                  img = np.rot90(img,1)

          if(self.mirroring):
              b = random.random()
              if(b<0.5):
                  img = np.flip(img,axis=0)

          imgfile.append(img)

        with open(self.label_path) as js_file:
            labels_total = json.load(js_file)
        
        label_for_img = []
        for i in index_batch:
            label_for_img.append(labels_total[str(i)])#jsonfile[string] will return the class/label value of the str

        images = np.copy(imgfile)
        labels = np.copy(label_for_img)

        return images, labels

    def augment(self,img):
        # this function takes a single image as an input and performs a random transformation
        # (mirroring and/or rotation) on it and outputs the transformed image
        #TODO: implement augmentation function
        a = random.random()
        b = random.random()

        if(a<=0.25):
            img = np.rot90(img,3)
        elif(0.25<a<=0.5):
            img = np.rot90(img,2)
        elif(0.5<a<=0.75):
            img = np.rot90(img,1)

        if(b<0.5):
            img = np.flip(img,axis=0)

        return img

    def class_name(self, x):
        # This function returns the class name for a specific input
        #TODO: implement class name function
        return self.class_dict[x]

    def show(self):
        # In order to verify that the generator creates batches as required, this functions calls next to get a
        # batch of images and labels and visualizes it.
        #TODO: implement show method
        return
