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

        with open(label_path,"r") as js_file:
          self.labelfile = json.load(js_file)

        self.batch_taken = 0
        self.alldata = []
        self.label = []

        labelfile = self.labelfile
        length = len(self.imagedirs)
        print(length)
        arrlen = np.arange(0,length)
        for idx in arrlen:
            imgdata = np.load(self.file_path+str(idx)+".npy")
            imgdata = self.augment(imgdata)
            imgdata = np.resize(imgdata,self.image_size)
            self.alldata.append(imgdata)#put all data into a big array by the order of idx
            a = labelfile[str(idx)]#label of the correspond npy
            self.label.append(a)
        print(np.size(self.alldata))

        
        self.class_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog',
                           7: 'horse', 8: 'ship', 9: 'truck'}
        #TODO: implement constructor

    def next(self):
        # This function creates a batch of images and corresponding labels and returns them.
        # In this context a "batch" of images just means a bunch, say 10 images that are forwarded at once.
        # Note that your amount of total data might not be divisible without remainder with the batch_size.
        # Think about how to handle such cases
        #TODO: implement next method
        imagedata = []
        labeldata = []

        num = len(self.alldata)
        arr = np.arange(0,num)

        if(self.shuffle):
            np.random.shuffle(arr)
           
        
        if(self.batch_taken+self.batch_size<=num):
            idx_in_batch = np.arange(0,self.batch_size)
            for i in idx_in_batch:
                idx_in_arr = self.batch_taken+i
                pos = arr[idx_in_arr]
                imagedata.append(self.alldata[pos])
                labeldata.append(self.label[pos])
            self.batch_taken+=self.batch_size

            plt.figure()
            plt.imshow(self.alldata[56])
            plt.show()

        else:
            idx_before = num-self.batch_taken
            idx_in_batch = np.arange(0,idx_before)
            for i in idx_in_batch:
                idx_in_arr = self.batch_taken+i
                pos = arr[idx_in_arr] 
                imagedata.append(self.alldata[pos])
                labeldata.append(self.label[pos])
            print(np.shape(imagedata))
            idx_after = np.arange(0,self.batch_size-idx_before)
            tempimage =[]
            templabel =[]
            for i in idx_after:#continue with the beginning
                idx_in_arr = i
                pos = arr[idx_in_arr]
                tempimage.append(self.alldata[pos])
                templabel.append(self.label[pos])

            print(np.shape(tempimage))

            imagedata = np.concatenate([imagedata,tempimage])
            labeldata = np.concatenate([labeldata,templabel])

            self.batch_taken = self.batch_size-idx_before




            

        #if(self.batch_taken+self.batch_size>num):#make sure that length of each batch is the same
        #    index1 = arr_idx[self.batch_taken:num]
        #    num_before = num- self.batch_taken
        #    num_after = self.batch_size-num_before
        #    index2 = arr_idx[0:num_after]
        #    index_batch = np.concatenate((index1,index2))
        #    self.batch_taken = num_after
        #else:
        #    index_batch = arr_idx[self.batch_taken : self.batch_taken+self.batch_size]
        #    self.batch_taken += self.batch_size
        #    print(self.batch_taken)
        images = np.copy(imagedata)
        labels = np.copy(labeldata)
       
        return images, labels

    def augment(self,img):
        # this function takes a single image as an input and performs a random transformation
        # (mirroring and/or rotation) on it and outputs the transformed image
        #TODO: implement augmentation function
        a = random.random()
        b = random.random()
        if(self.rotation):
            if(0.75<a):
               img = np.rot90(img,3)
            elif(0.25<a<=0.5):
               img = np.rot90(img,1)
            elif(0.5<a<=0.75):
               img = np.rot90(img,2)
               
        if(self.mirroring):
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

        images,labels = self.next()

        plt.figure()
        lines = int(np.ceil(self.batch_size/4))

        for i,image in enumerate(images):

            label = self.class_name( int(labels[i]) )

            plt.subplot(lines , 4 , i+1)
            plt.imshow( image )
            plt.title(label)
        plt.show()

        return
