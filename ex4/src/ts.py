import unittest
import torch as t
import pandas as pd
from skimage.io import imread
from skimage.color import gray2rgb
import numpy as np
import os
import torchvision as tv


# locate the csv file in file system and read it
csv_path = ''
for root, _, files in os.walk('.'):
    for name in files:
        if name == 'data.csv':
            csv_path = os.path.join(root, name)
tab = pd.read_csv(csv_path, sep=';')
path = tab.filename[23]
img = gray2rgb(imread(path))
print(np.shape(img))

transform = tv.transforms.Compose([tv.transforms.ToPILImage(), tv.transforms.RandomVerticalFlip(0.5), tv.transforms.ToTensor()])

imgtrans = transform(img)
print(type(imgtrans))
print(imgtrans.shape)
#width, height = imgtrans.size
#print(width)
#print(height)

x = np.zeros((1))
y = np.ones((1))
w = np.concatenate((x, y))

z = t.tensor([0, 1], dtype=t.float)
print(z)
print(type(z))
print(z[0])
k = t.from_numpy(w)
print(k)