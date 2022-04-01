import unittest
import torch as t
import pandas as pd
from skimage.io import imread
from skimage.color import gray2rgb
import numpy as np
import os
import torchvision as tv



check_path = 'checkpoints/'
dir = os.listdir(check_path)
for file in dir:
    print(file)
