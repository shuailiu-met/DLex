from torch.utils.data import Dataset
import torch
from pathlib import Path
from skimage.io import imread
from skimage.color import gray2rgb
import numpy as np
import torchvision as tv

train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]  # std and mean are given here


##the datacsv is shaped in name : crack, inactive    crack and inactive are bool values 0/1

class ChallengeDataset(Dataset):
    # TODO implement the Dataset class according to the description
    def __init__(self, data, mode):
        Dataset.__init__(self)
        self.data = data
        self.mode = mode
        if self.mode == 'val':
            self._transform = tv.transforms.Compose([
                tv.transforms.ToPILImage(), tv.transforms.ToTensor(),
                tv.transforms.Normalize(train_mean, train_std)  # first turn nd array to PILimg, then tensor
            ])
        else:  # train
            self._transform = tv.transforms.Compose([
                tv.transforms.ToPILImage(), tv.transforms.RandomVerticalFlip(0.5), tv.transforms.RandomHorizontalFlip(0.5),
                tv.transforms.ToTensor(), tv.transforms.Normalize(train_mean, train_std)
                # first turn nd array to PILimg, then tensor, during training, give more transforms
            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        path = self.data.filename[index]
        img_gray = imread(path)
        img_rgb = gray2rgb(img_gray)
        img_trans = self._transform(img_rgb)
        crack = self.data.crack[index]
        inactive = self.data.inactive[index]

        return img_trans, torch.tensor([crack, inactive])
