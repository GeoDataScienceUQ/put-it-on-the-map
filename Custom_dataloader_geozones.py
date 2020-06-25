import os
from skimage import transform
import cv2 as cv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
import time 
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class MapDataset(Dataset):

    def __init__(self, coordinateDf, root_dir, transform=None):
        """
        Args:
            coordinateDf (pd.DataFrame): DataFrame with image id and geographic coordinates.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.map_coordinates = coordinateDf
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.map_coordinates)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.map_coordinates.iloc[idx, 0] + '.png')
#         startTime = time.time()
        image = cv.imread(img_name)
#         print("Image reading time: ", time.time()-startTime)
        
        #startTime = time.time()
        ###image preprocessing
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        # image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        # ret,thresh_img = cv.threshold(image, 0, 255, cv.THRESH_BINARY_INV|cv.THRESH_OTSU)
        # kernel = cv.getStructuringElement(cv.MORPH_RECT, (4,8))
        # morph_img = cv.morphologyEx(thresh_img, cv.MORPH_CLOSE, kernel)
        #print("Image preprocessing time: ", time.time()-startTime) 

        if self.transform:
            image = self.transform(image)

        coordinates_crnr = np.array([self.map_coordinates['llcrnrlon'].iloc[idx], 
                            self.map_coordinates['llcrnrlat'].iloc[idx],
                            self.map_coordinates['urcrnrlon'].iloc[idx], 
                            self.map_coordinates['urcrnrlat'].iloc[idx]]).astype('float')
        
        sample = {'image': image,
                  'Geo_zone': self.map_coordinates['Geo_zone'].iloc[idx],
                  'image_path': img_name,
                  'coordinates_crnr': coordinates_crnr}

        return sample

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, image):

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size 
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        # image = transform.resize(image, (new_h, new_w), mode='reflect', anti_aliasing=True)
        image = cv.resize(image, (new_h, new_w))
        return image

class SquareRescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, image):
        
        image = cv.resize(image, (self.output_size, self.output_size))
        # image = transform.resize(image, (self.output_size, self.output_size), mode='reflect', anti_aliasing=True)

        return image


class CenterCrop(object):
    """Crop the image in a sample centered to the middle of the image.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, image):
        
        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        image = image[((h - new_h)//2): ((h - new_h)//2) + new_h,
                      ((w - new_w)//2): ((w - new_w)//2) + new_w]
        return image


class Normalize(object):
    """Normalize the image in a sample."""

    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def __call__(self, image):
        image = cv.normalize(image, None, alpha=self.alpha, beta=self.beta, 
                    norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
        return image


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, image):

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        # image = image.reshape(1, image.shape[0], image.shape[1])
        return torch.from_numpy(image)

