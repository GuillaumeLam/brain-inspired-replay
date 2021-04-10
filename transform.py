#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import numpy as np
import torch
import torchvision
import argparse
from torchvision import transforms, models, datasets
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image
from sklearn.model_selection import train_test_split
from scipy import ndimage
from skimage.util import random_noise

# Class that applies Gaussian Blurring 
class GaussianBlur():
    def __init__(self, kernel_size, sigma_min=0.1, sigma_max=2.0):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.kernel_size = kernel_size

    def __call__(self, img):
        sigma = np.random.uniform(self.sigma_min, self.sigma_max)
        img = cv2.GaussianBlur(np.array(img), (self.kernel_size, self.kernel_size), sigma)
        return Image.fromarray(img.astype(np.uint8))

# Class that applies Gaussian Noise     
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)    

# Choosing dataset
if dataset_name == 'mnist':  
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), 
                                        torchvision.transforms.Normalize((0.5,), (0.5,)),
                                        transforms.RandomApply([GaussianBlur(kernel_size=23)], p=0.1),
                                        transforms.RandomApply([AddGaussianNoise(args.mean, args.std)], p=0.3), 
                                        transforms.ColorJitter(brightness=0.4, contrast=0.5, saturation=0.3, hue=0.2),

                                       ])  

if dataset_name == 'cifar10':
    transform = transforms.Compose([torchvision.transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                    transforms.RandomApply([GaussianBlur(kernel_size=23)], p=0.1),
                                    transforms.RandomApply([AddGaussianNoise(args.mean, args.std)], p=0.5), 
                                    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
                                    
                                   ])
  
