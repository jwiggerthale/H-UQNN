#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  3 06:38:20 2025

@author: jwiggerthale
"""

from torch.utils.data import Dataset, DataLoader
import torch
import torchvision.transforms as transforms

import numpy as np
import pandas as pd
import os


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_cifar(fp: str = '/data/Uncertainty/cifar_modeling/cifar-10-python/cifar-10-batches-py'):
    data_files = os.listdir(fp)
    data_files = [f'{fp}/{f}' for f in data_files if '_batch' in f]
    
    train_ims = {b'labels': [], 
                 b'data': []}
    test_ims = {}
    for f in data_files:
        new_ims = unpickle(f)
        if ('test' in f):
            test_ims.update(new_ims)
        else:
            train_ims[b'labels'].extend(new_ims[b'labels'])
            train_ims[b'data'].extend(new_ims[b'data'])
        
    return train_ims, test_ims


#class to convert images to certain model (e.g. RGB)
class Convert:
    def __init__(self, mode='RGB'):
        self.mode = mode

    def __call__(self, image):
        return image.convert(self.mode)

'''
dataset for cifar images
transformations defined accoring to the dataset
call with: 
  data: dict --> data with images and labels
  num_ims --> number of images for ds
'''
class cifar_ds(Dataset):
    def __init__(self, 
                 data: dict, 
                 num_ims: int = 15000):
        self.labels = data[b'labels'][:num_ims]
        self.ims = data[b'data'][:num_ims]
        self.transforms = transforms.Compose([transforms.ToPILImage(),
                                            Convert('RGB'),
                                            transforms.Resize(32, interpolation=transforms.InterpolationMode.BILINEAR),
                                            transforms.CenterCrop(32),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]),
                                        ])
                                        
    def __len__(self):
        return(len(self.ims))
    def __getitem__(self, idx):
        im = self.ims[idx].astype(np.float32)
        channels = [im[:1024].reshape(32, 32), im[1024:2048].reshape(32, 32), im[2048:].reshape(32, 32)]
        my_im = np.stack(channels)
        my_im = np.transpose(my_im, axes=(1, 2, 0))
        my_im = self.transforms(my_im)
        label = self.labels[idx]
        return(my_im, label)
    

'''
dataset for mnist images
call with: 
  df: pd.DataFrame --> image data + labels in data frame (last column is label)
  noisy_classes: list --> list of classes to return with noise
'''
class mnist_dataset(Dataset): 
    def __init__(self, 
                 df: pd.DataFrame, 
                 noisy_classes: list = []):
        self.data = df
        self.transforms = transforms.Compose([transforms.ToPILImage(),
                          #transforms.RandomHorizontalFlip(p=0.5),
                          #transforms.RandomRotation(degrees=15),
                          #transforms.ColorJitter(brightness=0.3, contrast=0.3),
                          transforms.ToTensor(),
                          transforms.Normalize(mean=[ 0.1307],std=[0.3081]),
                          ])
        self.noisy_classes = noisy_classes
    def __len__(self):
        return(len(self.data))
    def __getitem__(self, idx):
        im = self.data[idx][:-1]
        im = np.array(im, dtype = np.uint8).reshape((28, 28, 1))
        im = self.transforms(im)
        im += 0.5
        label = self.data[idx][-1]
        if label in self.noisy_classes:
            z = torch.randn_like(im) * np.random.rand()
            im += z
        return(im, label)
    

'''
function to get train and test loader for mnist 
call with: 
  train_path: str --> path to train data (.csv file)
 test_path: str = None --> path to test data (.csv file)
 classes_to_neglect: list = [] --> classes to exclude from training
 classes_to_neglect_test: list = [] --> classes to exclude from testing 
 noisy_classes: list = None --> classes to return with noise added
 train_size: int = 10000 --> number of training samples
 test_size: int = 1000 --> number of test samples
 batch_size: int = 16 --> batch size to use
'''
def get_minst_dataloader(train_path: str,
                         test_path: str = None,
                         classes_to_neglect: list = [], 
                         classes_to_neglect_test: list = [], 
                         noisy_classes: list = None,
                         train_size: int = 10000, 
                         test_size: int = 1000, 
                         batch_size: int = 16):
    train = pd.read_csv(train_path).drop('Unnamed: 0', axis = 1)
    train_samples = []
    
    classes_to_train = np.arange(10)
    classes_to_train = [c for c in classes_to_train if c not in classes_to_neglect]
    
    classes_to_test = np.arange(10)
    classes_to_test = [c for c in classes_to_test if c not in classes_to_neglect_test]
    samples_per_class = int(train_size / len(classes_to_train))
    for c in classes_to_train:
        rows = []
        for i in range(len(train)):
            row = list(train.iloc[i, :])
            if row[-1] == c:
                rows.append(row)
            if len(rows) >= samples_per_class:
                break
        train_samples.extend(rows)        
        
    train_set = mnist_dataset(train_samples, noisy_classes = noisy_classes)
    train_loader = DataLoader(train_set, 
                              batch_size = batch_size, 
                              shuffle = True, 
                              drop_last = True)
    
    
    if test_path != None and test_size > 0:
        test = pd.read_csv(test_path).drop('Unnamed: 0', axis = 1)
        test_samples = []
        
        samples_per_class = int(test_size / len(classes_to_test))
        for c in classes_to_test:
            rows = []
            for i in range(len(test)):
                row = list(test.iloc[i, :])
                if row[-1] == c:
                    rows.append(row)
                if len(rows) >= samples_per_class:
                    break
            test_samples.extend(rows)
            
        test_set = mnist_dataset(test_samples)
        test_loader = DataLoader(test_set, batch_size = batch_size, shuffle = True, drop_last = True)    
     
        return train_loader, test_loader
    return train_loader
