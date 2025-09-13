#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 20 18:31:05 2025

@author: jwiggerthale
"""

'''
this script implements a test of H-UQNN on noise vs MNIST-images as described in Sec. 4B of our paper
'''


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from HUQNN import HUQNN
import pandas as pd
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
 


class mnist_dataset(Dataset): 
    def __init__(self, df: pd.DataFrame):
        self.data = df
        self.transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize(mean=[ 0.1307],std=[0.3081]),
                             ])
    def __len__(self):
        return(len(self.data))
    def __getitem__(self, idx):
        im = self.data.iloc[idx, :-1]
        im = np.array(im).reshape((28, 28))
        im= im/255
        im = self.transform(im).double()
        label = self.data.iloc[idx, -1]
        return(im.float(), label)
    
    


np.random.seed(1)
#test images (mnist)
test = pd.read_csv('/t1/erlangen/users/jwiggerthale/TestDatasets/MNIST/mnist_test.csv').drop('Unnamed: 0', axis = 1)
test = test.iloc[:100]

# uniform noise
noise = pd.read_csv('/t1/erlangen/users/jwiggerthale/TestDatasets/MNIST/noise.csv').drop('Unnamed: 0', axis = 1)

# get data loader
test_set = mnist_dataset(test)
test_loader = DataLoader(test_set, batch_size = 1, shuffle = False)

noise_set = mnist_dataset(noise)
noise_loader = DataLoader(noise_set, batch_size = 1, shuffle = False)


#Model information
hidden_dim = 50
hidden_layers = 1
num_epochs = 100
dropout = 0.01
tau = 0.15
n_hidden = [50]
lambda_u = 0.1
lambda_c = 1.2
num_samples  = 100
num_classes = 10


model = HUQNN(file_path = 'TestModel')
fp = './MNIST_HUQNN/Model.pth'
model.load_state_dict(torch.load(fp))
model.eval()    

# get preds, aus and eus for test ims
test_uns = []
test_eus = []
test_aus = []
for im, label in iter(test_loader):
    pred, au, eu = model.forward(im)
    test_eus.extend(eu.flatten().tolist())
    test_aus.extend(au.flatten().tolist())
    predicted_class = pred.argmax().item()
    test_uns.append(eu[0][predicted_class].item())

# get uncertainties for noise
noise_uns = []
for im, label in iter(noise_loader):
    pred, au, eu = model.forward(im)
    predicted_class = pred.argmax().item()
    noise_uns.append(eu[0][predicted_class].item())
    

# plot results
plt.rcParams['xtick.labelsize'] = 20  # X-axis tick labels
plt.rcParams['ytick.labelsize'] = 20
fig, ax = plt.subplots(figsize = (12,12))
ax.scatter(np.arange(1, 101), test_uns, label = 'Test set', color = 'blue')
ax.scatter(np.arange(1, 101), noise_uns, label = 'Noise', color = 'red')
ax.set_title('EU of H-UQNN on Test Images (Blue) and Noise (Red)', fontsize = 26)
ax.set_xlabel('Index', fontsize = 20)
ax.set_ylabel('Uncertainty', fontsize = 20)
