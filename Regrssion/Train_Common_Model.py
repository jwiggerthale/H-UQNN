#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 12 15:33:10 2025

@author: jwiggerthale
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import numpy as np
from scipy.stats import zscore

from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import explained_variance_score

import pandas as pd


import plotly.io as pio
pio.renderers.default = 'browser'  


from utils import get_data_loader_from_pandas
from modules import EasyNN

    
    
def train_model(model, 
                train_loader,
                test_loader,
                optimizer, 
                num_epochs: int = 20, 
                patience: int = 10):
    counter = 0
    model.train()
    best_loss = np.inf
    for epoch in range(num_epochs):
        running_loss = 0.0
        for x, y in iter(train_loader):
            optimizer.zero_grad()
            pred = model(x)
            loss = F.mse_loss(pred, y)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
        running_loss/=len(train_loader)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in iter(test_loader):
                pred = model(x)
                loss = F.mse_loss(pred, y)
                val_loss += loss.item()
        val_loss /= len(test_loader)
        print(f'Epoch {epoch + 1}\nValidation loss: {val_loss}\nTrain loss: {running_loss}\n-------------------------------------')
        if(val_loss < best_loss):
            torch.save(model.state_dict(), f'./common_regression_models_gal_split/model_epoch_{epoch + 1}_loss_{int(val_loss*1000)}.pth')
            best_loss = val_loss
            counter = 0
        else:
            counter += 1
            if counter > patience:
                 print(f'Early stopping triggered in epoch {epoch +1}')
                 break


#Boston Housing dataset split according to https://github.com/yaringal/DropoutUncertaintyExps, split 0)
test = pd.read_csv('TestData.csv')
train = pd.read_csv('TrainData.csv')

train_loader, test_loader, y_train_mean, y_train_std, x_train_mean, x_train_std = get_data_loader_from_pandas(train, test)


dropout = 0.2
n_hidden = [50]
n_epochs = 50
model = EasyNN(13, n_hidden, dropout)
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_model(model, 
            train_loader = train_loader,
            test_loader = test_loader,
            optimizer = optimizer, 
            num_epochs = n_epochs)
