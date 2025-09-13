#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 12 15:32:49 2025

@author: jwiggerthale
"""



import torch
import torch.nn as nn
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


from utils import het_loss,  get_data_loader_from_pandas
from modules import BayesianNN

    
    
def train_model(model, 
                train_loader,
                test_loader,
                optimizer, 
                criterion, 
                num_epochs: int = 20, 
                patience: int = 10):
    counter = 0
    model.train()
    best_loss = np.inf
    for epoch in range(num_epochs):
        running_loss = 0.0
        for x, y in iter(train_loader):
            optimizer.zero_grad()
            mu, sigma = model(x)
            loss = het_loss(mu, sigma, y)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
        running_loss/=len(train_loader)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in iter(test_loader):
                mu, sigma = model(x)
                loss = het_loss(mu, sigma, y)
                val_loss += loss.item()
        val_loss /= len(test_loader)
        print(f'Epoch {epoch + 1}\nValidation loss: {val_loss}\nTrain loss: {running_loss}\n-------------------------------------')
        if(val_loss < best_loss):
            torch.save(model.state_dict(), f'./regression_models/model_epoch_{epoch + 1}_loss_{int(val_loss*1000)}.pth')
            best_loss = val_loss
            counter = 0
        else:
            counter += 1
            if counter > patience:
                 print(f'Early stopping triggered in epoch {epoch +1}')
                 break



#Common dataset
train = pd.read_csv('./data/TrainDataGal.csv')
test = pd.read_csv('./data/TestDataGal.csv')

train_loader, test_loader, y_train_mean, y_train_std, x_train_mean, x_train_std = get_data_loader_from_pandas(train, test)


#Dataset with uncertainty challenges as described in Sec. 4A of our paper
test = pd.read_csv('./data/TestDataOutliers.csv').drop('Unnamed: 0', axis = 1)
train = pd.read_csv('./data/TrainDataNoise.csv').drop('Unnamed: 0', axis = 1)

train_loader, test_loader, y_train_mean, y_train_std, x_train_mean, x_train_std = get_data_loader_from_pandas(train, test)

dropout = 0.2
n_hidden = [50]
n_epochs = 50
model = BayesianNN(len(x_train[0]), n_hidden, dropout)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_model(model, 
            train_loader = train_loader,
            test_loader = test_loader,
            optimizer = optimizer, 
            criterion = criterion, 
            num_epochs = n_epochs)
