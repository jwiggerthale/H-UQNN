#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  2 11:21:31 2025

@author: jwiggerthale
"""

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

def het_loss(mu, sigma, y):
    """Negative Log-Likelihood Loss für Gaussian Distribution"""
    return torch.mean(0.5 * torch.log(2 * np.pi * sigma**2) + 0.5 * (y - mu)**2 / sigma**2)



'''
Class which implements a dataset for regression
call with np.arrays from x and y
'''
class my_dataset(Dataset): 
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype = torch.float32)
        self.y = torch.tensor(y, dtype = torch.float32)
    def __len__(self):
        return(len(self.x))
    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        return(x,y)


'''
Function wich processes raw data frame to dataset
call with: 
        train --> data frame containing the train data 
        test --> data frame containing the test data
        normalize: bool --> whether to normalize data or not
returns:
    train_loader --> data loader for training
    val_loader --> data loader for validation
    y_train_mean --> mean of y_train
    y_train_mean --> mean of y_train
    y_train_std --> std of y_train
    x_train_mean --> mean of x_train
    x_train_std --> std of x_train
Note: data is assumed to have X in columns data[:-1] and y in columndata[-1]
'''
def get_data_loader_from_pandas(train: pd.DataFrame, 
                                 test: pd.DataFrame, 
                                 normalize: bool = True):
    x_train = np.array(train.iloc[:, :-1])
    y_train = np.array(train.iloc[:, -1])
    x_val = np.array(test.iloc[:, :-1])
    y_val = np.array(test.iloc[:, -1])
    
    x_train_mean = np.mean(x_train, axis=0)    
    x_train_std = np.std(x_train, axis=0)
    x_train_std[x_train_std == 0] = 1
    
    if(normalize == True):
        x_train = (x_train - x_train_mean) / x_train_std
        x_val = (x_val - x_train_mean) / x_train_std
        
    y_train_mean = np.mean(y_train)
    y_train_std = np.std(y_train)
    y_train_norm = (y_train - y_train_mean)/ y_train_std
    y_val_norm = (y_val - y_train_mean)/ y_train_std
    
    train_set = my_dataset(x_train, y_train_norm)
    val_set = my_dataset(x_val, y_val_norm)
    val_loader = DataLoader(val_set)
    train_loader =  DataLoader(train_set)
    return train_loader, val_loader,  y_train_mean, y_train_std, x_train_mean, x_train_std
    


    


