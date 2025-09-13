#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  2 15:52:12 2025

@author: jwiggerthale
"""


from HUQNN import HUQNN as UQNN
import torch
import pandas as pd
from utils import get_data_loader_from_pandas
import argparse
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
import os

parser = argparse.ArgumentParser()
parser.add_argument('--lambda_r', type = float, default = 0.5)
parser.add_argument('--lambda_u', type = float, default = 0.5)
parser.add_argument('--model_path', type = str, default = 'TestUQNN')

args = parser.parse_args()

lambda_r = args.lambda_r
lambda_u = args.lambda_u

if args.model_path: 
    model_path = args.model_path    
else:
    model_path = f'UQNN_lambda_r_{lambda_r}_lambda_u_{lambda_u}'


'''
This is the OOD data
'''
test = pd.read_csv('./data/TestDataOutliers.csv').drop('Unnamed: 0', axis = 1)
train = pd.read_csv('./data/TrainDataNoise.csv').drop('Unnamed: 0', axis = 1)
train = train.iloc[:-60, :]

train_loader, test_loader, y_train_mean, y_train_std, x_train_mean, x_train_std = get_data_loader_from_pandas(train, test)

model = UQNN(lambda_r = lambda_r, 
             lambda_u = lambda_u, 
             file_path = model_path)

#load model
f = f'UQNN_lambda_r_0.4_lambda_u_3.0_noise/combined_epoch_36_ev_reg_-118_ev_uncertainty_-245.pth'
model.load_state_dict(torch.load(f))
mean_src= 0.0


all_preds = []
all_aus = []
all_eus = []
all_labels = []
for x, y in iter(test_loader):
    mu, au, eu = model.forward(x)
    all_preds.extend(mu.tolist())
    all_eus.extend(eu.reshape(-1).tolist())
    all_aus.extend(au.reshape(-1).tolist())
    all_labels.extend(y.tolist())
    
  
all_preds_train = []
all_aus_train = []
all_eus_train = []
all_labels_train = []
for x, y in iter(train_loader):
    mu, au, eu = model.forward(x)
    all_preds_train.extend(mu.tolist())
    all_eus_train.extend(eu.reshape(-1).tolist())
    all_aus_train.extend(au.reshape(-1).tolist())
    all_labels_train.extend(y.tolist())
   
    
#Create plot and add uncertainties on OOD data
fig, axes = plt.subplots(2, 2, figsize = (16, 16))
axes[0][0].boxplot([all_eus_train, all_eus])
axes[0][0].set_xticklabels(['Train', 'OOD'])
axes[0][0].set_xlabel('Dataset')
axes[0][0].set_ylabel('Epistemic Uncertainty')
axes[0][0].set_title('Epistemic Uncertainty of Model on Training Data and OOD Data')  

axes[0][1].boxplot([all_aus_train, all_aus])
axes[0][1].set_xticklabels(['Train', 'OOD'])
axes[0][1].set_xlabel('Dataset')
axes[0][1].set_ylabel('Aleatoric Uncertainty')
axes[0][1].set_title('Aleatoric Uncertainty of Model on Training Data and OOD Data')    


#noisy data
train = pd.read_csv('./data/TrainDataNoise.csv').drop('Unnamed: 0', axis = 1)
test = train.iloc[-60:, :]
train = train.iloc[:-60, :]
train_loader, test_loader, y_train_mean, y_train_std, x_train_mean, x_train_std = get_data_loader_from_pandas(train, test)



all_preds = []
all_aus = []
all_eus = []
all_labels = []
for x, y in iter(test_loader):
    mu, au, eu = model.forward(x)
    all_preds.extend(mu.tolist())
    all_eus.extend(eu.reshape(-1).tolist())
    all_aus.extend(au.reshape(-1).tolist())
    all_labels.extend(y.tolist())


    
    
all_preds_train = []
all_aus_train = []
all_eus_train = []
all_labels_train = []
for x, y in iter(train_loader):
    mu, au, eu = model.forward(x)
    all_preds_train.extend(mu.tolist())
    all_eus_train.extend(eu.reshape(-1).tolist())
    all_aus_train.extend(au.reshape(-1).tolist())
    all_labels_train.extend(y.tolist())
    

 
#Add uncertainties on noisy data to the plot 
axes[1][0].boxplot([all_eus_train, all_eus])
axes[1][0].set_xticklabels(['Normal', 'Noisy'])
axes[1][0].set_xlabel('Dataset')
axes[1][0].set_ylabel('Epistemic Uncertainty')
axes[1][0].set_title('Epistemic Uncertainty of Model on  Normal Data and Noisy Data')  

axes[1][1].boxplot([all_aus_train, all_aus])
axes[1][1].set_xticklabels(['Normal', 'Noisy'])
axes[1][1].set_xlabel('Dataset')
axes[1][1].set_ylabel('Aleatoric Uncertainty')
axes[1][1].set_title('Aleatoric Uncertainty of Model on Normal Data and Noisy Data')    
fig.suptitle('Distributions of AU and EU for predictions on noisy data and OOD data', fontsize = 16, y = 0.92)
plt.show()
