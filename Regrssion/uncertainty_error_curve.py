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



test = pd.read_csv('TestDataOutliers.csv').drop('Unnamed: 0', axis = 1)
train = pd.read_csv('TrainDataNoise_2.csv').drop('Unnamed: 0', axis = 1)
train = train.iloc[:-60, :]

train = pd.read_csv('TrainDataGal.csv')
test = pd.read_csv('TestDataGal.csv')

train_loader, test_loader, y_train_mean, y_train_std, x_train_mean, x_train_std = get_data_loader_from_pandas(train, test)

model = UQNN(lambda_r = lambda_r, 
             lambda_u = lambda_u, 
             file_path = model_path)


f = 'QNN_softplus_lambda_r_0.4_lambda_u_5.0_gal_split/reg_epoch_0_loss_54.pth'
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
    
    
    
test_labels = np.array(all_labels).reshape(-1)
test_preds = np.array(all_preds).reshape(-1)
v = np.array(all_eus).reshape(-1) + np.array(all_aus).reshape(-1)
tau = 1.0
ll = -0.5 * np.log(2 * np.pi * (1/tau + v)) - 0.5 * ((test_labels - test_preds) ** 2) / (1/tau + v)
print(f'log likelihood on test dara: {ll.mean()}')

errors = abs(test_labels - test_preds)
test_eus = np.array(all_eus).reshape(-1)
test_aus = np.array(all_aus).reshape(-1)
uns = test_eus + test_aus
src = spearmanr(test_eus, test_aus)
mean_src = src.correlation
  
print(f'SRC between test eus and test aus: {mean_src}')

plt.scatter(errors, uns)
plt.xlabel('Prediction Error')
plt.ylabel('Uncertainty')
plt.show()    


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
       
  
train_labels = np.array(all_labels_train).reshape(-1)
train_preds = np.array(all_preds_train).reshape(-1)
v_train = np.array(all_eus_train).reshape(-1) + np.array(all_aus_train).reshape(-1)
ll_train = -0.5 * np.log(2 * np.pi * (1/tau + v_train)) - 0.5 * ((train_labels - train_preds) ** 2) / (1/tau + v_train)
print(f'log likelihood on train data: {ll.mean()}')

errors = abs(train_labels - train_preds)
train_eus = np.array(all_eus).reshape(-1)
train_aus = np.array(all_aus).reshape(-1)
src = spearmanr(test_eus, test_aus)
mean_src = src.correlation
print(f'SRC between test eus and test aus: {mean_src}')
