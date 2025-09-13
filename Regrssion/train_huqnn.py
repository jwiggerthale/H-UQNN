#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  2 14:16:58 2025

@author: jwiggerthale
"""


from HUQNN import HUQNN
import pandas as pd
from utils import get_data_loader_from_pandas
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--lambda_r', type = float, default = 0.4)
parser.add_argument('--lambda_u', type = float, default = 8)
parser.add_argument('--model_path', type = str)

args = parser.parse_args()

lambda_r = args.lambda_r
lambda_u = args.lambda_u

if args.model_path: 
    model_path = args.model_path    
else:
    model_path = f'H_UQNN_lambda_r_{lambda_r}_lambda_u_{lambda_u}_noise'



# Select the dataset you want to use 
# 1) prepared dataset with noisy data in training data and test data only outliers
test = pd.read_csv('TestDataOutliers.csv').drop('Unnamed: 0', axis = 1)
train = pd.read_csv('TrainDataNoise_2.csv').drop('Unnamed: 0', axis = 1)

# 2) common data (split following https://github.com/yaringal/DropoutUncertaintyExps
#test = pd.read_csv('TestDataGal.csv')
#train = pd.read_csv('TrainDataGal.csv')

# Create data loader from the data 
train_loader, test_loader, y_train_mean, y_train_std, x_train_mean, x_train_std = get_data_loader_from_pandas(train, test)

# create model
model = HUQNN(lambda_r = lambda_r, 
             lambda_u = lambda_u, 
             file_path = model_path)

# train model (rest hapens automatically)
model.train_model(train_loader, test_loader)

#subimt script to queue (parse args if you want to change hyperparameters)
#qsub -m ae -j oe -q atlasrh8 -N TrainClf -l select=1:ncpus=1:mem=64gb -- /t1/erlangen/users/jwiggerthale/det_au_model/train_huqnn.sh
