#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 19 18:47:55 2025

@author: jwiggerthale
"""


from DataUtils import get_minst_dataloader,load_cifar, cifar_ds
from HUQNN import HUQNN
import torch
from torch.utils.data import DataLoader


import argparse

# arguents for model and training
parser = argparse.ArgumentParser()
parser.add_argument('--lambda_c', type = float, default = 1.0) # lambda_het in our paper (here c for classification)
parser.add_argument('--lambda_eu', type = float, default = 0.01) #lambda eu 
parser.add_argument('--batch_size', type = int, default = 16) #batch size to use
parser.add_argument('--train_size', type = int, default = 15000) #number of images for training (max 50.000)

args = parser.parse_args()
lambda_eu = args.lambda_eu
lambda_c = args.lambda_c




train_path = '/t1/erlangen/users/jwiggerthale/TestDatasets/MNIST/mnist_train.csv'
test_path = '/t1/erlangen/users/jwiggerthale/TestDatasets/MNIST/mnist_test.csv'
classes_to_neglect = []
noisy_classes = [] 
train_size = args.train_size
test_size = 2000
batch_size = args.batch_size



train_loader, test_loader = get_minst_dataloader(train_path = train_path,
                                                   test_path = test_path,
                                                   classes_to_neglect = classes_to_neglect,
                                                   noisy_classes= noisy_classes,
                                                   train_size = train_size, 
                                                   test_size = test_size, 
                                                   batch_size = batch_size)

    
model_name = f'./MNIST_HUQNN/HUQNN_lambda_eu_{lambda_eu}_c_{lambda_c}'
model = UQNN(file_path = model_name,
                  lambda_eu = lambda_eu, 
                  lambda_c = lambda_c, 
                  task = 'mnist')

#load state dict if you have a good baseline classification model and want to train without pretraining
#model.load_state_dict(torch.load(f'ClfPretrained/Model.pth'))
model.train_model(train_loader,
                  test_loader, 
                  use_pretraining = True
                  )
#command to add trainig to queue
#qsub -m ae -j oe -q bigmemrh8 -N CreateDS -l select=1:ncpus=1:mem=64gb -- /t1/erlangen/users/jwiggerthale/mnist_det_au_model_2/train_huqnn.sh --batch_size 16 --lambda_eu 0.5 --lambda_c 2
