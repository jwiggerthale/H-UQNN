#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 20 05:12:15 2025

@author: jwiggerthale
"""

'''
This script plots the uncertainty accuracy curve for a classification model using: 
  heteroscedastic model
  H-UQNN
'''

#Import libraries
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from typing import Union, Callable, Tuple, Optional, Text, Sequence, Dict
import numpy as np
import pandas as pd
from HUQNN import HUQNN
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import transforms
from PIL import Image
import scipy
import os
import datetime
from modules.lenet import Het_Model
from modules.uncertainty_utils import mc_dropout_with_heteroscedastic
from DataUtils import load_cifar, cifar_ds
from scipy.stats import spearmanr

import warnings


warnings.filterwarnings("ignore")

'''
Function which evaluates a H-UQNN either on AUC or accuracy
Call with: 
  model --> model to be tested
  dataloader --> dataloader containing data for test
Retruns: 
  dictionary containing two DataFrames (one for accuracy and one for AUC)
    --> each DataFrame has column mean and fraction
    --> mean is AUC / accuracy of model for appropriiate fraction
'''
def evaluate_HUQNN(
      model: nn.Module, 
      dataloader: DataLoader,
    ) -> Dict[Text, float]:
   
    # Containers used for caching performance evaluation
    y_true = list()
    labels = []
    y_pred = list()
    preds = []
    classes = []
    y_uncertainty = list()
    uncertainties = []
    aus = []
    eus = []

    #Get prediction and uncertainy for each sample
    start = datetime.datetime.now()
    for x, y in iter(dataloader):
      if isinstance(x, list):
          for im in x:
              pred, au, eu = model.forward(im)
              
              uncertainty= eu + au
              labels.extend([elem.int().item() for elem in y])
              np_pred = F.softmax(pred).cpu().detach().numpy()
              preds.extend([list(elem) for elem in np_pred])
              
              predictions = [elem.argmax().item() for elem in pred]
              classes.extend([elem.argmax().item() for elem in pred])
              eus.extend([eu[i][elem].item() for i, elem in enumerate(predictions)])
              aus.extend([au[i][elem].item() for i, elem in enumerate(predictions)])
              uncertainties.extend([u[predictions[i]].item() for i, u in enumerate(uncertainty)])
            
      else:
        pred, au, eu = model.forward(x)
        uncertainty = au + eu
        labels.extend([elem.int().item() for elem in y])
        np_pred = F.softmax(pred).cpu().detach().numpy()
        preds.extend([list(elem) for elem in np_pred])
        predictions = [elem.argmax().item() for elem in pred]
        eus.extend([eu[i][elem].item() for i, elem in enumerate(predictions)])
        aus.extend([au[i][elem].item() for i, elem in enumerate(predictions)])
        classes.extend([elem.argmax().item() for elem in pred])
        uncertainties.extend([u[predictions[i]].item() for i, u in enumerate(uncertainty)])
    end = datetime.datetime.now()
    duration = end -start
    src = spearmanr(eus, aus)
    
    
    # Use vectorized NumPy containers
    labels = np.array(labels)
    preds = np.array(preds)
    classes = np.array(classes)
    uncertainties = np.array(uncertainties)
    #y_true = np.concatenate(y_true).flatten()
    #y_pred = np.concatenate(y_pred).flatten()
    #y_uncertainty = np.concatenate(y_uncertainty).flatten()
    fractions = np.asarray([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    
  
    return {
        'acc': _evaluate_metric(
            labels,
            classes,
            uncertainties,
            fractions,
            accuracy_score
        ), 

        'auc': _evaluate_metric(
            labels,
            preds,
            uncertainties,
            fractions,
            roc_auc_score, 
            metric = 'roc'
        ),
        'src': src
    }



#
def evaluate_Heteroscedastic(
      model: nn.Module, 
      dataloader: DataLoader
    ) -> Dict[Text, float]:
   
    # Containers used for caching performance evaluation
    y_true = list()
    labels = []
    y_pred = list()
    preds = []
    classes = []
    y_uncertainty = list()
    uncertainties = []
    aus = []
    eus = []


    #Get prediction and uncertainy for each sample
    start = datetime.datetime.now()
    for x, y in iter(dataloader):
      if isinstance(x, list):
          for im in x:
              pred = mc_dropout_with_heteroscedastic(model, im)

              eu = pred['eu']
              au = pred['au']
              uncertainty= eu + au
              pred = pred['probs'] 
              labels.extend([elem.int().item() for elem in y])
              np_pred = F.softmax(pred).cpu().detach().numpy()
              preds.extend([list(elem) for elem in np_pred])
              predictions = [elem.argmax().item() for elem in pred]
              eus.extend([eu[i][elem].item() for i, elem in enumerate(predictions)])
              aus.extend([au[i][elem].item() for i, elem in enumerate(predictions)])
              classes.extend([elem.argmax().item() for elem in pred])
              uncertainties.extend([u[predictions[i]].item() for i, u in enumerate(uncertainty)])
      else:
        pred, uncertainty = mc_dropout_with_heteroscedastic(model, x)
        labels.extend([elem.int().item() for elem in y])
        np_pred = F.softmax(pred).cpu().detach().numpy()
        preds.extend([list(elem) for elem in np_pred])
        predictions = [elem.argmax().item() for elem in pred]
        classes.extend([elem.argmax().item() for elem in pred])
        eus.extend([eu[i][elem].item() for i, elem in enumerate(predictions)])
        aus.extend([au[i][elem].item() for i, elem in enumerate(predictions)])
        uncertainties.extend([u[predictions[i]].item() for i, u in enumerate(uncertainty)])
    end = datetime.datetime.now()
    duration = end -start
    src = spearmanr(aus, eus)
    print(f'Evaluation of heteroscedatstic model took {duration}\n\n')
    print(f'SRC for Heteroscedastic: {src}')
    
    # Use vectorized NumPy containers
    labels = np.array(labels)
    preds = np.array(preds)
    classes = np.array(classes)
    uncertainties = np.array(uncertainties)
    #y_true = np.concatenate(y_true).flatten()
    #y_pred = np.concatenate(y_pred).flatten()
    #y_uncertainty = np.concatenate(y_uncertainty).flatten()
    fractions = np.asarray([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    
  
    return {
        'acc': _evaluate_metric(
            labels,
            classes,
            uncertainties,
            fractions,
            accuracy_score
        ), 

        'auc': _evaluate_metric(
            labels,
            preds,
            uncertainties,
            fractions,
            roc_auc_score, 
            metric = 'roc'
        )
    }


'''
Function which evaluates models predictions on a metric function
Called automatically from function evaluate 
Call with: 
  y_true --> labels
  y_pred --> predictions
  y_uncertainty --> uncertainties
  fractions --> list of fractions to be retained
  metric_fn --> sklearn.metrics.accuracy_score or sklearn.metrics.roc_auc_score
Returns: 
  DataFrames for performance
    --> each DataFrame has column mean and fraction
    --> mean is AUC / accuracy of model for appropriiate fraction
'''
def _evaluate_metric(
      y_true: np.ndarray,
      y_pred: np.ndarray,
      y_uncertainty: np.ndarray,
      fractions: Sequence[float],
      metric_fn: Callable[[np.ndarray, np.ndarray], float],
      metric: str = 'acc'
    ) -> pd.DataFrame:
    """Evaluate model predictive distribution on `metric_fn` at data retain
    `fractions`.
    
    Args:
      y_true: `numpy.ndarray`, the ground truth labels, with shape [N].
      y_pred: `numpy.ndarray`, the model predictions, with shape [N].
      y_uncertainty: `numpy.ndarray`, the model uncertainties,
        with shape [N].
      fractions: `iterable`, the percentages of data to retain for
        calculating `metric_fn`.
      metric_fn: `lambda(y_true, y_pred) -> float`, a metric
        function that provides a score given ground truths
        and predictions.
      name: (optional) `str`, the name of the method.
    
    Returns:
      A `pandas.DataFrame` with columns ["retained_data", "mean", "std"],
      that summarizes the scores at different data retained fractions.
    """
    
    N = y_true.shape[0]
    
    # Sorts indexes by ascending uncertainty
    I_uncertainties = np.argsort(y_uncertainty)
    
    # Score containers
    mean = np.empty_like(fractions)
    # TODO(filangel): do bootstrap sampling and estimate standard error
    std = np.zeros_like(fractions)
    
    for i, frac in enumerate(fractions):
      # Keep only the %-frac of lowest uncertainties
      I = np.zeros(N, dtype=bool)
      I[I_uncertainties[:int(N * frac)]] = True
      I = np.array(I)
      if metric == 'roc':
          mean[i] = metric_fn(y_true[I], 
                                y_pred[I], 
                                multi_class="ovr",
                                average="macro")
      else:
          mean[i] = metric_fn(y_true[I], y_pred[I])
    
    # Store
    df = pd.DataFrame(dict(retained_data=fractions, mean=mean, std=std))
    
    return df


'''
Our mnist datset 
--> for more information see train_HUQNN.py
'''
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
        return(im, label)


'''
Similar as mnist_dataset but returns list of images roteted by certain angles
Call with: 
  df --> DataFrame containing Images and labels
  angles --> list of rotation angles to be used for testing
'''
class rotation_mnist_dataset(Dataset):
    def __init__(self, df: pd.DataFrame, angles: list):
        self.data = df
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.1307], std=[0.3081]),
        ])
        self.rotation_angles = angles # 0 bis 95 Grad in 5-Grad-Schritten

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        im = self.data.iloc[idx, :-1].values.astype(np.uint8)
        im = im.reshape((28, 28))
        im = Image.fromarray(im)

        rotated_images = []
        for angle in self.rotation_angles:
            rotated_im = im.rotate(angle)
            rotated_im = self.transform(rotated_im)
            rotated_images.append(rotated_im)

        label = self.data.iloc[idx, -1]
        return rotated_images, label


#Reproducibility
torch.manual_seed(1)    
np.random.seed(1)


train = pd.read_csv('/t1/erlangen/users/jwiggerthale/TestDatasets/MNIST/mnist_train.csv').drop('Unnamed: 0', axis = 1)
train = train.iloc[:6000, :]
test = pd.read_csv('/t1/erlangen/users/jwiggerthale/TestDatasets/MNIST/mnist_test.csv').drop('Unnamed: 0', axis = 1)
test = test.iloc[:1000]


angles = [20]
train_set = rotation_mnist_dataset(train, angles = angles)
test_set = rotation_mnist_dataset(test, angles = angles)


train_loader = DataLoader(train_set, batch_size = 16, shuffle = True)
test_loader = DataLoader(test_set, batch_size = 16, shuffle = False)
'''


entropies = pd.read_csv('annotation_entropies.csv')
train_data, test_data = load_cifar()
train_set = cifar_ds(train_data, 
                     num_ims = 15000)
train_loader= DataLoader(train_set, batch_size = 16)
test_set = cifar_ds(test_data,
                     num_ims = 2000)
test_loader= DataLoader(test_set, batch_size = 16)
'''
#Define parameters for H-UQNN model
lambda_u = 0.1
lambda_c = 1.2
num_samples  = 100
num_classes = 10


#evaluate heteroscedastic model
model = Het_Model()
fp = './MNIST_HetModels/Model.pth'
model.load_state_dict(torch.load(fp))
model.train() 
results_heteroscedastic = evaluate_Heteroscedastic(model, train_loader)




# evaluate HUQNN
f = './MNIST_HUQNN/HUQNN.pth'
model = UQNN(file_path = 'TestModel',
                    task = 'mnist')

model.load_state_dict(torch.load(f))
  
      
results = evaluate_HUQNN(model, train_loader)
mean_src = results['src'].correlation

print(f'#\n{H-UQNN}: {mean_src}\n')+

#Get metrics and plot them 
acc = results['acc']
auc = results['auc']
acc_heteroscedastic= results_heteroscedastic['acc']
auc_heteroscedastic = results_heteroscedastic['auc']

fig, ax = plt.subplots(figsize =(6, 6))
ax.plot(acc['retained_data'], acc['mean'], color = 'red', label = 'H-UQNN')
ax.plot(acc_heteroscedastic['retained_data'], acc_heteroscedastic['mean'], color = 'blue', label = 'Heteroscedastic')
ax.set_xlabel('Classified Data')
ax.set_ylabel('Accuracy')
ax.set_ylim(ymin = 0.87)
fig.suptitle('Development of Accuracy in Abstained Prediction Test')
fig.legend(loc = 'lower left')
plt.savefig(f'HUQNN.png')
plt.show()

    
    
