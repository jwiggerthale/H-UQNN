#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 20 05:11:41 2025

@author: jwiggerthale
"""

'''
This code serves to evaluate H-UQNN trained on modified dataset
Create boxplots as presented in Fig. 9 of our paper
'''


from modules.data_utils import get_minst_dataloader
from HUQNN import HUQNN
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


# get data
train_path = '/t1/erlangen/users/jwiggerthale/TestDatasets/MNIST/mnist_train.csv'
test_path = '/t1/erlangen/users/jwiggerthale/TestDatasets/MNIST/mnist_test.csv'

# we dont't neglect classes for testing and also add no noise to make sure the effects get visible
classes_to_neglect = []
noisy_classes = [] 
train_size = 500
test_size = 500
batch_size = 16

# get data loader 
train_loader, test_loader = get_minst_dataloader(train_path = train_path,
                                                 test_path = test_path,
                                                 classes_to_neglect = classes_to_neglect,
                                                 noisy_classes= noisy_classes,
                                                 train_size = train_size, 
                                                 test_size = test_size, 
                                                 batch_size = 16)



#create model
model = HUQNN(file_path = 'TestModel')


model_name = './MNIST_HUQNN/ModelModifiedData.pth
    
model.load_state_dict(torch.load(model_name))



all_eus = []
all_aus = []
all_ys = []
all_eu_labels = []
all_au_labels = []
all_classes = []
real_classes = []

# eus and aus for each class in a dictionary
class_eus = {}
class_aus = {}
for c in np.arange(10):
  class_eus[c] = []
  class_aus[c] = []



# get predictions for all data points (either train or test loader)
for x, y in iter(train_loader):
  pred, au, eu = model.forward(x)
  pred_cls = pred.argmax(dim = 1).tolist()
  eu_vals = [eu[i][c].item() for i, c in enumerate(pred_cls)]
  au_vals = [au[i][c].item() for i, c in enumerate(pred_cls)] 
  all_eus.extend(eu_vals)
  all_aus.extend(au_vals)

  # add au and eu for each element of the data loader to dictionary for aus and eus
  for i, pred_c in enumerate(pred_cls):
      pred_val = pred[i][pred_c]
      un_ale = au[i][pred_c]/pred_val
      un_epi = eu[i][pred_c]/pred_val        
      class_aus[y[i].item()].append(un_ale.item())
      class_eus[y[i].item()].append(un_epi.item())
  
  
  all_classes.extend(pred_cls)
  real_classes.extend(y.tolist())



    
# plot results in boxplot
plt.boxplot([class_aus[i] for i in class_aus], labels = [i for i in class_aus])
plt.ylabel('Uncertainty')
plt.title('Aleatoric Uncertainty for Different Classes - UQNN')
plt.savefig(f"{model_name.replace('pth', '')}_AU.png")
plt.show()

plt.boxplot([class_eus[i] for i in class_eus], labels = [i for i in class_eus])
plt.ylabel('Uncertainty')
plt.title('Epistemic Uncertainty for Different Classes - UQNN')
plt.savefig(f"{model_name.replace('pth', '')}_EU.png")
plt.show()

    
cm = confusion_matrix(real_classes, all_classes)
disp = ConfusionMatrixDisplay(cm, display_labels = np.arange(10))
plt.savefig(f"{model_name.replace('pth', '')}_CM.png")
disp.plot()
plt.show()

