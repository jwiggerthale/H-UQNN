#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  3 06:01:26 2025

@author: jwiggerthale
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms





'''
Model with dropout following LeNet5 (c.f. LeCun, Y.; Boser, B.; Denker, J. S.; Henderson, D.; Howard, R. E.; Hubbard, W.; Jackel, L. D. (December 1989). "Backpropagation Applied to Handwritten Zip Code Recognition". Neural Computation. 1 (4): 541–551. doi:10.1162/neco.1989.1.4.541. ISSN 0899-7667. S2CID 41312633)
Note: Only feature extractor 
--> classification is done by upstream models using LeNet5 as feature extractor (H-UQNN and heteroscedastic model)
'''
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.fc = nn.Linear(256, 120)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(120, 84)
        self.relu1 = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

        
    def forward(self, x):
        out = self.dropout(self.layer1(x))
        out = self.dropout(self.layer2(out))
        out = out.reshape(out.size(0), -1)
        out = self.dropout(self.fc(out))
        out = self.relu(out)
        out = self.dropout(self.fc1(out))
        out = self.relu1(out)
        return out
    
    
class LeNet_Classifier(nn.Module):
    def __init__(self, 
                     num_classes: int = 10):
        super(LeNet_Classifier, self).__init__()
        self.feature_extractor = LeNet5()
        self.mu = nn.Sequential(nn.Linear(84, 60), 
                                                 nn.ReLU(), 
                                                 nn.Linear(60, num_classes)
                                                 )
        
    def forward(self, x):
        pred = self.feature_extractor(x)
        pred = self.mu(pred)
        return pred
 
    
    
'''
Class which implements a heteroscedastic Model for multi class classification
Initialize with: 
    num_classes: int = 10, --> number of classes to be identified, adapt to your dataset
'''
class Het_Model(nn.Module):
    def __init__(self, 
                 num_classes: int = 10):
        super(Het_Model, self).__init__()
        #Define model structure
        #--> feature extractor = LeNet5 with dropout, heads = fully conneted layers
        self.feature_extractor = LeNet5()
        self.mu = nn.Sequential(nn.Linear(84, 80), 
                                nn.ReLU(), 
                                nn.Linear(80, 60), 
                                nn.ReLU(), 
                                nn.Linear(60, num_classes)
                                                 )
        self.log_var = nn.Sequential(nn.Linear(84, 80), 
                                                 nn.ReLU(),
                                                 nn.Linear(80, 60), 
                                                 nn.ReLU(),  
                                                 nn.Linear(60, num_classes), 
                                                 nn.Softplus()
                                                 )
            
    def forward(self, x):
        features = self.feature_extractor(x)
        mu = self.mu(features)
        sigma = self.log_var(features)
        return mu, sigma
    
   
