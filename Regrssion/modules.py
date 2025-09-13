#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  2 11:20:22 2025

@author: jwiggerthale
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class BayesianNN(nn.Module):
    def __init__(self, input_dim, n_hidden, dropout):
        super(BayesianNN, self).__init__()
        self.dropout = dropout
        
        layers = []
        layers.append(nn.Linear(input_dim, n_hidden[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(p=dropout))
        
        for i in range(len(n_hidden) - 1):
            layers.append(nn.Linear(n_hidden[i], n_hidden[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))
        
        self.feature_extractor = nn.Sequential(*layers)
        self.mu = nn.Linear(n_hidden[-1], 1)
        self.sigma = nn.Linear(n_hidden[-1], 1)
        

    def forward(self, x):
        h = self.feature_extractor(x)
        mu = self.mu(h)
        sigma = self.sigma(h)
        sigma = F.softplus(sigma) +1e-6
        
        return mu, sigma
    
    
class EasyNN(nn.Module):
    def __init__(self, input_dim, n_hidden, dropout):
        super(EasyNN, self).__init__()
        self.dropout = dropout
        
        layers = []
        layers.append(nn.Linear(input_dim, n_hidden[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(p=dropout))
        
        for i in range(len(n_hidden) - 1):
            layers.append(nn.Linear(n_hidden[i], n_hidden[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))
        
        self.feature_extractor = nn.Sequential(*layers)
        self.out = nn.Linear(n_hidden[-1], 1)
        

    def forward(self, x):
        h = self.feature_extractor(x)
        mu = self.out(h)
        return mu
