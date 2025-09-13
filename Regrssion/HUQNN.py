#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  2 14:01:05 2025

@author: jwiggerthale
"""



#Import libraries
import pandas as pd
import os
import numpy as np
from sklearn.metrics import explained_variance_score
import plotly.express as px
import plotly.graph_objects as go


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from utils import het_loss


'''
Class which implements a H-UQNN for regression
Initialize with:  
     input_dim: int = 13, --> number of input features
     n_hidden: list = [50], --> list of number of output features for each hidden layer (longer list = more complex model)
     dropout_rate: float =0.01, --> dropout rate to be used
     tau:float = 0.15, --> precision parameter, for more information see Gal et al. Dropout as Bayesian Approximation
     lambda_u: float = 0.2, --> factor to multiply uncertainty loss (see function combibed_loss)
     lambda_c: float = 0.4, --> factor to multiply classificatuion loss (see function combibed_loss)
     num_samples: int = 10000, --> Number of forward passes conducted with MC dropout
     file_path: str = 'Model' --> path where models and model definition are stored
'''
class HUQNN(nn.Module):
    def __init__(self, 
                 input_dim: int = 13, 
                 n_hidden: list = [50], 
                 dropout_rate: float =0.01, 
                 tau:float = 0.15,
                 lambda_u: float = 0.2, 
                 lambda_r: float = 0.4, 
                num_samples: int = 50, 
                file_path: str = 'Model'):
        super(HUQNN, self).__init__()

        #Define model structure
        layers = []
        prev_dim = input_dim
        self.dropout = dropout_rate
        self.tau = tau
        
        self.fc_1 = nn.Linear(input_dim, 50)
        
        
        layers.append(nn.Dropout(dropout_rate))
        for h in n_hidden:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = h
        
        self.feature_extractor = nn.Sequential(*layers)
        
        self.mu =  nn.Sequential(nn.Linear(prev_dim, 80), 
                                              nn.ReLU(), 
                                              nn.Linear(80, 60),
                                              nn.ReLU(),
                                              nn.Linear(60, 1))
        
        self.sigma =  nn.Sequential(nn.Linear(prev_dim, 80), 
                                              nn.ReLU(), 
                                              nn.Linear(80, 60),
                                              nn.ReLU(),
                                              nn.Linear(60, 1), 
                                              nn.Softplus())
        self.uncertainty_head = nn.Sequential(nn.Linear(prev_dim, 80), 
                                              nn.ReLU(), 
                                              nn.Linear(80, 60),
                                              nn.ReLU(),
                                              nn.Linear(60, 1),
                                              nn.Softplus())
        
    
  

        #Parameters for monitoring training
        self.is_converged = False
        self.reg_losses = []
        self.combined_losses = []
        self.num_samples = num_samples

        #Parameters for training
        self.lambda_u = lambda_u
        self.lambda_r = lambda_r
                  
        #Write architecture of model to .txt-file
        wd = os.getcwd()
        self.file_path = f'{wd}/{file_path}'
        if not os.path.isdir(self.file_path):
            os.mkdir(self.file_path)
        
        self.write_params()
     
    '''
    Function which writes parameters of model in  .txt-file
    Automatically called when initializing H-UQNN
    --> useful for reconstructing model when you test different configurations
    '''   
    def write_params(self):
        with open(f'{self.file_path}/definition.txt', 'a') as f:
            for attribute, value in self.__dict__.items():
                f.write(f'{attribute}: {value}\n')

    '''
    Function which determines if regressor converged
    Automatically called during training
    Works based on variation in last losses
        --> value for variation may be adapted when mregressor does not converge or converges to fast 
    Minimum of 5 epochs necessary
        --> may be reduced if model converges quickly
    '''      
    def regressor_converged(self):
        if(len(self.reg_losses) < 15):
            return
        else:
            self.is_converged = True
        '''
        elif(len(self.combined_losses) > 0):
            self.is_converged = True
        else:
            min_loss = np.min(self.reg_losses[-20:])
            max_loss = np.max(self.reg_losses[-20:])
            variation = (max_loss - min_loss)/min_loss
            if(variation > 0.02):
                return
            else:
                self.is_converged = True
        '''


    '''
    Loss function for main training stage
    Automatically called in training
    Called with: 
        mu: mean prediction from MC dropout
        label: actual label
        sigma: variance from MC dropout
        uncertainty: uncertainty estiation from uncertainty head
    Returns:
        combined loss as weighted sum of mse loss for regression and mse loss for uncertainty estimation
    '''
    def combined_loss(self, mu, sigma, label, eu, mc_uncertainty):
        reg_loss = het_loss(mu, sigma, label) * self.lambda_r
        uncertainty_loss = F.mse_loss(eu, mc_uncertainty) * self.lambda_u
        return(uncertainty_loss + reg_loss).mean(), uncertainty_loss, reg_loss



    '''
    Function which trains the model
    Call with: 
        train_loader: Dataloader --> DataLoader containing the train set
        test_loader: Dataloader --> DataLoader containing the test set
        num_epochs: int --> number of epochs 
    Function 
        Conducts pre-training
        Conducts main training
        Saves model if performance is sufficient
        Writes performance in console after every epoch
    '''
    def train_model(self, 
              train_loader,
              test_loader: DataLoader = None,
              num_epochs: int = 100):
        best_loss_reg = np.inf
        best_ll_combined = -np.inf
        best_loss_combined = np.inf
                   
        reg_params = list(self.feature_extractor.parameters()) + list(self.sigma.parameters()) + list(self.mu.parameters())
        self.reg_optim = optim.Adam(reg_params, lr=0.001, weight_decay=1e-2, betas=(0.9, 0.999), eps=1e-7)
        self.uncertainty_optim = optim.Adam(self.parameters(), lr=0.001)#, weight_decay=1e-2, betas=(0.9, 0.999), eps=1e-7)
        for epoch in range(num_epochs):
            train_loss = 0.0
            test_loss = 0.0
            train_loss_un = 0.0
            train_loss_reg = 0.0
            test_loss_un = 0.0
            test_loss_reg = 0.0
            ll = 0.0
            if(self.is_converged == False):
                self.regressor_converged()
            if(self.is_converged == False):
                for x, y in iter(train_loader):
                    mu, sigma, _ = self.single_pass(x)
                    loss = het_loss(mu, sigma, y)
                    train_loss += loss.item()
                    self.reg_optim.zero_grad()
                    loss.backward()
                    self.reg_optim.step()
                if test_loader is  not None:
                    for x, y in iter(test_loader):
                        with torch.no_grad():
                            mu, sigma, _ = self.single_pass(x)
                            loss = het_loss(mu, sigma, y)
                            test_loss += loss.item()
                    test_loss /= len(test_loader)
                else:
                    test_loss = train_loss / len(train_loader)
                train_loss /= len(train_loader)
                
                if(test_loss < best_loss_reg):
                    torch.save(self.state_dict(), f'{self.file_path}/reg_epoch_{epoch}_loss_{int(test_loss * 100)}.pth')
                    best_loss_reg = test_loss
                with open(f'{self.file_path}/protocol.txt', 'a', encoding = 'utf-8') as out_file:
                    out_file.write(f'Epoch {epoch + 1} - Regressor not yet converged\nLL: {int(ll * 100)}\nValidation Loss: {test_loss}\nTrain Loss: {train_loss}\n\n')
                self.reg_losses.append(test_loss)
            else:
                
                for x, y in iter(train_loader):
                    mus = []
                    sigmas = []
                    features = []
                    for _ in range(self.num_samples):
                        mu, sigma, feature = self.single_pass(x)
                        mus.append(mu)
                        sigmas.append(sigma)
                        features.append(feature)
                    mus= torch.stack(mus)
                    sigmas = torch.stack(sigmas)
                    features = torch.stack(features)
                    features = features.mean(dim = 0)
                    mu = mus.mean(dim = 0)
                    mc_eu = mus.std(dim = 0)
                    au = sigmas.mean(dim = 0)
                    pred_eu = self.uncertainty_head(features)
                    target = y.view(-1, 1).float()
                    loss, loss_un, loss_reg = self.combined_loss( mu = mu, 
                                                                 sigma = au, 
                                                                 label = y, 
                                                                 eu = pred_eu, 
                                                                 mc_uncertainty = mc_eu)
                    
                    train_loss_un += loss_un.item()
                    train_loss_reg += loss_reg.item()
                    train_loss += loss.item()
                    
                    self.uncertainty_optim.zero_grad()
                    loss.backward()
                    self.uncertainty_optim.step()
                train_loss /= len(train_loader)
                train_loss_reg /= len(train_loader)
                train_loss_un /= len(train_loader)

                mc_eus = []
                pred_eus = []
                mc_preds = []
                ys = []
                if(test_loader is not None):
                    for x, y in iter(test_loader):
                        mus = []
                        sigmas = []
                        features = []
                        ys.extend(y.tolist())
                        with torch.no_grad():
                            for _ in range(self.num_samples):
                                mu, sigma, feature = self.single_pass(x)
                                mus.append(mu)
                                sigmas.append(sigma)
                                features.append(feature)
                            mus= torch.stack(mus)
                            sigmas = torch.stack(sigmas)
                            features = torch.stack(features)
                            features = features.mean(dim = 0)
                            mu = mus.mean(dim = 0)
                            mc_eu = mus.std(dim = 0)
                            au = sigmas.mean(dim = 0)
                            mc_eus.extend(mc_eu.tolist())
                            mc_preds.extend(mu.tolist())
                            uncertainty = self.uncertainty_head(features)
                            pred_eus.extend(uncertainty.tolist())
                            target = y.view(-1, 1).float()
                            loss, loss_un, loss_reg = self.combined_loss( mu = mu, sigma = au, label = y, eu = uncertainty, mc_uncertainty = mc_eu)

                            #loss = F.mse_loss(sigma, uncertainty)
                            test_loss += loss.item()
                            test_loss_un += loss_un.item()
                            test_loss_reg += loss_reg.item()
                            
                    test_loss /= len(test_loader)
                    test_loss_un /= len(test_loader)
                    test_loss_reg /= len(test_loader)
                    
                    ev_uncertainty = explained_variance_score(pred_eus, mc_eus)
                    ev_reg = explained_variance_score(mc_preds, ys)
                    variance = np.var(mc_preds)
                    ys = np.array(ys)
                    mc_preds = np.array(mc_preds)
                    ll = -0.5 * np.log(2 * np.pi * (1/self.tau + variance)) - 0.5 * ((ys - mc_preds) ** 2) / (1/self.tau + variance)
                    ll = np.mean(ll)
                else:
                    test_loss = train_loss
                    ll = 0
                    ev_reg = 0
                    ev_uncertainty = 0
                if(test_loss < best_loss_combined):
                    torch.save(self.state_dict(), f'{self.file_path}/combined_epoch_{epoch}_ev_reg_{int(ev_reg*100)}_ev_uncertainty_{int(ev_uncertainty*100)}.pth')
                    best_loss_combined = test_loss
                elif(ll > best_ll_combined):
                    best_ll_combined = ll
                    torch.save(self.state_dict(), f'{self.file_path}/combined_epoch_{epoch}_ev_reg_{int(ev_reg*100)}_ev_uncertainty_{int(ev_uncertainty*100)}.pth')
                with open(f'{self.file_path}/protocol.txt', 'a', encoding = 'utf-8') as out_file:
                    out_file.write(f'''Epoch {epoch + 1} - Regressor convergedLog Likelihood: {ll}
                                   Explained Variance EU: {ev_uncertainty}
                                   Explained Variance Regression: {ev_reg}
                                   Validation Loss: {test_loss}
                                       Validation Loss Reg: {test_loss_reg}
                                       Validation Loss Un: {test_loss_un}
                                   Train Loss: {train_loss}
                                       Train Loss Reg: {train_loss_reg}
                                       Train Loss Un: {train_loss_un}\n\n''') 
                self.combined_losses.append(test_loss)
    
        pred, features = self.single_pass(x)
        uncertainty = self.uncertainty_head(features)
        return pred, uncertainty

    '''
    Function which predicts label for batch of samples and features that can be used by classification head
    Necessary in different other functions
        Call with: 
        x --> batch of samples to be classified
    Returns: 
        pred --> prediction on batch 
        features --> features extracted by the feature extractor
    '''
    def single_pass(self, x):
        features = F.relu(self.feature_extractor(x))
        mu = self.mu(features)
        sigma = self.sigma(features)
        return mu, sigma, features
         
    '''
    Forward pass of model
    Makes prediction on data point and estimates uncertainty
    Call with: 
        x --> batch of samples to be classified
    Returns: 
        pred --> prediction on batch 
        uncertainty --> uncertainty for predictions
    '''
    def forward(self, x):
        x = F.relu(self.feature_extractor(x))
        mu = self.mu(x)
        sigma = self.sigma(x)
        eu = self.uncertainty_head(x)
        return mu, sigma, eu
         
 


    
    
