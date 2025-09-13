#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  4 16:24:48 2025

@author: jwiggerthale
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader


import numpy as np
import os

from modules.lenet import LeNet5 #feature extractor for mnist
from modules.ResNet18 import ResNet18 #feature extractor for cifar-10
from modules.uncertainty_utils import heteroscedastic_ce


class HUQNN(nn.Module):
    def __init__(self, 
                 mc_samples: int = 20, 
                 num_augment: int = 15, 
                 lambda_c: float = 0.5, 
                 lambda_eu: float = 0.5, 
                 device: str = 'cpu', #Implement usage of correct device
                 steps: int = 1000, 
                 num_classes: int = 10, 
                 file_path: str = 'HUQNN', 
                 task: str = 'mnist'
                ):    
        super(HUQNN, self).__init__()
        #Hyperparameters
        self.lambda_c = lambda_c
        self.lambda_eu = lambda_eu
                
        self.num_classes = num_classes
        
        self.num_augment= num_augment
        self.num_mc_samples = mc_samples
        
        
        #Model Architecture depending on the dataset
        if task.lower() == 'mnist':
            self.feature_extractor = LeNet5()
            num_features = 84
        elif task.lower() == 'cifar':
            self.feature_extractor = ResNet18()
            num_features = 512
        self.mu = nn.Sequential(nn.Linear(num_features, 80), 
                                nn.ReLU(), 
                                nn.Dropout(0.2),
                                nn.Linear(80, 60),
                                nn.ReLU(), 
                                nn.Dropout(0.2),
                                nn.Linear(60, num_classes)
                                )
        self.log_var = nn.Sequential(nn.Linear(num_features, 80),
                                    nn.ReLU(), 
                                     nn.Dropout(0.2),
                                    nn.Linear(80, 60),
                                    nn.ReLU(), 
                                     nn.Dropout(0.2),
                                    nn.Linear(60, num_classes), 
                                    nn.Softplus()
                                    )
        self.eu_head = nn.Sequential(nn.Linear(num_features, 80), 
                                    nn.ReLU(), 
                                    nn.Linear(80, 60), 
                                    nn.ReLU(), 
                                    nn.Linear(60, num_classes), 
                                    nn.Softplus()
                                    )

        # optimizers for each training stage
        clf_params = list(self.feature_extractor.parameters()) + list(self.mu.parameters())  + list(self.log_var.parameters())
        self.clf_optim = optim.Adam(clf_params, lr=0.001)
        self.uncertainty_optim = optim.Adadelta(self.parameters(), lr=0.001)

      
        self.is_converged = False
        self.losses = []
        
        self.file_path = file_path
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
    Function to determine wheteher model is converged
    Automatically called before each epoch of training

    '''
    def model_converged(self, 
                     threshold: float = 0.0001, 
                     min_epochs: int = 2,
                     max_epochs: int = 2):
        if len(self.losses) < min_epochs:
            self.is_converged = False
            return
        # make sure model moves to main training somewhen regardless of loss development
        elif len(self.losses) > max_epochs:
            self.is_converged = True
            return
        elif (np.max(self.losses[-min_epochs:]) - np.min(self.losses[-min_epochs:])  > threshold):
            self.is_converged = False
            return
        else:
            self.is_converged = True
            return

    # combined loss function as described in Sec. 3 B of our paper
    def combined_loss(self, 
                      mu: torch.tensor, 
                      y: torch.tensor, 
                      pred_eu: torch.tensor, 
                      eu: torch.tensor, 
                      log_var: torch.tensor):
        
        ce_loss = heteroscedastic_ce(mu, log_var, target = y)
        #eu = eu / (eu.max() + 1e-8)
        
        eu_loss = nn.functional.mse_loss(pred_eu, eu)
        total_loss = (ce_loss * self.lambda_c + 
                      eu_loss * self.lambda_eu)
        loss = {'total': total_loss, 
                'ce': ce_loss,
                'eu': eu_loss}
        return loss

  
    '''
    Function which calculates how uncertain the model is regarding a certain prediction
    Within function:
        h-uqnn.num_samples forward passes are conducted
        mean features and mean prediction are calculated
        variance from forward passes is calculated
        uncertainty head predicts uncertainty using mean features
    Returns prediction, estimated uncertainty (from uncertainty head) but also variance from MC dropout
    Call with: 
        x: torch.tensor --> sample to be classified
    '''
    def get_uncertainty_metrics(self, 
                                x: torch.tensor, 
                                num_samples: int = 40, 
                                n_noise: int = 50):
        self.eval()
        for m in self.feature_extractor.modules():
            if type(m) == nn.Dropout:
                m.train()
        for m in self.mu.modules():
            if type(m) == nn.Dropout:
                m.train()
        for m in self.log_var.modules():
            if type(m) == nn.Dropout:
                m.train()

        def entropy(p, dim=-1, eps=1e-12):
            p = p.clamp_min(eps)
            return -(p * p.log())#.sum(dim=dim)

        probs_mean_per_d = []
    
        exp_entropy_accum = 0.0  # E_{w,z}[H[p]]
        
        
        log_vars = []
        features = []
        for _ in range(num_samples):
            feature = self.feature_extractor(x)
            mu = self.mu(feature)
            logvar = self.log_var(feature)
            std = torch.exp(0.5 * logvar)
            B, C = mu.shape
    
            eps = torch.randn(n_noise, B, C, device=mu.device)
            logits = mu.unsqueeze(0) + eps * std.unsqueeze(0)  # (S, B, C)
            probs  = F.softmax(logits, dim=-1)                 # (S, B, C)
    
            probs_mean_d = probs.mean(dim=0)                   # (B, C)  -> speichere!
            probs_mean_per_d.append(probs_mean_d)
    
            exp_entropy_accum += entropy(probs, dim=-1).mean(dim=0)  # (B,)
            
            features.append(feature)
            log_vars.append(logvar)
    
        probs_mean_stack = torch.stack(probs_mean_per_d, dim=0)      # (D, B, C)
        mean_probs = probs_mean_stack.mean(dim=0)                    # (B, C)
                          # H[E_{w,z}[p]]
        expected_entropy = exp_entropy_accum / num_samples# AU             # E_{w,z}[H[p]]
        eu = probs_mean_stack.std(dim= 0)#.mean(dim = -1)
        features = torch.stack(features)
        return mean_probs, eu, expected_entropy, features


    '''
    Normal forward function of the model
    Apply once model is trained to get mu, log_var(sigma_au), sigma_eu
    Call with: 
      x: torch.tensor --> data point(s) you want to make a prediction for
      num_sugment: int -->number of samples to get estimated distribution of target variable (note: has nothing to do with MC dropout, cf. https://arxiv.org/pdf/1703.04977, Sec. 3.3
    Returns: 
      pred: torch.tensor --> mean of estimated distribution
      aleatoric_entropy: torch.tensor --> aleatrocic uncertainty of the prediction 
      eu: torch.tensor --> epistemic uncertainty of the prediction
    '''
    def forward(self, 
               x: torch.tensor, 
               num_augment: int = 50):
        def entropy(p, dim=-1, eps=1e-12):
            p = p.clamp_min(eps)
            return -(p * p.log())#.sum(dim=dim)
        
        feature = self.feature_extractor(x)
        mu = self.mu(feature)
        logvar = self.log_var(feature)
        std = torch.exp(0.5 * logvar)
        B, C = mu.shape
    
        eps = torch.randn(num_augment, B, C, device=mu.device)
        logits = mu.unsqueeze(0) + eps * std.unsqueeze(0)  # (S, B, C)
        probs  = F.softmax(logits, dim=-1)                 # (S, B, C)
    
        pred = probs.mean(dim=0)                   # (B, C)  -> speichere!
        
        aleatoric_entropy = entropy(probs, dim=-1).mean(dim=0)  # (B,) == AU
        eu = self.eu_head(feature)
        return pred, aleatoric_entropy, eu        

    '''
    Function to train the model
    Call with:
      train_loader: DataLoader --> data loader to be used for training
      test_loader: DataLoader --> data loader to be used for validation / hyper parameter selection
      num_epochs: int = 100 --> number of epochs to train
      early_stopping: int = 10 --> early stopping triggered when no improvement in early_stopping epochs
      use_pretraining: bool = True --> whether to use pretraining or not
    Returns: 
      None
    Function: 
      - trains model
      - saves model after each epoch if loss or accuracy on validation data improved
      - writes training protocol to file after every epoch
    '''
    def train_model(self, 
                    train_loader: DataLoader, 
                    test_loader: DataLoader, 
                    num_epochs: int = 100, 
                    early_stopping: int = 10, 
                    use_pretraining: bool = True):
        best_loss = np.inf
        best_loss_combined = np.inf
        best_acc = 0.0
        for epoch in range(num_epochs):
            self.model_converged()
            if self.is_converged == False and use_pretraining == True:# or epoch%10 == 0:
                #we train on dataset without class 7
                running_loss = 0.0
                self.train()
                for x, y in iter(train_loader):
                    features = self.feature_extractor(x)
                    mu = self.mu(features)
                    log_var = self.log_var(features)
                    loss = heteroscedastic_ce(target = y, 
                                                                    mu = mu, 
                                                                    log_var = log_var)
                    self.clf_optim.zero_grad()
                    loss.backward()
                    self.clf_optim.step()
                    running_loss += loss.item()
                running_loss /= len(train_loader)
                out_text = f'Training in epoch {epoch} finished - Model not yet converged\n\ttraining_loss: {running_loss}'
                #We test on data loader containing all classes
                if test_loader is None:
                    self.losses.append(running_loss)
                    if running_loss < best_loss:
                        best_loss = running_loss
                        torch.save(self.state_dict(), f'{self.file_path}/HUQNN_epoch_{epoch}_loss_{int(running_loss *100)}.pth')
                else:
                    test_loss = 0.0
                    test_acc = 0.0
                    test_ce = 0.0
                    test_regularization = 0.0
                    for x, y in iter(test_loader):
                        with torch.no_grad():
                            features = self.feature_extractor(x)
                            mu = self.mu(features)
                            log_var = self.log_var(features)
                            loss = heteroscedastic_ce(target = y, 
                                                                            mu = mu, 
                                                                            log_var = log_var)
                            test_loss += loss.item()
                            #test_regularization += regularization.item()
                            #test_ce += ce.item()
                            correct = (mu.argmax(dim =1)==y).sum(dtype = float)
                            test_acc += correct.item()
                    test_loss /= len(test_loader)
                    test_regularization /= len(test_loader)
                    test_ce /= len(test_loader)
                    test_acc /= len(test_loader)
                    test_acc /= len(y)
                    out_text += f'\n\ttest loss: {test_loss}\n\ttest accuracy: {test_acc}\n\ttest ce: {test_ce}\n\ttest regularization: {test_regularization}'
                    self.losses.append(test_loss)

                    if test_loss < best_loss:
                        best_loss = test_loss
                        torch.save(self.state_dict(), f'{self.file_path}/HUQNN_epoch_{epoch}_loss_{int(test_loss *100)}.pth')
                    elif test_acc > best_acc:
                        best_acc = test_acc
                        torch.save(self.state_dict(), f'{self.file_path}/HUQNN_epoch_{epoch}_loss_{int(test_loss *100)}.pth')
                out_text += '\n\n'
                with open(f'{self.file_path}/training_protocol.txt', 'a', encoding = 'utf-8') as out_file:
                    out_file.write(out_text)
                
                    
            else:
                #we train on dataset without class 7
                running_loss = 0.0
                clf_loss = 0.0
                eu_loss = 0.0
                reg_component = 0.0
                ce_component = 0.0
                for x, y in iter(train_loader):
                    mc_pred, mc_eu, mc_au, features = self.get_uncertainty_metrics(x)
                    pred_eu = self.eu_head(features)
                    loss = self.combined_loss(mu = mc_pred, 
                                              y = y, 
                                              pred_eu = pred_eu, 
                                              eu = mc_eu, 
                                              log_var = mc_au)
                    self.train()
                    total = loss['total']
                    self.uncertainty_optim.zero_grad()
                    total.backward()
                    self.uncertainty_optim.step()
                    running_loss += total.item()
                    clf_loss += loss['ce'].item()
                    eu_loss += loss['eu'].item()
                running_loss /= len(train_loader)
                clf_loss /= len(train_loader)
                eu_loss /= len(train_loader)
                reg_component /= len(train_loader)
                ce_component /= len(train_loader)
        
    
                
                out_text = f''''Training in epoch {epoch} finished - Model converged
                                training_loss: {running_loss}
                                clf loss: {clf_loss}
                                    regularzation component: {reg_component}
                                    ce_component: {ce_component}
                                eu_loss: {eu_loss}
                                '''
                #We test on data loader containing all classes
                if test_loader is None:
                    self.losses.append(running_loss)
                    if running_loss < best_loss_combined:
                        best_loss_combined = running_loss
                        torch.save(self.state_dict(), f'{self.file_path}/HUQNN_epoch_{epoch}_loss_{int(running_loss *100)}.pth')
                else:
                    test_loss = 0.0
                    test_acc = 0.0
                    clf_loss = 0.0
                    reg_component = 0.0
                    ce_component = 0.0
                    eu_loss = 0.0
                    self.eval()
                    for x, y in iter(test_loader):
                        with torch.no_grad():
                            mc_pred, mc_eu, mc_au, features = self.get_uncertainty_metrics(x)
                            pred_eu = self.eu_head(features)
                            loss = self.combined_loss(mu = mc_pred, 
                                                      y = y, 
                                                      pred_eu = pred_eu, 
                                                      eu = mc_eu, 
                                                      log_var = mc_au)
                            
                            self.eval()
                            total = loss['total']
                            test_loss += total.item()
                            clf_loss += loss['ce'].item()
                            eu_loss += loss['eu'].item()
                            correct = (mc_pred.argmax(dim =1)==y).sum(dtype = float)/len(y)
                            test_acc += correct.item()
                    test_loss /= len(test_loader)
                    test_acc /= len(test_loader)
                    clf_loss /= len(test_loader)
                    eu_loss /= len(test_loader)
                    reg_component /= len(test_loader)
                    ce_component /= len(test_loader)
                    self.losses.append(test_loss)
                    out_text += f'\n\ttest loss: {test_loss}\n\ttest accuracy: {test_acc}'
                    out_text += f'''
                                    clf loss: {clf_loss}
                                        regularzation component: {reg_component}
                                        ce_component: {ce_component}
                                    eu_loss: {eu_loss}
                                
                                '''
                    self.losses.append(test_loss)
                    if test_loss < best_loss_combined:
                        best_loss_combined = test_loss
                        torch.save(self.state_dict(), f'{self.file_path}/HUQNN_epoch_{epoch}_loss_{int(test_loss *100)}.pth')
 
                    
                out_text += '\n\n'
                with open(f'{self.file_path}/training_protocol.txt', 'a', encoding = 'utf-8') as out_file:
                    out_file.write(out_text)
                    

    
    
    
#comand to add task to queue (add args as required)
#qsub -m ae -j oe -q bigmemrh8 -N CreateDS -l select=1:ncpus=1:mem=64gb -- /t1/erlangen/users/jwiggerthale/AU_DiffusionClassifier/train_huqnn.sh

                    
            
    
    
                
