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
import os

from openood.evaluator_api.Evaluator import Evaluator


import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type = int, default = 16)
parser.add_argument('--train_size', type = int, default = 15000)

args = parser.parse_args()

# common data
'''
train_data, test_data = load_cifar()
train_set = cifar_ds(train_data, 
                     num_ims = 60000)
train_loader= DataLoader(train_set, batch_size = 16)
test_set = cifar_ds(test_data, 
                     num_ims = 2000)
test_loader= DataLoader(test_set, batch_size = 16)
'''


'''
workaround to get exactly the train data used in OpenOOD
  - @article{zhang2023openood,
  title={OpenOOD v1.5: Enhanced Benchmark for Out-of-Distribution Detection},
  author={Zhang, Jingyang and Yang, Jingkang and Wang, Pengyun and Wang, Haoqi and Lin, Yueqian and Zhang, Haoran and Sun, Yiyou and Du, Xuefeng and Li, Yixuan and Liu, Ziwei and Chen, Yiran and Li, Hai},
  journal={arXiv preprint arXiv:2306.09301},
  year={2023}}
  - https://github.com/Jingkang50/OpenOOD/blob/a485cbb49d692b5f88e38b5ae2b68331f277b9ea/openood/networks/palm_net.py#L34
'''
evaluator = Evaluator(HUQNN(task = 'mnist'), 
                id_name = 'cifar10', 
                data_root = './benchmark_data', 
                config_root = None, 
                preprocessor = None, 
                postprocessor_name = 'uqnn', 
                postprocessor = None, 
                batch_size = 16, 
                shuffle = False, 
                num_workers = 1, 
                model_path=None)

train_loader = evaluator.dataloader_dict['id']['train']
test_loader = evaluator.dataloader_dict['id']['test']

# grid search (alternatively add args for lambda_eu and lambda_c [lamda_het] to argparser)
for lambda_eu in [0.1, 0.2, 0.5, 1, 2]:
    for lambda_c in [0.1, 0.2, 0.5, 1, 2]:
      model_name = f'./CIFAR_HUQNN/HUQNN_vgg_lambda_eu_{lambda_eu}_c_{lambda_c}'
      if os.path.isdir(model_name):
        my_var = 0
      else:
        print(model_name)
        model = HUQNN(file_path = model_name,
                  lambda_eu = lambda_eu, 
                  lambda_c = lambda_c, 
                  task = 'cifar')
        # load baseline classifier --> no pretraining necessarc
        model.load_state_dict(torch.load(f'/data/Uncertainty/cifar_modeling/cifar_results_3/VGG_BaseClf.pth'))
        model.to('cuda')
        model.train_model(train_loader,
                  test_loader, 
                  use_pretraining = False, 
                  num_epochs = 60
                  )
        del model 
        torch.cuda.empty_cache()

# add job to queue (adapt args)
#qsub -m ae -j oe -q bigmemrh8 -N CreateDS -l select=1:ncpus=1:mem=64gb -- /t1/erlangen/users/jwiggerthale/mnist_det_au_model_2/train_huqnn.sh --batch_size 16 
