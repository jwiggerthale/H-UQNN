from resnet_18 import ResNet18
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os
from openood.evaluation_api import Evaluator


'''
Class which implements heteroscedastic model 
Uses ResNet18 as backbone
'''
class het_model(nn.Module):
    def __init__(self, num_classes=10):
        super(het_model, self).__init__()
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
    def forward(self, 
                x: torch.tensor):
        pred = self.feature_extractor(x)
        pred = self.mu(pred)
        return pred

'''
Function to train heteroscedastic model
call with: 
  model: nn.Module --> model to be trained
  optimizer: nn.Module --> optimizer to use for training
  train_loader: DataLoader --> DataLoader with train data
  test_loader: DataLoader --> DataLoader with validation data (to determine performance and tune hyper parameters)
  num_epochs: int = 100
  early_stopping: int = 10 --> interrupt training if model does not improve in early_stopping epochs
  out_dirs: str = het_model --> model to store weights and training protocol
'''
def train(model: nn.Module, 
          optimizer: nn.Module, 
          train_loader: DataLoader, 
          test_loader: DataLoader,
          num_epochs: int = 100,
          early_stopping: int = 10, 
          out_dir: str = 'het_model'  
          ):
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    best_acc = 0.0
    best_loss = np.inf
    counter = 0
    for epoch in range(num_epochs):
        running_loss = 0.0
        test_loss = 0.0
        acc = 0.0

        for elem in iter(train_loader):
            x = elem['data'].to('cuda:2')
            y = elem['label'].to('cuda:2')
            pred = model.forward(x)
            loss = nn.functional.cross_entropy(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        running_loss /= len(train_loader)
        test_samples = 0
        for elem in iter(test_loader):
            x = elem['data'].to('cuda:2')
            y = elem['label'].to('cuda:2')
            with torch.no_grad():
                pred = model.forward(x)
            loss = nn.functional.cross_entropy(pred, y)
            test_loss += loss.item()
            pred_cls = pred.argmax(dim = 1)
            acc += (pred_cls == y).sum().item()
            test_samples += len(y)
        acc /= test_samples
        test_loss /= len(test_loader)
        counter += 1
        print(f'Training in epoch {epoch +1} finished: Acc: {acc}; Test Loss: {test_loss}, Train Loss: {running_loss}')
        with open(f'{out_dir}/protocol.csv', 'a', encoding='utf-8') as out_file:
            out_file.write(f'{epoch+1},{running_loss},{test_loss},{acc}\n')
        if test_loss < best_loss or acc > best_acc:
            torch.save(model.state_dict(), f'{out_dir}/model_epoch_{epoch+1}.pth')
            counter = 0
        elif counter > early_stopping:
            print(f'Early stopping in epoch {epoch +1}')
            break

        
    
'''
Little workaround to get exactly the same dataset for training as used in: 
  - @article{zhang2023openood,
  title={OpenOOD v1.5: Enhanced Benchmark for Out-of-Distribution Detection},
  author={Zhang, Jingyang and Yang, Jingkang and Wang, Pengyun and Wang, Haoqi and Lin, Yueqian and Zhang, Haoran and Sun, Yiyou and Du, Xuefeng and Li, Yixuan and Liu, Ziwei and Chen, Yiran and Li, Hai},
  journal={arXiv preprint arXiv:2306.09301},
  year={2023}}
  - https://github.com/Jingkang50/OpenOOD/blob/a485cbb49d692b5f88e38b5ae2b68331f277b9ea/openood/networks/palm_net.py#L34
'''
evaluator = Evaluator(MC_Model(), 
                id_name = 'cifar10', 
                data_root = './benchmark_data', 
                config_root = None, 
                preprocessor = None, 
                postprocessor_name = 'uqnn', 
                postprocessor = None, 
                batch_size = 16, 
                shuffle = False, 
                num_workers = 1)

train_loader = evaluator.dataloader_dict['id']['train']
test_loader = evaluator.dataloader_dict['id']['test']

# create model and set to device
model = het_model().to('cuda:2')
optim = torch.optim.SGD(model.parameters())

train(model=model, 
      optimizer=optim, 
      train_loader=train_loader, 
      test_loader=test_loader, 
      out_dir='MC_Model_SGD')
