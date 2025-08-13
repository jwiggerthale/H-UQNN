from modules import BayesianNN, EasyNN
from UQNN_det_au_model_reg import UQNN
from utils import nll_loss, get_index_train_test_path, get_dataloader, get_index_train_test_path, get_data_loader_from_pandas

import pandas as pd
import torch


import matplotlib.pyplot as plt
from sklearn.metrics import explained_variance_score

torch.manual_seed(1)

#test = pd.read_csv('TestDataOutliers.csv').drop('Unnamed: 0', axis = 1)
#train = pd.read_csv('TrainDataNoise.csv').drop('Unnamed: 0', axis = 1)


test = pd.read_csv('TestDataGal.csv')
train = pd.read_csv('TrainDataGal.csv')

train_loader, test_loader, y_train_mean, y_train_std, x_train_mean, x_train_std = get_data_loader_from_pandas(train, test)




dropout = 0.2
n_hidden = [50]
n_epochs = 50
model = BayesianNN(13, n_hidden, dropout)
model.load_state_dict(torch.load('./regression_models_gal_split/model_epoch_5_loss_194.pth'))


H_preds = []
H_labels = []

with torch.no_grad():
    for x, y in iter(test_loader):
        mu, sigma = model.forward(x)
        H_preds.extend(mu.tolist())
        H_labels.extend(y.tolist())
        
        
H_ev = explained_variance_score(H_preds, H_labels)
plt.scatter(H_labels, H_preds)
plt.title(f'Predictions of Common Heteroscedastic Model Over Real Values (EV: {H_ev})')
plt.xlabel('Real Value')
plt.ylabel('Prediction')
plt.show()



model = EasyNN(13, n_hidden, dropout)
model.load_state_dict(torch.load('./common_regression_models_gal_split/model_epoch_2_loss_107.pth'))
preds = []
labels = []

with torch.no_grad():
    for x, y in iter(test_loader):
        mu = model.forward(x)
        preds.extend(mu.tolist())
        labels.extend(y.tolist())
        
        
ev = explained_variance_score(preds, labels)
plt.scatter(labels, preds)
plt.title(f'Predictions of Common Model Over Real Values (EV: {ev})')
plt.xlabel('Real Value')
plt.ylabel('Prediction')
plt.show()


UQNN_preds = []
UQNN_labels = []
model = UQNN(file_path = 'TestUQNN')

model.load_state_dict(torch.load(f'./UQNN_softplus_lambda_r_0.4_lambda_u_6_spli_gal/combined_epoch_48_ev_reg_86_ev_uncertainty_40.pth'))

with torch.no_grad():
    for x, y in iter(test_loader):
        mu, au, eu = model.forward(x)
        UQNN_preds.extend(mu.tolist())
        UQNN_labels.extend(y.tolist())
        
        
ev_UQNN = explained_variance_score(UQNN_preds, UQNN_labels)
plt.scatter(UQNN_labels, UQNN_preds)
plt.title(f'Predictions of UQNNN Over Real Values (EV: {ev_UQNN})')
plt.xlabel('Real Value')
plt.ylabel('Prediction')
plt.show()


fig, axes = plt.subplots(1,3, figsize = (30, 10))

axes[0].scatter(labels, preds)
axes[0].set_title(f'Common Model (EV: {ev:.4f})', fontsize = 20)
axes[0].set_xlabel('Real Value', fontsize = 20)
axes[0].set_ylabel('Prediction', fontsize = 20)

axes[1].scatter(H_labels, H_preds)
axes[1].set_title(f'Heteroscedastic Model (EV: {H_ev:.4f})', fontsize = 20)
axes[1].set_xlabel('Real Value', fontsize = 20)
axes[1].set_ylabel('Prediction', fontsize = 20)

axes[2].scatter(UQNN_labels, UQNN_preds)
axes[2].set_title(f'H-UQNN (EV: {ev_UQNN:.4f})', fontsize = 20)
axes[2].set_xlabel('Real Value', fontsize = 20)
axes[2].set_ylabel('Prediction', fontsize = 20)

fig.suptitle('Comparison of Predictions of Different Methods on Boston Housing Dataset', fontsize = 32)
plt.show()
#
