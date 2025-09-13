from modules import BayesianNN, EasyNN
from HUQNN import HUQNN
from utils import get_data_loader_from_pandas

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


HUQNN_preds = []
HUQNN_labels = []
model = HUQNN(file_path = 'TestHUQNN')

model.load_state_dict(torch.load(f'./HUQNN_softplus_lambda_r_0.4_lambda_u_6_spli_gal/combined_epoch_48_ev_reg_86_ev_uncertainty_40.pth'))

with torch.no_grad():
    for x, y in iter(test_loader):
        mu, au, eu = model.forward(x)
        HUQNN_preds.extend(mu.tolist())
        HUQNN_labels.extend(y.tolist())
        
        
ev_HUQNN = explained_variance_score(HUQNN_preds, HUQNN_labels)
plt.scatter(HUQNN_labels, HUQNN_preds)
plt.title(f'Predictions of H-UQNNN Over Real Values (EV: {ev_HUQNN})')
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

axes[2].scatter(HUQNN_labels, HUQNN_preds)
axes[2].set_title(f'H-UQNN (EV: {ev_HUQNN:.4f})', fontsize = 20)
axes[2].set_xlabel('Real Value', fontsize = 20)
axes[2].set_ylabel('Prediction', fontsize = 20)

fig.suptitle('Comparison of Predictions of Different Methods on Boston Housing Dataset', fontsize = 32)
plt.show()
#
