#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 20 18:17:14 2025

@author: jwiggerthale
"""


import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from HUQNN import HUQNN


# get nodel
model = HUQNN(file_path = 'TestModel')
fp = './MNIST_HUQNN/Model.pth'
model.load_state_dict(torch.load(fp))
    
'''
Class which provides list of rotated images
initialize with: 
  df: pd.DataFrame --> data frame with images and labels (labels = last column)
__getitem__ returns: 
  list of rotated images(0 - 100 degree in 5 degree steps)
  label
'''
class rotation_mnist_dataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.data = df
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.1307], std=[0.3081]),
        ])
        self.rotation_angles = range(0, 100, 5)  # 0 bis 95 Grad in 5-Grad-Schritten

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


# get data
np.random.seed(1)
train = pd.read_csv('/t1/erlangen/users/jwiggerthale/TestDatasets/MNIST/mnist_train.csv').drop('Unnamed: 0', axis = 1)
train = train.iloc[:6000, :]
test = pd.read_csv('/t1/erlangen/users/jwiggerthale/TestDatasets/MNIST/mnist_test.csv').drop('Unnamed: 0', axis = 1)
test = test.iloc[:1000]

# create dataset and dataloader
train_set = rotation_mnist_dataset(train)
test_set = rotation_mnist_dataset(test)

train_loader = DataLoader(train_set, batch_size = 1, shuffle = True)
test_loader = DataLoader(test_set, batch_size = 1, shuffle = False)


# get some images (20, adapt as required)
all_ims = []
all_labels = []
for i in range(20):
    ims, label = next(iter(train_loader))
    all_ims.append(ims)
    all_labels.append(label)


# make predictions on each image
all_preds = []
all_uncertainties = []
all_predicted_classes = []
all_eus = []
all_aus = []
for ims in all_ims:
    preds = []
    labels = []
    uncertainties = []
    predicted_classes = []
    eus= []
    aus = []
    for im in ims:
        with torch.no_grad():
            pred, au, eu = model.forward(im)
        un = au + eu
        preds.append(pred.cpu().numpy())
        aus.append(au)
        eus.append(eu)
        uncertainties.append(un)
        predicted_class = pred.argmax(dim = 1)
        predicted_classes.extend([p.item() for p in predicted_class])
    all_preds.append(preds)
    all_uncertainties.append(uncertainties)
    all_predicted_classes.append(predicted_classes)
    all_aus.append(aus)
    all_eus.append(eus)
        
    



# plot results 
for i, predicted_classes in enumerate(all_predicted_classes):
    angles = range(0, 100, 5)  # Erzeugt eine Winkelreihe (wenn nÃƒÂ¶tig anpassen)
    num_classes = 10  # Anzahl der Klassen
    classes = np.unique(predicted_classes)
    colors = plt.cm.jet(np.linspace(0, 1, num_classes))
    
    # eus to np.array
    uncertainties = [u.cpu().detach().numpy() for u in all_eus[i]]  
    softmax_outputs = np.array(uncertainties)
    
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    # (a) Softmax Input Scatter (Logits)
    for class_idx in classes:
        logits_per_class = [pred[0][class_idx].item() for pred in all_preds[i]] # get logits for predicted class
        axes[0].scatter(angles, logits_per_class, color=colors[class_idx], alpha=0.5, label=f'Class {class_idx}')
        probs_per_class = softmax_outputs.reshape(20, 10)[:, class_idx] # get eus for predicted class
        normalized_probs = probs_per_class# / logits_per_class
        axes[1].scatter(angles, abs(normalized_probs), color=colors[class_idx], alpha=0.5, label=f'Class {class_idx}')

    # logits on first plot
    axes[0].set_title('Predicted Value for Class', fontsize = 20)
    axes[0].set_xlabel("Angle", fontsize = 16)
    axes[0].set_ylabel("Predicted value",  fontsize = 16)
    axes[0].legend(loc="lower left",  fontsize = 16)
    
    # EUs in second plot
    axes[1].set_title("Epistemic Uncertainty", fontsize = 20)
    axes[1].set_xlabel("Angle",  fontsize = 16)
    axes[1].set_ylabel("Uncertainty",  fontsize = 16)
    axes[1].legend(loc="upper left",  fontsize = 16)

    # little version of rotated image under main plot
    ims_to_plot = [all_ims[i][0], all_ims[i][4], all_ims[i][9], all_ims[i][14], all_ims[i][19]]
    for i, img in enumerate(ims_to_plot):
        ax = fig.add_axes([0.09 + i * 0.08, -0.1, 0.07, 0.07], anchor='NE', zorder=1)
        ax.imshow(img.squeeze(), cmap='gray')
        ax.axis("off")
        ax = fig.add_axes([0.51 + i * 0.08, -0.1, 0.07, 0.07], anchor='NE', zorder=1)
        ax.imshow(img.squeeze(), cmap='gray')
        ax.axis("off")  
    plt.show()




