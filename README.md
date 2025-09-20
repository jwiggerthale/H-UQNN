This repo contains code for our Paper "Heteroscedastic Uncertainty Quantifying Neural Networks - Learning to Predict Aleatoric and Epistemic Uncertainty".

The repo contains one folder for regression (Boston Housing dataset) and one folder for classificatio (MNIST and CIFAR-10 dataset). 
To reproduce our results just clone the repo and execute the apprpriate test_xx.py files.


# Idea of H-UQNN
H-UQNN is a three-headed model with a shared feature
extractor. Given an input vector $ x ∈ R^n $, the feature extractor
maps it to a representation $f ∈ R^m $. $f$ is an internal representation of the model. Given this representation, the task of the
heads is to predict the distribution of the target variable as well
as the EU of the model. The distribution of the target variable
is given by the estimated mean $\mu(f)$ and the estimated variance
 $log\sigma_{au}^2(f)$ of the target variable and therefore includes AU.
To do so, the three heads operate as follows:
- The µ-head predicts the target mean $\mu(f)$.
- The $log\sigma_{au}^2$ head predicts $log\sigma_{au}^2(f)$ , forming together
with $\mu(f)$ a heteroscedastic predictive distribution.
- The §\sigma_{eu}$-head predicts the epistemic uncertainty $\sigma_{MC-drop eu}(f)$ obtained from MC dropout.


The structure of H-UQNN is presented below: 

<img src="./Ims/H-UQNN Structure.jpg" alt="Structure of H-UQNN" style="width:50px; vertical-align:middle;">
