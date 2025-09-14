import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from openood.utils import Config
from keras.utils import to_categorical
from .lr_scheduler import cosine_annealing


# KL divergence of predicted parameters from uniform Dirichlet distribution
# from https://arxiv.org/abs/1806.01768
# code based on:
# https://bariskurt.com/kullback-leibler-divergence-between-two-dirichlet-and-beta-distributions/
def dirichlet_reg(alpha, y):
    # dirichlet parameters after removal of non-misleading evidence (from the label)
    alpha = y + (1 - y) * alpha

    # uniform dirichlet distribution
    beta = torch.ones_like(alpha)

    sum_alpha = alpha.sum(-1)
    sum_beta = beta.sum(-1)

    t1 = sum_alpha.lgamma() - sum_beta.lgamma()
    t2 = (alpha.lgamma() - beta.lgamma()).sum(-1)
    t3 = alpha - beta
    t4 = alpha.digamma() - sum_alpha.digamma().unsqueeze(-1)

    kl = t1 - t2 + (t3 * t4).sum(-1)
    return kl.mean()


# Eq. (5) from https://arxiv.org/abs/1806.01768:
# Sum of squares loss
def dirichlet_mse(alpha, y):
    sum_alpha = alpha.sum(-1, keepdims=True)
    p = alpha / sum_alpha
    t1 = (y - p).pow(2).sum(-1)
    t2 = ((p * (1 - p)) / (sum_alpha + 1)).sum(-1)
    mse = t1 + t2
    return mse.mean()


def evidential_classification(alpha, y, lamb=1.0):
    num_classes = alpha.shape[-1]
    y = F.one_hot(y, num_classes)
    return dirichlet_mse(alpha, y) + lamb * dirichlet_reg(alpha, y)



def KL(alpha, nb_classes):
    alpha = torch.clamp(alpha, min=1e-7, max=1e7)
    alpha = alpha.cuda()
    beta = torch.ones((1, nb_classes)).cuda()
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)
    lnB = torch.lgamma(S_alpha).cuda() - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True).cuda()
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True).cuda() - torch.lgamma(S_beta).cuda()

    dg0 = torch.digamma(S_alpha).cuda()
    dg1 = torch.digamma(alpha).cuda()

    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
    return kl




def dirichlet_loss(nb_classes):
    def func(target, evidential_output, global_step, annealing_step):
        alpha, preds, epistemic_uncertainty, aleatoric_uncertainty = evidential_output

        S = torch.sum(alpha, dim=1, keepdim=True)
        E = alpha - 1
        m = alpha / S

        A = torch.sum((target - m) ** 2, dim=1, keepdim=True)
        B = torch.sum(alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True)

        annealing_coef = torch.minimum(torch.tensor(1.0), torch.tensor(global_step / annealing_step))

        alp = E * (1 - target) + 1
        C = annealing_coef * KL(alp, nb_classes)
        return torch.mean((A + B) + C)

        """
        check nb_classes
        check softmax for alp
        kl divergence
        """

    return func


class EDL_Trainer:
    def __init__(self, net: nn.Module, train_loader: DataLoader,
                 config: Config) -> None:
        print(config)
        self.lambda_c = config.uncertainty_lambda_c
        self.lambda_eu = config.uncertainty_lambda_eu
        self.net = net
        self.train_loader = train_loader
        self.config = config
        self.p = config.trainer.dropout_p

        self.optimizer = torch.optim.SGD(
            net.parameters(),
            config.optimizer.lr,
            momentum=config.optimizer.momentum,
            weight_decay=config.optimizer.weight_decay,
            nesterov=True,
        )

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: cosine_annealing(
                step,
                config.optimizer.num_epochs * len(train_loader),
                1,
                1e-6 / config.optimizer.lr,
            ),
        )
        

    def train_epoch(self, 
                    epoch_idx):
        self.net.train()
        loss_avg = 0.0
        train_dataiter = iter(self.train_loader)

        for train_step in tqdm(range(1,
                                     len(train_dataiter) + 1),
                               desc='Epoch {:03d}: '.format(epoch_idx),
                               position=0,
                               leave=True):
            batch = next(train_dataiter)
            data = batch['data'].cuda()
            target = batch['label'].cuda()
            # forward
            pred = self.net.forward(data)
            loss = evidential_classification(pred, target, lamb=min(1, epoch_idx / 10))
            
            
            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            # exponential moving average, show smooth values
            with torch.no_grad():
                loss_avg = loss_avg * 0.8 + float(loss) * 0.2

        metrics = {}
        metrics['epoch_idx'] = epoch_idx
        metrics['loss'] = loss_avg

        return self.net, metrics