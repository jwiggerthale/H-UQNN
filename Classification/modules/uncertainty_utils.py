#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 19 18:03:12 2025

@author: jwiggerthale
"""

import torch
import torch.nn.functional as F

'''
Function which implements heteroscedastische cross-entropy via logit-sampling
call with:
    mu: torch.tensor --> logits of mu head
    log_var: torch.tensor --> predictted log_var for logis
    target: torch.tensor --> real classes
    num_samples: int --> number of samples to sample predicted distibution 
    var_epsilon: float --> bias for var (to avoid zero var)
    penalty: regulariuer for avoid strong increase in log_var
'''
def heteroscedastic_ce(mu: torch.tensor, 
                       log_var: torch.tensor, 
                       target: torch.tensor, 
                       num_samples: int = 5, 
                       reduction: str = "mean", 
                       var_epsilon: float =1e-6, 
                       penalty: float =1e-6):
    var = F.softplus(log_var) + var_epsilon
    std = torch.sqrt(var)

    # reparametrize: z = mu + std * eps, eps ~ N(0, I)
    # stack along new dimension
    B, C = mu.shape
    eps = torch.randn(num_samples, B, C, device=mu.device, dtype=mu.dtype)
    logits_samples = mu.unsqueeze(0) + std.unsqueeze(0) * eps  # (S, B, C)

    # cross entropy per sample
    ce = []
    for s in range(num_samples):
        ce_s = F.cross_entropy(logits_samples[s], target, reduction="none")  # (B,)
        ce.append(ce_s)
    ce = torch.stack(ce, dim=0).mean(dim=0)  # (B,)

    # penalty to avoid strong increase in log_var
    var_pen = penalty * var.mean(dim=1)  # (B,)

    loss = ce + var_pen
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss


'''
Function for heteroscedastic model to get EU via MC dropout and AU via heteroscedastic loss (sampling)
call with: 
  model: nn.Module --> model to use
  x: torch.tensor --> data point(s) you want to make a prediction on
  n_dropout: int = 30 --> number of forward passes to conduct (more takses longer)
  n_noise: int = 10 --> number of samples to sample distribution of target variable (c.f. https://arxiv.org/pdf/1703.04977, Sec. 3.3)
returns: 
  dict with: 
        "probs": mean_probs,
        "pred_entropy": pred_entropy,
        "au": expected_entropy, 
        "eu": eu, 
        "probs_var_across_dropout": variance across droput forward passes,
'''
@torch.no_grad()
def mc_dropout_with_heteroscedastic(model: nn.Module, 
                                    x: torch.tensor, 
                                    n_dropout: int = 30, 
                                    n_noise: int = 10):
    import torch.nn.functional as F
    def entropy(p, dim=-1, eps=1e-12):
        p = p.clamp_min(eps)
        return -(p * p.log())#.sum(dim=dim)

    model.train()  # Dropout an
    probs_mean_per_d = []

    exp_entropy_accum = 0.0 
    for _ in range(n_dropout):
        pred = model(x)
        if len(pred) == 3:
            mu, logvar, _ = pred
        else:
            mu, logvar = pred                     # (B, C)
        std = torch.exp(0.5 * logvar)
        B, C = mu.shape

        eps = torch.randn(n_noise, B, C, device=mu.device)
        logits = mu.unsqueeze(0) + eps * std.unsqueeze(0)  # (S, B, C)
        probs  = F.softmax(logits, dim=-1)                 # (S, B, C)

        probs_mean_d = probs.mean(dim=0)                   # (B, C)  -> speichere!
        probs_mean_per_d.append(probs_mean_d)

        exp_entropy_accum += entropy(probs, dim=-1).mean(dim=0)  # (B,)

    probs_mean_stack = torch.stack(probs_mean_per_d, dim=0)      # (D, B, C)
    mean_probs = probs_mean_stack.mean(dim=0)                    # (B, C)

    pred_entropy = entropy(mean_probs)                           # H[E_{w,z}[p]]
    expected_entropy = exp_entropy_accum / n_dropout             # E_{w,z}[H[p]]
    epistemic_entropy = (pred_entropy - expected_entropy).clamp_min(0.0)
    eu = probs_mean_stack.std(dim= 0)#.mean(dim = -1)
    probs_var_across_dropout = probs_mean_stack.var(dim=0, unbiased=False).mean(dim=-1)  # (B,)

    return {
        "probs": mean_probs,
        "pred_entropy": pred_entropy,
        "au": expected_entropy, 
        "eu": eu, 
        "probs_var_across_dropout": probs_var_across_dropout,
    }

