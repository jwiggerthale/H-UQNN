import torch.nn as nn
import torch.nn.functional as F
import torch

def entropy(p, dim=-1, eps=1e-12):
    p = p.clamp_min(eps)
    return -(p * p.log())#.sum(dim=dim)

class UQNNNet(nn.Module):
    def __init__(self, backbone, dropout_p):
        super(UQNNNet, self).__init__()
        self.backbone = backbone
        self.dropout_p = dropout_p

    def forward(self, x, use_dropout=True):
        if use_dropout:
            return self.forward_with_dropout(x)
        else:
            return self.backbone(x)

    def forward_with_dropout(self, x):
        _, feature = self.backbone(x, return_feature=True)
        feature = F.dropout2d(feature, self.dropout_p, training=True)
        mu = self.backbone.fc(feature)
        log_var  = self.backbone.fc_log_var(feature)
        eu = self.backbone.fc_eu(feature)

        #process mu +log var 
        std = torch.exp(0.5 * log_var)
        B, C = mu.shape
        eps = torch.randn(40, B, C, device=mu.device)
        logits = mu.unsqueeze(0) + eps * std.unsqueeze(0)  # (S, B, C)
        probs  = F.softmax(logits, dim=-1)                 # (S, B, C)
        logits_cls = probs.mean(dim=0)                   # (B, C)  -> speichere!
        aleatoric_entropy = entropy(probs, dim=-1).mean(dim=0)  # (B,) == AU

        

        return logits_cls, aleatoric_entropy, eu
