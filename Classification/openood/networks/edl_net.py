import torch.nn as nn


class EDLNet(nn.Module):
    def __init__(self, backbone):
        super(EDLNet, self).__init__()
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)


