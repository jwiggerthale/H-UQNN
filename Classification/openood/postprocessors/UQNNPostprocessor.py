from typing import Any

import torch
from torch import nn

from .base_postprocessor import BasePostprocessor


class UQNNPostprocessor(BasePostprocessor):
    def __init__(self, config):
        self.config = config
        self.args = config.postprocessor.postprocessor_args
        self.dropout_times = self.args.dropout_times

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        logits_mean, au, eu = net.forward(data)
        au = au.mean(dim = 1)
        eu = eu.mean(dim = 1)
        pred = torch.argmax(logits_mean, dim = 1)
        uncertainty = eu + au
        return pred, uncertainty * -1