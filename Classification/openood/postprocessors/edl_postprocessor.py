from typing import Any

import torch
from torch import nn

from tqdm import tqdm
from torch.utils.data import DataLoader
import openood.utils.comm as comm
from .base_postprocessor import BasePostprocessor


class EDLPostprocessor(BasePostprocessor):
    def __init__(self, config):
        self.config = config
        self.args = config.postprocessor.postprocessor_args


    def inference(self,
                  net: nn.Module,
                  data_loader: DataLoader,
                  progress: bool = True):
        pred_list, conf_list, label_list = [], [], []
        for batch in tqdm(data_loader,
                          disable=not progress or not comm.is_main_process()):
            data = batch['data'].cuda()
            label = batch['label'].cuda()
            pred, conf = self.postprocess(net, data)

            pred_list.append(pred.cpu())
            conf_list.append(conf.cpu())
            label_list.append(label.cpu())

        # convert values into numpy array
        pred_list = torch.cat(pred_list).numpy().astype(int)
        conf_list = torch.cat(conf_list).numpy()
        label_list = torch.cat(label_list).numpy().astype(int)
        print(conf_list.shape)
        print(pred_list.shape)
        print(label_list.shape)
        return pred_list, conf_list, label_list

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        alpha = net(data).squeeze().cpu()
        prob = alpha / alpha.sum()
        uncertainty = 10 / alpha.sum(dim = -1)
        pred = torch.argmax(alpha, dim = -1)
        return pred, uncertainty * -1