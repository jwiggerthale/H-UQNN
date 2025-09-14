import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from openood.utils import Config
from modules.uncertainty_utils import heteroscedastic_ce

from .lr_scheduler import cosine_annealing


class UQNN_Trainer:
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
    def combined_loss(self, 
                      mu: torch.tensor, 
                      y: torch.tensor, 
                      pred_eu: torch.tensor, 
                      eu: torch.tensor, 
                      log_var: torch.tensor):
        
        ce_loss = heteroscedastic_ce(mu, log_var, target = y)
        #eu = eu / (eu.max() + 1e-8)
        
        eu_loss = nn.functional.mse_loss(pred_eu, eu)
        total_loss = (ce_loss * self.lambda_c + 
                      eu_loss * self.lambda_eu)
        loss = {'total': total_loss, 
                'ce': ce_loss,
                'eu': eu_loss}
        return loss
    

    def mc_predict(self, 
                   data, 
                   num_samples: int = 20):
        preds  = []
        for _ in range(num_samples):
            pred, _, _ = self.net.forward_with_dropout(data)
            preds.append(pred)
        preds = torch.stack(preds, dim = 0)
        pred = torch.mean(preds, dim = 0)
        mc_eu = preds.std(dim = 0)
        return preds, mc_eu
        

    def train_epoch(self, 
                    epoch_idx):
        self.net.train()
        pretrain = False
        if epoch_idx < 51: 
            pretrain = True
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
            if pretrain == True: 
                mu, log_var, eu = self.net.forward(data, self.p)
                loss = heteroscedastic_ce(target = target, 
                                            mu = mu, 
                                            log_var = log_var)
            else:
                mu, pred_au, pred_eu = self.net.forward_with_dropout(data)
                _, mc_eu = self.mc_predict(data = data)
                loss = self.combined_loss(mu = mu, 
                                            y = target, 
                                            pred_eu = pred_eu, 
                                            eu = mc_eu, 
                                            log_var = pred_au)
                with open(f'{self.config.output_dir}/losses.csv' , 'a', encoding='utf-8') as out_file:
                    out_file.write(f"{loss['total']},{loss['ce']},{loss['eu']}\n")
                loss = loss['total']
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