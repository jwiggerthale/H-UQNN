from openood.evaluation_api import Evaluator
from UQNN import UQNN
from resnet_18 import ResNet18, MC_Model
import torch
import os


from openood.utils.config import setup_config_manual
from openood.networks.utils import get_network

fp = None


postprocessor_name = 'uqnn'
#postprocessor_name = 'dropout'
#postprocessor_name = 'edl'

# adapt configs according to model you want to test (refer to the original OpenOOD repo for more information)
cfgs = ['configs/datasets/cifar10/cifar10.yml', 
        'configs/networks/uqnn_net.yml', 
        'configs/pipelines/train/baseline.yml',
        'configs/preprocessors/base_preprocessor.yml']

config = setup_config_manual(cfgs= cfgs)
model = get_network(config)

#path to your model
fp = './CIFAR_HUQNN/Model.pth
#path where you want to store csv file of results
dest_path = f"./benchmark_results_2/{fp.replace('/', '_').replace('.pth', '.csv')}"

#create evaluator
evaluator = Evaluator(model, 
      id_name = 'cifar10', 
      data_root = './benchmark_data', 
      config_root = None, 
      preprocessor = None, 
      postprocessor_name = postprocessor_name, 
      postprocessor = None, 
      batch_size = 200, 
      shuffle = False, 
      num_workers = 1)

# load state dict to evaluator's model
evaluator.net.load_state_dict(torch.load(fp))
evaluator.net.to('cuda')
evaluator.net.eval()
metrics = evaluator.eval_ood(fsood = False)
metrics.to_csv(dest_path)

