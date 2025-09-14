from openood.evaluators.mos_evaluator import MOSEvaluator
from openood.utils import Config

from .ad_evaluator import ADEvaluator
from .arpl_evaluator import ARPLEvaluator
from .base_evaluator import BaseEvaluator
from .ece_evaluator import ECEEvaluator
from .fsood_evaluator import FSOODEvaluator
from .ood_evaluator import OODEvaluator
from .osr_evaluator import OSREvaluator
from .patchcore_evaluator import PatchCoreEvaluator
from .uqnn_evaluator import UQNNEvaluator
from.edl_evaluator import EDLEvaluator



def get_evaluator(config: Config):
    evaluators = {
        'uqnn': UQNNEvaluator, 
        'base': BaseEvaluator,
        'ood': OODEvaluator,
        'fsood': FSOODEvaluator,
        'patch': PatchCoreEvaluator,
        'arpl': ARPLEvaluator,
        'ad': ADEvaluator,
        'mos': MOSEvaluator,
        'ece': ECEEvaluator,
        'osr': OSREvaluator, 
        'edl': EDLEvaluator
    }
    return evaluators[config.evaluator.name](config)
