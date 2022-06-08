from contextlib import contextmanager


# other
import random
from collections import OrderedDict
import numpy as np
import time
import os
from sklearn.metrics import roc_auc_score
import datetime
import os
from tqdm import tqdm

# torch
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torch.nn.parallel.replicate import replicate
import torch.distributed as dist


from copy import deepcopy
import math
import time
import torch
import os




def print_cfg(logger, cfg):
	logger.info('')
	logger.info('*' * 50)
	for key, value in vars(cfg).items():
		if key.startswith('_'):
			continue
		logger.info('{key}: {value}'.format(key=key, value=value))
	logger.info('*' * 50)




@contextmanager
def torch_distributed_zero_first(local_rank: int):
	"""
	Decorator to make all processes in distributed training wait for each local_master to do something.
	"""
	if local_rank not in [-1, 0]:
		torch.distributed.barrier()
	yield
	if local_rank == 0:
		torch.distributed.barrier()



def seed_everything(seed):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)

	# faster, less reproducible
	#torch.backends.cudnn.benchmark = True
	#torch.backends.cudnn.deterministic = False

	# slower, more reproducible
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True






def is_parallel(model):
    # is model is parallel with DP or DDP
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)


def copy_attr(a, b, include=(), exclude=()):
    # Copy attributes from b to a, options to only include [...] and to exclude [...]
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith('_') or k in exclude:
            continue
        else:
            setattr(a, k, v)

class ModelEMA:
    """ Model Exponential Moving Average from https://github.com/rwightman/pytorch-image-models
    Keep a moving average of everything in the model state_dict (parameters and buffers).
    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    A smoothed version of the weights is necessary for some training schemes to perform well.
    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    """

    def __init__(self, model, decay=0.9999, updates=0):
        # Create EMA
        self.ema = deepcopy(model.module if is_parallel(model) else model).eval()  # FP32 EMA
        # if next(model.parameters()).device.type != 'cpu':
        #     self.ema.half()  # FP16 EMA
        self.updates = updates  # number of EMA updates
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000))  # decay exponential ramp (to help early epochs)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        # Update EMA parameters
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)

            msd = model.module.state_dict() if is_parallel(model) else model.state_dict()  # model state_dict
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1. - d) * msd[k].detach()

    def update_attr(self, model, include=(), exclude=('process_group', 'reducer')):
        # Update EMA attributes
        copy_attr(self.ema, model, include, exclude)









	