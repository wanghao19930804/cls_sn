# other
import random
from collections import OrderedDict
from matplotlib.patches import FancyArrow
import numpy as np
import time
import os
from regex import F
from sklearn.metrics import roc_auc_score
import datetime
import os
from tqdm import tqdm
import argparse
import pandas as pd

# torch
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torch.nn.parallel.replicate import replicate
import torch.distributed as dist
import torchvision


# 
from utils.logger import init_logger
from utils.util import print_cfg


import timm
import cv2

cv2.setNumThreads(0)

class Cfg():
    def __init__(self):
        self.root = './input'

        self.use_TTA = False
        



        #self.backbone = 'swin_base_patch4_window7_224_in22k'
        #self.backbone = 'swin_base_patch4_window12_384_in22k'
        #self.backbone = 'swin_large_patch4_window7_224_in22k'
        #self.backbone = 'swin_large_patch4_window12_384_in22k'  


        #self.backbone = 'tf_efficientnet_b7_ns'

        #self.backbone = 'eca_nfnet_l2'
        #self.backbone = 'eca_nfnet_l1'



        self.backbone = ''

        
        self.use_meta = False
        self.fix_backbone = False

        
        self.use_all_data = True
        #self.max_lr = 1e-3
        self.max_lr = 1e-4
        self.fold = 0
        
        self.max_epoch = 20
        self.warmup_iters = 500
        

        self.arcface_m_x =  0.45
        self.arcface_m_y = 0.05


        self.num_classes = 1572
        self.image_size = 224
        #self.image_size = 384

        self.use_cutmix = False
        self.cutmix_prob = 0.4
        self.mix_ratio = 1
        


        self.use_amp = True
        
        self.embedding_size = 512

        self.samples_per_gpu = 32
        self.workers_per_gpu = 8
        
        
        self.info = f'{self.image_size}'

        if self.use_all_data:
            self.info += '+all_data'
     
        #if self.fix_backbone:
        #    self.info += '+fix_backbone'
        
        #if self.use_meta:
        #    self.info += '+meta'

            
        if self.use_cutmix:
            self.info += '+cutmix'

        self.work_dir = None
        self.logger = None        
        
        
        
   
        









