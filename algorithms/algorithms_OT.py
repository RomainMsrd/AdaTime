import torch
import torch.nn as nn
import numpy as np
import itertools

from models.models import classifier, ReverseLayerF, Discriminator, RandomLayer, Discriminator_CDAN, \
    codats_classifier, AdvSKM_Disc, CNN_ATTN
from models.loss import SliceWassersteinDiscrepancy
from utils import EMA
from torch.optim.lr_scheduler import StepLR
from copy import deepcopy
import torch.nn. functional as F
from algorithms import Algorithm, MCD
class SWD(MCD):
    def __init__(self, backbone, configs, hparams, device):
        super().__init__(configs, backbone)
        self.N = self.hparams['N']
        self.swd_loss= SliceWassersteinDiscrepancy(self.N)
    def discrepancy(self, out1, out2):
        return self.swd_loss(out1, out2)