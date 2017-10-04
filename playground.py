from __future__ import print_function, absolute_import

import os
import argparse
import time
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torchvision.datasets as datasets

from pose import Bar
from pose.utils.logger import Logger, savefig
from pose.utils.evaluation import accuracy, AverageMeter, final_preds
from pose.utils.misc import save_checkpoint, save_pred, adjust_learning_rate
from pose.utils.osutils import mkdir_p, isfile, isdir, join
from pose.utils.imutils import batch_with_heatmap
from pose.utils.transforms import fliplr, flip_back
import pose.models as models
import pose.datasets as datasets

# parameters
arch = 'hg'
stacks = 1
blocks = 1
classes = 16
weights_path = '/home/ubuntu/cvlab/meego/pytorch-pose/pretrained_weights/hg_s1_b1/model_best.pth.tar'

base_model = models.__dict__[arch](num_stacks=stacks, num_blocks=blocks, num_classes=classes) # load the base model
base_model = torch.nn.DataParallel(base_model).cuda()
checkpoint = torch.load(weights_path)
base_model.load_state_dict(checkpoint['state_dict'])
#import pdb; pdb.set_trace()
# construct the pose feature extraction component from stacked hourglass networks
extraction_component = nn.Sequential(*list(base_model.children())[:10])

