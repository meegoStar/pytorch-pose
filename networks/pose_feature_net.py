from __future__ import print_function, absolute_import

import torch
import torch.nn as nn

import pose.models as models
import pose.datasets as datasets



# parameters
arch = 'hg'
stacks = 1
blocks = 1
classes = 16
weights_path = '/home/ubuntu/cvlab/meego/pytorch-pose/pretrained_weights/hg_s1_b1/model_best.pth.tar'

# construct the base Stacked Hourglass Nets model
stacked_hourglass_nets = models.__dict__[arch](num_stacks=stacks, num_blocks=blocks, num_classes=classes)
stacked_hourglass_nets = torch.nn.DataParallel(stacked_hourglass_nets).cuda()
checkpoint = torch.load(weights_path)
stacked_hourglass_nets.load_state_dict(checkpoint['state_dict']) # load pretrained weights
base_model = stacked_hourglass_nets.module



class PoseFeatureNet(nn.Module):
    def __init__(self, base_model=base_model):
        super(PoseFeatureNet, self).__init__()

        self.inplanes = base_model.inplanes
        self.num_feats = base_model.num_feats
        self.num_stacks = base_model.num_stacks

        self.conv1 = base_model.conv1
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.maxpool = base_model.maxpool

        self.hg = base_model.hg
        self.res = base_model.res
        self.fc = base_model.fc
        self.score = base_model.score
        self.fc_ = base_model.fc_
        self.score_ = base_model.score_

    def forward(self, x):
        # TODO: (to think)
        # feature maps(256 channels) v.s. result confidence maps(16 channels),
        # what's the difference between them w.r.t. the information contained in them?
        out = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.maxpool(x)
        x = self.layer2(x)
        x = self.layer3(x)

        fc_feature = []
        for i in range(self.num_stacks):
            y = self.hg[i](x)
            fc_feature.append(y)
            y = self.res[i](y)
            y = self.fc[i](y)

            score = self.score[i](y)
            out.append(score)
            if i < self.num_stacks-1:
                fc_ = self.fc_[i](y)
                score_ = self.score_[i](score)
                x = x + fc_ + score_

        #return out
        return fc_feature
