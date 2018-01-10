import torch
import torch.nn as nn
import torchvision.models as models

from pose_feature_net import PoseFeatureNet
import resnet


def convert_conv1_weight(conv1_weight, original_channels_num=3, new_channels_num=10):
    weight_sum = 0.
    for i in range(original_channels_num):
        weight_sum += conv1_weight[:, i, :, :]

    weight_avg = weight_sum / float(original_channels_num)

    new_conv1_weight = torch.FloatTensor(64, new_channels_num, 7, 7) # 64 is the number of filters
    for i in range(new_channels_num):
        new_conv1_weight[:, i, :, :] = weight_avg

    return new_conv1_weight


def make_pose_resnet(num_classes, input_channels):
    # Not using default pretrained weights, since pretrained weights need some
    # adjustments in the first layer
    pose_resnet = resnet.resnet18(input_channels=input_channels,
                                  pretrained=False)

    # Adjust pretrained weights here
    resnet_3channels = models.resnet18(pretrained=True)

    # Get the weight of first convolution layer (torch.FloatTensor of size 64x3x7x7)
    state_dict = resnet_3channels.state_dict()
    conv1_weight3 = state_dict['conv1.weight']

    # Average across RGB channel and replicate this average by the channel number of target network
    conv1_weight_expanded = convert_conv1_weight(conv1_weight3,
                                                 new_channels_num=input_channels)
    state_dict['conv1.weight'] = conv1_weight_expanded
    pose_resnet.load_state_dict(state_dict)

    # Replace fc1000 with fc12
    num_features = pose_resnet.fc.in_features
    pose_resnet.fc = nn.Linear(num_features, num_classes)

    return pose_resnet


class PoseStreamNet(nn.Module):
    def __init__(self, num_classes, freeze_pose_feature_net=True):
        super(PoseStreamNet, self).__init__()

        self.pose_feature_net = PoseFeatureNet()
        if freeze_pose_feature_net:
            self.freeze_pose_feature_net()

        # Appened sub net to perform inference of action labels from features extracted by PoseFeatureNet
        input_channels = self.pose_feature_net.num_classes
        self.pose_classification_sub_net = PoseClassificationSubNet(num_classes=num_classes, input_channels=input_channels)

    def freeze_pose_feature_net(self):
        for param in self.pose_feature_net.parameters():
            param.requires_grad = False

    def forward(self, x):
        pose_feature_list = self.pose_feature_net(x)
        best_pose_feature = pose_feature_list[-1]  # use the last stack of pose estimation
        y_head = self.pose_classification_sub_net(best_pose_feature)
        return y_head

class PoseClassificationSubNet(nn.Module):
    def __init__(self, num_classes, input_channels):
        super(PoseClassificationSubNet, self).__init__()

        self.upsample_1 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.conv_1 = nn.Conv2d(in_channels=input_channels, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)  # apply dimension reduction to simulate the idea of "joint position"
        self.relu_1 = nn.ReLU(inplace=True)
        #self.pose_resnet = make_pose_resnet(num_classes=num_classes, input_channels=1)    # for upsample_1 --> conv_1/relu_1 --> pose_resnet
        self.pose_resnet = make_pose_resnet(num_classes=num_classes, input_channels=16)  # for upsample_1 --> pose_resnet

    def forward(self, x):
        x = self.upsample_1(x)
        #x = self.conv_1(x)
        #x = self.relu_1(x)
        x = self.pose_resnet(x)
        return x
