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

        self.num_classes = num_classes
        self.pose_feature_net = PoseFeatureNet()

        self.upsample_1 = nn.Upsample(scale_factor=4, mode='bilinear')

        # Appened some new layers to perform inference of action labels from
        # features extracted by PoseFeatureNet
        input_channels = self.pose_feature_net.num_classes
        self.pose_resnet = make_pose_resnet(self.num_classes, input_channels)

        if freeze_pose_feature_net:
            self.freeze_pose_feature_net()

    def freeze_pose_feature_net(self):
        for param in self.pose_feature_net.parameters():
            param.requires_grad = False

    def forward(self, x):
        pose_feature_list = self.pose_feature_net(x)
        best_pose_feature = pose_feature_list[-1]  # use the last stack of pose estimation
        upsampled = self.upsample_1(best_pose_feature)
        y_head = self.pose_resnet(upsampled)
        return y_head
