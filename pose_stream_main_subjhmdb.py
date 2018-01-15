import os
import argparse

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
import torchvision.transforms as transforms

from settings import visible_gpus
from settings.paths_and_names import paths_names_instance

from utils.training_flow_simplified import TrainingFlowSimplified
from utils.accuracy_simplified import Accuracy
from utils.datasets import jhmdb
from utils import pickle_tools

#from networks.pose_stream_net import PoseStreamNet
from networks.pose_stream_net_upgraded import PoseStreamNet


parser = argparse.ArgumentParser(description='PyTorch-pose Pose Stream Training')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to the latest checkpoint (default: none)')


if __name__ == '__main__':
    # Parameters
    os.environ['OMP_NUM_THREADS'] = '1' # for preventing dataloader stuck issue
    SUB_JHMDB_CLASSES = 12  # sub JHMDB contains 12 classes

    JHMDB_RGB_ROOT = paths_names_instance.JHMDB_RGB_ROOT
    #JHMDB_RGB_ROOT = paths_names_instance.JHMDB_CROPPED_RGB_ROOT
    if JHMDB_RGB_ROOT == paths_names_instance.JHMDB_CROPPED_RGB_ROOT:
        CROPPED_VERSION = True
    else:
        CROPPED_VERSION = False

    JHMDB_TRAIN_DICT_PATH = paths_names_instance.JHMDB_TRAIN_DICT_PATH
    JHMDB_TEST_DICT_PATH = paths_names_instance.JHMDB_TEST_DICT_PATH
    
    CHECKPOINT_NAME = paths_names_instance.CHECKPOINT_NAME
    BEST_MODEL_NAME = paths_names_instance.BEST_MODEL_NAME
    CSV_LOG_NAME = paths_names_instance.CSV_LOG_NAME

    # Hyper parameters
    num_classes = SUB_JHMDB_CLASSES
    epochs = 50
    batch_size = 8
    lr = 1e-5
    saturate_patience = 12
    reduce_patience = 4
    freeze_pose_feature_net = False

    arch = 'PoseFeatureNet-BilinearUpsampling-ResNet18'

    # Initialize
    args = parser.parse_args()

    pose_stream_net = PoseStreamNet(num_classes, freeze_pose_feature_net=freeze_pose_feature_net)
    pose_stream_net = nn.DataParallel(pose_stream_net).cuda()

    loss_function = nn.CrossEntropyLoss().cuda()
    compute_batch_accuracy = Accuracy().compute_batch_accuracy

    if freeze_pose_feature_net:
        #params_to_optimize = pose_stream_net.module.pose_resnet.parameters()
        params_to_optimize = pose_stream_net.module.pose_classification_sub_net.parameters()
    else:
        params_to_optimize = pose_stream_net.parameters()

    #optimizer = SGD(params_to_optimize, lr=lr, momentum=0.9, weight_decay=5e-4)
    optimizer = Adam(params_to_optimize, lr=lr)

    train_dict = pickle_tools.load_pickle(JHMDB_TRAIN_DICT_PATH)
    test_dict = pickle_tools.load_pickle(JHMDB_TEST_DICT_PATH)
    
    train_set = jhmdb.JhmdbRgbData(train_dict, JHMDB_RGB_ROOT,
                                   cropped_version=CROPPED_VERSION,
                                   transform=transforms.Compose([
                                       transforms.Scale([256, 256]),
                                       transforms.ToTensor()]))

    test_set = jhmdb.JhmdbRgbData(test_dict,JHMDB_RGB_ROOT,
                                  train=False,
                                  cropped_version=CROPPED_VERSION,
                                  transform=transforms.Compose([
                                      transforms.Scale([256, 256]),
                                      transforms.ToTensor()]))

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)
    
    training_flow = TrainingFlowSimplified(model=pose_stream_net,
                                           params_to_optimize=params_to_optimize,
                                           loss_function=loss_function,
                                           compute_batch_accuracy=compute_batch_accuracy,
                                           optimizer=optimizer,
                                           train_loader=train_loader,
                                           test_loader=test_loader,
                                           epochs=epochs,
                                           saturate_patience=saturate_patience,
                                           reduce_patience=reduce_patience,
                                           freeze_pose_feature_net=freeze_pose_feature_net,
                                           csv_log_name=CSV_LOG_NAME,
                                           checkpoint_name=CHECKPOINT_NAME,
                                           best_model_name=BEST_MODEL_NAME,
                                           arch=arch,
                                           args=args)

    # Train
    training_flow.train()
