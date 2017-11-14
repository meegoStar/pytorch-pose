import os
import argparse

import torch.nn as nn

from settings import visible_gpus
from settings.paths_and_names import paths_names_instance
from networks.pose_stream_net import PoseStreamNet
from utils.training_flow import TrainingFlow


parser = argparse.ArgumentParser(description='PyTorch-pose Pose Stream Training')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to the latest checkpoint (default: none)')


if __name__ == '__main__':
    # Parameters
    os.environ['OMP_NUM_THREADS'] = '1' # for preventing dataloader stuck issue
    SUB_JHMDB_CLASSES = 12  # sub JHMDB contains 12 classes
    TRAIN_DICT_PATH = paths_names_instance.TRAIN_DICT_PATH
    TEST_DICT_PATH = paths_names_instance.TEST_DICT_PATH
    JHMDB_RGB_ROOT = paths_names_instance.JHMDB_RGB_ROOT
    CHECKPOINT_NAME = paths_names_instance.CHECKPOINT_NAME
    BEST_MODEL_NAME = paths_names_instance.BEST_MODEL_NAME
    TRAIN_CSV_NAME = paths_names_instance.TRAIN_CSV_NAME
    TEST_CSV_NAME = paths_names_instance.TEST_CSV_NAME

    # Hyper parameters
    epochs = 50
    batch_size = 16
    lr = 1e-5
    num_classes = SUB_JHMDB_CLASSES
    num_workers = 2
    freeze_pose_feature_net = False

    # Initialize
    pose_stream_net = PoseStreamNet(num_classes, freeze_pose_feature_net=freeze_pose_feature_net)
    pose_stream_net = nn.DataParallel(pose_stream_net).cuda()

    args = parser.parse_args()
    training_flow = TrainingFlow(epochs, batch_size, lr, num_classes,
                                 pose_stream_net, num_workers,
                                 train_dict_path=TRAIN_DICT_PATH,
                                 test_dict_path=TEST_DICT_PATH,
                                 jhmdb_rgb_root=JHMDB_RGB_ROOT,
                                 checkpoint_name=CHECKPOINT_NAME,
                                 best_model_name=BEST_MODEL_NAME,
                                 train_csv_name=TRAIN_CSV_NAME,
                                 test_csv_name=TEST_CSV_NAME,
                                 args=args)

    # Train
    training_flow.train()
