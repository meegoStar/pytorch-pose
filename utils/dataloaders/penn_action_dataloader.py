from __future__ import print_function

import os

from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from utils import pickle_tools
from utils.datasets import penn_action


class PennActionRgbDataLoader():
    def __init__(self, batch_size, num_workers, data_path, dict_path):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_path = data_path
        self.dict_path = dict_path

        self.load_dicts()  #load data dicts

    def load_dicts(self):
        train_video_dict_path = os.path.join(self.dict_path, 'train_video.pickle')
        test_video_dict_path = os.path.join(self.dict_path, 'test_video.pickle')
        frame_count_dict_path = os.path.join(self.dict_path, 'frame_count.pickle')

        self.train_video = pickle_tools.load_pickle(train_video_dict_path)
        self.test_video = pickle_tools.load_pickle(test_video_dict_path)
        self.frame_count = pickle_tools.load_pickle(frame_count_dict_path)

    def run(self):
        self.test_frame_sampling()
        self.train_video_labeling()
        train_loader = self.train()
        val_loader = self.val()
        return train_loader, val_loader
    
    def test_frame_sampling(self):  # uniformly sample 18 frames and  make a video level consenus
        self.dic_test_idx = {}
        for video in self.test_video:  # dic[video] = label
            nb_frames = int(self.frame_count[video]) - 1  # currently, the number of generated cropped images for each video is one less than the number of original frames
            interval = int(nb_frames / 18)
            for i in range(18):
                frame_idx = i * interval + 1
                key = video + '[@]' + str(frame_idx)
                self.dic_test_idx[key] = self.test_video[video]

    def train_video_labeling(self):
        self.dic_video_train = {}
        for video in self.train_video:  # dic[video] = label
            nb_frames = int(self.frame_count[video]) - 1  # currently, the number of generated cropped images for each video is one less than the number of original frames
            key = video +'[@]' + str(nb_frames)
            self.dic_video_train[key] = self.train_video[video]
                            
    def train(self):
        training_set = penn_action.PennActionRgbData(penn_action_dict=self.dic_video_train,
                                                     root_dir=self.data_path,
                                                     transform=transforms.Compose([
                                                         transforms.Scale((256, 256)),
                                                         #transforms.RandomCrop(224),
                                                         transforms.RandomHorizontalFlip(),
                                                         transforms.ToTensor(),
                                                         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                                         ]))
        print('==> Training data :', len(training_set),' videos')
        print(training_set[1][0].size())

        train_loader = DataLoader(dataset=training_set,
                                  batch_size=self.batch_size,
                                  shuffle=True,
                                  num_workers=self.num_workers)
        return train_loader

    def val(self):
        validation_set = penn_action.PennActionRgbData(penn_action_dict=self.dic_test_idx,
                                                       root_dir=self.data_path,
                                                       train=False,
                                                       transform=transforms.Compose([
                                                           transforms.Scale((256, 256)),
                                                           #transforms.CenterCrop(224),
                                                           transforms.ToTensor(),
                                                           transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                                           ]))
        print('==> Validation data :', len(validation_set),' clips')
        print(validation_set[1][1].size())

        val_loader = DataLoader(dataset=validation_set,
                                batch_size=self.batch_size,
                                shuffle=False,
                                num_workers=self.num_workers)
        return val_loader