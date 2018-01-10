from PIL import Image
import os
from random import randint

import torch
from torch.utils.data import Dataset


class PennActionRgbData(Dataset):
    def __init__(self, penn_action_dict, root_dir, transform=None, train=True):
        self.keys = penn_action_dict.keys()
        self.values = penn_action_dict.values()
        self.root_dir = root_dir
        self.transform = transform
        self.train = train  # set dataset instance to training or testing mode

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        if self.train:
            video, nb_frames = self.keys[idx].split('[@]')
            frame_idx = randint(1, int(nb_frames))
        else:
            video, frame_idx = self.keys[idx].split('[@]')

        post_fix = os.path.join(video, str(frame_idx).zfill(6) + '.jpg')
        img_path = os.path.join(self.root_dir, post_fix)

        with Image.open(img_path) as img:
            transformed_img = self.transform(img)
            
            label = self.values[idx]
            label = int(label) - 1
            if self.train:
                sample = (transformed_img, label)
            else:
                sample = (video, transformed_img, label)

            return sample
