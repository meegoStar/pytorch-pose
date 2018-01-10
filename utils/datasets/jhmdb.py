from PIL import Image
import os

import torch
from torch.utils.data import Dataset


class JhmdbRgbData(Dataset):
    def __init__(self, jhmdb_dict, root_dir, transform=None, train=True, cropped_version=False):
        self.keys = jhmdb_dict.keys()
        self.values = jhmdb_dict.values()
        self.root_dir = root_dir
        self.transform = transform
        self.train = train # set dataset instance to training or testing mode

        if cropped_version:
            self.extension = '.jpg'  # cropped version images are saved in '.jpg'
        else:
            self.extension = '.png'  # no-cropped version images are saved in '.png'

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        key_without_extension = key[:key.rfind('.')]
        img_path = os.path.join(self.root_dir, key_without_extension + self.extension)

        with Image.open(img_path) as img:
            transformed_img = self.transform(img)

            label = self.values[idx]
            label = int(label) - 1
            if self.train:
                sample = (transformed_img, label)
            else:
                videoname = key.split('/')[1]
                sample = (videoname, transformed_img, label)

            return sample
