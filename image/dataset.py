import os
import torch
import prnu
import pyvips
import pickle
import numpy as np
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self, pkl_dir, vision_dir, transforms=None):
        self.data = []
        self.vision_dir = vision_dir
        self.length = 512
        self.transforms = transforms
        with open(pkl_dir, "rb") as f:
            self.data = pickle.load(f)

        self.n_classes = len(np.unique([d['target'] for d in self.data]))
        self.crop_size = (256, 256, 3)

    def __len__(self):
        return len(self.data)

    def usingVIPSandShrink(self, f):
        image = pyvips.Image.new_from_file(f, access="sequential")  # RGB
        return image.numpy(dtype=np.float32)

    def __getitem__(self, idx):
        entry = self.data[idx]
        img_path = os.path.join(self.vision_dir, entry['device_name'], entry['image_name'] + '.jpg')

        values_cropped = prnu.cut_ctr(self.usingVIPSandShrink(img_path), self.crop_size)
        values = np.transpose(values_cropped, (2, 0, 1))

        # values = torch.Tensor(values)
        values = torch.from_numpy(values)
        if self.transforms:
            values = self.transforms(values)
        target = torch.LongTensor([entry["target"]])
        return values, target, entry['device_name']
