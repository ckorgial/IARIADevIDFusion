from torch.utils.data import Dataset
import numpy as np
import pickle
import torch
import os
import pandas as pd


class AudioDataset(Dataset):
    def __init__(self, split_dir, mel_dir, transforms=None):
        self.mel_dir = mel_dir
        self.data = []
        self.length = 1500
        self.transforms = transforms
        self.data = pd.read_csv(split_dir)
        self.n_classes = len(np.unique(self.data['dev_numerical_id'].values))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        mel_path = os.path.join(self.mel_dir,
                                item['dev_alphabetical_id'],
                                item['dev_alphabetical_id'] + '.pkl')
        with open(mel_path, "rb") as f:
            values = pickle.load(f)

        mel = values['mel'].reshape(-1, 128, self.length)
        mel = torch.Tensor(mel)
        if self.transforms:
            mel = self.transforms(mel)
        target = torch.LongTensor([item['dev_numerical_id']])
        return mel, target, item['dev_alphabetical_id']
