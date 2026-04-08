# -*- coding: utf-8 -*-
import numpy as np
import torch
from torch.utils.data import Dataset


class PressSequenceDataset(Dataset):

    def __init__(self, img_path, label_path, seq_transform=None,
                 normalize_labels=True, label_stds=None, label_means=None):
        # Memory-mapped loading for efficiency
        self.imgs   = np.load(img_path, mmap_mode='r')   # (N, T, 1, H, W)
        self.labels = np.load(label_path)                # (N, D)
        self.seq_transform = seq_transform
        self.normalize_labels = normalize_labels

        if normalize_labels:
            assert label_stds is not None and label_means is not None, \
                "normalize_labels=True requires label_stds and label_means."
            self.label_stds  = torch.tensor(label_stds,  dtype=torch.float32)
            self.label_means = torch.tensor(label_means, dtype=torch.float32)

    def __len__(self):
        return int(self.imgs.shape[0])

    def __getitem__(self, idx): #각각 하나씩
        # Convert memmap array to writable numpy array
        seq_np = np.array(self.imgs[idx], copy=True)          # (T, 1, H, W)
        seq = torch.from_numpy(seq_np).float().div_(255.0)    # Normalize to [0, 1]
        seq = seq.repeat(1, 3, 1, 1)                          # Convert to (T, 3, H, W)

        if self.seq_transform is not None:
            seq = self.seq_transform(seq)

        y = torch.tensor(self.labels[idx], dtype=torch.float32)

        if self.normalize_labels:
            y = (y - self.label_means) / self.label_stds

        return seq, y #tensor form img, label array
