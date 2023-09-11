import torch
import torch.nn as nn

class ImageDataset(torch.utils.data.Dataset):

    def __init__(self):
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()

    def __getitem__(self, idx):
        raise NotImplementedError()