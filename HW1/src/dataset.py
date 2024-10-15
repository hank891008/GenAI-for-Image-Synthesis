import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image

from torchvision import transforms
from torch.utils.data import Dataset


train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

test_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

class PetDataset(Dataset):
    def __init__(self, root, type='train', only_use_british_shorthair=True):
        self.root = root
        if type == 'train':
            self.transform = train_transform
        elif type == 'test':
            self.transform = test_transform
        else:
            raise ValueError('type should be "train" or "test"')
        
        self.images = os.listdir(root)
        if only_use_british_shorthair:
            self.images = [img for img in self.images if img.startswith('British_Shorthair')]
        self.length = len(self.images)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.root, img_name)
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img
