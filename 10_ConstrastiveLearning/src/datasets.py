import pickle
import numpy as np
import random

from src.augmentations import Cutout, AddGaussianNoise
import torch
from torch.utils.data import Dataset
from torchvision import transforms

def unpickle(file):
    with open(file, 'rb') as fin:
        data = pickle.load(fin, encoding='bytes')
    return data

lab_dict = {
    0: 'airplane',
    1: 'automobile',
    2: 'bird',
    3: 'cat',
    4: 'deer',
    5: 'dog',
    6: 'frog',
    7: 'horse',
    8: 'ship',
    9: 'truck'}


class Pretrained_Dset(Dataset):
    def __init__(self, img_arr, label_arr, mode, mean, std, s=0.5):
        self.mode = mode
        self.img_arr = img_arr
        self.label_arr = label_arr

        # Augmentations
        self.s = s
        self.transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomResizedCrop(32, (0.8,1.0), antialias=False),
            transforms.Compose([
                transforms.RandomApply([
                    transforms.ColorJitter(0.8*self.s, 0.8*self.s, 0.8*self.s, 0.2*self.s)], p = 0.8),
                transforms.RandomGrayscale(p=0.2)]),
            transforms.RandomApply([Cutout()], p=0.7),
            ])

        # Stats
        self.mean, self.std = mean, std

    def __len__(self):
        return self.img_arr.shape[0]

    def __getitem__(self, idx):
        x = self.img_arr[idx]

        # Scale
        x = torch.from_numpy(
            x.astype(np.float32) / 255.0)

        # Augment to create ~x1, ~x2
        x1 = self.__augment(x)
        x2 = self.__augment(x)

        # Normalize
        x = self.__normalize(x)
        x1 = self.__normalize(x1)
        x2 = self.__normalize(x2)

        return {
            'img': x.squeeze(dim=0),
            'label': torch.tensor(self.label_arr[idx], dtype=torch.long),
            'x1': x1.squeeze(dim=0),
            'x2': x2.squeeze(dim=0)}


    def __normalize(self, frame):
        return (frame - self.mean) / self.std

    def __augment(self, frame):
        """Applies randomly selected augmentations to each clip (same for each frame in the clip)
        """
        if self.mode == 'train':
            frame = self.transforms(frame)
        else:
            return frame
        return frame

    def on_epoch_end(self):
        """Shuffles the dataset at the end of each epoch"""
        idx = random.sample(
            population=list(range(self.__len__())),
            k=self.__len__())
        self.img_arr = self.img_arr[idx]
        self.label_arr = self.label_arr[idx]


class Downstream_Dset(Dataset):
    def __init__(self, img_arr, label_arr, mode, mean, std):
        self.mode = mode
        self.img_arr = img_arr
        self.label_arr = label_arr

        # Augmentations
        self.randomcrop = transforms.RandomResizedCrop(32, (0.8,1.0), antialias=False)

        # Stats
        self.mean, self.std = mean, std

    def __len__(self):
        return self.img_arr.shape[0]

    def __getitem__(self, idx):
        x = self.img_arr[idx]

        # Scale
        x = torch.from_numpy(
            x.astype(np.float32) / 255.0)

        # Augment
        if self.mode == 'train':
            x  = self.randomcrop(x)

        # Normalize
        x = self.__normalize(x)

        return {
            'img': x.squeeze(dim=0),
            'label': torch.tensor(self.label_arr[idx], dtype=torch.long),
        }

    def on_epoch_end(self):
        idx = random.sample(population=list(range(self.__len__())),k = self.__len__())
        self.img_arr = self.img_arr[idx]
        self.label_arr = self.label_arr[idx]

    def __normalize(self, frame):
        return (frame - self.mean) / self.std
