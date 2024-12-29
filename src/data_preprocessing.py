# data_preprocessing.py

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from albumentations import Compose, Normalize, HorizontalFlip, ShiftScaleRotate, CoarseDropout, RandomCrop, PadIfNeeded
from albumentations.pytorch import ToTensorV2
from config import CIFAR_MEAN, CIFAR_STD
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

from torchvision.datasets import CIFAR10
import torch
from torch.utils.data import Dataset
import numpy as np

class CIFAR10Dataset(Dataset):
    def __init__(self, train=True, transforms=None):
        """
        Args:
            train (bool): If True, load the training dataset; otherwise, load the test dataset.
            transforms (callable, optional): A function/transform to apply to the images (e.g., Albumentations pipeline).
        """
        # Load CIFAR-10 dataset
        self.dataset = CIFAR10(root="./data", train=train, download=True)
        self.transforms = transforms
    
    def __len__(self):
        """
        Returns:
            int: The total number of samples in the dataset.
        """
        return len(self.dataset)
    
    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the data sample to fetch.

        Returns:
            tuple: (image, label) where image is the transformed image and label is its corresponding class label.
        """
        image, label = self.dataset[idx]
        image = np.array(image)  # Convert PIL Image to NumPy array
        if self.transforms:
            # Apply Albumentations transformations
            image = self.transforms(image=image)["image"]
        return image, label

def get_train_transforms():
    return Compose([
        PadIfNeeded(min_height=36, min_width=36, border_mode=0, p=1.0),  # Padding to 36x36
        RandomCrop(height=32, width=32, p=1.0),  # Random crop back to 32x32
        HorizontalFlip(p=0.5),  # Flip horizontally
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),  # Random affine transformations
        CoarseDropout(max_holes=1, max_height=8, max_width=8, min_holes=1, min_height=8, min_width=8, p=0.5),  # Cutout
        Normalize(mean=CIFAR_MEAN, std=CIFAR_STD),  # Normalize
        ToTensorV2(),  # Convert to tensor
    ])

# Normalization and tensor conversion for testing
def get_test_transforms():
    return Compose([
        Normalize(mean=CIFAR_MEAN, std=CIFAR_STD),  # Normalization for test data
        ToTensorV2(),
    ])

def get_data_loaders(batch_size=64):

    
    train_dataset = CIFAR10Dataset(train=True, transforms=get_train_transforms())
    test_dataset = CIFAR10Dataset(train=False, transforms=get_test_transforms())

    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return trainloader, testloader

