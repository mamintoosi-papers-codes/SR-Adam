"""
Module: data.py
Description: Data loading and preprocessing utilities
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


class AddGaussianNoise(object):
    """Add Gaussian noise to tensor images for data augmentation."""
    def __init__(self, mean=0., std=0.):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + f'(mean={self.mean}, std={self.std})'


def get_data_loaders(dataset_name='CIFAR10', batch_size=512, noise_std=0.0, num_workers=8):
    """
    Load dataset with optimized settings for GPU.
    
    Args:
        dataset_name (str): 'CIFAR10' or 'CIFAR100'
        batch_size (int): Batch size for data loaders
        noise_std (float): Standard deviation for Gaussian noise augmentation
        num_workers (int): Number of worker processes for data loading
    
    Returns:
        tuple: (train_loader, test_loader, num_classes)
    """
    if dataset_name == 'CIFAR10':
        DatasetClass = torchvision.datasets.CIFAR10
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        num_classes = 10
    elif dataset_name == 'CIFAR100':
        DatasetClass = torchvision.datasets.CIFAR100
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
        num_classes = 100
    else:
        raise ValueError("Unsupported dataset: choose 'CIFAR10' or 'CIFAR100'")

    # Training transforms with augmentation
    train_transform_list = [
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
    if noise_std > 0:
        train_transform_list.append(AddGaussianNoise(mean=0., std=noise_std))

    train_transform = transforms.Compose(train_transform_list)
    
    # Test transforms (no augmentation, only normalization)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    
    # Load datasets
    train_set = DatasetClass(root='./data', train=True, download=True, transform=train_transform)
    test_set = DatasetClass(root='./data', train=False, download=True, transform=test_transform)
    
    # Create data loaders
    train_loader = DataLoader(
        train_set, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,  
        pin_memory=True  
    )
    
    test_loader = DataLoader(
        test_set, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader, num_classes
