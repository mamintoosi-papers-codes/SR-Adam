"""
Module: model.py
Description: Model definitions (SimpleCNN, ResNet-18)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class SimpleCNN(nn.Module):
    """
    Simple Convolutional Neural Network for classification.
    
    Architecture:
    Conv2d(3, 32) -> ReLU -> MaxPool2d(2, 2)
    -> Conv2d(32, 64) -> ReLU -> MaxPool2d(2, 2)
    -> Flatten -> Linear(64*8*8, 128) -> ReLU -> Dropout(0.2)
    -> Linear(128, num_classes)
    
    Input: (batch_size, 3, 32, 32) - CIFAR-10/100 images
    Output: (batch_size, num_classes) - class logits
    """
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        """Forward pass through the network."""
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def get_resnet18(num_classes=10, pretrained=False):
    """
    Returns a ResNet-18 model adapted for CIFAR datasets.
    
    Args:
        num_classes: Number of output classes (10 for CIFAR-10, 100 for CIFAR-100)
        pretrained: Whether to load ImageNet pretrained weights (default: False)
    
    Returns:
        ResNet-18 model with modified final layer for num_classes
    
    Note:
        - ResNet-18 has ~11M parameters (44x larger than SimpleCNN)
        - Original designed for ImageNet (224x224), works well on CIFAR (32x32)
        - Final FC layer adapted from 1000 classes → num_classes
    """
    model = models.resnet18(pretrained=pretrained)
    
    # Replace final fully connected layer
    # ResNet-18 final layer: Linear(512, 1000) → Linear(512, num_classes)
    model.fc = nn.Linear(512, num_classes)
    
    return model


def get_model(model_name, num_classes=10):
    """
    Factory function to create models.
    
    Args:
        model_name (str): 'simplecnn' or 'resnet18'
        num_classes (int): Number of output classes
        
    Returns:
        nn.Module: The requested model
    """
    if model_name.lower() == "simplecnn":
        return SimpleCNN(num_classes=num_classes)
    
    elif model_name.lower() == "resnet18":
        # Use torchvision's ResNet-18 (without pretrained weights)
        model = models.resnet18(pretrained=False)
        # Adjust final layer for num_classes
        model.fc = nn.Linear(512, num_classes)
        return model
    
    else:
        raise ValueError(f"Unknown model: {model_name}. Choose 'simplecnn' or 'resnet18'")
