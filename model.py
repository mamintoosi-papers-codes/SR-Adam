"""
Module: model.py
Description: Model definitions (SimpleCNN)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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
