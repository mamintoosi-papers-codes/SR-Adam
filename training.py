"""
Module: training.py
Description: Training and evaluation functions
"""

import torch
import torch.nn as nn
import time
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast


def train_epoch(model, train_loader, optimizer, criterion, device, epoch, num_epochs):
    """
    Train the model for one epoch.
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        optimizer: Optimizer instance
        criterion: Loss function
        device: torch.device (cuda or cpu)
        epoch: Current epoch number (1-indexed)
        num_epochs: Total number of epochs
    
    Returns:
        tuple: (train_loss, train_acc)
    """
    scaler = GradScaler()
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch}/{num_epochs}'):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    train_loss = running_loss / len(train_loader)
    train_acc = 100.0 * correct / total
    
    return train_loss, train_acc


def evaluate(model, test_loader, criterion, device):
    """
    Evaluate the model on the test set.
    
    Args:
        model: PyTorch model
        test_loader: Test data loader
        criterion: Loss function
        device: torch.device (cuda or cpu)
    
    Returns:
        tuple: (test_loss, test_acc)
    """
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    test_loss = test_loss / len(test_loader)
    test_acc = 100.0 * correct / total
    
    return test_loss, test_acc


def train_model(model, train_loader, test_loader, optimizer, criterion, num_epochs, device):
    """
    Train the model for multiple epochs with mixed precision.
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        test_loader: Test data loader
        optimizer: Optimizer instance
        criterion: Loss function
        num_epochs: Number of training epochs
        device: torch.device (cuda or cpu)
    
    Returns:
        dict: Metrics including train_loss, test_loss, train_acc, test_acc, epoch_time
    """
    metrics = {
        'train_loss': [],
        'test_loss': [],
        'train_acc': [],
        'test_acc': [],
        'epoch_time': []
    }
    
    model.to(device)
    
    for epoch in range(1, num_epochs + 1):
        start_time = time.time()
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch, num_epochs
        )
        
        # Evaluate
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        
        epoch_time = time.time() - start_time
        
        # Store metrics
        metrics['train_loss'].append(train_loss)
        metrics['test_loss'].append(test_loss)
        metrics['train_acc'].append(train_acc)
        metrics['test_acc'].append(test_acc)
        metrics['epoch_time'].append(epoch_time)
        
        print(f'Epoch {epoch}/{num_epochs} | '
              f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | '
              f'Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}% | '
              f'Time: {epoch_time:.2f}s')
    
    return metrics
