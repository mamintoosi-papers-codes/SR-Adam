import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import time
from tqdm import tqdm
import math
from torch.cuda.amp import GradScaler, autocast
import os
import pandas as pd
import json

# Set device for GPU support
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# ================== OPTIMIZER DEFINITIONS ==================

class SGD(optim.SGD):
    """Standard Stochastic Gradient Descent optimizer"""
    def __init__(self, params, lr=0.01, momentum=0):
        super().__init__(params, lr=lr, momentum=momentum)

class Momentum(optim.SGD):
    """SGD with Momentum optimizer"""
    def __init__(self, params, lr=0.01, momentum=0.9):
        super().__init__(params, lr=lr, momentum=momentum)

class Adam(optim.Adam):
    """Standard Adam optimizer"""
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8):
        super().__init__(params, lr=lr, betas=betas, eps=eps)

class SRAdamFixed(optim.Optimizer):
    """
    Stein-Rule Adam with fixed sigma^2 parameter.
    Applies James-Stein shrinkage globally across all parameters.
    """
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, 
                 weight_decay=0, stein_sigma_sq=1e-6):  # stein_sigma_sq = sigma^2
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, stein_sigma_sq=stein_sigma_sq)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Collect all gradients, exp_avg, exp_avg_sq
        all_grad = []
        all_exp_avg = []
        all_exp_avg_sq = []
        step = None
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                all_grad.append(p.grad.view(-1))

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                all_exp_avg.append(state['exp_avg'].view(-1))
                all_exp_avg_sq.append(state['exp_avg_sq'].view(-1))
                if step is None:
                    step = state['step']

        if len(all_grad) == 0:
            return loss

        all_grad = torch.cat(all_grad)
        all_exp_avg = torch.cat(all_exp_avg)
        all_exp_avg_sq = torch.cat(all_exp_avg_sq)

        beta1, beta2 = self.param_groups[0]['betas']  # Assume same for all groups

        # Increment step for all states
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    self.state[p]['step'] += 1

        step += 1

        if step > 1:
            p_dim = all_grad.numel()
            diff = all_grad - all_exp_avg
            dist_sq = torch.sum(diff * diff).item()
            stein_sigma_sq = self.param_groups[0]['stein_sigma_sq']
            numerator = (p_dim - 2) * stein_sigma_sq
            shrinkage = 1.0 - (numerator / (dist_sq + 1e-10))
            shrinkage = max(0.0, shrinkage)
        else:
            shrinkage = 1.0

        # Apply to each param
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]
                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']

                grad_to_use = exp_avg + shrinkage * (grad - exp_avg) if step > 1 else grad

                if group['weight_decay'] != 0:
                    grad_to_use = grad_to_use.add(p, alpha=group['weight_decay'])

                exp_avg.mul_(beta1).add_(grad_to_use, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad_to_use, grad_to_use, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step
                step_size = group['lr'] / bias_correction1

                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss

class SRAdamDynamic(optim.Optimizer):
    """
    Stein-Rule Adam with dynamic noise variance estimation.
    Estimates noise variance globally using Adam's moment buffers.
    """
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Collect all
        all_grad = []
        all_exp_avg = []
        all_exp_avg_sq = []
        step = None
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                all_grad.append(p.grad.view(-1))

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                all_exp_avg.append(state['exp_avg'].view(-1))
                all_exp_avg_sq.append(state['exp_avg_sq'].view(-1))
                if step is None:
                    step = state['step']

        if len(all_grad) == 0:
            return loss

        all_grad = torch.cat(all_grad)
        all_exp_avg = torch.cat(all_exp_avg)
        all_exp_avg_sq = torch.cat(all_exp_avg_sq)

        beta1, beta2 = self.param_groups[0]['betas']

        # Increment step
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    self.state[p]['step'] += 1

        step += 1

        if step > 1:
            p_dim = all_grad.numel()
            element_wise_var = all_exp_avg_sq - all_exp_avg.pow(2)
            element_wise_var = torch.clamp(element_wise_var, min=0)
            dynamic_sigma_sq = element_wise_var.mean().item()
            diff = all_grad - all_exp_avg
            dist_sq = torch.sum(diff * diff).item()
            numerator = (p_dim - 2) * dynamic_sigma_sq
            shrinkage = 1.0 - (numerator / (dist_sq + 1e-10))
            shrinkage = max(0.0, shrinkage)
        else:
            shrinkage = 1.0

        # Apply
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]
                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']

                grad_to_use = exp_avg + shrinkage * (grad - exp_avg) if step > 1 else grad

                if group['weight_decay'] != 0:
                    grad_to_use = grad_to_use.add(p, alpha=group['weight_decay'])

                exp_avg.mul_(beta1).add_(grad_to_use, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad_to_use, grad_to_use, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step
                step_size = group['lr'] / bias_correction1

                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss

# ================== MODEL DEFINITION ==================

class SimpleCNN(nn.Module):
    """
    Simple Convolutional Neural Network for CIFAR-10 classification.
    Architecture: Conv -> ReLU -> MaxPool -> Conv -> ReLU -> MaxPool -> FC -> ReLU -> Dropout -> FC
    """
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# ================== DATA LOADING ==================

def get_data_loaders(batch_size=512):
    """
    Load CIFAR-10 dataset with optimized settings for GPU.
    """
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
    
    train_loader = DataLoader(
        train_set, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=8,  
        pin_memory=True  
    )
    
    test_loader = DataLoader(
        test_set, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=8,
        pin_memory=True
    )
    
    return train_loader, test_loader

# ================== TRAINING FUNCTION ==================

def train_model(model, train_loader, test_loader, optimizer, criterion, num_epochs=15):
    """
    Train the model with the given optimizer and evaluate on test set.
    Uses mixed precision training for faster GPU computation.
    """
    scaler = GradScaler()
    
    metrics = {
        'train_loss': [],
        'test_loss': [],
        'train_acc': [],
        'test_acc': [],
        'epoch_time': []
    }
    
    model.to(device)
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
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
        train_acc = 100. * correct / total
        metrics['train_loss'].append(train_loss)
        metrics['train_acc'].append(train_acc)
        
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
        test_acc = 100. * correct / total
        metrics['test_loss'].append(test_loss)
        metrics['test_acc'].append(test_acc)
        
        epoch_time = time.time() - start_time
        metrics['epoch_time'].append(epoch_time)
        
        print(f'Epoch {epoch+1}/{num_epochs} | '
              f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | '
              f'Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}% | '
              f'Time: {epoch_time:.2f}s')
    
    return metrics

# ================== VISUALIZATION FUNCTION ==================

def plot_results(results, optimizers_names):
    """
    Plot training and testing metrics for all optimizers.
    """
    plt.figure(figsize=(20, 12))
    
    plt.subplot(2, 2, 1)
    for i, metrics in enumerate(results):
        plt.plot(metrics['train_loss'], label=optimizers_names[i])
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    for i, metrics in enumerate(results):
        plt.plot(metrics['test_loss'], label=optimizers_names[i])
    plt.title('Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 3)
    for i, metrics in enumerate(results):
        plt.plot(metrics['train_acc'], label=optimizers_names[i])
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    for i, metrics in enumerate(results):
        plt.plot(metrics['test_acc'], label=optimizers_names[i])
    plt.title('Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('optimizer_comparison.png', dpi=300)
    plt.show()

# ================== SAVE RESULTS FUNCTION ==================

def save_results(results, optimizers_names, batch_size, num_epochs):
    """
    Save results to a folder with an Excel file (one sheet per optimizer) and a config.json file.
    """
    # Create results folder if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Save to Excel
    excel_filename = f"optimizer_comparison_batch{batch_size}_epochs{num_epochs}.xlsx"
    excel_path = os.path.join('results', excel_filename)
    
    with pd.ExcelWriter(excel_path) as writer:
        for i, name in enumerate(optimizers_names):
            df = pd.DataFrame({
                'Epoch': range(1, num_epochs + 1),
                'Train Loss': results[i]['train_loss'],
                'Test Loss': results[i]['test_loss'],
                'Train Acc': results[i]['train_acc'],
                'Test Acc': results[i]['test_acc'],
                'Epoch Time': results[i]['epoch_time']
            })
            df.to_excel(writer, sheet_name=name.replace(' ', '_').replace('(', '').replace(')', ''), index=False)
    
    # Save config
    config = {
        'batch_size': batch_size,
        'num_epochs': num_epochs,
        'optimizers': optimizers_names
    }
    config_path = os.path.join('results', 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"Results saved to {excel_path} and {config_path}")

# ================== MAIN EXECUTION ==================

if __name__ == "__main__":
    batch_size = 512
    num_epochs = 15
    
    train_loader, test_loader = get_data_loaders(batch_size=batch_size)
    
    optimizers = [
        ('SGD', lambda model: SGD(model.parameters(), lr=0.01)),
        ('Momentum', lambda model: Momentum(model.parameters(), lr=0.01, momentum=0.9)),
        ('Adam', lambda model: Adam(model.parameters(), lr=0.001)),
        ('SR-Adam (Fixed)', lambda model: SRAdamFixed(model.parameters(), lr=0.001, stein_sigma_sq=1e-6)),
        ('SR-Adam (Dynamic)', lambda model: SRAdamDynamic(model.parameters(), lr=0.001))
    ]
    
    results = []
    optimizers_names = []
    
    for name, optimizer_fn in optimizers:
        print(f"\n{'='*60}")
        print(f"Training with {name} optimizer")
        print(f"{'='*60}")
        
        model = SimpleCNN()
        
        criterion = nn.CrossEntropyLoss()
        
        optimizer = optimizer_fn(model)
        
        metrics = train_model(
            model, train_loader, test_loader, 
            optimizer, criterion, num_epochs=num_epochs
        )
        
        results.append(metrics)
        optimizers_names.append(name)
    
    plot_results(results, optimizers_names)
    
    save_results(results, optimizers_names, batch_size, num_epochs)
    
    print(f"\n{'='*60}")
    print("FINAL TEST ACCURACIES")
    print(f"{'='*60}")
    for i, name in enumerate(optimizers_names):
        final_acc = results[i]['test_acc'][-1]
        print(f"{name}: {final_acc:.2f}%")