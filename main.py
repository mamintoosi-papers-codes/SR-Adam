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
    Stein-Rule Adam with fixed sigma parameter.
    Applies James-Stein shrinkage to gradients using a fixed noise variance estimate.
    """
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, 
                 weight_decay=0, stein_sigma=1e-3):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, stein_sigma=stein_sigma)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('SRAdam does not support sparse gradients')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Apply Stein-Rule correction for steps > 1
                if state['step'] > 1:
                    # Calculate dimension (p)
                    p_dim = grad.numel()
                    
                    # Calculate squared Euclidean distance between current gradient and momentum
                    diff = grad - exp_avg
                    dist_sq = torch.sum(diff * diff)
                    
                    # Calculate shrinkage factor using fixed sigma
                    stein_sigma = group['stein_sigma']
                    numerator = (p_dim - 2) * stein_sigma
                    
                    # James-Stein shrinkage factor with positive part rule
                    shrinkage = 1.0 - (numerator / (dist_sq + 1e-10))
                    shrinkage = max(0.0, shrinkage)
                    
                    # Apply shrinkage to gradient
                    grad_stein = exp_avg + shrinkage * (grad - exp_avg)
                    grad_to_use = grad_stein
                else:
                    grad_to_use = grad

                # Apply weight decay if needed
                if group['weight_decay'] != 0:
                    grad_to_use = grad_to_use.add(p, alpha=group['weight_decay'])

                # Update biased first moment estimate
                exp_avg.mul_(beta1).add_(grad_to_use, alpha=1 - beta1)
                # Update biased second raw moment estimate
                exp_avg_sq.mul_(beta2).addcmul_(grad_to_use, grad_to_use, value=1 - beta2)

                # Compute bias-corrected first moment estimate
                bias_correction1 = 1 - beta1 ** state['step']
                # Compute bias-corrected second raw moment estimate
                bias_correction2 = 1 - beta2 ** state['step']
                
                # Compute step size
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1
                
                # Update parameters
                denom = exp_avg_sq.sqrt().add_(group['eps'])
                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss

class SRAdamDynamic(optim.Optimizer):
    """
    Stein-Rule Adam with dynamic noise variance estimation.
    Estimates noise variance online using Adam's moment buffers.
    """
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('SRAdam does not support sparse gradients')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                state['step'] += 1

                # Apply Stein-Rule correction for steps > 1
                if state['step'] > 1:
                    # Calculate dimension (p)
                    p_dim = grad.numel()
                    
                    # Estimate dynamic noise variance using Adam's moments
                    # Var(g) = E[g^2] - (E[g])^2
                    element_wise_var = state['exp_avg_sq'] - state['exp_avg'].pow(2)
                    # Ensure non-negativity for numerical stability
                    element_wise_var = torch.clamp(element_wise_var, min=0)
                    # Get scalar variance estimate by averaging across parameters
                    dynamic_sigma_sq = element_wise_var.mean().item()
                    
                    # Calculate squared Euclidean distance
                    diff = grad - exp_avg
                    dist_sq = torch.sum(diff * diff)

                    # Calculate adaptive shrinkage factor
                    numerator = (p_dim - 2) * dynamic_sigma_sq
                    shrinkage = 1.0 - (numerator / (dist_sq + 1e-10))
                    shrinkage = max(0.0, shrinkage)
                    
                    # Apply shrinkage to gradient
                    grad_stein = exp_avg + shrinkage * (grad - exp_avg)
                    grad_to_use = grad_stein
                else:
                    grad_to_use = grad

                # Apply weight decay if needed
                if group['weight_decay'] != 0:
                    grad_to_use = grad_to_use.add(p, alpha=group['weight_decay'])

                # Update biased first moment estimate
                exp_avg.mul_(beta1).add_(grad_to_use, alpha=1 - beta1)
                # Update biased second raw moment estimate
                exp_avg_sq.mul_(beta2).addcmul_(grad_to_use, grad_to_use, value=1 - beta2)

                # Compute bias-corrected moments
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                # Compute step size
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1
                
                # Update parameters
                denom = exp_avg_sq.sqrt().add_(group['eps'])
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
    
    Args:
        batch_size: Batch size for training (larger for GPU efficiency)
    
    Returns:
        train_loader, test_loader: PyTorch DataLoaders
    """
    # Data augmentation and normalization for training
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # Normalization for testing
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # Load datasets
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
    
    # Create data loaders with optimized settings
    train_loader = DataLoader(
        train_set, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=8,  # Increased for faster data loading
        pin_memory=True  # Faster data transfer to GPU
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
    
    Args:
        model: Neural network model
        train_loader: Training data loader
        test_loader: Test data loader
        optimizer: Optimization algorithm
        criterion: Loss function
        num_epochs: Number of training epochs
    
    Returns:
        Dictionary with training and testing metrics
    """
    # Initialize gradient scaler for mixed precision
    scaler = GradScaler()
    
    # Initialize metrics storage
    metrics = {
        'train_loss': [],
        'test_loss': [],
        'train_acc': [],
        'test_acc': [],
        'epoch_time': []
    }
    
    # Move model to device
    model.to(device)
    
    # Training loop
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Mixed precision forward pass
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            # Mixed precision backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Track statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        # Calculate training metrics
        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total
        metrics['train_loss'].append(train_loss)
        metrics['train_acc'].append(train_acc)
        
        # Testing phase
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Mixed precision forward pass
                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        # Calculate test metrics
        test_loss = test_loss / len(test_loader)
        test_acc = 100. * correct / total
        metrics['test_loss'].append(test_loss)
        metrics['test_acc'].append(test_acc)
        
        # Calculate epoch time
        epoch_time = time.time() - start_time
        metrics['epoch_time'].append(epoch_time)
        
        # Print progress
        print(f'Epoch {epoch+1}/{num_epochs} | '
              f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | '
              f'Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}% | '
              f'Time: {epoch_time:.2f}s')
    
    return metrics

# ================== VISUALIZATION FUNCTION ==================

def plot_results(results, optimizers_names):
    """
    Plot training and testing metrics for all optimizers.
    
    Args:
        results: List of metrics dictionaries
        optimizers_names: List of optimizer names
    """
    plt.figure(figsize=(20, 12))
    
    # Plot training loss
    plt.subplot(2, 2, 1)
    for i, metrics in enumerate(results):
        plt.plot(metrics['train_loss'], label=optimizers_names[i])
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot testing loss
    plt.subplot(2, 2, 2)
    for i, metrics in enumerate(results):
        plt.plot(metrics['test_loss'], label=optimizers_names[i])
    plt.title('Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot training accuracy
    plt.subplot(2, 2, 3)
    for i, metrics in enumerate(results):
        plt.plot(metrics['train_acc'], label=optimizers_names[i])
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    # Plot testing accuracy
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

# ================== MAIN EXECUTION ==================

if __name__ == "__main__":
    # Configuration
    batch_size = 1024  # Large batch size for GPU efficiency
    num_epochs = 5   # Sufficient epochs for comparison
    
    # Get data loaders
    train_loader, test_loader = get_data_loaders(batch_size=batch_size)
    
    # Define optimizers to compare
    optimizers = [
        ('SGD', lambda model: SGD(model.parameters(), lr=0.01)),
        ('Momentum', lambda model: Momentum(model.parameters(), lr=0.01, momentum=0.9)),
        ('Adam', lambda model: Adam(model.parameters(), lr=0.001)),
        ('SR-Adam (Fixed)', lambda model: SRAdamFixed(model.parameters(), lr=0.001, stein_sigma=1e-3)),
        ('SR-Adam (Dynamic)', lambda model: SRAdamDynamic(model.parameters(), lr=0.001))
    ]
    
    # Store results
    results = []
    optimizers_names = []
    
    # Train and evaluate each optimizer
    for name, optimizer_fn in optimizers:
        print(f"\n{'='*60}")
        print(f"Training with {name} optimizer")
        print(f"{'='*60}")
        
        # Create fresh model for each optimizer
        model = SimpleCNN()
        
        # Define loss function
        criterion = nn.CrossEntropyLoss()
        
        # Initialize optimizer
        optimizer = optimizer_fn(model)
        
        # Train model
        metrics = train_model(
            model, train_loader, test_loader, 
            optimizer, criterion, num_epochs=num_epochs
        )
        
        # Store results
        results.append(metrics)
        optimizers_names.append(name)
    
    # Plot comparison results
    plot_results(results, optimizers_names)
    
    # Print final test accuracies
    print(f"\n{'='*60}")
    print("FINAL TEST ACCURACIES")
    print(f"{'='*60}")
    for i, name in enumerate(optimizers_names):
        final_acc = results[i]['test_acc'][-1]
        print(f"{name}: {final_acc:.2f}%")