import torch
import torch.nn as nn
import torch.nn.functional as F
# import torch.optim as optim
from torch.optim import Optimizer
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
import argparse

# Set device for GPU support
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# ================== OPTIMIZER DEFINITIONS ==================

# ============================================================
# 1. SGD (Baseline)
# ============================================================

class SGDManual(Optimizer):
    """
    Vanilla Stochastic Gradient Descent
    """
    def __init__(self, params, lr=0.01, weight_decay=0):
        defaults = dict(lr=lr, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            wd = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if wd != 0:
                    grad = grad.add(p, alpha=wd)

                p.add_(grad, alpha=-lr)

        return loss


# ============================================================
# 2. SGD with Momentum (Baseline)
# ============================================================

class MomentumManual(Optimizer):
    """
    SGD with classical heavy-ball momentum
    """
    def __init__(self, params, lr=0.01, momentum=0.9, weight_decay=0):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            mu = group['momentum']
            wd = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state['velocity'] = torch.zeros_like(p)

                v = state['velocity']

                if wd != 0:
                    grad = grad.add(p, alpha=wd)

                v.mul_(mu).add_(grad)
                p.add_(v, alpha=-lr)

        return loss


# ============================================================
# 3. Adam (Baseline – manual)
# ============================================================

class AdamBaseline(Optimizer):
    """
    Manual Adam implementation (baseline for fair comparison)
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999),
                 eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas,
                        eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.global_step = 0

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self.global_step += 1

        for group in self.param_groups:
            beta1, beta2 = group['betas']
            lr = group['lr']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)

                m, v = state['exp_avg'], state['exp_avg_sq']

                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])

                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bc1 = 1 - beta1 ** self.global_step
                bc2 = 1 - beta2 ** self.global_step

                step_size = lr / bc1
                denom = (v.sqrt() / math.sqrt(bc2)).add_(group['eps'])

                p.addcdiv_(m, denom, value=-step_size)

        return loss


# ============================================================
# 4. SR-Adam (Fixed Sigma, Global)
# ============================================================

class SRAdamFixedGlobal(Optimizer):
    """
    Stein-Rule Adam with fixed sigma^2, global shrinkage
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999),
                 eps=1e-8, weight_decay=0, stein_sigma=1e-3):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay,
                        stein_sigma=stein_sigma)
        super().__init__(params, defaults)
        self.global_step = 0

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self.global_step += 1

        grads, moms = [], []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                grads.append(p.grad.view(-1))
                moms.append(state['exp_avg'].view(-1))

        g = torch.cat(grads)
        m = torch.cat(moms)

        if self.global_step > 1:
            diff = g - m
            p_dim = g.numel()
            sigma2 = self.param_groups[0]['stein_sigma']
            shrink = 1 - (p_dim - 2) * sigma2 / (diff.pow(2).sum() + 1e-12)
            shrink = torch.clamp(shrink, 0.0, 1.0)
        else:
            shrink = 1.0

        for group in self.param_groups:
            beta1, beta2 = group['betas']
            lr = group['lr']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                m_t, v_t = state['exp_avg'], state['exp_avg_sq']

                g_hat = m_t + shrink * (grad - m_t)

                m_t.mul_(beta1).add_(g_hat, alpha=1 - beta1)
                v_t.mul_(beta2).addcmul_(g_hat, g_hat, value=1 - beta2)

                bc1 = 1 - beta1 ** self.global_step
                bc2 = 1 - beta2 ** self.global_step
                step_size = lr / bc1
                denom = (v_t.sqrt() / math.sqrt(bc2)).add_(group['eps'])

                p.addcdiv_(m_t, denom, value=-step_size)

        return loss


# ============================================================
# 5. SR-Adam (Adaptive Sigma, Global)
# ============================================================

class SRAdamAdaptiveGlobal(Optimizer):
    """
    Stein-Rule Adam (Adaptive, Global, Stable)

    - Global James–Stein shrinkage
    - Noise variance estimated from Adam second moments
    - Warm-up + shrinkage clipping for stability
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        warmup_steps=20,
        shrink_clip=(0.1, 1.0)
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            warmup_steps=warmup_steps,
            shrink_clip=shrink_clip,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # -------- collect global buffers --------
        all_grad, all_m, all_v = [], [], []
        step = None

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)

                all_grad.append(p.grad.view(-1))
                all_m.append(state['exp_avg'].view(-1))
                all_v.append(state['exp_avg_sq'].view(-1))

                if step is None:
                    step = state['step']

        if len(all_grad) == 0:
            return loss

        g = torch.cat(all_grad)
        m = torch.cat(all_m)
        v = torch.cat(all_v)

        # -------- increment step --------
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    self.state[p]['step'] += 1

        step += 1
        beta1, beta2 = self.param_groups[0]['betas']
        warmup = self.param_groups[0]['warmup_steps']
        clip_lo, clip_hi = self.param_groups[0]['shrink_clip']

        # -------- Stein shrinkage (global) --------
        if step <= warmup:
            shrink = 1.0
        else:
            # noise variance from Adam moments
            sigma2 = (v - m.pow(2)).clamp(min=0).mean().item()

            diff = g - m
            dist_sq = diff.pow(2).sum().item()
            p_dim = g.numel()

            raw = 1.0 - ((p_dim - 2) * sigma2) / (dist_sq + 1e-12)
            shrink = max(clip_lo, min(clip_hi, raw))

        # -------- apply Adam update --------
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                grad_hat = exp_avg + shrink * (grad - exp_avg)

                if group['weight_decay'] != 0:
                    grad_hat = grad_hat.add(p, alpha=group['weight_decay'])

                exp_avg.mul_(beta1).add_(grad_hat, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad_hat, grad_hat, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step

                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1
                denom = exp_avg_sq.sqrt().add_(group['eps'])

                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss



# ============================================================
# 6. SR-Adam (Adaptive Sigma, Local / Group-wise)
# ============================================================

class SRAdamAdaptiveLocal(Optimizer):
    """
    Stein-Rule Adam with adaptive sigma^2, local (per param_group) shrinkage
    
    FIXES:
    - Uses Adam moments (v_t - m_t^2) to estimate sigma^2
    - Applies shrinkage clipping to prevent instability
    - Per-parameter step counting
    - Warm-up for first few steps
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999),
                 eps=1e-8, weight_decay=0, warmup_steps=20, shrink_clip=(0.1, 1.0)):
        defaults = dict(lr=lr, betas=betas,
                        eps=eps, weight_decay=weight_decay,
                        warmup_steps=warmup_steps, shrink_clip=shrink_clip)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group['betas']
            lr = group['lr']
            warmup = group['warmup_steps']
            clip_lo, clip_hi = group['shrink_clip']

            # Initialize step counter for this group if needed
            if 'group_step' not in group:
                group['group_step'] = 0
            
            group['group_step'] += 1
            step = group['group_step']

            grads, moms, vars_list = [], [], []
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)

                grads.append(p.grad.view(-1))
                moms.append(state['exp_avg'].view(-1))
                vars_list.append(state['exp_avg_sq'].view(-1))

            if not grads:
                continue

            g = torch.cat(grads)
            m = torch.cat(moms)
            v = torch.cat(vars_list)

            # -------- Compute adaptive shrinkage factor --------
            if step <= warmup:
                shrink = 1.0
            else:
                # Estimate noise variance from Adam moments: sigma^2 = E[g^2] - E[g]^2
                sigma2 = (v - m.pow(2)).clamp(min=0).mean().item()
                
                diff = g - m
                dist_sq = diff.pow(2).sum().item()
                p_dim = g.numel()
                
                # James-Stein shrinkage factor
                raw = 1.0 - ((p_dim - 2) * sigma2) / (dist_sq + 1e-12)
                
                # Clip to ensure stability
                shrink = max(clip_lo, min(clip_hi, raw))

            # -------- Apply updates to each parameter --------
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                m_t, v_t = state['exp_avg'], state['exp_avg_sq']

                # Stein-rule corrected gradient
                g_hat = m_t + shrink * (grad - m_t)

                if group['weight_decay'] != 0:
                    g_hat = g_hat.add(p, alpha=group['weight_decay'])

                # Update moments
                m_t.mul_(beta1).add_(g_hat, alpha=1 - beta1)
                v_t.mul_(beta2).addcmul_(g_hat, g_hat, value=1 - beta2)

                # Bias correction
                bc1 = 1 - beta1 ** step
                bc2 = 1 - beta2 ** step
                step_size = lr * math.sqrt(bc2) / bc1
                denom = v_t.sqrt().add_(group['eps'])

                p.addcdiv_(m_t, denom, value=-step_size)

        return loss


# ================== MODEL DEFINITION ==================

class SimpleCNN(nn.Module):
    """
    Simple Convolutional Neural Network for classification.
    Architecture: Conv -> ReLU -> MaxPool -> Conv -> ReLU -> MaxPool -> FC -> ReLU -> Dropout -> FC
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

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=0.):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

def get_data_loaders(dataset_name='CIFAR10', batch_size=512, noise_std=0.0):
    """
    Load dataset with optimized settings for GPU.
    Supports CIFAR10 or CIFAR100.
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

    train_transform_list = [
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
    if noise_std > 0:
        train_transform_list.append(AddGaussianNoise(mean=0., std=noise_std))

    train_transform = transforms.Compose(train_transform_list)
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    
    train_set = DatasetClass(root='./data', train=True, download=True, transform=train_transform)
    test_set = DatasetClass(root='./data', train=False, download=True, transform=test_transform)
    
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
    
    return train_loader, test_loader, num_classes

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

def save_results(results, optimizers_names, dataset_name, batch_size, num_epochs, noise_std):
    """
    Save results to a folder with an Excel file (one sheet per optimizer) and a config.json file.
    Includes dataset_name and noise_std in filenames.
    """
    # Create results folder if it doesn't exist
    folder_name = f"results_{dataset_name}_noise{noise_std}"
    os.makedirs(folder_name, exist_ok=True)
    
    # Save to Excel
    excel_filename = f"optimizer_comparison_{dataset_name}_batch{batch_size}_epochs{num_epochs}_noise{noise_std}.xlsx"
    excel_path = os.path.join(folder_name, excel_filename)
    
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
        'dataset_name': dataset_name,
        'batch_size': batch_size,
        'num_epochs': num_epochs,
        'noise_std': noise_std,
        'optimizers': optimizers_names
    }
    config_path = os.path.join(folder_name, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"Results saved to {excel_path} and {config_path}")

# ================== MAIN EXECUTION ==================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model on CIFAR datasets with various optimizers.")
    parser.add_argument('--dataset', type=str, default='CIFAR10', choices=['CIFAR10', 'CIFAR100'], help="Dataset to use: CIFAR10 or CIFAR100")
    parser.add_argument('--batch_size', type=int, default=512, help="Batch size for data loaders")
    parser.add_argument('--num_epochs', type=int, default=15, help="Number of training epochs")
    parser.add_argument('--noise', type=float, default=0.0, help="Standard deviation for Gaussian noise (0.0 for no noise)")

    args = parser.parse_args()

    dataset_name = args.dataset
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    noise_std = args.noise
    
    train_loader, test_loader, num_classes = get_data_loaders(dataset_name=dataset_name, batch_size=batch_size, noise_std=noise_std)
    
    optimizers = [
        ('SGD', 
        lambda model: SGDManual(
            model.parameters(), 
            lr=0.01
        )),

        ('Momentum', 
        lambda model: MomentumManual(
            model.parameters(), 
            lr=0.01, 
            momentum=0.9
        )),

        ('Adam', 
        lambda model: AdamBaseline(
            model.parameters(), 
            lr=0.001
        )),

        ('SR-Adam (Fixed, Global)', 
        lambda model: SRAdamFixedGlobal(
            model.parameters(), 
            lr=0.001, 
            stein_sigma=1e-6
        )),

        ('SR-Adam (Adaptive, Global)', 
        lambda model: SRAdamAdaptiveGlobal(
            model.parameters(),
            lr=1e-3,
            warmup_steps=20,
            shrink_clip=(0.2, 1.0)
        )),

        ('SR-Adam (Adaptive, Local)', 
        lambda model: SRAdamAdaptiveLocal(
            model.parameters(), 
            lr=0.001
        )),
    ]

    
    results = []
    optimizers_names = []
    
    for name, optimizer_fn in optimizers:
        print(f"\n{'='*60}")
        print(f"Training with {name} optimizer")
        print(f"{'='*60}")
        
        model = SimpleCNN(num_classes=num_classes)
        
        criterion = nn.CrossEntropyLoss()
        
        optimizer = optimizer_fn(model)
        
        metrics = train_model(
            model, train_loader, test_loader, 
            optimizer, criterion, num_epochs=num_epochs
        )
        
        results.append(metrics)
        optimizers_names.append(name)
    
    plot_results(results, optimizers_names)
    
    save_results(results, optimizers_names, dataset_name, batch_size, num_epochs, noise_std)
    
    print(f"\n{'='*60}")
    print("FINAL TEST ACCURACIES")
    print(f"{'='*60}")
    for i, name in enumerate(optimizers_names):
        final_acc = results[i]['test_acc'][-1]
        print(f"{name}: {final_acc:.2f}%")