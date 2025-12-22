"""
SR-Adam: Stein-Rule Adaptive Moment Estimation
Main entry point for training models with various optimizers.

Modular structure:
- optimizers.py: Optimizer implementations
- model.py: Model architecture
- data.py: Data loading utilities
- training.py: Training and evaluation loops
- utils.py: Utilities for saving results and visualization
"""

import torch
import torch.nn as nn
import numpy as np
import argparse

# Import modules
from optimizers import (
    SGDManual, 
    MomentumManual, 
    AdamBaseline,
    SRAdamFixedGlobal,
    SRAdamAdaptiveGlobal,
    SRAdamAdaptiveLocal
)
from model import SimpleCNN
from data import get_data_loaders
from training import train_model
from utils import (
    create_results_directory,
    save_all_results,
    plot_results,
    print_summary,
    count_parameters
)


def setup_device():
    """Setup and display device information."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    return device


def set_seeds(seed=42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"Random seeds set to {seed}")


def parse_optimizer_list(raw, all_names, alias_map):
    """Parse user-specified optimizer list allowing custom delimiters and aliases."""
    if raw.lower() == 'all':
        return all_names

    # Prefer separators that do not conflict with commas inside names
    for sep in [';', '|', '\n']:
        if sep in raw:
            tokens = [t.strip() for t in raw.split(sep) if t.strip()]
            break
    else:
        # Fallback: single token (could still be comma-delimited; advise user to use ';')
        tokens = [t.strip() for t in raw.split(',') if t.strip()]

    resolved = []
    for tok in tokens:
        key = alias_map.get(tok, tok)
        if key not in all_names:
            raise ValueError(f"Unknown optimizer requested: {tok}")
        resolved.append(key)
    return resolved


def create_optimizer(name, model, num_classes):
    """
    Create optimizer instance by name.
    
    Args:
        name (str): Optimizer name
        model: PyTorch model
        num_classes (int): Number of classes
    
    Returns:
        tuple: (optimizer_name, optimizer_instance)
    """
    optimizer_configs = {
        'SGD': lambda: SGDManual(model.parameters(), lr=0.01),
        'Momentum': lambda: MomentumManual(model.parameters(), lr=0.01, momentum=0.9),
        'Adam': lambda: AdamBaseline(model.parameters(), lr=0.001),
        'SR-Adam (Fixed, Global)': lambda: SRAdamFixedGlobal(
            model.parameters(), 
            lr=0.001, 
            stein_sigma=1e-6
        ),
        'SR-Adam (Adaptive, Global)': lambda: SRAdamAdaptiveGlobal(
            model.parameters(),
            lr=1e-3,
            warmup_steps=5,
            shrink_clip=(0.2, 1.0)
        ),
        'SR-Adam (Adaptive, Local)': lambda: SRAdamAdaptiveLocal(
            model.parameters(), 
            lr=0.001,
            warmup_steps=5,
            shrink_clip=(0.1, 1.0)
        ),
    }
    
    if name not in optimizer_configs:
        raise ValueError(f"Unknown optimizer: {name}")
    
    return name, optimizer_configs[name]()


def main():
    """Main execution function."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train models with various optimizers on CIFAR datasets.")
    parser.add_argument('--dataset', type=str, default='CIFAR10', 
                        choices=['CIFAR10', 'CIFAR100'], 
                        help='Dataset: CIFAR10 or CIFAR100')
    parser.add_argument('--batch_size', type=int, default=512, 
                        help='Batch size for data loaders')
    parser.add_argument('--num_epochs', type=int, default=15, 
                        help='Number of training epochs')
    parser.add_argument('--noise', type=float, default=0.0, 
                        help='Standard deviation for Gaussian noise (0.0 for no noise)')
    parser.add_argument('--seed', type=int, default=42, 
                        help='Random seed for reproducibility')
    parser.add_argument('--optimizers', type=str, default='all',
                        help='List of optimizers to run or "all"; separate by ";" or "|" when names contain commas')
    
    args = parser.parse_args()
    
    # Setup
    device = setup_device()
    set_seeds(args.seed)
    
    # Create results directory
    results_dir = create_results_directory(args.dataset, args.noise)
    
    # Load data
    print(f"\nLoading {args.dataset} dataset...")
    train_loader, test_loader, num_classes = get_data_loaders(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        noise_std=args.noise
    )
    print(f"Dataset loaded: {num_classes} classes")
    
    # Define optimizers to test
    all_optimizer_names = [
        'SGD',
        'Momentum',
        'Adam',
        'SR-Adam (Fixed, Global)',
        'SR-Adam (Adaptive, Global)',
        'SR-Adam (Adaptive, Local)',
    ]

    optimizer_settings = {
        'SGD': {'lr': 0.01, 'weight_decay': 0},
        'Momentum': {'lr': 0.01, 'momentum': 0.9, 'weight_decay': 0},
        'Adam': {'lr': 0.001, 'betas': (0.9, 0.999), 'eps': 1e-8, 'weight_decay': 0},
        'SR-Adam (Fixed, Global)': {'lr': 0.001, 'betas': (0.9, 0.999), 'eps': 1e-8, 'stein_sigma': 1e-6},
        'SR-Adam (Adaptive, Global)': {'lr': 0.001, 'betas': (0.9, 0.999), 'eps': 1e-8, 'warmup_steps': 5, 'shrink_clip': (0.2, 1.0)},
        'SR-Adam (Adaptive, Local)': {'lr': 0.001, 'betas': (0.9, 0.999), 'eps': 1e-8, 'warmup_steps': 5, 'shrink_clip': (0.1, 1.0)},
    }

    alias_map = {
        'sgd': 'SGD',
        'momentum': 'Momentum',
        'adam': 'Adam',
        'sradam_fixed': 'SR-Adam (Fixed, Global)',
        'sradam_global': 'SR-Adam (Adaptive, Global)',
        'sradam_local': 'SR-Adam (Adaptive, Local)',
    }

    optimizer_names = parse_optimizer_list(args.optimizers, all_optimizer_names, alias_map)
    
    # Training loop
    results = []
    
    for optimizer_name in optimizer_names:
        print(f"\n{'='*80}")
        print(f"Training with {optimizer_name} optimizer")
        print(f"{'='*80}")
        
        # Create fresh model
        model = SimpleCNN(num_classes=num_classes)
        param_count = count_parameters(model)
        print(f"Trainable parameters: {param_count:,}")
        
        # Create optimizer
        _, optimizer = create_optimizer(optimizer_name, model, num_classes)
        
        # Loss function
        criterion = nn.CrossEntropyLoss()
        
        # Train model
        metrics = train_model(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            optimizer=optimizer,
            criterion=criterion,
            num_epochs=args.num_epochs,
            device=device
        )
        
        results.append(metrics)
    
    # Save and visualize results
    print(f"\n{'='*80}")
    print("Saving results...")
    print(f"{'='*80}")
    
    save_all_results(
        results=results,
        optimizers_names=optimizer_names,
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        noise_std=args.noise,
        optimizer_params={name: optimizer_settings[name] for name in optimizer_names}
    )
    
    # Plot results
    plot_results(results, optimizer_names, save_path=f'{results_dir}/optimizer_comparison.png')
    
    # Print summary
    print_summary(results, optimizer_names)


if __name__ == "__main__":
    main()
