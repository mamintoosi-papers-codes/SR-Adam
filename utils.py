"""
Module: utils.py
Description: Utility functions for saving/loading results and visualization
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import torch


def save_multirun_summary(summary_stats, results_dir):
    """
    Save mean/std statistics over multiple runs.
    """
    import pandas as pd
    import os

    df = pd.DataFrame.from_dict(summary_stats, orient="index")
    path = os.path.join(results_dir, "summary_statistics.csv")
    df.to_csv(path)
    print(f"Multi-run summary saved to {path}")


def count_parameters(model):
    """Return the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_results_directory(dataset_name, noise_std):
    """
    Create results directory with appropriate naming.
    
    Args:
        dataset_name (str): Name of dataset (CIFAR10, CIFAR100)
        noise_std (float): Standard deviation of Gaussian noise
    
    Returns:
        str: Path to results directory
    """
    folder_name = f"results_{dataset_name}_noise{noise_std}"
    os.makedirs(folder_name, exist_ok=True)
    return folder_name


def save_checkpoint(model, optimizer, epoch, metrics, checkpoint_dir):
    """
    Save model checkpoint with metrics.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer instance
        epoch: Current epoch number
        metrics: Dictionary of metrics up to this epoch
        checkpoint_dir (str): Directory to save checkpoint
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
    
    torch_checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    
    torch.save(torch_checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")


def save_intermediate_results(results_dict, optimizer_name, epoch, results_dir):
    """
    Save intermediate results after each epoch.
    
    Args:
        results_dict (dict): Metrics dictionary with train/test loss and accuracy
        optimizer_name (str): Name of optimizer
        epoch (int): Current epoch number
        results_dir (str): Directory to save results
    """
    os.makedirs(results_dir, exist_ok=True)
    
    # Save as CSV for quick access
    csv_path = os.path.join(results_dir, f"{optimizer_name}_metrics.csv")
    df = pd.DataFrame({
        'Epoch': list(range(1, epoch + 1)),
        'Train Loss': results_dict['train_loss'],
        'Test Loss': results_dict['test_loss'],
        'Train Acc': results_dict['train_acc'],
        'Test Acc': results_dict['test_acc'],
        'Epoch Time': results_dict['epoch_time']
    })
    df.to_csv(csv_path, index=False)


def save_all_results(results, optimizers_names, dataset_name, batch_size, num_epochs, noise_std, optimizer_params=None):
    """
    Save final results to Excel and JSON config.
    
    Args:
        results (list): List of metrics dictionaries for each optimizer
        optimizers_names (list): List of optimizer names
        dataset_name (str): Name of dataset
        batch_size (int): Batch size used
        num_epochs (int): Number of epochs
        noise_std (float): Noise standard deviation
    """
    folder_name = create_results_directory(dataset_name, noise_std)
    
    # Save to Excel
    excel_filename = f"optimizer_comparison_{dataset_name}_batch{batch_size}_epochs{num_epochs}_noise{noise_std}.xlsx"
    excel_path = os.path.join(folder_name, excel_filename)
    
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        for i, name in enumerate(optimizers_names):
            df = pd.DataFrame({
                'Epoch': list(range(1, num_epochs + 1)),
                'Train Loss': results[i]['train_loss'],
                'Test Loss': results[i]['test_loss'],
                'Train Acc': results[i]['train_acc'],
                'Test Acc': results[i]['test_acc'],
                'Epoch Time': results[i]['epoch_time']
            })
            # Clean sheet name: remove special characters
            sheet_name = name.replace(' ', '_').replace('(', '').replace(')', '').replace(',', '')[:31]
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    
    print(f"Results saved to {excel_path}")
    
    # Save config
    config = {
        'dataset_name': dataset_name,
        'batch_size': batch_size,
        'num_epochs': num_epochs,
        'noise_std': noise_std,
        'optimizers': optimizers_names,
        'optimizer_params': optimizer_params or {},
        'final_test_accuracies': {name: results[i]['test_acc'][-1] for i, name in enumerate(optimizers_names)}
    }
    config_path = os.path.join(folder_name, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"Config saved to {config_path}")


def plot_results(results, optimizers_names, save_path='optimizer_comparison.png'):
    """
    Plot training and testing metrics for all optimizers.
    
    Args:
        results (list): List of metrics dictionaries
        optimizers_names (list): List of optimizer names
        save_path (str): Path to save the plot
    """
    plt.figure(figsize=(20, 12))
    
    # Training Loss
    plt.subplot(2, 2, 1)
    for i, metrics in enumerate(results):
        plt.plot(metrics['train_loss'], label=optimizers_names[i], linewidth=2)
    plt.title('Training Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Test Loss
    plt.subplot(2, 2, 2)
    for i, metrics in enumerate(results):
        plt.plot(metrics['test_loss'], label=optimizers_names[i], linewidth=2)
    plt.title('Test Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Training Accuracy
    plt.subplot(2, 2, 3)
    for i, metrics in enumerate(results):
        plt.plot(metrics['train_acc'], label=optimizers_names[i], linewidth=2)
    plt.title('Training Accuracy', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Test Accuracy
    plt.subplot(2, 2, 4)
    for i, metrics in enumerate(results):
        plt.plot(metrics['test_acc'], label=optimizers_names[i], linewidth=2)
    plt.title('Test Accuracy', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    plt.show()


def print_summary(results, optimizers_names):
    """
    Print summary statistics for all optimizers.
    
    Args:
        results (list): List of metrics dictionaries
        optimizers_names (list): List of optimizer names
    """
    print("\n" + "=" * 80)
    print("FINAL TEST ACCURACIES AND STATISTICS")
    print("=" * 80)
    
    for i, name in enumerate(optimizers_names):
        final_acc = results[i]['test_acc'][-1]
        best_acc = max(results[i]['test_acc'])
        avg_time = sum(results[i]['epoch_time']) / len(results[i]['epoch_time'])
        
        print(f"\n{name}:")
        print(f"  Final Test Accuracy: {final_acc:.2f}%")
        print(f"  Best Test Accuracy:  {best_acc:.2f}%")
        print(f"  Avg Epoch Time:      {avg_time:.2f}s")
    
    print("\n" + "=" * 80)

def plot_mean_std(results_per_optimizer, optimizer_names, save_path):
    """
    Plot mean ± std test accuracy over epochs.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))

    for name in optimizer_names:
        runs = results_per_optimizer[name]
        acc = np.array([r["test_acc"] for r in runs])  # [runs, epochs]

        mean = acc.mean(axis=0)
        std = acc.std(axis=0)

        plt.plot(mean, label=name)
        plt.fill_between(
            range(len(mean)),
            mean - std,
            mean + std,
            alpha=0.2
        )

    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy (%)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
