"""
Module: utils.py
Description: Utility functions for saving/loading results and visualization
"""

import os
import json
import glob
import time
import pandas as pd
import numpy as np
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


def save_run_metrics(metrics, dataset, model, noise, optimizer_name, run_id, base_dir='results'):
    """Save a single run's metrics as CSV and metadata JSON.

    Path: results/{dataset}/{model}/noise_{noise}/{optimizer_name}/run_{run_id}.csv
    """
    folder = os.path.join(base_dir, dataset, model, f"noise_{noise}", optimizer_name)
    os.makedirs(folder, exist_ok=True)

    csv_path = os.path.join(folder, f"run_{run_id}.csv")
    # Avoid overwriting existing run files
    if os.path.exists(csv_path):
        csv_path = os.path.join(folder, f"run_{run_id}_{int(time.time())}.csv")
    df = pd.DataFrame({
        'Epoch': list(range(1, len(metrics['test_acc']) + 1)),
        'Train Loss': metrics['train_loss'],
        'Test Loss': metrics['test_loss'],
        'Train Acc': metrics['train_acc'],
        'Test Acc': metrics['test_acc'],
        'Epoch Time': metrics['epoch_time']
    })
    df.to_csv(csv_path, index=False)

    meta = {
        'dataset': dataset,
        'model': model,
        'noise': noise,
        'optimizer': optimizer_name,
        'run_id': run_id,
        'final_test_acc': float(metrics['test_acc'][-1]),
        'best_test_acc': float(max(metrics['test_acc']))
    }
    with open(os.path.join(folder, f"run_{run_id}_meta.json"), 'w') as f:
        json.dump(meta, f, indent=2)

    return csv_path


def _to_cpu_state_dict(state_dict):
    """Detach and move model state dict tensors to CPU for portable checkpoints."""
    return {k: (v.detach().cpu() if hasattr(v, 'detach') else v) for k, v in state_dict.items()}


def save_run_checkpoints(dataset, model, noise, optimizer_name, run_id, metrics, base_dir='results'):
    """Save best and last model checkpoints for a single run.

    Paths:
      results/{dataset}/{model}/noise_{noise}/{optimizer_name}/run_{run_id}_best.pt
      results/{dataset}/{model}/noise_{noise}/{optimizer_name}/run_{run_id}_last.pt
    """
    folder = os.path.join(base_dir, dataset, model, f"noise_{noise}", optimizer_name)
    os.makedirs(folder, exist_ok=True)

    best_sd = metrics.get('best_state_dict')
    final_sd = metrics.get('final_state_dict')
    if best_sd is None and final_sd is None:
        # Nothing to save
        return None, None

    best_epoch = metrics.get('best_epoch')
    best_acc = metrics.get('best_test_acc')
    final_acc = float(metrics['test_acc'][-1]) if metrics.get('test_acc') else None

    best_path = None
    last_path = None

    if best_sd is not None:
        best_path = os.path.join(folder, f"run_{run_id}_best.pt")
        torch.save({
            'epoch': best_epoch,
            'test_acc': best_acc,
            'model_state_dict': _to_cpu_state_dict(best_sd),
            'meta': {
                'dataset': dataset,
                'model': model,
                'noise': noise,
                'optimizer': optimizer_name,
                'run_id': run_id,
            }
        }, best_path)

    if final_sd is not None:
        last_path = os.path.join(folder, f"run_{run_id}_last.pt")
        torch.save({
            'epoch': len(metrics['test_acc']) if metrics.get('test_acc') else None,
            'test_acc': final_acc,
            'model_state_dict': _to_cpu_state_dict(final_sd),
            'meta': {
                'dataset': dataset,
                'model': model,
                'noise': noise,
                'optimizer': optimizer_name,
                'run_id': run_id,
            }
        }, last_path)

    if best_path:
        print(f"Saved best checkpoint to {best_path}")
    if last_path:
        print(f"Saved last checkpoint to {last_path}")

    return best_path, last_path


def aggregate_runs_and_save(dataset, model, noise, optimizer_name, base_dir='results'):
    """Aggregate all run CSVs for an optimizer and save summary JSON and Excel.

    Expects files: results/{dataset}/{model}/noise_{noise}/{optimizer_name}/run_{k}.csv
    """
    folder = os.path.join(base_dir, dataset, model, f"noise_{noise}", optimizer_name)
    pattern = os.path.join(folder, "run_*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No run CSV files found in {folder}")

    runs = [pd.read_csv(f) for f in files]

    # Final and best accuracies across runs
    final_accs = [float(r['Test Acc'].iloc[-1]) for r in runs]
    best_accs = [float(r['Test Acc'].max()) for r in runs]

    summary = {
        'num_runs': len(runs),
        'final_test_acc_mean': float(np.mean(final_accs)),
        'final_test_acc_std': float(np.std(final_accs, ddof=0)),
        'best_test_acc_mean': float(np.mean(best_accs)),
        'best_test_acc_std': float(np.std(best_accs, ddof=0)),
    }

    # Save summary JSON
    summary_path = os.path.join(folder, 'aggregated_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    # Save per-epoch aggregated mean/std for Test Acc
    acc_arrays = [r['Test Acc'].values for r in runs]
    # pad to same length if necessary
    max_len = max(len(a) for a in acc_arrays)
    acc_mat = np.zeros((len(acc_arrays), max_len), dtype=float)
    for i, a in enumerate(acc_arrays):
        acc_mat[i, :len(a)] = a

    epoch_idx = list(range(1, max_len + 1))
    mean = acc_mat.mean(axis=0)
    std = acc_mat.std(axis=0)
    df_epoch = pd.DataFrame({'Epoch': epoch_idx, 'Test Acc Mean': mean, 'Test Acc Std': std})
    df_epoch.to_csv(os.path.join(folder, 'aggregated_epoch_stats.csv'), index=False)

    # Save Excel with per-run sheets + aggregate
    excel_path = os.path.join(folder, 'runs_and_aggregate.xlsx')
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        for idx, r in enumerate(runs):
            sheet = f'run_{idx+1}'
            r.to_excel(writer, sheet_name=sheet, index=False)
        df_epoch.to_excel(writer, sheet_name='aggregate', index=False)

    return summary_path, excel_path


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
