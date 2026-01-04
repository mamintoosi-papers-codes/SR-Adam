"""
Ablation study utilities: batch size sensitivity, result reuse, checkpoint loading.
Keeps main.py clean by centralizing these concerns.
"""

import os
import json
import glob
import torch
import numpy as np
from pathlib import Path


def check_run_exists(dataset_name, model_name, noise, optimizer_name, run_id, base_dir='results', batch_size=512):
    """Check if a run's metrics have already been computed and saved.
    
    Path: results/{dataset}/{model}/noise_{noise}/{optimizer}/batch_size_{bs}/run_{id}_metrics.csv
    
    Args:
        batch_size: Batch size for this run (default: 512)
    
    Returns:
        (bool, dict) - (exists, metrics_dict or None)
    """
    # Always use new format with batch_size subdirectory
    folder = os.path.join(base_dir, dataset_name, model_name, f'noise_{noise}', optimizer_name, f'batch_size_{batch_size}')
    
    csv_path = os.path.join(folder, f'run_{run_id}_metrics.csv')
    meta_path = os.path.join(folder, f'run_{run_id}_meta.json')
    
    if os.path.exists(csv_path) and os.path.exists(meta_path):
        try:
            import pandas as pd
            df = pd.read_csv(csv_path)
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            
            metrics = {
                'train_loss': df['Train Loss'].tolist(),
                'test_loss': df['Test Loss'].tolist(),
                'train_acc': df['Train Acc'].tolist(),
                'test_acc': df['Test Acc'].tolist(),
                'epoch_time': df['Epoch Time'].tolist(),
                'final_test_acc': meta.get('final_test_acc'),
                'best_test_acc': meta.get('best_test_acc'),
            }
            
            # Preserve ablation metadata
            if 'ablation' in meta:
                metrics['ablation'] = meta['ablation']
            if 'batch_size' in meta:
                metrics['batch_size'] = meta['batch_size']
            
            return True, metrics
        except Exception as e:
            print(f"Warning: failed to load existing metrics from {csv_path}: {e}")
            return False, None
    
    return False, None


def try_load_checkpoint_metrics(dataset_name, model_name, noise, optimizer_name, run_id, 
                                num_epochs, base_dir='runs', batch_size=512):
    """If run metrics don't exist in results/ but checkpoint exists in runs/, 
    reconstruct minimal metrics from checkpoint metadata.
    
    Path: runs/{dataset}/{model}/noise_{noise}/{optimizer}/batch_size_{bs}/run_{id}_final.pt
    
    Args:
        batch_size: Batch size for this run (default: 512)
    
    Returns:
        (bool, dict) - (found, metrics_dict or None)
    """
    # Always use new format with batch_size subdirectory
    folder = os.path.join(base_dir, dataset_name, model_name, f'noise_{noise}', optimizer_name, f'batch_size_{batch_size}')
    
    last_ckpt = os.path.join(folder, f'run_{run_id}_final.pt')
    
    if not os.path.exists(last_ckpt):
        return False, None
    
    try:
        ckpt = torch.load(last_ckpt, map_location='cpu')
        test_acc = ckpt.get('test_acc', None)
        
        if test_acc is None:
            return False, None
        
        # Reconstruct minimal metrics (placeholder epochs)
        metrics = {
            'train_loss': [0.0] * num_epochs,
            'test_loss': [0.0] * num_epochs,
            'train_acc': [0.0] * num_epochs,
            'test_acc': [test_acc] * num_epochs,  # Use loaded acc for last epoch
            'epoch_time': [0.0] * num_epochs,
            'final_test_acc': test_acc,
            'best_test_acc': test_acc,
            'best_state_dict': None,
            'final_state_dict': ckpt.get('model_state_dict'),
        }
        
        # Always tag with batch_size
        metrics['batch_size'] = batch_size
        
        return True, metrics
    except Exception as e:
        print(f"Warning: failed to load checkpoint metrics from {last_ckpt}: {e}")
        return False, None


def should_skip_run(dataset_name, model_name, noise, optimizer_name, run_id, 
                    num_epochs, clean_previous, batch_size=None, base_dir='results'):
    """
    Decide whether to skip training for a run (already computed or in checkpoint).
    
    Args:
        batch_size: If provided, check in batch_size_{bs}/ subdirectory
    
    Returns:
        (bool, dict or None) - (skip, metrics_dict or None)
    """
    if clean_previous:
        # User explicitly wants to recompute
        return False, None
    
    # Check if metrics exist
    exists, metrics = check_run_exists(dataset_name, model_name, noise, optimizer_name, run_id, base_dir, batch_size)
    if exists:
        print(f"  [Reuse] Found existing metrics for run_{run_id}")
        return True, metrics
    
    # Fallback: try loading from checkpoint
    found, metrics = try_load_checkpoint_metrics(dataset_name, model_name, noise, optimizer_name, 
                                                 run_id, num_epochs, 'runs', batch_size)
    if found:
        print(f"  [Reuse] Reconstructed metrics from checkpoint for run_{run_id}")
        return True, metrics
    
    return False, None


def get_batch_ablation_configs(ablation_enabled, optimizers_requested, default_batch_size=512):
    """Return list of (optimizer, batch_size) tuples for ablation.
    
    If ablation enabled:
        - Adam and SR-Adam: batch sizes [256, 512, 2048]
        - Other optimizers: default_batch_size
    Else:
        - All optimizers: default_batch_size
        
    Args:
        default_batch_size: Default batch size when not ablated (default: 512)
    
    Returns:
        List of (optimizer_name, batch_size) tuples
    """
    if not ablation_enabled:
        # No ablation: all optimizers use default batch size
        return [(opt, default_batch_size) for opt in optimizers_requested]
    
    # Ablation enabled
    batch_sizes = [256, 512, 2048]
    ablation_optimizers = ["Adam", "SR-Adam"]
    
    configs = []
    for opt in optimizers_requested:
        if opt in ablation_optimizers:
            # Ablate these optimizers across multiple batch sizes
            for bs in batch_sizes:
                configs.append((opt, bs))
        else:
            # Other optimizers use default batch size
            configs.append((opt, default_batch_size))
    
    return configs


def tag_ablation_metadata(metrics_dict, batch_size=None):
    """
    Add ablation metadata to metrics dict before saving.
    """
    if batch_size is not None:
        metrics_dict['ablation'] = 'batch_size'
        metrics_dict['batch_size'] = batch_size
    else:
        metrics_dict['ablation'] = None
    
    return metrics_dict
