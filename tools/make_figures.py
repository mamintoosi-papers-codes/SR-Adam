import os
import json
import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
from pathlib import Path

def parse_list_arg(raw, value_type=str):
    """Parse pipe/semicolon-separated list arguments."""
    if raw.upper() == "ALL":
        return None
    
    for sep in ["|", ";", ","]:
        if sep in raw:
            tokens = [t.strip() for t in raw.split(sep) if t.strip()]
            break
    else:
        tokens = [raw.strip()]
    
    if value_type == float:
        return [float(t) for t in tokens]
    elif value_type == int:
        return [int(t) for t in tokens]
    else:
        return tokens

def get_optimizer_order():
    """Return the standard optimizer ordering from main.py"""
    return [
        "SGD",
        "Momentum",
        "Adam",
        "SR-Adam",
        "SR-Adam-All-Weights",
    ]

def get_optimizer_colors():
    """Return consistent colors for optimizers"""
    colors = {
        "SGD": "#1f77b4",
        "Momentum": "#ff7f0e",
        "Adam": "#2ca02c",
        "SR-Adam": "#d62728",
        "SR-Adam-All-Weights": "#9467bd",
    }
    return colors

def generate_output_dir(batch_sizes, output_dir):
    """
    Generate unique output directory based on batch sizes.
    
    Examples:
    - [512] -> "paper/bs512"
    - [256, 512, 2048] -> "paper/bs256-512-2048"
    """
    if not batch_sizes or output_dir == "paper":
        base_dir = output_dir
    else:
        bs_suffix = "_bs" + "-".join(str(bs) for bs in sorted(batch_sizes))
        base_dir = output_dir + bs_suffix
    
    return base_dir

def main():
    parser = argparse.ArgumentParser(
        description="Generate publication figures from results"
    )
    parser.add_argument("--dataset", type=str, default="ALL",
                        help='Dataset(s) to plot')
    parser.add_argument("--noise", type=str, default="ALL",
                        help='Noise level(s) to plot')
    parser.add_argument("--batch_size", type=str, default="512",
                        help='Batch size(s) to include in plots')
    parser.add_argument("--output_dir", type=str, default="paper",
                        help='Output directory for figures (batch sizes will be appended)')
    
    args = parser.parse_args()
    
    # Parse filters
    datasets = parse_list_arg(args.dataset) or ["CIFAR10", "CIFAR100"]
    noise_levels = parse_list_arg(args.noise, float) or [0.0, 0.05, 0.1]
    batch_sizes = parse_list_arg(args.batch_size, int) or [512]
    
    # Get standard orderings
    optimizer_order = get_optimizer_order()
    optimizer_colors = get_optimizer_colors()
    
    # Generate unique output directory
    output_dir = generate_output_dir(batch_sizes, args.output_dir)
    
    print(f"\n{'='*80}")
    print("Generating Publication Figures")
    print(f"{'='*80}")
    print(f"Datasets:    {datasets}")
    print(f"Noise:       {noise_levels}")
    print(f"Batch sizes: {batch_sizes}")
    print(f"Output dir:  {output_dir}")
    print(f"{'='*80}\n")
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Collect data
    figure_data = {}
    
    for dataset in datasets:
        for noise in noise_levels:
            for batch_size in batch_sizes:
                results_dir = Path('results') / dataset / 'simplecnn' / f'noise_{noise}'
                
                if not results_dir.exists():
                    continue
                
                for opt_dir in results_dir.iterdir():
                    if not opt_dir.is_dir():
                        continue
                    
                    opt_name = opt_dir.name
                    batch_dir = opt_dir / f'batch_size_{batch_size}'
                    summary_file = batch_dir / 'aggregated_summary.json'
                    
                    if not summary_file.exists():
                        continue
                    
                    try:
                        with open(summary_file) as f:
                            summary = json.load(f)
                        
                        key = (dataset, noise, batch_size, opt_name)
                        figure_data[key] = summary
                    except Exception as e:
                        print(f"[ERROR] Failed to read {summary_file}: {e}")
                        continue
    
    if not figure_data:
        print("❌ No data found! Run regenerate_aggregates.py first.\n")
        return
    
    # Generate comparison figures
    for dataset in datasets:
        print(f"\nGenerating figures for {dataset}:")
        
        fig, axes = plt.subplots(1, len(noise_levels), figsize=(5*len(noise_levels), 4))
        if len(noise_levels) == 1:
            axes = [axes]
        
        for noise_idx, noise in enumerate(noise_levels):
            ax = axes[noise_idx]
            
            # Collect data for this noise level
            data_points = []
            
            for batch_size in batch_sizes:
                for opt_name in optimizer_order:
                    key = (dataset, noise, batch_size, opt_name)
                    if key in figure_data:
                        summary = figure_data[key]
                        data_points.append({
                            'optimizer': opt_name,
                            'batch_size': batch_size,
                            'best_acc': summary['best_test_acc_mean'],
                            'std': summary['best_test_acc_std']
                        })
            
            if not data_points:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center')
                ax.set_title(f'Noise = {noise}')
                continue
            
            # Plot
            batch_sizes_unique = sorted(set(dp['batch_size'] for dp in data_points))
            x_pos = np.arange(len(batch_sizes_unique))
            width = 0.15
            
            for idx, opt_name in enumerate(optimizer_order):
                means = []
                stds = []
                
                for bs in batch_sizes_unique:
                    dp = next((d for d in data_points 
                              if d['optimizer'] == opt_name and d['batch_size'] == bs), None)
                    if dp:
                        means.append(dp['best_acc'])
                        stds.append(dp['std'])
                    else:
                        means.append(0)
                        stds.append(0)
                
                offset = (idx - len(optimizer_order)/2) * width
                color = optimizer_colors.get(opt_name, None)
                ax.bar(x_pos + offset, means, width, label=opt_name, 
                       color=color, alpha=0.8, yerr=stds, capsize=3)
            
            ax.set_xlabel('Batch Size')
            ax.set_ylabel('Test Accuracy (%)')
            ax.set_title(f'{dataset} - Noise = {noise}')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(batch_sizes_unique)
            ax.legend(fontsize=8, loc='lower right')
            ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure with batch_size info
        bs_suffix = "_bs" + "-".join(str(bs) for bs in sorted(batch_sizes))
        fig_path = Path(output_dir) / f"{dataset}_comparison{bs_suffix}.png"
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved: {fig_path}")
        plt.close()
    
    print(f"\n✓ All figures saved to: {output_dir}")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()