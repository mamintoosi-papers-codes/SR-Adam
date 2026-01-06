import os
import json
import pandas as pd
import numpy as np
import argparse
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

def generate_output_filename(batch_sizes, output_template):
    """
    Generate unique output filename based on batch sizes.
    
    Examples:
    - [512] -> "minimal-tables-content_bs512.tex"
    - [256, 512, 2048] -> "minimal-tables-content_bs256-512-2048.tex"
    """
    if not batch_sizes:
        return output_template
    
    # Remove .tex extension if present
    if output_template.endswith('.tex'):
        base = output_template[:-4]
    else:
        base = output_template
    
    # Create batch_size suffix
    bs_suffix = "_bs" + "-".join(str(bs) for bs in sorted(batch_sizes))
    
    return base + bs_suffix + ".tex"

def main():
    parser = argparse.ArgumentParser(
        description="Generate LaTeX tables from aggregated results with filters"
    )
    parser.add_argument("--dataset", type=str, default="ALL",
                        help='Dataset(s): "ALL", "CIFAR10", "CIFAR100"')
    parser.add_argument("--noise", type=str, default="ALL",
                        help='Noise level(s): "ALL" or specific values')
    parser.add_argument("--batch_size", type=str, default="512",
                        help='Batch size(s) to include')
    parser.add_argument("--optimizers", type=str, default="ALL",
                        help='Optimizer(s): "ALL" or specific names')
    parser.add_argument("--output", type=str, default="paper_figures/minimal-tables-content.tex",
                        help='Output .tex file path (batch sizes will be appended)')
    
    args = parser.parse_args()
    
    # Parse arguments
    datasets = parse_list_arg(args.dataset) or ["CIFAR10", "CIFAR100"]
    noise_levels = parse_list_arg(args.noise, float) or [0.0, 0.05, 0.1]
    batch_sizes = parse_list_arg(args.batch_size, int) or [512]
    optimizers = parse_list_arg(args.optimizers)
    
    # Get standard optimizer order
    optimizer_order = get_optimizer_order()
    
    print(f"\n{'='*80}")
    print("Generating LaTeX Tables")
    print(f"{'='*80}")
    print(f"Datasets:    {datasets}")
    print(f"Noise:       {noise_levels}")
    print(f"Batch sizes: {batch_sizes}")
    print(f"Optimizers:  {optimizers if optimizers else 'ALL (in standard order)'}")
    print(f"Output:      {args.output}")
    print(f"{'='*80}\n")
    
    # Collect data from aggregated summaries
    table_data = {}
    found_optimizers = set()
    
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
                    found_optimizers.add(opt_name)
                    
                    # Filter optimizer
                    if optimizers and opt_name not in optimizers:
                        continue
                    
                    # Look for batch_size_X subdirectory
                    batch_dir = opt_dir / f'batch_size_{batch_size}'
                    summary_file = batch_dir / 'aggregated_summary.json'
                    
                    if not summary_file.exists():
                        continue
                    
                    try:
                        with open(summary_file) as f:
                            summary = json.load(f)
                        
                        key = (dataset, noise, batch_size, opt_name)
                        table_data[key] = summary
                    except Exception as e:
                        print(f"[ERROR] Failed to read {summary_file}: {e}")
                        continue
    
    if not table_data:
        print("❌ No data found!")
        print("   Make sure to run regenerate_aggregates.py first:")
        print("   python tools/regenerate_aggregates.py --batch_size 512\n")
        return
    
    # Generate unique output filename with batch_size info
    output_file = generate_output_filename(batch_sizes, args.output)
    
    # Generate LaTeX content
    latex_lines = []
    latex_lines.append("% Auto-generated by make_minimal_tables.py\n")
    latex_lines.append("% Filters applied:\n")
    latex_lines.append(f"% Datasets: {datasets}\n")
    latex_lines.append(f"% Noise: {noise_levels}\n")
    latex_lines.append(f"% Batch sizes: {batch_sizes}\n")
    latex_lines.append(f"% Optimizers: {optimizers if optimizers else 'ALL (standard order)'}\n\n")
    
    # Group data by batch_size and generate tables
    for batch_size in batch_sizes:
        print(f"\nGenerating table for batch_size={batch_size}:")
        
        for dataset in datasets:
            print(f"  {dataset}:")
            
            latex_lines.append(f"% {dataset} (batch_size={batch_size})\n")
            latex_lines.append("\\begin{table}[h]\n")
            latex_lines.append("\\centering\n")
            latex_lines.append("\\begin{tabular}{lcccc}\n")
            latex_lines.append("\\toprule\n")
            latex_lines.append("Optimizer & Noise=0.0 & Noise=0.05 & Noise=0.1 \\\\\n")
            latex_lines.append("\\midrule\n")
            
            # Get unique optimizers for this dataset that exist in data
            available_opts = set(k[3] for k in table_data.keys() 
                                if k[0] == dataset and k[2] == batch_size)
            
            # Use standard optimizer order, but only for those available
            opt_names_ordered = [opt for opt in optimizer_order if opt in available_opts]
            
            # Add any found optimizers not in standard order (shouldn't happen normally)
            for opt in sorted(available_opts):
                if opt not in opt_names_ordered:
                    opt_names_ordered.append(opt)
            
            for opt_name in opt_names_ordered:
                row = [opt_name]
                found = False
                
                for noise in noise_levels:
                    key = (dataset, noise, batch_size, opt_name)
                    
                    if key in table_data:
                        data = table_data[key]
                        acc = data['best_test_acc_mean']
                        std = data['best_test_acc_std']
                        row.append(f"{acc:.2f} $\\pm$ {std:.2f}")
                        found = True
                    else:
                        row.append("--")
                
                if found:
                    latex_lines.append(" & ".join(row) + " \\\\\n")
                    print(f"    ✓ {opt_name}")
            
            latex_lines.append("\\bottomrule\n")
            latex_lines.append("\\end{tabular}\n")
            latex_lines.append(f"\\caption{{Test Accuracy Results - {dataset} (batch\\_size={batch_size})}}\n")
            latex_lines.append(f"\\label{{tab:{dataset.lower()}_bs{batch_size}}}\n")
            latex_lines.append("\\end{table}\n\n")
        
        print(f"  ✓ Table generated for batch_size={batch_size}")
    
    # Write output
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        f.writelines(latex_lines)
    
    print(f"\n✓ Saved: {output_file}")
    print(f"  {len(table_data)} data points")
    print(f"  Optimizers (in order): {', '.join(opt_names_ordered)}\n")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()