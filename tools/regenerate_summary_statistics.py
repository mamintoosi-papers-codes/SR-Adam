import argparse
import glob
import json
import os
import pandas as pd
import numpy as np
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

def main():
    parser = argparse.ArgumentParser(
        description="Regenerate summary statistics CSV with optional filters"
    )
    parser.add_argument("--dataset", type=str, default="ALL")
    parser.add_argument("--noise", type=str, default="ALL")
    parser.add_argument("--batch_size", type=str, default="ALL")
    parser.add_argument("--optimizers", type=str, default="ALL")
    parser.add_argument("--model", type=str, default="simplecnn")
    parser.add_argument("--output", type=str, default="results/summary_statistics.csv")
    
    args = parser.parse_args()
    
    datasets = parse_list_arg(args.dataset) or ["CIFAR10", "CIFAR100"]
    noise_levels = parse_list_arg(args.noise, float) or [0.0, 0.05, 0.1]
    batch_sizes = parse_list_arg(args.batch_size, int)
    optimizers = parse_list_arg(args.optimizers)
    
    print(f"\n{'='*80}")
    print("Regenerating Summary Statistics")
    print(f"{'='*80}")
    print(f"Datasets:    {datasets}")
    print(f"Noise:       {noise_levels}")
    print(f"Batch sizes: {batch_sizes if batch_sizes else 'ALL'}")
    print(f"Optimizers:  {optimizers if optimizers else 'ALL'}")
    print(f"{'='*80}\n")
    
    rows = []
    
    for dataset in datasets:
        for noise in noise_levels:
            results_dir = Path('results') / dataset / args.model / f'noise_{noise}'
            
            if not results_dir.exists():
                continue
            
            for opt_dir in results_dir.iterdir():
                if not opt_dir.is_dir():
                    continue
                
                opt_name = opt_dir.name
                
                if optimizers and opt_name not in optimizers:
                    continue
                
                summary_file = opt_dir / 'aggregated_summary.json'
                if not summary_file.exists():
                    continue
                
                # Check batch_size filter
                run_files = list(opt_dir.glob('run_*.csv'))
                if run_files and batch_sizes:
                    df = pd.read_csv(run_files[0])
                    if 'batch_size' in df.columns:
                        run_bs = int(df['batch_size'].iloc[0])
                        if run_bs not in batch_sizes:
                            continue
                        batch_size = run_bs
                    else:
                        batch_size = None
                else:
                    batch_size = None
                
                with open(summary_file) as f:
                    summary = json.load(f)
                
                row = {
                    'dataset': dataset,
                    'model': args.model,
                    'noise': noise,
                    'optimizer': opt_name,
                    'batch_size': batch_size,
                    'final_acc_mean': summary['final_test_acc_mean'],
                    'final_acc_std': summary['final_test_acc_std'],
                    'best_acc_mean': summary['best_test_acc_mean'],
                    'best_acc_std': summary['best_test_acc_std'],
                    'num_runs': summary['num_runs']
                }
                rows.append(row)
    
    if rows:
        df = pd.DataFrame(rows)
        df = df.sort_values(['dataset', 'noise', 'batch_size', 'optimizer'])
        
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        df.to_csv(args.output, index=False)
        print(f"\n✓ Saved: {args.output}")
        print(f"  {len(df)} rows\n")
    else:
        print("\n[WARNING] No matching results found\n")

if __name__ == "__main__":
    main()
