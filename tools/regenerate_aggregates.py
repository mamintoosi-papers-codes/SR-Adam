import argparse
import os
import json
import numpy as np
import pandas as pd
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

def find_accuracy_column(df):
    """Find the accuracy column (flexible naming)."""
    for col_name in ['test_acc', 'Test Acc', 'test_accuracy', 'Test Accuracy', 'accuracy']:
        if col_name in df.columns:
            return col_name
    # اگر پیدا نشد، چاپ کنید
    return None

def main():
    parser = argparse.ArgumentParser(
        description="Regenerate aggregated results from individual run CSVs with optional filters"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="ALL",
        help='Dataset(s): "ALL", "CIFAR10", "CIFAR100", or pipe-separated list'
    )
    parser.add_argument(
        "--noise",
        type=str,
        default="ALL",
        help='Noise level(s): "ALL" or specific values like "0.05" or "0.0|0.05|0.1"'
    )
    parser.add_argument(
        "--batch_size",
        type=str,
        default="ALL",
        help='Batch size(s): "ALL" or specific values like "512" or "256|512|2048"'
    )
    parser.add_argument(
        "--optimizers",
        type=str,
        default="ALL",
        help='Optimizer(s): "ALL" or specific names like "Adam|SR-Adam"'
    )
    parser.add_argument(
        "--model",
        type=str,
        default="simplecnn",
        help='Model architecture (default: simplecnn)'
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    # Parse filters
    datasets = parse_list_arg(args.dataset) or ["CIFAR10", "CIFAR100"]
    noise_levels = parse_list_arg(args.noise, float) or [0.0, 0.05, 0.1]
    batch_sizes = parse_list_arg(args.batch_size, int)  # None = all
    optimizers = parse_list_arg(args.optimizers)  # None = all
    
    print(f"\n{'='*80}")
    print("Regenerating Aggregated Results")
    print(f"{'='*80}")
    print(f"Datasets:    {datasets}")
    print(f"Noise:       {noise_levels}")
    print(f"Batch sizes: {batch_sizes if batch_sizes else 'ALL'}")
    print(f"Optimizers:  {optimizers if optimizers else 'ALL'}")
    print(f"Model:       {args.model}")
    print(f"{'='*80}\n")
    
    processed_count = 0
    skipped_count = 0
    
    # First pass: detect column name from first file
    acc_column = None
    for dataset in datasets:
        for noise in noise_levels:
            results_dir = Path('results') / dataset / args.model / f'noise_{noise}'
            
            if not results_dir.exists():
                continue
            
            for opt_dir in results_dir.iterdir():
                if not opt_dir.is_dir():
                    continue
                
                for batch_dir in opt_dir.iterdir():
                    if not batch_dir.is_dir():
                        continue
                    
                    run_files = sorted(batch_dir.glob('run_*_metrics.csv'))
                    if run_files:
                        try:
                            df = pd.read_csv(run_files[0])
                            acc_column = find_accuracy_column(df)
                            if acc_column:
                                print(f"✓ Detected accuracy column: '{acc_column}'")
                                print(f"  (from {run_files[0].name})\n")
                                break
                        except Exception as e:
                            pass
                
                if acc_column:
                    break
            
            if acc_column:
                break
        
        if acc_column:
            break
    
    if not acc_column:
        print("❌ Could not find accuracy column!")
        print("   Checked for: test_acc, Test Acc, test_accuracy, Test Accuracy, accuracy")
        print("\n   Please check your CSV files:\n")
        # List a sample
        for dataset in datasets[:1]:
            for noise in noise_levels[:1]:
                results_dir = Path('results') / dataset / args.model / f'noise_{noise}'
                if results_dir.exists():
                    for opt_dir in list(results_dir.iterdir())[:1]:
                        for batch_dir in list(opt_dir.iterdir())[:1]:
                            run_files = sorted(batch_dir.glob('run_*_metrics.csv'))
                            if run_files:
                                print(f"   Sample file: {run_files[0]}")
                                try:
                                    df = pd.read_csv(run_files[0])
                                    print(f"   Columns: {df.columns.tolist()}\n")
                                except Exception as e:
                                    print(f"   Error: {e}\n")
        return
    
    # Second pass: aggregate with detected column
    for dataset in datasets:
        for noise in noise_levels:
            results_dir = Path('results') / dataset / args.model / f'noise_{noise}'
            
            if not results_dir.exists():
                if args.verbose:
                    print(f"[SKIP] Directory not found: {results_dir}")
                continue
            
            for opt_dir in results_dir.iterdir():
                if not opt_dir.is_dir():
                    continue
                
                opt_name = opt_dir.name
                
                # Filter optimizer
                if optimizers and opt_name not in optimizers:
                    if args.verbose:
                        print(f"[SKIP] Optimizer filtered: {opt_name}")
                    skipped_count += 1
                    continue
                
                # Iterate through batch_size subdirectories
                for batch_dir in opt_dir.iterdir():
                    if not batch_dir.is_dir() or not batch_dir.name.startswith('batch_size_'):
                        continue
                    
                    batch_size_str = batch_dir.name.replace('batch_size_', '')
                    try:
                        batch_size = int(batch_size_str)
                    except ValueError:
                        continue
                    
                    # Filter batch_size if specified
                    if batch_sizes and batch_size not in batch_sizes:
                        if args.verbose:
                            print(f"[SKIP] Batch size {batch_size} filtered")
                        skipped_count += 1
                        continue
                    
                    # Find run CSV files
                    run_files = sorted(batch_dir.glob('run_*_metrics.csv'))
                    if not run_files:
                        if args.verbose:
                            print(f"[SKIP] No run files in {batch_dir}")
                        continue
                    
                    print(f"Processing: {dataset}/{args.model}/noise_{noise}/{opt_name}/batch_size_{batch_size}")
                    print(f"  Found {len(run_files)} run files")
                    
                    # Aggregate runs
                    all_runs = []
                    for run_file in run_files:
                        try:
                            df = pd.read_csv(run_file)
                            all_runs.append(df)
                        except Exception as e:
                            print(f"  ⚠️ Error reading {run_file.name}: {e}")
                            continue
                    
                    if not all_runs:
                        print(f"  ⚠️ No valid run data")
                        continue
                    
                    # Extract final and best accuracies
                    finals_acc = []
                    bests_acc = []
                    
                    for df in all_runs:
                        if acc_column in df.columns:
                            finals_acc.append(float(df[acc_column].iloc[-1]))
                            bests_acc.append(float(df[acc_column].max()))
                    
                    if not finals_acc:
                        print(f"  ⚠️ No valid accuracy data")
                        continue
                    
                    # Compute statistics with ddof=1 (sample std)
                    summary = {
                        'dataset': dataset,
                        'model': args.model,
                        'noise': float(noise),
                        'optimizer': opt_name,
                        'batch_size': batch_size,
                        'final_test_acc_mean': float(np.mean(finals_acc)),
                        'final_test_acc_std': float(np.std(finals_acc, ddof=1)) if len(finals_acc) > 1 else 0.0,
                        'best_test_acc_mean': float(np.mean(bests_acc)),
                        'best_test_acc_std': float(np.std(bests_acc, ddof=1)) if len(bests_acc) > 1 else 0.0,
                        'num_runs': len(all_runs)
                    }
                    
                    # Save aggregated summary
                    summary_path = batch_dir / 'aggregated_summary.json'
                    with open(summary_path, 'w') as f:
                        json.dump(summary, f, indent=2)
                    
                    print(f"  ✓ Saved: {summary_path}")
                    print(f"    Final Acc: {summary['final_test_acc_mean']:.4f} ± {summary['final_test_acc_std']:.4f}")
                    print(f"    Best Acc:  {summary['best_test_acc_mean']:.4f} ± {summary['best_test_acc_std']:.4f}\n")
                    
                    processed_count += 1
    
    print(f"{'='*80}")
    print(f"✓ Processed {processed_count} configurations")
    print(f"⊘ Skipped {skipped_count} configurations")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
