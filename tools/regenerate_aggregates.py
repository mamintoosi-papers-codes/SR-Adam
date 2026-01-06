import argparse
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path

def parse_list_arg(raw, value_type=str):
    """Parse pipe/semicolon-separated list arguments."""
    if raw.upper() == "ALL":
        return None  # None means process all
    
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

def aggregate_runs_and_save(folder):
    pattern = os.path.join(folder, 'run_*.csv')
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No run CSV files found in {folder}")

    runs = [pd.read_csv(f) for f in files]

    final_accs = [float(r['Test Acc'].iloc[-1]) for r in runs]
    best_accs = [float(r['Test Acc'].max()) for r in runs]

    summary = {
        'num_runs': len(runs),
        'final_test_acc_mean': float(np.mean(final_accs)),
        'final_test_acc_std': float(np.std(final_accs, ddof=1)),
        'best_test_acc_mean': float(np.mean(best_accs)),
        'best_test_acc_std': float(np.std(best_accs, ddof=1)),
    }

    summary_path = os.path.join(folder, 'aggregated_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    acc_arrays = [r['Test Acc'].values for r in runs]
    max_len = max(len(a) for a in acc_arrays)
    acc_mat = np.zeros((len(acc_arrays), max_len), dtype=float)
    for i, a in enumerate(acc_arrays):
        acc_mat[i, :len(a)] = a

    epoch_idx = list(range(1, max_len + 1))
    mean = acc_mat.mean(axis=0)
    std = acc_mat.std(axis=0)
    df_epoch = pd.DataFrame({'Epoch': epoch_idx, 'Test Acc Mean': mean, 'Test Acc Std': std})
    df_epoch.to_csv(os.path.join(folder, 'aggregated_epoch_stats.csv'), index=False)

    excel_path = os.path.join(folder, 'runs_and_aggregate.xlsx')
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        for idx, r in enumerate(runs):
            sheet = f'run_{idx+1}'
            r.to_excel(writer, sheet_name=sheet, index=False)
        df_epoch.to_excel(writer, sheet_name='aggregate', index=False)

    return summary_path, excel_path


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
    
    for dataset in datasets:
        for noise in noise_levels:
            results_dir = Path('results') / dataset / args.model / f'noise_{noise}'
            
            if not results_dir.exists():
                print(f"[SKIP] Directory not found: {results_dir}")
                continue
            
            for opt_dir in results_dir.iterdir():
                if not opt_dir.is_dir():
                    continue
                
                opt_name = opt_dir.name
                
                # Filter optimizer
                if optimizers and opt_name not in optimizers:
                    continue
                
                # Find run CSV files
                run_files = sorted(opt_dir.glob('run_*.csv'))
                if not run_files:
                    continue
                
                # Filter by batch_size if specified
                if batch_sizes:
                    first_df = pd.read_csv(run_files[0])
                    if 'batch_size' in first_df.columns:
                        run_bs = int(first_df['batch_size'].iloc[0])
                        if run_bs not in batch_sizes:
                            continue
                
                print(f"Processing: {dataset}/{args.model}/noise_{noise}/{opt_name}")
                
                # Aggregate runs
                all_runs = []
                for run_file in run_files:
                    df = pd.read_csv(run_file)
                    all_runs.append(df)
                
                if not all_runs:
                    continue
                
                # Extract final and best accuracies
                finals_acc = [df['test_acc'].iloc[-1] for df in all_runs]
                bests_acc = [df['test_acc'].max() for df in all_runs]
                
                # Compute statistics with ddof=1 (sample std)
                summary = {
                    'final_test_acc_mean': float(np.mean(finals_acc)),
                    'final_test_acc_std': float(np.std(finals_acc, ddof=1)),
                    'best_test_acc_mean': float(np.mean(bests_acc)),
                    'best_test_acc_std': float(np.std(bests_acc, ddof=1)),
                    'num_runs': len(all_runs)
                }
                
                # Save aggregated summary
                summary_path = opt_dir / 'aggregated_summary.json'
                with open(summary_path, 'w') as f:
                    json.dump(summary, f, indent=2)
                
                print(f"  ✓ Saved: {summary_path}")
                processed_count += 1
    
    print(f"\n{'='*80}")
    print(f"Processed {processed_count} optimizer configurations")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
