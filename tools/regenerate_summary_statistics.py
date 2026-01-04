import os
import glob
import json
import numpy as np
import pandas as pd

RESULTS_ROOT = os.path.join(os.path.dirname(__file__), 'results')

if not os.path.isdir(RESULTS_ROOT):
    raise SystemExit(f'results folder not found: {RESULTS_ROOT}')

summary_stats = {}

for dataset in os.listdir(RESULTS_ROOT):
    dataset_path = os.path.join(RESULTS_ROOT, dataset)
    if not os.path.isdir(dataset_path):
        continue

    for model in os.listdir(dataset_path):
        model_path = os.path.join(dataset_path, model)
        if not os.path.isdir(model_path):
            continue

        ds_model_key = f'{dataset}|{model}'

        for noise_dir in os.listdir(model_path):
            if not noise_dir.startswith('noise_'):
                continue
            noise_path = os.path.join(model_path, noise_dir)
            if not os.path.isdir(noise_path):
                continue

            noise_value = noise_dir.replace('noise_', '')

            for optimizer in os.listdir(noise_path):
                opt_path = os.path.join(noise_path, optimizer)
                if not os.path.isdir(opt_path):
                    continue

                # Check for batch_size subdirectories
                for batch_dir in os.listdir(opt_path):
                    batch_path = os.path.join(opt_path, batch_dir)
                    if not os.path.isdir(batch_path):
                        continue
                    if not batch_dir.startswith('batch_size_'):
                        continue

                    run_files = sorted(glob.glob(os.path.join(batch_path, 'run_*_metrics.csv')))
                    if not run_files:
                        continue

                    runs = [pd.read_csv(f) for f in run_files]
                    final_accs = [float(r['Test Acc'].iloc[-1]) for r in runs]
                    best_accs = [float(r['Test Acc'].max()) for r in runs]

                    batch_size = batch_dir.replace('batch_size_', '')
                    stats_key = f"{dataset}|noise_{noise_value}|{optimizer}|batch_{batch_size}"
                    summary_stats[stats_key] = {
                        'final_mean': float(np.mean(final_accs)),
                        'final_std': float(np.std(final_accs, ddof=0)),
                        'best_mean': float(np.mean(best_accs)),
                        'best_std': float(np.std(best_accs, ddof=0)),
                        'num_runs': len(runs),
                        'batch_size': batch_size,
                    }

                    if ds_model_key not in summary_stats:
                        summary_stats[ds_model_key] = {}
                    if 'noise_table' not in summary_stats[ds_model_key]:
                        summary_stats[ds_model_key]['noise_table'] = {}
                    opt_batch_key = f"{optimizer}_batch{batch_size}"
                    if opt_batch_key not in summary_stats[ds_model_key]['noise_table']:
                        summary_stats[ds_model_key]['noise_table'][opt_batch_key] = []
                    summary_stats[ds_model_key]['noise_table'][opt_batch_key].append({
                        'noise': noise_value,
                        'final_mean': summary_stats[stats_key]['final_mean'],
                        'final_std': summary_stats[stats_key]['final_std'],
                    })

# Save summary_statistics.csv in the same format used by save_multirun_summary
summary_df = pd.DataFrame.from_dict(summary_stats, orient='index')
summary_csv_path = os.path.join(RESULTS_ROOT, 'summary_statistics.csv')
summary_df.to_csv(summary_csv_path)
print(f'Regenerated summary statistics at {summary_csv_path}')
