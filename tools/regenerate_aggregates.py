import os
import sys
import glob
import json
import numpy as np
import pandas as pd

# Lightweight reimplementation of aggregate_runs_and_save to avoid torch import.
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


ROOT = os.path.join(os.path.dirname(__file__), 'results')

if not os.path.isdir(ROOT):
    sys.exit('results folder not found: {}'.format(ROOT))

updated = []
for dirpath, dirnames, filenames in os.walk(ROOT):
    if not any(f.startswith('run_') and f.endswith('.csv') for f in filenames):
        continue
    rel = os.path.relpath(dirpath, ROOT)
    parts = rel.split(os.sep)
    if len(parts) < 4:
        continue
    dataset, model, noise_dir, optimizer = parts[:4]
    if not noise_dir.startswith('noise_'):
        continue
    aggregate_runs_and_save(dirpath)
    updated.append((dataset, model, noise_dir, optimizer))
    print(f'Regenerated aggregates for {dataset}/{model}/{noise_dir}/{optimizer}')

if not updated:
    print('No run_* CSV folders found; nothing regenerated')
else:
    print(f'Total regenerated: {len(updated)}')
