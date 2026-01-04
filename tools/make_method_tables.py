import os
import glob
import pandas as pd
import numpy as np

RESULTS_ROOT = os.path.join(os.path.dirname(__file__), 'results')
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'paper')
os.makedirs(OUT_DIR, exist_ok=True)

METHOD_ORDER = ["SGD", "Momentum", "Adam", "SR-Adam"]
NOISES = ["0.0", "0.05", "0.1"]
DATASETS = ["CIFAR10", "CIFAR100"]
MODEL = "simplecnn"


def collect_runs(dataset, noise, optimizer, batch_size=512):
    folder = os.path.join(RESULTS_ROOT, dataset, MODEL, f"noise_{noise}", optimizer, f"batch_size_{batch_size}")
    files = sorted(glob.glob(os.path.join(folder, 'run_*_metrics.csv')))
    return [pd.read_csv(f) for f in files]


def compute_stats(runs):
    # Each run is a DF with columns: Train Loss, Test Loss, Train Acc, Test Acc
    finals_acc = [float(r['Test Acc'].iloc[-1]) for r in runs]
    bests_acc = [float(r['Test Acc'].max()) for r in runs]
    finals_loss = [float(r['Test Loss'].iloc[-1]) for r in runs]
    bests_loss = [float(r['Test Loss'].min()) for r in runs]  # lower is better
    return {
        'final_acc_mean': float(np.mean(finals_acc)),
        'final_acc_std': float(np.std(finals_acc, ddof=0)),
        'best_acc_mean': float(np.mean(bests_acc)),
        'best_acc_std': float(np.std(bests_acc, ddof=0)),
        'final_loss_mean': float(np.mean(finals_loss)),
        'final_loss_std': float(np.std(finals_loss, ddof=0)),
        'best_loss_mean': float(np.mean(bests_loss)),
        'best_loss_std': float(np.std(bests_loss, ddof=0)),
    }


def format_cell(mean, std, bold=False):
    cell = f"{mean:.2f} \\pm {std:.2f}"
    return f"\\textbf{{{cell}}}" if bold else cell


def build_table(metric_key, metric_label, higher_is_better, out_tex, batch_size=512):
    # columns: CIFAR10 (0.0,0.05,0.1) | CIFAR100 (0.0,0.05,0.1)
    lines = []
    lines.append("\\documentclass[varwidth=\\maxdimen]{standalone}")
    lines.append("\\usepackage{booktabs}")
    lines.append("\\begin{document}")
    lines.append("\\begin{tabular}{l ccc ccc}")
    lines.append("\\toprule")
    lines.append(" & \\multicolumn{3}{c}{CIFAR10} & \\multicolumn{3}{c}{CIFAR100} \\")
    lines.append("Method & 0.0 & 0.05 & 0.1 & 0.0 & 0.05 & 0.1 \\\\")
    lines.append("\\midrule")

    for method in METHOD_ORDER:
        row_cells = [method]
        values = {}
        # Compute per dataset/noise
        for dataset in DATASETS:
            for noise in NOISES:
                runs = collect_runs(dataset, noise, method, batch_size)
                if not runs:
                    values[(dataset, noise)] = (float('nan'), float('nan'))
                    continue
                stats = compute_stats(runs)
                mean = stats[f'{metric_key}_mean']
                std = stats[f'{metric_key}_std']
                values[(dataset, noise)] = (mean, std)
        # Determine best per column
        for dataset in DATASETS:
            for noise in NOISES:
                col_vals = {m: values[(dataset, noise)][0] for m in METHOD_ORDER}
                if higher_is_better:
                    best_method = max(col_vals, key=lambda k: col_vals[k])
                else:
                    best_method = min(col_vals, key=lambda k: col_vals[k])
                mean, std = values[(dataset, noise)]
                row_cells.append(format_cell(mean, std, bold=(method == best_method)))
        lines.append(" ".join(row_cells) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append(f"\\end{{document}}")

    out_filename = out_tex.replace('.tex', f'_bs{batch_size}.tex')
    with open(os.path.join(OUT_DIR, out_filename), 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    print(f"Wrote {out_filename}")


def main():
    batch_size = 512  # Default batch size
    # Two tables: best accuracy (higher better) and best loss (lower better)
    build_table('best_acc', 'Best Accuracy', True, 'methods_best_acc.tex', batch_size)
    build_table('best_loss', 'Best Loss', False, 'methods_best_loss.tex', batch_size)
    # Optionally, also final
    build_table('final_acc', 'Final Accuracy', True, 'methods_final_acc.tex', batch_size)
    build_table('final_loss', 'Final Loss', False, 'methods_final_loss.tex', batch_size)


if __name__ == '__main__':
    main()
