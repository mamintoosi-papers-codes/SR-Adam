import os
import glob
import pandas as pd
import numpy as np

RESULTS_ROOT = os.path.join(os.path.dirname(__file__), 'results')
OUT_DIR = os.path.join(os.path.dirname(__file__), 'paper_figures')
os.makedirs(OUT_DIR, exist_ok=True)

METHOD_ORDER = ["SGD", "Momentum", "Adam", "SR-Adam"]
NOISES = ["0.0", "0.05", "0.1"]
DATASETS = ["CIFAR10", "CIFAR100"]
MODEL = "simplecnn"


def collect_runs(dataset, noise, optimizer):
    folder = os.path.join(RESULTS_ROOT, dataset, MODEL, f"noise_{noise}", optimizer)
    files = sorted(glob.glob(os.path.join(folder, 'run_*.csv')))
    return [pd.read_csv(f) for f in files]


def compute_stats(runs):
    finals_acc = [float(r['Test Acc'].iloc[-1]) for r in runs]
    bests_acc = [float(r['Test Acc'].max()) for r in runs]
    finals_loss = [float(r['Test Loss'].iloc[-1]) for r in runs]
    bests_loss = [float(r['Test Loss'].min()) for r in runs]
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
    if np.isnan(mean) or np.isnan(std):
        return "--"
    cell = f"{mean:.2f} $\\pm$ {std:.2f}"
    return f"\\textbf{{{cell}}}" if bold else cell


def gather_values(metric_key, higher_is_better):
    # Build a dict: method -> {(dataset, noise): (mean,std)} and best markers
    values = {m: {} for m in METHOD_ORDER}
    for method in METHOD_ORDER:
        for dataset in DATASETS:
            for noise in NOISES:
                runs = collect_runs(dataset, noise, method)
                if not runs:
                    values[method][(dataset, noise)] = (float('nan'), float('nan'))
                    continue
                stats = compute_stats(runs)
                mean = stats[f'{metric_key}_mean']
                std = stats[f'{metric_key}_std']
                values[method][(dataset, noise)] = (mean, std)
    # Determine best per column
    best_flags = {m: {} for m in METHOD_ORDER}
    for dataset in DATASETS:
        for noise in NOISES:
            col_vals = {m: values[m][(dataset, noise)][0] for m in METHOD_ORDER}
            if higher_is_better:
                best_method = max(col_vals, key=lambda k: col_vals[k])
            else:
                best_method = min(col_vals, key=lambda k: col_vals[k])
            for m in METHOD_ORDER:
                best_flags[m][(dataset, noise)] = (m == best_method)
    return values, best_flags


def build_table_tex(metric_title, values, best_flags, label=None):
    lines = []
    lines.append("% " + metric_title)
    lines.append("\\begin{table}[t]")
    lines.append("  \\centering")
    lines.append("  \\begin{tabular}{l ccc ccc}")
    lines.append("    \\toprule")
    lines.append("     & \\multicolumn{3}{c}{CIFAR10} & \\multicolumn{3}{c}{CIFAR100} \\\\")
    lines.append("     \\cmidrule(lr){2-4} \\cmidrule(lr){5-7}")
    lines.append("    Method & 0.0 & 0.05 & 0.1 & 0.0 & 0.05 & 0.1 \\\\")
    lines.append("    \\midrule")
    for method in METHOD_ORDER:
        row = [method]
        for dataset in DATASETS:
            for noise in NOISES:
                mean, std = values[method][(dataset, noise)]
                bold = best_flags[method][(dataset, noise)]
                row.append(format_cell(mean, std, bold=bold))
        lines.append("    " + " & ".join(row) + " \\\\")
    lines.append("    \\bottomrule")
    lines.append("  \\end{tabular}")
    lines.append(f"  \\caption{{{metric_title}}}")
    if label:
        lines.append(f"  \\label{{{label}}}")
    lines.append("\\end{table}")
    lines.append("")
    return "\n".join(lines)


def main():
    parts = []
    body_parts = []

    # Best Accuracy (higher better)
    vals, flags = gather_values('best_acc', True)
    body_parts.append(build_table_tex('Best test accuracy (mean $\\pm$ std) over epochs; higher is better.', vals, flags, label='tab:method_best_acc'))

    # Final Accuracy (higher better)
    vals, flags = gather_values('final_acc', True)
    body_parts.append(build_table_tex('Final test accuracy (mean $\\pm$ std) at last epoch; higher is better.', vals, flags, label='tab:method_final_acc'))

    # Best Loss (lower better)
    vals, flags = gather_values('best_loss', False)
    body_parts.append(build_table_tex('Best test loss (mean $\\pm$ std) over epochs; lower is better.', vals, flags, label='tab:method_best_loss'))

    # Final Loss (lower better)
    vals, flags = gather_values('final_loss', False)
    body_parts.append(build_table_tex('Final test loss (mean $\\pm$ std) at last epoch; lower is better.', vals, flags, label='tab:method_final_loss'))
    # Write full standalone doc
    parts.append("\\documentclass{article}")
    parts.append("\\usepackage{booktabs}")
    parts.append("\\usepackage[margin=1in]{geometry}")
    parts.append("\\begin{document}")
    parts.append("\n".join(body_parts))
    parts.append("\\end{document}")
    out_path = os.path.join(OUT_DIR, 'minimal-tables.tex')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(parts))
    print(f"Wrote {out_path}")

    # Write body-only content for \input into the main paper
    body_out_path = os.path.join(OUT_DIR, 'minimal-tables-content.tex')
    with open(body_out_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(body_parts))
    print(f"Wrote {body_out_path}")


if __name__ == '__main__':
    main()
