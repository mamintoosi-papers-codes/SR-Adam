import os
import glob
import pandas as pd
import numpy as np
import argparse
import json
from pathlib import Path

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
    finals_acc = [float(r['Test Acc'].iloc[-1]) for r in runs]
    bests_acc = [float(r['Test Acc'].max()) for r in runs]
    finals_loss = [float(r['Test Loss'].iloc[-1]) for r in runs]
    bests_loss = [float(r['Test Loss'].min()) for r in runs]
    return {
        'final_acc_mean': float(np.mean(finals_acc)),
        'final_acc_std': float(np.std(finals_acc, ddof=1)),
        'best_acc_mean': float(np.mean(bests_acc)),
        'best_acc_std': float(np.std(bests_acc, ddof=1)),
        'final_loss_mean': float(np.mean(finals_loss)),
        'final_loss_std': float(np.std(finals_loss, ddof=1)),
        'best_loss_mean': float(np.mean(bests_loss)),
        'best_loss_std': float(np.std(bests_loss, ddof=1)),
    }


def format_cell(mean, std, bold=False):
    if np.isnan(mean) or np.isnan(std):
        return "--"
    cell = f"{mean:.2f} $\\pm$ {std:.2f}"
    return f"\\textbf{{{cell}}}" if bold else cell


def gather_values(metric_key, higher_is_better, batch_size=512):
    # Build a dict: method -> {(dataset, noise): (mean,std)} and best markers
    values = {m: {} for m in METHOD_ORDER}
    for method in METHOD_ORDER:
        for dataset in DATASETS:
            for noise in NOISES:
                runs = collect_runs(dataset, noise, method, batch_size)
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


def parse_list_arg(raw, value_type=str):
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
    parts = []
    body_parts = []

    batch_size = 512  # Default batch size

    # Best Accuracy (higher better)
    vals, flags = gather_values('best_acc', True, batch_size)
    body_parts.append(build_table_tex('Best test accuracy (mean $\\pm$ std) over epochs; higher is better.', vals, flags, label='tab:method_best_acc'))

    # Final Accuracy (higher better)
    vals, flags = gather_values('final_acc', True, batch_size)
    body_parts.append(build_table_tex('Final test accuracy (mean $\\pm$ std) at last epoch; higher is better.', vals, flags, label='tab:method_final_acc'))

    # Best Loss (lower better)
    vals, flags = gather_values('best_loss', False, batch_size)
    body_parts.append(build_table_tex('Best test loss (mean $\\pm$ std) over epochs; lower is better.', vals, flags, label='tab:method_best_loss'))

    # Final Loss (lower better)
    vals, flags = gather_values('final_loss', False, batch_size)
    body_parts.append(build_table_tex('Final test loss (mean $\\pm$ std) at last epoch; lower is better.', vals, flags, label='tab:method_final_loss'))
    # Write full standalone doc
    parts.append("\\documentclass{article}")
    parts.append("\\usepackage{booktabs}")
    parts.append("\\usepackage[margin=1in]{geometry}")
    parts.append("\\begin{document}")
    parts.append("\n".join(body_parts))
    parts.append("\\end{document}")
    out_path = os.path.join(OUT_DIR, f'minimal-tables_bs{batch_size}.tex')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(parts))
    print(f"Wrote {out_path}")

    # Write body-only content for \input into the main paper
    body_out_path = os.path.join(OUT_DIR, f'minimal-tables-content_bs{batch_size}.tex')
    with open(body_out_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(body_parts))
    print(f"Wrote {body_out_path}")

    parser = argparse.ArgumentParser(
        description="Generate LaTeX tables from aggregated results with filters"
    )
    parser.add_argument("--dataset", type=str, default="ALL")
    parser.add_argument("--noise", type=str, default="ALL")
    parser.add_argument("--batch_size", type=str, default="512")
    parser.add_argument("--optimizers", type=str, default="ALL")
    parser.add_argument("--output", type=str, default="paper_figures/minimal-tables-content.tex")

    args = parser.parse_args()

    datasets = parse_list_arg(args.dataset) or ["CIFAR10", "CIFAR100"]
    noise_levels = parse_list_arg(args.noise, float) or [0.0, 0.05, 0.1]
    batch_sizes = parse_list_arg(args.batch_size, int) or [512]
    optimizers = parse_list_arg(args.optimizers)

    print(f"\n{'='*80}")
    print("Generating LaTeX Tables")
    print(f"{'='*80}")
    print(f"Datasets:    {datasets}")
    print(f"Noise:       {noise_levels}")
    print(f"Batch sizes: {batch_sizes}")
    print(f"Optimizers:  {optimizers if optimizers else 'ALL'}")
    print(f"Output:      {args.output}")
    print(f"{'='*80}\n")

    # Collect data
    table_data = {}

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

                    if optimizers and opt_name not in optimizers:
                        continue

                    summary_file = opt_dir / 'aggregated_summary.json'
                    if not summary_file.exists():
                        continue

                    with open(summary_file) as f:
                        summary = json.load(f)

                    key = (dataset, noise, batch_size, opt_name)
                    table_data[key] = summary

    # Generate LaTeX content
    latex_lines = []
    latex_lines.append("% Auto-generated by make_minimal_tables.py\n")
    latex_lines.append("% Filters applied:\n")
    latex_lines.append(f"% Datasets: {datasets}\n")
    latex_lines.append(f"% Noise: {noise_levels}\n")
    latex_lines.append(f"% Batch sizes: {batch_sizes}\n")
    latex_lines.append(f"% Optimizers: {optimizers if optimizers else 'ALL'}\n\n")

    # Generate table rows
    for dataset in datasets:
        latex_lines.append(f"% {dataset}\n")
        latex_lines.append("\\begin{tabular}{lcccc}\n")
        latex_lines.append("\\toprule\n")
        latex_lines.append("Optimizer & Noise=0.0 & Noise=0.05 & Noise=0.1 \\\\\n")
        latex_lines.append("\\midrule\n")

        opt_names = sorted(set(k[3] for k in table_data.keys() if k[0] == dataset))

        for opt_name in opt_names:
            row = [opt_name]
            for noise in noise_levels:
                key = (dataset, noise, batch_sizes[0], opt_name)
                if key in table_data:
                    data = table_data[key]
                    acc = data['best_test_acc_mean']
                    std = data['best_test_acc_std']
                    row.append(f"{acc:.2f} $\\pm$ {std:.2f}")
                else:
                    row.append("--")
            latex_lines.append(" & ".join(row) + " \\\\\n")

        latex_lines.append("\\bottomrule\n")
        latex_lines.append("\\end{tabular}\n\n")

    # Write output
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        f.writelines(latex_lines)

    print(f"✓ Saved: {args.output}\n")


if __name__ == '__main__':
    main()
