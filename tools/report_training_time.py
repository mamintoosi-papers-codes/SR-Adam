import argparse
import json
import os
from pathlib import Path
import pandas as pd
import numpy as np

def parse_list_arg(raw, value_type=str):
    if raw is None:
        return None
    raw = str(raw)
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
    return ["SGD", "Momentum", "Adam", "SR-Adam", "SR-Adam-All-Weights"]

def bs_suffix(batch_sizes):
    if not batch_sizes:
        return ""
    return "_bs" + "-".join(str(bs) for bs in sorted(batch_sizes))

def main():
    parser = argparse.ArgumentParser(description="Report mean training time per optimizer")
    parser.add_argument("--dataset", type=str, default="ALL")
    parser.add_argument("--noise", type=str, default="0.05")
    parser.add_argument("--batch_size", type=str, default="512")
    parser.add_argument("--model", type=str, default="simplecnn")
    parser.add_argument("--optimizers", type=str, default="ALL")
    parser.add_argument("--output_csv", type=str, default=None)
    parser.add_argument("--output_tex", type=str, default=None)
    args = parser.parse_args()

    datasets = parse_list_arg(args.dataset) or ["CIFAR10", "CIFAR100"]
    noise_levels = parse_list_arg(args.noise, float) or [0.05]
    batch_sizes = parse_list_arg(args.batch_size, int) or [512]
    optimizers = parse_list_arg(args.optimizers) or get_optimizer_order()

    rows = []
    for dataset in datasets:
        for noise in noise_levels:
            base = Path("results") / dataset / args.model / f"noise_{noise}"
            for opt_name in optimizers:
                opt_dir = base / opt_name
                if not opt_dir.exists():
                    continue
                for bs in batch_sizes:
                    bs_dir = opt_dir / f"batch_size_{bs}"
                    if not bs_dir.exists():
                        continue
                    run_files = sorted(bs_dir.glob("run_*_metrics.csv"))
                    if not run_files:
                        continue

                    per_run_means = []
                    per_run_totals = []
                    for rf in run_files:
                        try:
                            df = pd.read_csv(rf)
                        except Exception:
                            continue
                        if "Epoch Time" not in df.columns:
                            continue
                        per_run_means.append(float(df["Epoch Time"].mean()))
                        per_run_totals.append(float(df["Epoch Time"].sum()))

                    if not per_run_means:
                        continue

                    rows.append({
                        "dataset": dataset,
                        "noise": noise,
                        "batch_size": bs,
                        "optimizer": opt_name,
                        "num_runs": len(per_run_means),
                        "mean_epoch_time": float(np.mean(per_run_means)),
                        "std_epoch_time": float(np.std(per_run_means, ddof=1)) if len(per_run_means) > 1 else 0.0,
                        "mean_total_time": float(np.mean(per_run_totals)),
                        "std_total_time": float(np.std(per_run_totals, ddof=1)) if len(per_run_totals) > 1 else 0.0,
                    })

    if not rows:
        print("No training time data found.")
        return

    df = pd.DataFrame(rows).sort_values(["dataset", "noise", "batch_size", "optimizer"])
    print(df.to_string(index=False))

    # CSV
    if args.output_csv is None:
        args.output_csv = f"paper_figures/training_time_report{bs_suffix(batch_sizes)}.csv"
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print(f"Saved CSV: {args.output_csv}")

    # LaTeX (per dataset × batch_size for the given noise levels)
    if args.output_tex is None:
        args.output_tex = f"paper_figures/training_time_report{bs_suffix(batch_sizes)}.tex"
    lines = ["% Auto-generated training time report\n", "\\usepackage{booktabs}\n\n"]
    for dataset in datasets:
        for bs in batch_sizes:
            lines.append(f"% {dataset} (batch_size={bs})\n")
            lines.append("\\begin{table}[t]\n\\centering\n\\begin{tabular}{lcc}\n\\toprule\n")
            lines.append("Optimizer & Mean Epoch Time (s) & Mean Total Time (s) \\\\\n\\midrule\n")
            sub = df[(df.dataset == dataset) & (df.batch_size == bs)]
            # enforce optimizer order
            for opt in get_optimizer_order():
                row = sub[sub.optimizer == opt]
                if row.empty:
                    continue
                r = row.iloc[0]
                lines.append(f"{opt} & {r.mean_epoch_time:.2f} $\\pm$ {r.std_epoch_time:.2f} & {r.mean_total_time:.1f} $\\pm$ {r.std_total_time:.1f} \\\\\n")
            lines.append("\\bottomrule\n\\end{tabular}\n")
            lines.append(f"\\caption{{Mean training time per optimizer on {dataset} (batch\\_size={bs}).}}\n")
            lines.append(f"\\label{{tab:train_time_{dataset.lower()}_bs{bs}}}\n\\end{table}\n\n")
    Path(args.output_tex).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_tex, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"Saved LaTeX: {args.output_tex}")