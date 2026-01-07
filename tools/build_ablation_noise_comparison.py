import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

METHODS = ["Adam", "SR-Adam", "SR-Adam-All-Weights"]
NOISE_LEVELS = ["0.0", "0.05", "0.1"]
BATCH_SIZE = 512
DEFAULT_MODEL = "simplecnn"

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "paper")
os.makedirs(OUT_DIR, exist_ok=True)


def _pick_results_root():
    candidates = [
        os.path.join(os.path.dirname(__file__), "..", "results"),
        os.path.join(os.path.dirname(__file__), "results"),
    ]
    for p in candidates:
        if os.path.isdir(p):
            return os.path.abspath(p)
    raise SystemExit("results folder not found")


RESULTS_ROOT = _pick_results_root()


def _collect_runs(dataset, model, noise, optimizer):
    folder = os.path.join(
        RESULTS_ROOT,
        dataset,
        model,
        f"noise_{noise}",
        optimizer,
        f"batch_size_{BATCH_SIZE}",
    )
    files = sorted(glob.glob(os.path.join(folder, "run_*_metrics.csv")))
    return [pd.read_csv(f) for f in files]


def _compute_best_stats(runs):
    bests = [float(r["Test Acc"].max()) for r in runs]
    return np.mean(bests), np.std(bests, ddof=1)


def _format_cell(mean, std, bold=False):
    cell = f"{mean:.2f} $\\pm$ {std:.2f}"
    return f"\\textbf{{{cell}}}" if bold else cell


def build_table(dataset, model):
    values = {m: {} for m in METHODS}

    for noise in NOISE_LEVELS:
        for m in METHODS:
            runs = _collect_runs(dataset, model, noise, m)
            if not runs:
                values[m][noise] = (np.nan, np.nan)
            else:
                values[m][noise] = _compute_best_stats(runs)

    # determine winners per noise
    winners = {}
    for noise in NOISE_LEVELS:
        col = {m: values[m][noise][0] for m in METHODS}
        winners[noise] = max(col, key=col.get)

    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("  \\centering")
    lines.append("  \\begin{tabular}{lccc}")
    lines.append("    \\toprule")
    lines.append("    Method & Noise=0.0 & Noise=0.05 & Noise=0.1 \\\\")
    lines.append("    \\midrule")

    for m in METHODS:
        row = [m]
        for noise in NOISE_LEVELS:
            mean, std = values[m][noise]
            row.append(_format_cell(mean, std, bold=(winners[noise] == m)))
        lines.append("    " + " & ".join(row) + " \\\\")

    lines.append("    \\bottomrule")
    lines.append("  \\end{tabular}")
    lines.append(
        f"  \\caption{{Effect of noise level on best test accuracy "
        f"(mean $\\pm$ std) at batch size {BATCH_SIZE} on {dataset}.}}"
    )
    lines.append(f"  \\label{{tab:ablation-noise-{dataset.lower()}}}")
    lines.append("\\end{table}")

    out_path = os.path.join(OUT_DIR, f"ablation_noise_{dataset.lower()}.tex")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Wrote table to {out_path}")
    return values


def build_plot(dataset, values):
    plt.figure(figsize=(6.5, 4))
    x = np.arange(len(NOISE_LEVELS))

    colors = {
        "SGD": "#1f77b4",
        "Momentum": "#ff7f0e",
        "Adam": "#2ca02c",
        "SR-Adam": "#d62728",
        "SR-Adam-All-Weights": "#9467bd",
    }

    for m in METHODS:
        means = [values[m][n][0] for n in NOISE_LEVELS]
        stds = [values[m][n][1] for n in NOISE_LEVELS]
        col = colors.get(m, None)
        plt.errorbar(
            x,
            means,
            yerr=stds,
            marker="o",
            capsize=4,
            label=m,
            color=col,
        )


    plt.xticks(x, NOISE_LEVELS)
    plt.xlabel("Noise level")
    plt.ylabel("Best test accuracy (%)")
    # plt.title(f"{dataset}: Noise robustness (BS={BATCH_SIZE})")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out_pdf = os.path.join(OUT_DIR, f"{dataset.lower()}_noise_comparison_bs512.pdf")
    plt.savefig(out_pdf)
    plt.close()
    print(f"Wrote plot to {out_pdf}")


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="CIFAR10", choices=["CIFAR10", "CIFAR100"])
    p.add_argument("--model", default=DEFAULT_MODEL)
    args = p.parse_args()

    values = build_table(args.dataset, args.model)
    build_plot(args.dataset, values)


if __name__ == "__main__":
    main()
