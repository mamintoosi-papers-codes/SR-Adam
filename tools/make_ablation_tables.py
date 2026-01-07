import os
import re
import glob
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

METHODS = ["Adam", "SR-Adam"]
DEFAULT_NOISE = "0.05"
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
    raise SystemExit("results folder not found next to tools/ or under tools/")


RESULTS_ROOT = _pick_results_root()


def _available_batch_sizes(dataset, model, noise, optimizer):
    base = os.path.join(RESULTS_ROOT, dataset, model, f"noise_{noise}", optimizer)
    if not os.path.isdir(base):
        return []
    bs_dirs = [d for d in os.listdir(base) if d.startswith("batch_size_") and os.path.isdir(os.path.join(base, d))]
    sizes = []
    for d in bs_dirs:
        try:
            sizes.append(int(d.replace("batch_size_", "")))
        except ValueError:
            continue
    return sorted(sizes)


def _collect_runs(dataset, model, noise, optimizer, batch_size):
    folder = os.path.join(RESULTS_ROOT, dataset, model, f"noise_{noise}", optimizer, f"batch_size_{batch_size}")
    files = sorted(glob.glob(os.path.join(folder, "run_*_metrics.csv")))
    return [pd.read_csv(f) for f in files]


def _compute_stats(runs):
    finals_acc = [float(r["Test Acc"].iloc[-1]) for r in runs]
    bests_acc = [float(r["Test Acc"].max()) for r in runs]
    return {
        "final_mean": float(np.mean(finals_acc)) if finals_acc else float("nan"),
        "final_std": float(np.std(finals_acc, ddof=1)) if finals_acc else float("nan"),
        "best_mean": float(np.mean(bests_acc)) if bests_acc else float("nan"),
        "best_std": float(np.std(bests_acc, ddof=1)) if bests_acc else float("nan"),
        "num_runs": len(runs),
    }


def _format_cell(mean, std, bold=False):
    if np.isnan(mean) or np.isnan(std):
        return "--"
    cell = f"{mean:.2f} $\\pm$ {std:.2f}"
    return f"\\textbf{{{cell}}}" if bold else cell


def build_ablation_tables(dataset, model, noise, optimizers=None, batch_sizes=None):
    optimizers = optimizers or METHODS
    if batch_sizes is None:
        merged = set()
        for opt in optimizers:
            merged.update(_available_batch_sizes(dataset, model, noise, opt))
        batch_sizes = sorted(merged)

    values_final = {opt: {} for opt in optimizers}
    values_best = {opt: {} for opt in optimizers}

    for opt in optimizers:
        for bs in batch_sizes:
            runs = _collect_runs(dataset, model, noise, opt, bs)
            if not runs:
                values_final[opt][bs] = (float("nan"), float("nan"))
                values_best[opt][bs] = (float("nan"), float("nan"))
                continue
            stats = _compute_stats(runs)
            values_final[opt][bs] = (stats["final_mean"], stats["final_std"])
            values_best[opt][bs] = (stats["best_mean"], stats["best_std"])

    def _best_flags(values, higher_is_better=True):
        flags = {opt: {} for opt in optimizers}
        for bs in batch_sizes:
            col = {opt: values[opt][bs][0] for opt in optimizers}
            if higher_is_better:
                best_opt = max(col, key=lambda k: col[k]) if col else None
            else:
                best_opt = min(col, key=lambda k: col[k]) if col else None
            for opt in optimizers:
                flags[opt][bs] = (opt == best_opt)
        return flags

    flags_final = _best_flags(values_final, True)
    flags_best = _best_flags(values_best, True)

    def _table_tex(metric_title, values, flags, label_suffix):
        lines = []
        lines.append("% " + metric_title)
        lines.append("\\begin{table}[t]")
        lines.append("  \\centering")
        fmt = "l" + "c" * len(batch_sizes)
        lines.append(f"  \\begin{{tabular}}{{{fmt}}}")
        lines.append("    \\toprule")
        header = "    Method " + " ".join([f"& BS={bs}" for bs in batch_sizes]) + " \\\\" 
        lines.append(header)
        lines.append("    \\midrule")
        for opt in optimizers:
            row = [opt]
            for bs in batch_sizes:
                mean, std = values[opt][bs]
                row.append(_format_cell(mean, std, bold=flags[opt][bs]))
            lines.append("    " + " & ".join(row) + " " + "\\\\")
        lines.append("    \\bottomrule")
        lines.append("  \\end{tabular}")
        lines.append(f"  \\caption{{{metric_title} on {dataset} / {model} (noise={noise}).}}")
        lines.append(f"  \\label{{tab:ablation-bs-{dataset.lower()}-{label_suffix}}}")
        lines.append("\\end{table}")
        lines.append("")
        return "\n".join(lines)

    doc = []
    doc.append("% Auto-generated")
    doc.append("\\usepackage{booktabs}")
    doc.append("")
    doc.append(_table_tex("Final test accuracy (mean $\\pm$ std) vs batch size", values_final, flags_final, "final"))
    doc.append(_table_tex("Best test accuracy over epochs (mean $\\pm$ std) vs batch size", values_best, flags_best, "best"))
    stats = {
        "batch_sizes": batch_sizes,
        "values_final": values_final,
        "values_best": values_best,
        "flags_final": flags_final,
        "flags_best": flags_best,
        "optimizers": optimizers,
    }
    return "\n".join(doc), stats


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="CIFAR10")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--noise", default=DEFAULT_NOISE)
    p.add_argument("--optimizers", default="Adam|SR-Adam")
    p.add_argument("--batch_sizes", default="")
    args = p.parse_args()

    optimizers = args.optimizers.split("|") if args.optimizers else METHODS
    batch_sizes = [int(x) for x in args.batch_sizes.split(",") if x.strip().isdigit()] if args.batch_sizes else None

    tex, stats = build_ablation_tables(args.dataset, args.model, args.noise, optimizers, batch_sizes)

    out_path = os.path.join(OUT_DIR, f"ablation_bs_{args.dataset}_noise{args.noise.replace('.', 'p')}.tex")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(tex)
    print(f"Wrote ablation table(s) to {out_path}")

    # Also produce a plot for best accuracies vs batch size
    try:
        def _make_ablation_plot(stats, dataset, noise, out_dir=OUT_DIR):
            colors = {
                "SGD": "#1f77b4",
                "Momentum": "#ff7f0e",
                "Adam": "#2ca02c",
                "SR-Adam": "#d62728",
            }
            batch_sizes = stats["batch_sizes"]
            optimizers = stats["optimizers"]
            values_best = stats["values_best"]

            plt.figure(figsize=(7, 4))
            for opt in optimizers:
                means = [values_best[opt].get(bs, (np.nan, np.nan))[0] for bs in batch_sizes]
                stds = [values_best[opt].get(bs, (np.nan, np.nan))[1] for bs in batch_sizes]
                col = colors.get(opt, None)
                plt.errorbar(batch_sizes, means, yerr=stds, marker="o", label=opt, color=col, capsize=4)

            plt.xlabel("Batch size")
            plt.ylabel("Best test accuracy (%)")
            plt.title(f"Best test accuracy vs batch size — {dataset} (noise={noise})")
            plt.xticks(batch_sizes)
            plt.grid(alpha=0.25)
            plt.legend()

            filename = f"{dataset.lower()}_noise{noise}_acc_mean_std.pdf"
            out_file = os.path.join(out_dir, filename)
            plt.tight_layout()
            plt.savefig(out_file)
            # Also save PNG for quick previews
            png_file = os.path.join(out_dir, f"{dataset.lower()}_noise{noise}_acc_mean_std.png")
            plt.savefig(png_file)
            plt.close()
            print(f"Wrote ablation plot(s) to {out_file} and {png_file}")

        _make_ablation_plot(stats, args.dataset, args.noise)
    except Exception as e:
        print(f"Failed to produce ablation plot: {e}")


if __name__ == "__main__":
    main()
