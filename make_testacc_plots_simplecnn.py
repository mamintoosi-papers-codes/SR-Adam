import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_ROOT = os.path.join(os.path.dirname(__file__), "results")
FIG_DIR = os.path.join(os.path.dirname(__file__), "paper_figures")
os.makedirs(FIG_DIR, exist_ok=True)

OPT_ORDER = ["SGD", "Momentum", "Adam", "SR-Adam"]


def load_agg(opt_path):
    agg_path = os.path.join(opt_path, "aggregated_epoch_stats.csv")
    if not os.path.isfile(agg_path):
        return None
    return pd.read_csv(agg_path)


def plot_acc(dataset, noise, model="simplecnn"):
    noise_dir = f"noise_{noise}"
    base = os.path.join(RESULTS_ROOT, dataset, model, noise_dir)
    if not os.path.isdir(base):
        return None

    fig, ax = plt.subplots(figsize=(6, 4))
    for opt in OPT_ORDER:
        opt_path = os.path.join(base, opt)
        df = load_agg(opt_path)
        if df is None:
            continue
        ax.plot(df["Epoch"], df["Test Acc Mean"], label=opt, linewidth=2)
        ax.fill_between(
            df["Epoch"],
            df["Test Acc Mean"] - df["Test Acc Std"],
            df["Test Acc Mean"] + df["Test Acc Std"],
            alpha=0.2,
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title(f"{dataset} / {model} (noise={noise})")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()

    out_path = os.path.join(FIG_DIR, f"{dataset.lower()}_noise{noise}_acc_mean_std.pdf")
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def main():
    outputs = []
    for dataset in ["CIFAR10", "CIFAR100"]:
        for noise in ["0.0", "0.05", "0.1"]:
            out = plot_acc(dataset, noise)
            if out:
                outputs.append(out)
    if outputs:
        print("Saved acc figures:")
        for o in outputs:
            print(o)
    else:
        print("No figures generated (missing aggregated stats?)")


if __name__ == "__main__":
    main()
