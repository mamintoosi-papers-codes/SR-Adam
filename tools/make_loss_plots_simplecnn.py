import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

RESULTS_ROOT = os.path.join(os.path.dirname(__file__), "results")
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "paper")
os.makedirs(FIG_DIR, exist_ok=True)

OPT_ORDER = ["SGD", "Momentum", "Adam", "SR-Adam"]
COLORS = None  # use matplotlib default cycle; order controlled by OPT_ORDER


def load_runs(opt_path):
    run_files = sorted(glob.glob(os.path.join(opt_path, "run_*.csv")))
    if not run_files:
        return None
    runs = [pd.read_csv(f) for f in run_files]
    max_len = max(len(r) for r in runs)
    def pad_col(col):
        mat = np.zeros((len(runs), max_len))
        for i, r in enumerate(runs):
            arr = r[col].values
            mat[i, : len(arr)] = arr
        return mat
    train_loss = pad_col("Train Loss")
    test_loss = pad_col("Test Loss")
    epochs = np.arange(1, max_len + 1)
    return epochs, train_loss, test_loss


def plot_loss(dataset, noise, model="simplecnn"):
    noise_dir = f"noise_{noise}"
    base = os.path.join(RESULTS_ROOT, dataset, model, noise_dir)
    if not os.path.isdir(base):
        return None

    fig, ax = plt.subplots(figsize=(6, 4))
    for idx, opt in enumerate(OPT_ORDER):
        opt_path = os.path.join(base, opt)
        data = load_runs(opt_path)
        if data is None:
            continue
        epochs, train_loss, test_loss = data
        mean = test_loss.mean(axis=0)
        std = test_loss.std(axis=0)
        ax.plot(epochs, mean, label=opt, linewidth=2)
        ax.fill_between(epochs, mean - std, mean + std, alpha=0.2)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test Loss")
    ax.set_title(f"{dataset} / {model} (noise={noise})")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()

    out_path = os.path.join(FIG_DIR, f"{dataset.lower()}_noise{noise}_loss_mean_std.pdf")
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def main():
    outputs = []
    for dataset in ["CIFAR10", "CIFAR100"]:
        for noise in ["0.0", "0.05", "0.1"]:
            out = plot_loss(dataset, noise)
            if out:
                outputs.append(out)
    if outputs:
        print("Saved loss figures:")
        for o in outputs:
            print(o)
    else:
        print("No figures generated (missing runs?)")


if __name__ == "__main__":
    main()
