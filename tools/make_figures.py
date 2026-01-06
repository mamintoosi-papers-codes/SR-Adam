import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse

RESULTS_ROOT = os.path.join(os.path.dirname(__file__), "results")
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "paper")
os.makedirs(FIG_DIR, exist_ok=True)

plt.style.use('seaborn-v0_8')

OPT_ORDER = ["SGD", "Momentum", "Adam", "SR-Adam"]


def load_epoch_stats(dataset, model, noise, optimizer, batch_size=512):
    path = os.path.join(RESULTS_ROOT, dataset, model, f"noise_{noise}", optimizer, f"batch_size_{batch_size}", "aggregated_epoch_stats.csv")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"missing aggregated_epoch_stats: {path}")
    return pd.read_csv(path)


def figure_epoch_curve(batch_size=512):
    dataset = "CIFAR10"
    model = "simplecnn"
    noise = "0.1"
    optimizers = ["Adam", "SR-Adam"]

    fig, ax = plt.subplots(figsize=(5, 3.2))
    for opt in optimizers:
        df = load_epoch_stats(dataset, model, noise, opt, batch_size)
        ax.plot(df["Epoch"], df["Test Acc Mean"], label=opt, linewidth=2)
        ax.fill_between(df["Epoch"], df["Test Acc Mean"] - df["Test Acc Std"], df["Test Acc Mean"] + df["Test Acc Std"], alpha=0.2)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title(f"{dataset} (noise={noise})")
    ax.legend()
    fig.tight_layout()
    out_path = os.path.join(FIG_DIR, f"cifar10_noise0p1_epoch_bs{batch_size}.pdf")
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def load_summary():
    df = pd.read_csv(os.path.join(RESULTS_ROOT, "summary_statistics.csv"), index_col=0)
    df = df.reset_index().rename(columns={"index": "key"})
    parts = df["key"].str.split("|", expand=True)
    df["dataset"] = parts[0]
    df["noise"] = parts[1].str.replace("noise_", "", regex=False)
    df["optimizer"] = parts[2]
    return df


def figure_noise_sweep():
    df = load_summary()
    datasets = [
        ("CIFAR10", "simplecnn"),
        ("CIFAR100", "resnet18"),
    ]
    optimizers = ["Adam", "SR-Adam"]
    outputs = []

    for dataset, _model in datasets:
        fig, ax = plt.subplots(figsize=(5, 3.2))
        for opt in optimizers:
            sub = df[(df["dataset"] == dataset) & (df["optimizer"] == opt)]
            sub = sub[sub["noise"].isin(["0.0", "0.05", "0.1"])]
            sub = sub.sort_values("noise", key=lambda s: s.astype(float))
            ax.errorbar(sub["noise"].astype(float), sub["final_mean"], yerr=sub["final_std"], label=opt, linewidth=2, capsize=4, marker="o")
        ax.set_xlabel("Noise Std")
        ax.set_ylabel("Final Test Accuracy (%)")
        ax.set_title(dataset)
        ax.legend()
        fig.tight_layout()
        out_path = os.path.join(FIG_DIR, f"{dataset.lower()}_noise_sweep_bs512.pdf")
        fig.savefig(out_path)
        plt.close(fig)
        outputs.append(out_path)

    return outputs


def load_runs_for_loss(opt_path):
    """Load all run CSV files and pad to same length for loss plotting."""
    run_files = sorted(glob.glob(os.path.join(opt_path, "run_*_metrics.csv")))
    if not run_files:
        return None
    runs = [pd.read_csv(f) for f in run_files]
    max_len = max(len(r) for r in runs)
    
    def pad_col(col):
        mat = np.zeros((len(runs), max_len))
        for i, r in enumerate(runs):
            arr = r[col].values
            mat[i, :len(arr)] = arr
        return mat
    
    train_loss = pad_col("Train Loss")
    test_loss = pad_col("Test Loss")
    epochs = np.arange(1, max_len + 1)
    return epochs, train_loss, test_loss


def figure_loss_curves(dataset, model="simplecnn", batch_size=512):
    """Generate loss curves for all noise levels and optimizers."""
    outputs = []
    for noise in ["0.0", "0.05", "0.1"]:
        noise_dir = f"noise_{noise}"
        base = os.path.join(RESULTS_ROOT, dataset, model, noise_dir)
        if not os.path.isdir(base):
            continue

        fig, ax = plt.subplots(figsize=(6, 4))
        for opt in OPT_ORDER:
            opt_path = os.path.join(base, opt, f"batch_size_{batch_size}")
            data = load_runs_for_loss(opt_path)
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

        out_path = os.path.join(FIG_DIR, f"{dataset.lower()}_noise{noise}_loss_bs{batch_size}.pdf")
        fig.savefig(out_path)
        plt.close(fig)
        outputs.append(out_path)
    
    return outputs


def figure_acc_curves(dataset, model="simplecnn", batch_size=512):
    """Generate test accuracy curves for all noise levels and optimizers."""
    outputs = []
    for noise in ["0.0", "0.05", "0.1"]:
        noise_dir = f"noise_{noise}"
        base = os.path.join(RESULTS_ROOT, dataset, model, noise_dir)
        if not os.path.isdir(base):
            continue

        fig, ax = plt.subplots(figsize=(6, 4))
        for opt in OPT_ORDER:
            opt_path = os.path.join(base, opt, f"batch_size_{batch_size}")
            agg_path = os.path.join(opt_path, "aggregated_epoch_stats.csv")
            if not os.path.isfile(agg_path):
                continue
            df = pd.read_csv(agg_path)
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

        out_path = os.path.join(FIG_DIR, f"{dataset.lower()}_noise{noise}_acc_bs{batch_size}.pdf")
        fig.savefig(out_path)
        plt.close(fig)
        outputs.append(out_path)
    
    return outputs


def main():
    parser = argparse.ArgumentParser(description="Generate figures for CIFAR results")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size for loading results")
    args = parser.parse_args()

    batch_size = args.batch_size
    
    # Original figures
    epoch_fig = figure_epoch_curve(batch_size)
    noise_figs = figure_noise_sweep()
    
    # New comprehensive plots for all noise levels
    print("Generating figures...")
    print(f"  Using batch_size={batch_size}")
    
    loss_figs_cifar10 = figure_loss_curves("CIFAR10", "simplecnn", batch_size)
    loss_figs_cifar100 = figure_loss_curves("CIFAR100", "simplecnn", batch_size)
    
    acc_figs_cifar10 = figure_acc_curves("CIFAR10", "simplecnn", batch_size)
    acc_figs_cifar100 = figure_acc_curves("CIFAR100", "simplecnn", batch_size)
    
    print("\nSaved figures:")
    print(f"  Epoch curve: {epoch_fig}")
    for f in noise_figs:
        print(f"  Noise sweep: {f}")
    for f in loss_figs_cifar10 + loss_figs_cifar100:
        print(f"  Loss curves: {f}")
    for f in acc_figs_cifar10 + acc_figs_cifar100:
        print(f"  Acc curves: {f}")


if __name__ == "__main__":
    main()
