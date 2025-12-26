import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_ROOT = os.path.join(os.path.dirname(__file__), "results")
FIG_DIR = os.path.join(os.path.dirname(__file__), "paper_figures")
os.makedirs(FIG_DIR, exist_ok=True)

plt.style.use('seaborn-v0_8')


def load_epoch_stats(dataset, model, noise, optimizer):
    path = os.path.join(RESULTS_ROOT, dataset, model, f"noise_{noise}", optimizer, "aggregated_epoch_stats.csv")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"missing aggregated_epoch_stats: {path}")
    return pd.read_csv(path)


def figure_epoch_curve():
    dataset = "CIFAR10"
    model = "simplecnn"
    noise = "0.1"
    optimizers = ["Adam", "SR-Adam"]

    fig, ax = plt.subplots(figsize=(5, 3.2))
    for opt in optimizers:
        df = load_epoch_stats(dataset, model, noise, opt)
        ax.plot(df["Epoch"], df["Test Acc Mean"], label=opt, linewidth=2)
        ax.fill_between(df["Epoch"], df["Test Acc Mean"] - df["Test Acc Std"], df["Test Acc Mean"] + df["Test Acc Std"], alpha=0.2)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title(f"{dataset} (noise={noise})")
    ax.legend()
    fig.tight_layout()
    out_path = os.path.join(FIG_DIR, "cifar10_noise0p1_epoch.pdf")
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
        out_path = os.path.join(FIG_DIR, f"{dataset.lower()}_noise_sweep.pdf")
        fig.savefig(out_path)
        plt.close(fig)
        outputs.append(out_path)

    return outputs


def main():
    epoch_fig = figure_epoch_curve()
    noise_figs = figure_noise_sweep()
    print("Saved figures:")
    print(epoch_fig)
    for f in noise_figs:
        print(f)


if __name__ == "__main__":
    main()
