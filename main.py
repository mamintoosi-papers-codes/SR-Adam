"""
SR-Adam: Stein-Rule Adaptive Moment Estimation
Main entry point for multi-run experiments on CIFAR datasets.

Key features:
- Stein-rule applied ONLY to convolutional layers (Conv-only SR)
- Multiple independent runs with different random seeds
- Mean / Std statistics computed and saved
- Designed for fair optimizer comparison
"""

import torch
import torch.nn as nn
import numpy as np
import argparse
import os

from optimizers import (
    SGDManual,
    MomentumManual,
    AdamBaseline,
    SRAdamAdaptiveLocal
)

from model import SimpleCNN, get_model
from data import get_data_loaders
from training import train_model
from utils import (
    create_results_directory,
    save_all_results,
    plot_results,
    print_summary,
    count_parameters,
    save_multirun_summary,
    plot_mean_std
)


# ----------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------

def setup_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    return device


def set_seeds(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_optimizer_list(raw, all_names, alias_map):
    if raw.lower() == "all":
        return all_names

    for sep in [";", "|", "\n"]:
        if sep in raw:
            tokens = [t.strip() for t in raw.split(sep) if t.strip()]
            break
    else:
        tokens = [t.strip() for t in raw.split(",") if t.strip()]

    resolved = []
    for tok in tokens:
        key = alias_map.get(tok.lower(), tok)
        if key not in all_names:
            raise ValueError(f"Unknown optimizer requested: {tok}")
        resolved.append(key)
    return resolved


# ----------------------------------------------------------------------
# Optimizer factory
# ----------------------------------------------------------------------

def create_optimizer(name, model):
    """
    Create optimizer for given model.
    For SR-Adam, detect Conv2d modules programmatically and apply Stein-rule only to their params.
    """
    if name == "SGD":
        return SGDManual(model.parameters(), lr=0.01)

    if name == "Momentum":
        return MomentumManual(model.parameters(), lr=0.01, momentum=0.9)

    if name == "Adam":
        return AdamBaseline(model.parameters(), lr=1e-3)

    if name == "SR-Adam (Conv-only, Adaptive)":
        # Collect parameters belonging to nn.Conv2d modules (robust for ResNet downsample, etc.)
        conv_params_set = set()
        for module in model.modules():
            # Use explicit type check to avoid relying on parameter names
            if isinstance(module, nn.Conv2d):
                for p in module.parameters(recurse=False):
                    conv_params_set.add(p)

        conv_params = []
        other_params = []
        for p in model.parameters():
            if p in conv_params_set:
                conv_params.append(p)
            else:
                other_params.append(p)

        # Create parameter groups: Conv with Stein, others without
        param_groups = []
        if conv_params:
            param_groups.append({"params": conv_params, "stein": True})
        if other_params:
            param_groups.append({"params": other_params, "stein": False})

        return SRAdamAdaptiveLocal(
            param_groups,
            lr=1e-3,
            warmup_steps=5,
            shrink_clip=(0.1, 1.0),
        )

    raise ValueError(f"Unknown optimizer: {name}")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Multi-run SR-Adam experiments on CIFAR datasets"
    )

    parser.add_argument("--dataset", type=str, default="CIFAR10",
                        choices=["CIFAR10", "CIFAR100"])
    parser.add_argument("--model", type=str, default="simplecnn",
                        choices=["simplecnn", "resnet18"],
                        help="Model architecture to use")
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--noise", type=float, default=0.0)

    parser.add_argument("--num_runs", type=int, default=5,
                        help="Number of independent runs (different seeds)")
    parser.add_argument("--base_seed", type=int, default=42)

    parser.add_argument(
        "--optimizers",
        type=str,
        default="all",
        help='Optimizer list or "all"; separate with "|" or ";"'
    )

    args = parser.parse_args()

    device = setup_device()

    results_dir = create_results_directory(args.dataset, args.noise)

    print(f"\nLoading dataset: {args.dataset}")
    train_loader, test_loader, num_classes = get_data_loaders(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        noise_std=args.noise
    )

    all_optimizer_names = [
        "SGD",
        "Momentum",
        "Adam",
        "SR-Adam (Conv-only, Adaptive)",
    ]

    alias_map = {
        "sgd": "SGD",
        "momentum": "Momentum",
        "adam": "Adam",
        "sradam": "SR-Adam (Conv-only, Adaptive)",
    }

    optimizer_names = parse_optimizer_list(
        args.optimizers,
        all_optimizer_names,
        alias_map
    )

    # ------------------------------------------------------------------
    # Multi-run experiments
    # ------------------------------------------------------------------

    all_results = {}      # optimizer -> list of run metrics
    summary_stats = {}    # optimizer -> mean/std statistics

    for opt_name in optimizer_names:
        print("\n" + "=" * 80)
        print(f"Optimizer: {opt_name}")
        print("=" * 80)

        all_results[opt_name] = []

        for run in range(args.num_runs):
            seed = args.base_seed + run
            print(f"\nRun {run + 1}/{args.num_runs} | seed = {seed}")
            set_seeds(seed)

            model = get_model(args.model, num_classes=num_classes).to(device)
            print(f"Model: {args.model} | Trainable parameters: {count_parameters(model):,}")

            optimizer = create_optimizer(opt_name, model)
            criterion = nn.CrossEntropyLoss()

            metrics = train_model(
                model=model,
                train_loader=train_loader,
                test_loader=test_loader,
                optimizer=optimizer,
                criterion=criterion,
                num_epochs=args.num_epochs,
                device=device,
            )

            metrics["seed"] = seed
            all_results[opt_name].append(metrics)

        # ---- compute statistics for this optimizer ----
        final_accs = [m["test_acc"][-1] for m in all_results[opt_name]]
        best_accs = [max(m["test_acc"]) for m in all_results[opt_name]]

        summary_stats[opt_name] = {
            "final_mean": float(np.mean(final_accs)),
            "final_std": float(np.std(final_accs)),
            "best_mean": float(np.mean(best_accs)),
            "best_std": float(np.std(best_accs)),
            "num_runs": args.num_runs,
        }

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------

    os.makedirs(results_dir, exist_ok=True)

    # Save per-run results (reuse existing utility for first run curves)
    representative_results = [all_results[name][0] for name in optimizer_names]

    save_all_results(
        results=representative_results,
        optimizers_names=optimizer_names,
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        noise_std=args.noise,
        optimizer_params=summary_stats,
    )

    # Save summary statistics explicitly
    save_multirun_summary(summary_stats, results_dir)


    # Plot representative curves
    plot_results(
        representative_results,
        optimizer_names,
        save_path=f"{results_dir}/optimizer_comparison.png",
    )

    # Print concise summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY (mean ± std over runs)")
    print("=" * 80)
    for name, stats in summary_stats.items():
        print(
            f"{name}: "
            f"{stats['final_mean']:.2f} ± {stats['final_std']:.2f} "
            f"(best: {stats['best_mean']:.2f} ± {stats['best_std']:.2f})"
        )

    # plot_mean_std(results_per_optimizer, optimizer_names, save_path)

if __name__ == "__main__":
    main()
