"""
SR-Adam: Stein-Rule Adaptive Moment Estimation
Main entry point for training models with various optimizers.

Stein-rule is applied ONLY to convolutional layers via param_groups.
Fully-connected layers use pure Adam updates.
"""

import torch
import torch.nn as nn
import numpy as np
import argparse

from optimizers import (
    SGDManual,
    MomentumManual,
    AdamBaseline,
    SRAdamAdaptiveLocal
)

from model import SimpleCNN
from data import get_data_loaders
from training import train_model
from utils import (
    create_results_directory,
    save_all_results,
    plot_results,
    print_summary,
    count_parameters
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


def set_seeds(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"Random seeds set to {seed}")


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
    if name == "SGD":
        return SGDManual(model.parameters(), lr=0.01)

    if name == "Momentum":
        return MomentumManual(model.parameters(), lr=0.01, momentum=0.9)

    if name == "Adam":
        return AdamBaseline(model.parameters(), lr=1e-3)

    if name == "SR-Adam (Conv-only, Adaptive)":
        return SRAdamAdaptiveLocal(
            [
                {"params": model.conv1.parameters(), "stein": True},
                {"params": model.conv2.parameters(), "stein": True},
                {"params": model.fc1.parameters(), "stein": False},
                {"params": model.fc2.parameters(), "stein": False},
            ],
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
        description="SR-Adam experiments on CIFAR datasets"
    )

    parser.add_argument("--dataset", type=str, default="CIFAR10",
                        choices=["CIFAR10", "CIFAR100"])
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--optimizers",
        type=str,
        default="all",
        help='Optimizer list or "all"; separate with "|" or ";"'
    )

    args = parser.parse_args()

    device = setup_device()
    set_seeds(args.seed)

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
        "sradam": "SR-Adam (Conv-only, Adaptive)"
    }

    optimizer_names = parse_optimizer_list(
        args.optimizers,
        all_optimizer_names,
        alias_map
    )

    results = []

    for opt_name in optimizer_names:
        print("\n" + "=" * 80)
        print(f"Training with optimizer: {opt_name}")
        print("=" * 80)

        model = SimpleCNN(num_classes=num_classes).to(device)
        print(f"Trainable parameters: {count_parameters(model):,}")

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

        results.append(metrics)

    save_all_results(
        results=results,
        optimizers_names=optimizer_names,
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        noise_std=args.noise,
        optimizer_params={name: {} for name in optimizer_names},
    )

    plot_results(
        results,
        optimizer_names,
        save_path=f"{results_dir}/optimizer_comparison.png",
    )

    print_summary(results, optimizer_names)


if __name__ == "__main__":
    main()
