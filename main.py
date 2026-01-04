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
import shutil

from src.optimizers import (
    SGDManual,
    MomentumManual,
    AdamBaseline,
    SRAdamAdaptiveLocal
)

from src.model import SimpleCNN, get_model
from src.data import get_data_loaders
from src.training import train_model
from src.utils import (
    create_results_directory,
    save_all_results,
    plot_results,
    print_summary,
    count_parameters,
    save_multirun_summary,
    plot_mean_std,
    save_run_metrics,
    aggregate_runs_and_save,
    save_run_checkpoints,
)

# Import ablation and statistical utilities
from tools.ablation_utils import (
    should_skip_run,
    get_batch_ablation_configs,
    tag_ablation_metadata,
)
from tools.stat_testing import run_statistical_tests


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

    if name == "SR-Adam":
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

    if name == "SR-Adam-All-Weights":
        # apply Stein-rule to all parameters (single group with stein=True)
        all_params = list(model.parameters())
        param_groups = [{"params": all_params, "stein": True}]
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

    parser.add_argument("--dataset", type=str, default="ALL",
                        choices=["ALL", "CIFAR10", "CIFAR100"],
                        help="Dataset to run (use ALL to run both CIFAR10 and CIFAR100)")
    parser.add_argument("--model", type=str, default=None,
                        choices=[None, "simplecnn", "resnet18"],
                        help="Model architecture to use (overrides default per-dataset)")
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--noise", type=float, default=0.0)

    parser.add_argument("--num_runs", type=int, default=5,
                        help="Number of independent runs (different seeds)")
    parser.add_argument("--base_seed", type=int, default=42)

    parser.add_argument(
        "--optimizers",
        type=str,
        default="ALL",
        help='Optimizer list or "ALL" (case-insensitive); separate multiple with "|" or ";"'
    )

    parser.add_argument(
        "--clean_previous",
        action="store_true",
        help="Remove previous results for the selected optimizers (dataset/model/noise scope) before running"
    )

    parser.add_argument(
        "--batch_ablation",
        action="store_true",
        help="Enable batch-size ablation: test Adam and SR-Adam with batch_sizes=[256,512,2048]"
    )

    args = parser.parse_args()

    device = setup_device()

    # Decide dataset list and noise grid
    if args.dataset == "ALL":
        dataset_list = ["CIFAR10", "CIFAR100"]
    else:
        dataset_list = [args.dataset]

    noise_levels = [0.0, 0.05, 0.1]

    all_optimizer_names = [
        "SGD",
        "Momentum",
        "Adam",
        "SR-Adam",
        "SR-Adam-All-Weights",
    ]

    alias_map = {
        "sgd": "SGD",
        "momentum": "Momentum",
        "adam": "Adam",
        "sradam": "SR-Adam",
        "sradam_all": "SR-Adam-All-Weights",
    }

    optimizer_names = parse_optimizer_list(args.optimizers, all_optimizer_names, alias_map)

    # ------------------------------------------------------------------
    # Multi-run experiments
    # ------------------------------------------------------------------

    all_results = {}      # optimizer -> list of run metrics
    summary_stats = {}    # optimizer -> mean/std statistics

    # Determine batch-size configurations
    if args.batch_ablation:
        batch_ablation_configs = get_batch_ablation_configs(True, optimizer_names)
        print(f"[Ablation] Batch-size study enabled for: {batch_ablation_configs}")
    else:
        batch_ablation_configs = get_batch_ablation_configs(False, optimizer_names)

    # Iterate dataset x noise x optimizer grid
    for dataset_name in dataset_list:
        for noise in noise_levels:
            print(f"\n=== Dataset: {dataset_name} | Noise: {noise} ===")

            # select model per dataset unless overridden
            if args.model:
                model_name = args.model
            else:
                model_name = 'simplecnn' if dataset_name == 'CIFAR10' else 'resnet18'

            print(f"Using model: {model_name}")

            # Optional cleanup: remove previous results for chosen optimizers
            if args.clean_previous:
                for opt_name in optimizer_names:
                    folder = os.path.join('results', dataset_name, model_name, f'noise_{noise}', opt_name)
                    if os.path.isdir(folder):
                        try:
                            shutil.rmtree(folder)
                            print(f"[Clean] Removed previous results: {folder}")
                        except Exception as e:
                            print(f"[Clean] Failed to remove {folder}: {e}")

            # Load data for this dataset/noise
            # Use args.batch_size as default; will override in ablation loop if needed
            train_loader, test_loader, num_classes = get_data_loaders(
                dataset_name=dataset_name,
                batch_size=args.batch_size,
                noise_std=noise
            )

            # Prepare container for plotting epoch mean/std across optimizers
            results_per_optimizer = {}

            # Iterate over ablation configurations (batch_size always set)
            for opt_name, batch_size in batch_ablation_configs:
                
                # Reload data with correct batch size
                if batch_size != args.batch_size:
                    train_loader, test_loader, _ = get_data_loaders(
                        dataset_name=dataset_name,
                        batch_size=batch_size,
                        noise_std=noise
                    )

                print("\n" + "=" * 80)
                print(f"Optimizer: {opt_name} (batch_size: {batch_size})")
                print("=" * 80)

                all_results_for_opt = []

                for run in range(args.num_runs):
                    seed = args.base_seed + run
                    print(f"\nRun {run + 1}/{args.num_runs} | seed = {seed}")

                    # Check if run already exists; skip if so
                    skip_training, loaded_metrics = should_skip_run(
                        dataset_name, model_name, noise, opt_name, run + 1,
                        args.num_epochs, args.clean_previous, batch_size=batch_size
                    )

                    if skip_training and loaded_metrics:
                        print(f"  [Reuse] Skipping training")
                        metrics = loaded_metrics
                    else:
                        # Train normally
                        set_seeds(seed)

                        model = get_model(model_name, num_classes=num_classes).to(device)
                        print(f"Model params: {count_parameters(model):,}")

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

                    # Tag with ablation metadata (batch_size always set)
                    metrics = tag_ablation_metadata(metrics, batch_size=batch_size)

                    all_results_for_opt.append(metrics)

                    # Save per-run CSV + meta
                    save_run_metrics(metrics, dataset_name, model_name, noise, opt_name, run + 1)

                    # Save best and last model checkpoints (only if just trained)
                    if not skip_training:
                        try:
                            save_run_checkpoints(dataset_name, model_name, noise, opt_name, run + 1, metrics, base_dir='runs')
                        except Exception as e:
                            print(f"Checkpoint saving failed: {e}")

                # Aggregate runs for this optimizer/dataset/noise
                try:
                    # Run statistical tests if applicable
                    stat_tests = None
                    if opt_name in ["Adam", "SR-Adam", "SR-Adam-All-Weights"]:
                        # Will be computed after all optimizers for this grid point
                        stat_tests = None

                    summary_path, excel_path = aggregate_runs_and_save(
                        dataset_name, model_name, noise, opt_name, stat_tests=stat_tests
                    )
                    print(f"Aggregated results saved: {summary_path}, {excel_path}")
                except Exception as e:
                    print(f"Aggregation failed: {e}")

                # Compute and store summary stats for reporting across grid
                final_accs = [m['test_acc'][-1] for m in all_results_for_opt]
                best_accs = [max(m['test_acc']) for m in all_results_for_opt]

                summary_stats_key = f"{dataset_name}|noise_{noise}|{opt_name}"
                if batch_size:
                    summary_stats_key += f"|bs_{batch_size}"

                summary_stats[summary_stats_key] = {
                    'final_mean': float(np.mean(final_accs)),
                    'final_std': float(np.std(final_accs)),
                    'best_mean': float(np.mean(best_accs)),
                    'best_std': float(np.std(best_accs)),
                    'num_runs': args.num_runs,
                }

                # keep runs for plotting per-epoch mean/std
                results_per_optimizer[opt_name] = all_results_for_opt

                # accumulate per-dataset noise summary for later noise-vs-accuracy plot
                # initialize container at dataset level if missing
                dataset_level_key = f"{dataset_name}|{model_name}"
                if dataset_level_key not in summary_stats:
                    summary_stats[dataset_level_key] = {}
                if 'noise_table' not in summary_stats[dataset_level_key]:
                    summary_stats[dataset_level_key]['noise_table'] = {}
                if opt_name not in summary_stats[dataset_level_key]['noise_table']:
                    summary_stats[dataset_level_key]['noise_table'][opt_name] = []
                summary_stats[dataset_level_key]['noise_table'][opt_name].append({
                    'noise': noise,
                    'final_mean': summary_stats[summary_stats_key]['final_mean'],
                    'final_std': summary_stats[summary_stats_key]['final_std']
                })

            # After all optimizers for this dataset+noise: run statistical tests
            # and re-save aggregates with p-values
            try:
                stat_test_results = run_statistical_tests(results_per_optimizer, list(results_per_optimizer.keys()))
                
                # Re-aggregate with test results
                for opt_name in results_per_optimizer.keys():
                    # Filter tests for this optimizer (as comparator)
                    relevant_tests = [t for t in stat_test_results if opt_name in [t['optimizer_1'], t['optimizer_2']]]
                    if relevant_tests:
                        summary_path, _ = aggregate_runs_and_save(
                            dataset_name, model_name, noise, opt_name, 
                            stat_tests=relevant_tests
                        )
                        print(f"Updated {opt_name} aggregates with statistical test results")
            except Exception as e:
                print(f"Statistical testing failed: {e}")

            # After all optimizers for this dataset+noise: plot epoch mean ± std
            try:
                save_folder = os.path.join('results', dataset_name, model_name, f'noise_{noise}')
                os.makedirs(save_folder, exist_ok=True)
                plot_mean_std(results_per_optimizer, list(results_per_optimizer.keys()),
                              save_path=os.path.join(save_folder, 'test_acc_epoch_mean_std.png'))
                print(f"Saved epoch mean/std plot to {save_folder}")
            except Exception as e:
                print(f"Failed to plot epoch mean/std: {e}")

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------

    # Save global summary of all grid experiments
    os.makedirs('results', exist_ok=True)
    save_multirun_summary(summary_stats, 'results')

    # Print concise summary for all grid entries
    print("\n" + "=" * 80)
    print("FINAL GRID SUMMARY (mean ± std over runs)")
    print("=" * 80)
    for key, stats in summary_stats.items():
        # Skip non-summary entries (e.g., dataset-level containers)
        if not isinstance(stats, dict):
            continue

        if 'final_mean' in stats:
            print(
                f"{key}: "
                f"{stats['final_mean']:.2f} ± {stats['final_std']:.2f} "
                f"(best: {stats['best_mean']:.2f} ± {stats['best_std']:.2f})"
            )
        elif 'noise_table' in stats:
            # Print noise-vs-accuracy table for this dataset|model key
            print(f"\nNoise summary for {key}:")
            for opt_name, entries in stats['noise_table'].items():
                print(f"  Optimizer: {opt_name}")
                for e in entries:
                    print(f"    noise={e['noise']}: {e['final_mean']:.2f} ± {e['final_std']:.2f}")
        else:
            # Unknown dict shape; skip
            continue

if __name__ == "__main__":
    main()
