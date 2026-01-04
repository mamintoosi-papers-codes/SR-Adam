# Ablation Studies & Advanced Features

This document describes the new ablation study capabilities and result-reuse features integrated into the SR-Adam experiments pipeline.

## Overview

The pipeline now supports:
1. **Optimizer Ablation**: Compare SR-Adam (conv-only) vs SR-Adam-All-Weights
2. **Batch-Size Sensitivity**: Study optimizer performance across different batch sizes
3. **Result Reuse**: Skip redundant training runs if metrics already exist
4. **Statistical Significance Testing**: Paired t-tests and Wilcoxon signed-rank tests

## Quick Start

### Run with All Features (Recommended for Paper)

```bash
python main.py \
  --dataset CIFAR10 \
  --num_epochs 100 \
  --num_runs 5 \
  --batch_ablation \
  --optimizers "Adam|SR-Adam|SR-Adam-All-Weights"
```

This will:
- Train Adam and SR-Adam with batch sizes [256, 512, 2048]
- Train SR-Adam-All-Weights with default batch size
- Skip any runs that already have saved results
- Compute statistical significance tests after each optimizer group
- Save aggregated results with p-values

### Run Default Study (No Batch Ablation)

```bash
python main.py \
  --dataset CIFAR10 \
  --num_epochs 100 \
  --num_runs 5 \
  --optimizers "SGD|Momentum|Adam|SR-Adam|SR-Adam-All-Weights"
```

Trains all optimizers with default batch size (128).

## Feature Details

### 1. Batch-Size Sensitivity Study (`--batch_ablation`)

When enabled, Adam and SR-Adam are tested with batch sizes: **256, 512, 2048** (in addition to default 128).

**Output Structure**:
```
results/
  CIFAR10/
    simplecnn/
      noise_0.0/
        Adam/
          batch_size_256/
            run_1_metrics.csv
            aggregated_summary.json
          batch_size_512/
            ...
          batch_size_2048/
            ...
        SR-Adam/
          batch_size_256/
            ...
```

**CSV Metadata**: Each `run_*_metrics.csv` includes columns:
- `ablation`: "batch_size" (if ablated)
- `batch_size`: 256, 512, 2048, or null (default)

### 2. Result Reuse (Always Enabled)

Before training a run, the pipeline checks:
1. Does `results/.../run_{k}_metrics.csv` exist?
2. Does a checkpoint exist in `runs/.../run_*.pt`?

If either exists (and `--clean_previous` is NOT set):
- **CSV found**: Load metrics directly from CSV
- **Checkpoint found**: Reconstruct minimal metrics from checkpoint state

This avoids redundant training on the same machine or when resuming work.

### 3. Optimizer Ablation

Two SR-Adam variants are now supported:

| Optimizer | Stein Rule Applied To |
|-----------|----------------------|
| SR-Adam | Convolutional layers only |
| SR-Adam-All-Weights | All parameters (conv + fc) |

Use `--optimizers "SR-Adam|SR-Adam-All-Weights"` to compare them.

### 4. Statistical Significance Testing

After all runs for each optimizer+dataset+noise combination, the pipeline computes:
- **Paired t-test**: Tests if final test accuracies are significantly different
- **Wilcoxon signed-rank test**: Non-parametric alternative

**Output**: `aggregated_summary.json` includes:
```json
{
  "statistical_tests": [
    {
      "optimizer_1": "Adam",
      "optimizer_2": "SR-Adam",
      "t_test_pvalue": 0.0234,
      "wilcoxon_pvalue": 0.0625,
      "t_test_statistic": -2.456,
      "wilcoxon_statistic": 2.0
    }
  ]
}
```

**Interpretation**:
- p-value < 0.05: Statistically significant difference (at 95% confidence)
- p-value >= 0.05: No significant difference
- Wilcoxon test is more robust for small sample sizes (n < 30)

## Output Files

### Per-Run Metrics
- **File**: `results/{dataset}/{model}/noise_{noise}/{optimizer}/batch_size_{bs}/run_{k}_metrics.csv`
- **Contains**: Epoch-wise train/test loss and accuracy, final best accuracy, seed, ablation metadata

### Aggregated Results
- **File**: `results/{dataset}/{model}/noise_{noise}/{optimizer}/aggregated_summary.json`
- **Contains**: Mean/std final accuracy, best accuracy, statistical tests, metadata

### Excel Summary
- **File**: `results/{dataset}/{model}/noise_{noise}/{optimizer}/aggregated_results.xlsx`
- **Contains**: Formatted summary for easy review

## Examples

### Example 1: Batch-Size Sensitivity on CIFAR10

```bash
python main.py \
  --dataset CIFAR10 \
  --num_epochs 100 \
  --num_runs 3 \
  --batch_ablation \
  --optimizers "Adam|SR-Adam"
```

Produces 12 training jobs (2 optimizers × 3 batch sizes × 2 runs, but 1 run = default batch size):
- Adam bs=256, 512, 2048 (3 runs each = 9 total)
- SR-Adam bs=256, 512, 2048 (3 runs each = 9 total)
- Total time: ~2-3 hours (depending on hardware)

### Example 2: Reproduce Paper Results with Ablations

```bash
python main.py \
  --dataset ALL \
  --num_epochs 200 \
  --num_runs 5 \
  --batch_ablation \
  --optimizers "SGD|Momentum|Adam|SR-Adam|SR-Adam-All-Weights"
```

This reproduces the full paper including:
- Baseline optimizers: SGD, Momentum (default batch size only)
- Adaptive optimizers with batch ablation: Adam, SR-Adam (batch sizes 256, 512, 2048)
- SR-Adam variant: SR-Adam-All-Weights (default batch size)
- Statistical tests comparing all variants
- Both CIFAR10 and CIFAR100

### Example 3: Resume Interrupted Experiments

```bash
# First run (interrupted after 2 runs)
python main.py --dataset CIFAR10 --num_runs 5 --batch_ablation

# Resume: Same command, existing runs are skipped
python main.py --dataset CIFAR10 --num_runs 5 --batch_ablation
# Continues from run 3
```

### Example 4: Clean Previous Results and Restart

```bash
python main.py \
  --dataset CIFAR10 \
  --batch_ablation \
  --clean_previous \
  --optimizers "Adam|SR-Adam"
```

Removes all previous results for Adam and SR-Adam, then trains fresh.

## Troubleshooting

### Results Not Being Reused
- Check that `--clean_previous` is NOT set
- Verify CSV files exist in `results/{dataset}/{model}/noise_{noise}/{optimizer}/`
- Check file permissions and disk space

### Statistical Tests Not Running
- Ensure at least 2 runs completed for each optimizer
- Check that `run_*_metrics.csv` files have `test_acc` column

### Memory Issues with Large Batch Sizes
- Reduce `--num_runs`
- Use smaller dataset first: `--dataset CIFAR10` instead of `--dataset ALL`
- Run optimizers separately: `--optimizers "Adam"` then `--optimizers "SR-Adam"`

## Output Structure Summary

```
results/
  {DATASET}/
    {MODEL}/
      noise_{NOISE}/
        {OPTIMIZER}/
          batch_size_{BS}/  # Only if --batch_ablation
            run_1_metrics.csv
            run_2_metrics.csv
            ...
            aggregated_summary.json
            aggregated_results.xlsx
            test_acc_epoch_mean_std.png

runs/
  {DATASET}/
    {MODEL}/
      noise_{NOISE}/
        {OPTIMIZER}/
          batch_size_{BS}/  # Only if --batch_ablation
            run_1_best.pt
            run_1_final.pt
            run_2_best.pt
            run_2_final.pt
            ...
```

## Advanced: Reuse Results Across Machines

1. **On machine A**: Run experiments and save to `results/` and `runs/`
2. **Transfer**: Copy `runs/` folder to machine B
3. **On machine B**: Run same command; the pipeline will reconstruct metrics from checkpoints if CSV not found

This is useful for:
- Centralizing long training on compute server
- Continuing on local machine for analysis
- Sharing reproducible model checkpoints

## Key References

- Main script: [main.py](main.py)
- Ablation utilities: [tools/ablation_utils.py](tools/ablation_utils.py)
- Statistical testing: [tools/stat_testing.py](tools/stat_testing.py)
- Aggregation logic: [src/utils.py](src/utils.py) `aggregate_runs_and_save()`

