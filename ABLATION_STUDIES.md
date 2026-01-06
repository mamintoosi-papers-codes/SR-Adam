# Ablation Studies & Advanced Features

This document reflects the current ablation workflow, batch-size handling, and result reuse in the SR-Adam pipeline.

## Overview
- **Optimizer set (fixed order)**: `SGD`, `Momentum`, `Adam`, `SR-Adam`, `SR-Adam-All-Weights`
- **Batch-size sensitivity**: support for multiple batch sizes (e.g., 256, 512, 2048) with per-batch subfolders
- **Result reuse**: skips runs if metrics already exist
- **Statistical tests**: paired t-test and Wilcoxon after each optimizer group

## Quick Start

### Ablation run (multiple batch sizes)
```bash
python main.py \
  --dataset CIFAR10 \
  --num_epochs 100 \
  --num_runs 5 \
  --batch_size "256|512|2048" \
  --optimizers "Adam|SR-Adam|SR-Adam-All-Weights"