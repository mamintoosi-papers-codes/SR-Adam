# Stein-Rule Shrinkage for Stochastic Gradient Estimation in High Dimensions

This repository contains the experimental code for:

**Stein-Rule Shrinkage for Stochastic Gradient Estimation in High Dimensions**
(Arashi & Amintoosi, 2026, arXiv:2602.01777)

The project investigates stochastic gradient optimization from a **high-dimensional statistical decision-theoretic perspective**, connecting classical Stein-rule shrinkage (James–Stein estimation) with modern deep learning optimization. The proposed method, **SR-Adam**, incorporates adaptive shrinkage into Adam using online variance estimates derived from its moment statistics.

---

## Key Idea

We reinterpret the stochastic gradient as a noisy estimator of the true gradient and apply Stein-type shrinkage toward a stable momentum estimate:

$$
\hat{g}*t = m*{t-1} + c_t (g_t - m_{t-1})
$$

with adaptive shrinkage factor:

$$
c_t = \max\left(0,; 1 - \frac{(p-2)\sigma^2}{|g_t - m_{t-1}|^2}\right)
$$

where:

* $p$: parameter dimension
* $\sigma^2$: estimated noise variance (from Adam moments)
* $m_{t-1}$: exponential moving average of gradients

This yields **SR-Adam**, a lightweight modification of Adam with negligible computational overhead.

---

## Highlights

* Connects **Stein-rule estimation theory** with stochastic optimization
* Provides a **decision-theoretic justification** for gradient shrinkage
* Introduces **SR-Adam**, an adaptive optimizer built on Adam moments
* Requires **no additional tuning hyperparameters**
* Demonstrates empirical gains in controlled deep learning settings

---

## Experimental Setup

All experiments are designed for controlled comparison of optimizers under identical training conditions.

* Datasets: CIFAR-10, CIFAR-100
* Model: SimpleCNN (fixed architecture across all runs)
* Optimizers: SGD, Momentum, Adam, SR-Adam
* Noise injection: Gaussian input noise (0.0, 0.05, 0.1)
* Runs per configuration: 5 independent seeds
* Evaluation: test accuracy, loss, and aggregated statistics

---

## Quick Start

<!-- Open in Colab (runs `main.ipynb`):

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamintoosi-papers-codes/SR-Adam/blob/main/main.ipynb) -->

### Full reproducibility pipeline (Windows)


```bash
cd PATH-TO-SR-Adam-Folder
reproduce_all.bat
```

This executes:

1. Training across all configurations
2. Aggregation of results
3. Generation of plots and tables
4. Paper-ready outputs

---

### Run experiments manually

```bash
python main.py \
  --dataset ALL \
  --model simplecnn \
  --optimizers ALL \
  --num_epochs 20 \
  --num_runs 5 \
  --batch_size 512
```

Notes:
- Noise levels [0.0, 0.05, 0.1] are handled inside `main.py` and do not require `--noise`.
- To customize datasets/optimizers, adjust `--dataset` and `--optimizers` (e.g., `"Adam|SR-Adam"`).

---

## Project Structure

```
SR-Adam/
├── main.py                        # Experiment orchestrator (CIFAR10/100 grid)
├── src/                           # Core modules
│   ├── data.py
│   ├── model.py
│   ├── optimizers.py
│   ├── training.py
│   └── utils.py
├── tools/                         # Aggregation, plots, tables, reporting utilities
├── REPRODUCE.md                   # Full reproducibility guide
├── QUICK_START.md                 # Minimal quick start
├── reproduce_all.bat              # Windows automation script
├── paper/                         # LaTeX paper + generated tables/figures
├── results/                       # Metrics/plots (created automatically)
└── runs/                          # Model checkpoints (git-ignored)
```

---

## Implemented Optimizers

### Baselines
- **SGD** - Vanilla Stochastic Gradient Descent
- **Momentum** - SGD with heavy-ball momentum  
- **Adam** - Adaptive Moment Estimation

### Stein-Rule Optimizer
- **SR-Adam** - Adaptive Stein-rule gradient shrinkage using Adam moments; applied to convolutional layers where shrinkage is most effective



---

## Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--dataset` | `ALL` | `CIFAR10`, `CIFAR100`, or `ALL` (runs both) |
| `--model` | `None` | `simplecnn` (this study uses SimpleCNN throughout) |
| `--optimizers` | `ALL` | One or more of `SGD`, `Momentum`, `Adam`, `SR-Adam` (use `|` to separate) |
| `--num_runs` | `5` | Independent runs per configuration (seeded) |
| `--num_epochs` | `20` | Training epochs per run |
| `--batch_size` | `2048` | Batch size for training/eval |
| `--noise` | `0.0` | Ignored: noise levels are processed automatically in `main.py` |
| `--base_seed` | `42` | Base seed; per-run seeds = base+run_index |


---

## Output Files

Training produces two main output directories:

### `results/` - Metrics and Summaries
CSV files, JSON metadata, aggregated statistics, and plots (kept in version control):
```
results/
├── CIFAR10/
│   └── simplecnn/
│       ├── noise_0.0/
│       │   ├── Adam/
│       │   │   ├── run_1.csv              # Per-epoch metrics
│       │   │   ├── run_1_meta.json        # Final summary
│       │   │   └── aggregated_summary.json # Mean ± std across runs
│       │   └── SR-Adam/...
│       └── noise_0.1/test_acc_epoch_mean_std.png
└── summary_statistics.csv
```

### `runs/` - Model Checkpoints
PyTorch model files (excluded from git via `.gitignore` to save space):
```
runs/
├── CIFAR10/
│   └── simplecnn/
│       └── noise_0.05/
│           ├── Adam/
│           │   ├── run_1_best.pt   # Best epoch checkpoint
│           │   └── run_1_last.pt   # Final epoch checkpoint
│           └── SR-Adam/
│               ├── run_1_best.pt
│               └── run_5_last.pt
```

**Note:** Model checkpoints (.pt files) are automatically saved to `runs/` directory to keep the repository lightweight.

---

## Documentation

- **[REPRODUCE.md](REPRODUCE.md)** - Full pipeline and arguments
- **[QUICK_START.md](QUICK_START.md)** - Minimal checklist and examples
- **Module docstrings** - API documentation in each `.py` file

### Fairness and Reproducibility
- Identical training protocol across methods; only the optimizer differs.
- Fixed backbone, epochs, batch size, input-noise schedule, evaluation, and seed plan.
- In-house implementations with matching interfaces; no third-party optimizer packages used.
- Public repository with executable code: https://github.com/mamintoosi-papers-codes/SR-Adam

---

## Installation

### Using pre-configured requirements:

```bash
# GPU (CUDA 12.4)
pip install -r requirements.txt

# CPU only
pip install -r requirements-cpu.txt
```

### Or manually install core packages:

```bash
pip install torch torchvision matplotlib numpy pandas openpyxl tqdm seaborn scipy
```


---

```
main.py
└── src/
  ├── model.py (SimpleCNN)
  ├── data.py (CIFAR loaders with noise)
  ├── training.py (train_model, evaluate)
  ├── optimizers.py (SGD, Momentum, Adam, SR-Adam)
  └── utils.py (save, aggregate, plot)

Regeneration scripts (run from tools/):
├── regenerate_aggregates.py
├── regenerate_summary_statistics.py
├── regenerate_epoch_pngs.py

Table generators (tools/):
├── make_minimal_tables.py
├── generate_architecture_table.py
├── make_method_tables.py

Figure generators (tools/):
├── make_testacc_plots_simplecnn.py
├── make_loss_plots_simplecnn.py
├── make_figures.py

Paper:
└── paper/paper-draft.tex
  ├── \input{experimental_figures.tex}
  └── \input{minimal-tables-content.tex}

```

---

## Advanced Usage: Filtering Results

All result processing scripts support optional filters for flexible analysis:

### Filter Options

```bash
# Available filters for all tools:
--dataset       CIFAR10, CIFAR100, or ALL (default: ALL)
--noise         Specific noise level(s) or ALL (default: ALL)
--batch_size    Specific batch size(s) or ALL (default: ALL for aggregation, 512 for tables)
--optimizers    Specific optimizer(s) or ALL (default: ALL)
```


---

## Research Context

This code accompanies research on:
- High-dimensional shrinkage estimation
- Statistical risk reduction in stochastic optimization
- Bridging classical estimation theory and deep learning

---

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{arashi2026steinrule,  
	title={Stein-Rule Shrinkage for Stochastic Gradient Estimation in High Dimensions},  
	author={Arashi, M. and Amintoosi, M.},  
	year={2026},  
	eprint={2602.01777},  
	archivePrefix={arXiv},  
	primaryClass={cs.LG},  
	url={https://arxiv.org/abs/2602.01777}
}
```

---

## Repository Purpose

This repository is intended to provide:

* Fully reproducible experimental pipeline
* Reference implementation of SR-Adam
* Controlled benchmarking environment for optimizer comparison
* Supporting material for the associated manuscript