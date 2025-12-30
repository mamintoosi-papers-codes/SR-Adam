# SR-Adam: Stein-Rule Adaptive Moment Estimation

Experimental implementations of **Stein-rule–based gradient shrinkage** for deep learning optimization, studying how classical high-dimensional shrinkage estimation (James–Stein, Preliminary Test) can improve stochastic optimization in modern deep neural networks.

---

## Quick Start

Open in Colab (runs `main.ipynb`):

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamintoosi-papers-codes/SR-Adam/blob/main/main.ipynb)

```bash
# Windows: run full pipeline (experiments → aggregates → tables/figures → paper)
cd c:\git\mamintoosi-papers-codes\SR-Adam
reproduce_all.bat

# Or run experiments only (all noise levels processed automatically)
python main.py --dataset ALL --model simplecnn --optimizers ALL --num_epochs 20 --num_runs 5 --batch_size 512
```

Notes:
- Noise levels [0.0, 0.05, 0.1] are handled inside `main.py` and do not require `--noise`.
- To customize datasets/optimizers, adjust `--dataset` and `--optimizers` (e.g., `"Adam|SR-Adam"`).

---

## Project Structure

```
SR-Adam/
├── main.py                        # Experiment orchestrator (CIFAR10/100 grid)
├── optimizers.py                  # Baselines + SR-Adam implementations
├── model.py, data.py, training.py # Core pipeline
├── utils.py                       # I/O and plotting helpers
├── REPRODUCE.md                   # Full reproducibility guide
├── QUICK_START.md                 # Minimal quick start
├── reproduce_all.bat              # Windows automation script
├── paper_figures/                 # LaTeX paper + generated tables/figures
└── results/                       # Outputs (created automatically)
```

---

## Implemented Optimizers

### Baselines
- **SGD** - Vanilla Stochastic Gradient Descent
- **Momentum** - SGD with heavy-ball momentum  
- **Adam** - Adaptive Moment Estimation

### Stein-Rule Variants
### Stein-Rule Optimizer
- **SR-Adam** - Adaptive Stein-rule gradient shrinkage using Adam moments; applied to convolutional layers where shrinkage is most effective

---

## Core Idea

Treat stochastic gradient as a noisy estimator and shrink it toward stable momentum using James-Stein theory:

$$\hat{g}_t = m_{t-1} + c_t(g_t - m_{t-1})$$

Shrinkage factor:

$$c_t = \max\left(0, 1 - \frac{(p-2)\sigma^2}{\|g_t - m_{t-1}\|^2}\right)$$

Where:
- $p$ = dimensionality
- $\sigma^2$ = noise variance (estimated adaptively from Adam moments)

---

## Key Features

✅ **Modular Architecture** - 6 independent Python modules  
✅ **Automatic Result Saving** - CSV, Excel, JSON, plots after each epoch  
✅ **Numerical Stability** - 5 critical bugs fixed in SR-Adam implementation  
✅ **Mixed Precision** - AMP support for faster training  
✅ **Reproducibility** - Seed tracking, comprehensive configs  

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

## Expected Performance (Summary)

- On clean CIFAR10/CIFAR100, SR-Adam matches Adam within noise, staying competitive with standard baselines.
- Under label noise (0.05–0.1), SR-Adam typically improves test accuracy by roughly 1–3 percentage points over Adam/Momentum with SimpleCNN, reflecting better stability of noisy gradients.
- Variance across runs remains similar to Adam; seed-averaged mean ± std is reported in the generated tables.

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

**Note:** Model checkpoints (.pt files) are automatically saved to `runs/` directory to keep the repository lightweight. Use `move_models_to_runs.py` to migrate existing checkpoints from `results/` to `runs/` if needed.

---

## Documentation

- **[REPRODUCE.md](REPRODUCE.md)** - Full pipeline and arguments
- **[QUICK_START.md](QUICK_START.md)** - Minimal checklist and examples
- **Module docstrings** - API documentation in each `.py` file

### Fairness and Reproducibility
- Identical training protocol across methods; only the optimizer differs.
- Fixed backbone, epochs, batch size, label-noise schedule, evaluation, and seed plan.
- In-house implementations with matching interfaces; no third-party optimizer packages used.
- Public repository with executable code: https://github.com/mamintoosi-papers-codes/SR-Adam

---

## Installation

```bash
pip install torch torchvision matplotlib numpy pandas openpyxl tqdm
```

---

## Research Context

This code accompanies research on:
- High-dimensional shrinkage estimation
- Statistical risk reduction in stochastic optimization
- Bridging classical estimation theory and deep learning

The implementations are intentionally explicit to support experimentation and theoretical inspection.

### Key References
- **Stein (1956)** - Inadmissibility of the usual estimator
- **James & Stein (1961)** - Estimation with quadratic loss
- **Kingma & Ba (2014)** - Adam: A Method for Stochastic Optimization

---

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{sr-adam,
  title={SR-Adam: Stein-Rule Adaptive Moment Estimation},
  author={M.Arashi},
  year={2025},
  url={https://github.com/mamintoosi-papers-codes/SR-Adam}
}
```

---

## License

MIT License - See LICENSE file for details
