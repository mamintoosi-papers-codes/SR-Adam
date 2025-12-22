# SR-Adam: Stein-Rule Adaptive Moment Estimation

Experimental implementations of **Stein-rule–based gradient shrinkage** for deep learning optimization, studying how classical high-dimensional shrinkage estimation (James–Stein, Preliminary Test) can improve stochastic optimization in modern deep neural networks.

---

## Quick Start

```bash
# Run with default settings (CIFAR-10, 15 epochs)
python main.py

# Quick test (5 epochs)
python main.py --num_epochs 5

# Full customization
python main.py --dataset CIFAR100 --batch_size 256 --num_epochs 30 --noise 0.01
```

---

## Project Structure

```
SR-Adam/
├── main.py              # Entry point
├── optimizers.py        # 6 optimizer implementations
├── model.py             # SimpleCNN architecture
├── data.py              # Data loading utilities
├── training.py          # Training loops
├── utils.py             # Save/visualization
├── doc/                 # Documentation
│   ├── QUICKSTART.md    # Detailed command reference
│   ├── NOTES.md         # Technical details
│   └── main_original.py # Original monolithic version
└── results_*/           # Generated results
```

---

## Implemented Optimizers

### Baselines
- **SGD** - Vanilla Stochastic Gradient Descent
- **Momentum** - SGD with heavy-ball momentum  
- **Adam** - Adaptive Moment Estimation

### Stein-Rule Variants
- **SR-Adam (Fixed, Global)** - Fixed noise variance σ²
- **SR-Adam (Adaptive, Global)** - Adaptive variance from Adam moments (stable)
- **SR-Adam (Adaptive, Local)** - Per-group shrinkage (most faithful)

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
✅ **Numerical Stability** - 5 critical bugs fixed in SR-Adam (Adaptive, Local)  
✅ **Mixed Precision** - AMP support for faster training  
✅ **Reproducibility** - Seed tracking, comprehensive configs  

---

## Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--dataset` | `CIFAR10` | Dataset: CIFAR10 or CIFAR100 |
| `--batch_size` | `512` | Batch size for data loaders |
| `--num_epochs` | `15` | Number of training epochs |
| `--noise` | `0.0` | Gaussian noise std dev (0.0 = no noise) |
| `--seed` | `42` | Random seed for reproducibility |

---

## Expected Performance

Typical test accuracy on CIFAR-10 (15 epochs):

| Optimizer | Test Accuracy |
|-----------|---------------|
| SGD | 45-50% |
| Momentum | 68-72% |
| Adam | 72-75% |
| SR-Adam (Fixed, Global) | ? |
| SR-Adam (Adaptive, Global) | ? |
| SR-Adam (Adaptive, Local) | ? |

---

## Output Files

Results are saved to `results_{DATASET}_noise{NOISE}/`:

```
results_CIFAR10_noise0.0/
├── optimizer_comparison_CIFAR10_batch512_epochs15_noise0.0.xlsx
├── config.json
└── optimizer_comparison.png
```

- **Excel file**: One sheet per optimizer with all metrics
- **JSON config**: Experiment parameters and final accuracies  
- **PNG plot**: 4-panel visualization (train/test loss & accuracy)

---

## Documentation

- **[doc/QUICKSTART.md](doc/QUICKSTART.md)** - Detailed command reference & examples
- **[doc/NOTES.md](doc/NOTES.md)** - Technical details & implementation notes
- **Module docstrings** - API documentation in each `.py` file

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
