# Quick Start Guide - SR-Adam Refactored

## Installation

Ensure you have PyTorch and required packages:

```bash
pip install torch torchvision matplotlib numpy pandas openpyxl tqdm
```

## Running Experiments

### 1. Default Configuration (CIFAR-10, 15 epochs)

```bash
python main_refactored.py
```

### 2. Custom Dataset (CIFAR-100)

```bash
python main_refactored.py --dataset CIFAR100
```

### 3. Quick Test (5 epochs)

```bash
python main_refactored.py --num_epochs 5
```

### 4. Full Customization

```bash
python main_refactored.py \
    --dataset CIFAR10 \
    --batch_size 256 \
    --num_epochs 30 \
    --noise 0.01 \
    --seed 42
```

## Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | `CIFAR10` | Dataset to use: `CIFAR10` or `CIFAR100` |
| `--batch_size` | `512` | Batch size for data loaders |
| `--num_epochs` | `15` | Number of training epochs |
| `--noise` | `0.0` | Gaussian noise std dev (0.0 = no noise) |
| `--seed` | `42` | Random seed for reproducibility |

## Output Files

Results are saved to: `results_{DATASET}_noise{NOISE}/`

| File | Description |
|------|-------------|
| `optimizer_comparison_{DATASET}_batch{BS}_epochs{E}_noise{N}.xlsx` | Excel file with metrics for each optimizer |
| `config.json` | Experiment configuration and final accuracies |
| `optimizer_comparison.png` | 4-panel plot of metrics |
| `{optimizer}_metrics.csv` | Per-optimizer CSV (intermediate) |

## Excel Output Structure

Each Excel file contains one sheet per optimizer:

| Column | Description |
|--------|-------------|
| `Epoch` | Epoch number (1-indexed) |
| `Train Loss` | Training loss |
| `Test Loss` | Test loss |
| `Train Acc` | Training accuracy (%) |
| `Test Acc` | Test accuracy (%) |
| `Epoch Time` | Time per epoch (seconds) |

## Interpreting Results

### Training Curves

The generated `optimizer_comparison.png` shows 4 plots:

1. **Training Loss** - Should decrease smoothly
2. **Test Loss** - Should decrease initially then plateau
3. **Training Accuracy** - Should increase smoothly
4. **Test Accuracy** - Main metric for comparison

### Issues to Watch

| Issue | Likely Cause |
|-------|--------------|
| Loss = `nan` or `inf` | Numerical instability; check shrinkage clipping |
| No improvement | Learning rate too low; try `lr=0.01` or `lr=1e-2` |
| Divergence (loss increases) | Learning rate too high; try `lr=1e-4` or `lr=5e-4` |
| Accuracy at 10% | Check data loading; model may not be training |

## Module Structure

```
SR-Adam/
├── main_refactored.py         ← Entry point
├── optimizers.py               ← 6 optimizer classes
├── model.py                    ← SimpleCNN architecture
├── data.py                     ← Data loading
├── training.py                 ← Training loops
├── utils.py                    ← Results saving/visualization
├── README.md                   ← Original documentation
└── REFACTORING_SUMMARY.md     ← This refactoring
```

## Using Individual Modules

### Import and use optimizers directly

```python
from optimizers import SRAdamAdaptiveLocal
from model import SimpleCNN
from data import get_data_loaders
from training import train_model

# Load data
train_loader, test_loader, num_classes = get_data_loaders()

# Create model and optimizer
model = SimpleCNN(num_classes=10)
optimizer = SRAdamAdaptiveLocal(model.parameters(), lr=1e-3)

# Train
metrics = train_model(model, train_loader, test_loader, 
                      optimizer, criterion, num_epochs=15, device='cuda')
```

### Use data loading only

```python
from data import get_data_loaders

train_loader, test_loader, num_classes = get_data_loaders(
    dataset_name='CIFAR10',
    batch_size=512,
    noise_std=0.01,
    num_workers=8
)
```

## Troubleshooting

### Memory Issues

Reduce batch size:
```bash
python main_refactored.py --batch_size 128
```

### Slow Training

- Reduce number of workers in `data.py` (line ~73)
- Use larger batch size
- Reduce `num_epochs`

### Reproducibility

Always specify `--seed`:
```bash
python main_refactored.py --seed 42
```

### CUDA Not Available

The script automatically falls back to CPU. For GPU:
- Ensure NVIDIA drivers are installed
- Install `torch` with CUDA support:
  ```bash
  pip install torch --index-url https://download.pytorch.org/whl/cu118
  ```

## Expected Performance

Typical final test accuracy on CIFAR-10 (15 epochs):

| Optimizer | Accuracy |
|-----------|----------|
| SGD | 45-50% |
| Momentum | 68-72% |
| Adam | 72-75% |
| SR-Adam (Fixed) | 71-74% |
| SR-Adam (Adaptive, Global) | 72-76% |
| SR-Adam (Adaptive, Local) | 71-75% |

Note: Exact values depend on random seed and batch size.

## Contributing

To add a new optimizer:

1. Define class in `optimizers.py`
2. Add to `optimizer_names` in `main_refactored.py`
3. Add entry to `create_optimizer()` function
4. Run: `python main_refactored.py --num_epochs 3`

## References

- **Paper**: "Shrinkage Estimation in High-Dimensional Deep Learning: A Stein-Rule Approach"
- **James-Stein Shrinkage**: Stein (1956), James & Stein (1961)
- **Adam Optimizer**: Kingma & Ba (2014)

---

For more details, see:
- `README.md` - Project overview and theory
- `REFACTORING_SUMMARY.md` - Detailed refactoring documentation
- Individual module docstrings
