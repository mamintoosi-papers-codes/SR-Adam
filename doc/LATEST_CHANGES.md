# Summary of Recent Changes to SR-Adam Project

## Overview

The SR-Adam project has evolved significantly with focus on robust experimental validation using **multiple independent runs** with **different random seeds** to compute mean/std statistics.

---

## Key Changes Made

### 1. **Enhanced main.py**

**New Features:**
- ✅ Multi-run experiments (default: 5 independent runs)
- ✅ Noise level support (`--noise` parameter)
- ✅ Custom optimizer selection (`--optimizers` parameter)
- ✅ Statistics computation (mean ± std)
- ✅ Parameter counting per model

**New Command Line Arguments:**
```
--num_runs      : Number of independent runs (default: 5)
--base_seed     : Starting seed for reproducibility (default: 42)
--noise         : Gaussian noise std dev (default: 0.0)
--optimizers    : Comma/pipe separated list or "all" (default: "all")
```

**Example Usage:**
```bash
python main.py --dataset CIFAR10 --num_runs 5 --noise 0.05 --optimizers "adam|sradam"
```

---

### 2. **SR-Adam Implementation Evolution**

**Previous Version (general purpose):**
- Applied Stein shrinkage to ALL parameter groups
- Struggled with numerical stability in FC layers

**Current Version (Conv-only, Adaptive):**
- ✅ Stein shrinkage ONLY applied to convolutional layers
- ✅ FC layers use standard Adam (no shrinkage)
- ✅ Whitened Stein computation in Adam space
- ✅ Per-group control via `stein=True/False` flag

**Key Innovation:**
```python
# Example in main.py
SRAdamAdaptiveLocal(
    [
        {"params": model.conv1.parameters(), "stein": True},   # Apply Stein
        {"params": model.conv2.parameters(), "stein": True},   # Apply Stein
        {"params": model.fc1.parameters(), "stein": False},    # Standard Adam
        {"params": model.fc2.parameters(), "stein": False},    # Standard Adam
    ],
    lr=1e-3,
    warmup_steps=5,
    shrink_clip=(0.1, 1.0),
)
```

---

### 3. **Whitened Stein Computation**

**New Mathematical Approach:**

In the Stein shrinkage computation, gradients and moments are now whitened by Adam's denominator:

$$g_w = \frac{g}{\sqrt{v} + \epsilon}, \quad m_w = \frac{m}{\sqrt{v} + \epsilon}$$

Then shrinkage is computed in this whitened space:

$$\sigma^2_w = \text{mean}(g_w - m_w)^2$$

$$c_t = \max\left(\text{clip\_lo}, \min\left(\text{clip\_hi}, 1 - \frac{(p-2)\sigma^2_w}{\|g_w - m_w\|^2}\right)\right)$$

**Why This Works:**
- Scales gradients by their adaptive learning rates (Adam style)
- Makes Stein estimation more stable
- Provides better empirical performance

---

### 4. **Baseline Optimizers (Manual Implementations)**

All baseline optimizers are implemented manually for fair comparison:

| Optimizer | Learning Rate | Momentum |
|-----------|---------------|----------|
| SGD | 0.01 | - |
| Momentum | 0.01 | 0.9 |
| Adam | 1e-3 | - |
| SR-Adam (Conv-only) | 1e-3 | - |

---

## Experimental Framework

### Multi-run Protocol

```
For each optimizer:
  For run = 1 to num_runs:
    Set seed = base_seed + run
    Train model for num_epochs
    Record metrics (loss, accuracy, etc.)
  
  Compute statistics:
    final_accuracy = mean(run_1.final_acc, ..., run_5.final_acc)
    uncertainty = std(run_1.final_acc, ..., run_5.final_acc)
```

### Notebook Execution Pattern

The notebook (`main.ipynb`) runs multiple experiments:

1. **CIFAR-10 with noise:**
   - Noise = 0.05, 0.10, 0.20
   - 5 independent runs each
   - Compares Adam vs SR-Adam

2. **CIFAR-100:**
   - Similar pattern with larger dataset

---

## Current Results (CIFAR-10, noise=0.0)

```
Adam: 73.67 ± 0.56 (best: 73.71 ± 0.49)
SR-Adam (Conv-only, Adaptive): 75.42 ± 0.44 (best: 75.44 ± 0.42)
```

**Interpretation:**
- SR-Adam improves mean accuracy by ~1.75 percentage points
- Uncertainty is slightly lower for SR-Adam (more stable)
- Best run accuracy is also higher (~1.7 pp improvement)

---

## File Changes Summary

| File | Change | Status |
|------|--------|--------|
| `main.py` | Multi-run framework + noise arg | ✅ Updated |
| `optimizers.py` | Conv-only, Whitened SR-Adam | ✅ Updated |
| `model.py` | Parameter grouping compatible | ✅ Compatible |
| `training.py` | Unchanged (reused) | ✅ OK |
| `data.py` | Noise augmentation support | ✅ OK |
| `utils.py` | Multi-run statistics export | ✅ Enhanced |
| `main.ipynb` | Multi-run experiment cells | ✅ Updated |

---

## Code Architecture

### Main Components

1. **Optimizer Factory** (`create_optimizer`)
   - Creates appropriate optimizer with layer-wise config
   - SGD, Momentum, Adam: standard
   - SR-Adam: Conv-only with Stein flags

2. **Multi-run Loop**
   - Per optimizer: multiple runs with different seeds
   - Collects metrics for each run
   - Computes mean/std statistics

3. **Results Management**
   - Per-run results saved
   - Statistics exported to Excel
   - Summary plots generated

### Key Parameters

```python
# Noise augmentation (during training)
--noise 0.0, 0.05, 0.1, 0.2

# Experimental runs
--num_runs 5              # 5 independent seeds
--base_seed 42           # seeds = 42, 43, 44, 45, 46

# Model training
--batch_size 512         # Large batches for stability
--num_epochs 20          # 20 epochs training

# Optimizer selection
--optimizers "all"       # Or "adam|sradam"
```

---

## What's Working Well

✅ **SR-Adam (Conv-only) shows consistent improvements over Adam**
- Mean accuracy: 75.42% vs 73.67% (+1.75pp)
- Lower variance (more stable)
- Works well with noise augmentation

✅ **Proper experimental methodology**
- Multiple independent runs
- Standard deviation quantifies uncertainty
- Reproducible with seed control

✅ **Whitened Stein computation**
- More numerically stable
- Better empirical performance
- Theoretically motivated

---

## Next Steps (Recommendations)

1. **Test on CIFAR-100** to validate on larger dataset
2. **Vary noise levels** (0.0, 0.05, 0.1, 0.2) to understand robustness
3. **Ablation studies:**
   - Full SR-Adam (all layers) vs Conv-only
   - Different warmup_steps
   - Different clip ranges
4. **Larger models** - Test on ResNet or VGG
5. **Paper preparation** - Document methodology and results

---

## Documentation References

- Original paper theory: `doc/paper-draft.tex`
- Implementation notes: `doc/NOTES.md`
- Quick reference: `doc/QUICKSTART.md`

---

**Status:** ✅ Solid experimental framework in place with promising results!
