# SR-Adam: Reproducible Research Guide

## Overview

This guide enables you to reproduce the full SR-Adam research pipeline:
1. **Run experiments** on CIFAR10/CIFAR100 with label noise
2. **Aggregate results** from multiple runs (5 seeds per configuration)
3. **Generate publication-ready tables** (best/final accuracy and loss)
4. **Generate figures** (per-epoch plots and noise sweeps)
5. **Compile the paper** with all tables and figures

---

## Prerequisites

### Environment Setup

1. **Clone and navigate to the repository:**
   ```bash
   cd c:\git\mamintoosi-papers-codes\SR-Adam
   ```

2. **Create/activate conda environment with PyTorch:**
   ```bash
   # If you don't have the 'pth' environment yet:
   conda create -n pth python=3.10 pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
   
   # Activate:
   conda activate pth
   ```

3. **Install additional packages:**
   ```bash
   pip install pandas numpy matplotlib seaborn openpyxl
   ```

4. **Verify setup:**
   ```bash
   python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
   ```

---

## Quick Start: Full Reproducibility Pipeline

### Option A: Automated Windows Batch Script

Run the entire pipeline with one command:

```bash
cd c:\git\mamintoosi-papers-codes\SR-Adam
reproduce_all.bat
```

This will:
- Run all experiments (CIFAR10 & CIFAR100, 3 noise levels, 5 seeds each)
- Aggregate results
- Generate all tables and figures
- Compile the paper PDF
- Expected time: **2-4 hours** (depending on GPU)

### Option B: Manual Step-by-Step

---

## Detailed Workflow

### Step 1: Run Experiments

**Time required:** ~2 hours on NVIDIA GPU (varies by hardware)

Simple command to run all CIFAR10 and CIFAR100 experiments:

```bash
# Activate environment
conda activate pth

# Run all experiments (noise levels: 0.0, 0.05, 0.1 are processed automatically)
python main.py --dataset ALL --model simplecnn --optimizers ALL --num_epochs 20 --num_runs 5 --batch_size 512
```

**Important:** The `--noise` argument is automatically handled by `main.py`. All three noise levels [0.0, 0.05, 0.1] are processed in a single execution.

For argument details, see **Configuration: main.py Arguments** section below.
```
results/
├── CIFAR10/
│   └── simplecnn/
│       ├── noise_0.0/
│       │   ├── SGD/
│       │   │   ├── run_1.csv
│       │   │   ├── run_1_meta.json
│       │   │   └── ...
│       │   ├── Momentum/
│       │   ├── Adam/
│       │   └── SR-Adam/
│       ├── noise_0.05/
│       └── noise_0.1/
├── CIFAR100/
│   └── simplecnn/
│       ├── noise_0.0/
│       ├── noise_0.05/
│       └── noise_0.1/
└── summary_statistics.csv
```

---

### Step 2: Aggregate and Summarize Results

After experiments complete, aggregate per-run CSV files and generate summary statistics:

```bash
conda activate pth

# Regenerate aggregated files per optimizer per noise level
python regenerate_aggregates.py

# Rebuild root summary_statistics.csv
python regenerate_summary_statistics.py

# Regenerate epoch-level PNG plots
python regenerate_epoch_pngs.py
```

**Outputs:**
- `results/*/summarized_epoch_stats.csv` – per-optimizer aggregates
- `results/summary_statistics.csv` – root summary table
- `results/*/test_acc_epoch_mean_std.png` – epoch plots

---

### Step 3: Generate Publication-Ready Tables

```bash
conda activate pth

# Generate method comparison tables (best/final Acc/Loss with bold best)
python make_minimal_tables.py

# Generate SimpleCNN architecture table
python generate_architecture_table.py
```

**Outputs:**
- `paper_figures/minimal-tables.tex` – standalone compilable LaTeX
- `paper_figures/minimal-tables-content.tex` – body-only for \input{}
- `paper_figures/simplecnn_arch.tex` – architecture standalone
- `paper_figures/simplecnn_arch_content.tex` – architecture body-only

---

### Step 4: Generate Publication-Ready Figures

```bash
conda activate pth

# Generate loss and accuracy panels for SimpleCNN
python make_loss_plots_simplecnn.py
python make_testacc_plots_simplecnn.py

# Generate method comparison figures
python make_figures.py
```

**Outputs:**
- `paper_figures/cifar10_noise0.0_acc_mean_std.pdf` (and 0.05, 0.1)
- `paper_figures/cifar10_noise0.0_loss_mean_std.pdf` (and 0.05, 0.1)
- `paper_figures/cifar100_noise0.0_acc_mean_std.pdf` (and 0.05, 0.1)
- `paper_figures/cifar100_noise0.0_loss_mean_std.pdf` (and 0.05, 0.1)
- `paper_figures/figures.pdf` – noise sweep and epoch plots

---

### Step 5: Compile the Paper

```bash
cd paper_figures
pdflatex -interaction=nonstopmode paper-draft.tex
pdflatex -interaction=nonstopmode paper-draft.tex  # Run twice for cross-references
```

**Output:**
- `paper_figures/paper-draft.pdf` – final compiled paper with all tables/figures

---

## Configuration: main.py Arguments

All experiment configuration is controlled via `main.py` command-line arguments:

### **Core Arguments**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--dataset` | str | `ALL` | Dataset to use: `CIFAR10`, `CIFAR100`, or `ALL` (runs both) |
| `--model` | str | None (auto) | Model architecture: `simplecnn`. If None, SimpleCNN is used throughout this study |
| `--optimizers` | str | `ALL` | Optimizers to test: `SGD`, `Momentum`, `Adam`, `SR-Adam`, or `ALL` (runs all four). Use `\|` to separate: `"adam\|sradam"` |
| `--num_runs` | int | 5 | Number of independent runs per configuration (different seeds) |
| `--num_epochs` | int | 20 | Number of training epochs |
| `--batch_size` | int | 2048 | Batch size for training and evaluation |
| `--noise` | float | 0.0 | **DEPRECATED/IGNORED**: Noise levels [0.0, 0.05, 0.1] are automatically processed by main.py |

**Important Note on Noise Levels:**
The `--noise` argument is no longer used. In each execution of `main.py`, all three noise levels **[0.0, 0.05, 0.1]** are automatically tested for each dataset-optimizer combination. This is hardcoded in `main.py` (line 172: `noise_levels = [0.0, 0.05, 0.1]`).

### **Examples**

**Minimal (runs CIFAR10 with all noise levels):**
```bash
python main.py --dataset CIFAR10
```

**Full paper reproduction (all configurations):**
```bash
python main.py --dataset ALL --model simplecnn --optimizers ALL --num_epochs 20 --num_runs 5 --batch_size 512
```

**Optimizer comparison only (both datasets, all noise levels):**
```bash
python main.py --dataset ALL --optimizers "Adam|SR-Adam" --num_runs 3
```

**SimpleCNN on CIFAR100 (all noise levels):**
```bash
python main.py --dataset CIFAR100 --model simplecnn --num_epochs 20 --num_runs 3
```

### **Reproducibility Settings**

- **Seeds**: Automatically incremented (1, 2, 3, ..., num_runs)
- **GPU Usage**: Automatic (CPU fallback available)
- **Batch Size Impact**: Larger batch size = more stable gradients but less frequent updates

---

## Expected Results

### Accuracy (higher is better)
- **CIFAR10, noise=0.0**: SGD ~70%, Momentum ~75%, Adam ~80%, SR-Adam ~81%
- **CIFAR100, noise=0.1**: SR-Adam typically 2-3% better than Adam

### Computation Time
- Per-seed training: ~5-10 minutes (GPU)
- Full pipeline (30 runs): ~2-4 hours
- Table/figure generation: ~5 minutes
- Paper compilation: ~30 seconds

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'torch'"
**Solution:**
```bash
conda activate pth
pip install torch torchvision
```

### Issue: GPU out of memory
**Solution:** Reduce batch_size
```bash
python main.py --batch_size 1024
```

### Issue: "run_*.csv not found"
**Solution:** Ensure experiments completed successfully. Check:
```bash
ls results/CIFAR10/simplecnn/noise_0.0/SGD/
# Should show: run_1.csv, run_1_meta.json, run_2.csv, etc.
```

### Issue: Figures PDFs missing when generating tables
**Solution:** Re-run figure generation scripts first
```bash
python make_testacc_plots_simplecnn.py
python make_loss_plots_simplecnn.py
```

---

## File Dependencies

```
main.py
├── model.py (SimpleCNN)
├── data.py (CIFAR loaders with noise)
├── training.py (train_model, evaluate)
├── optimizers.py (SGD, Momentum, Adam, SR-Adam)
└── utils.py (save, aggregate, plot)

Regeneration scripts:
├── regenerate_aggregates.py
├── regenerate_summary_statistics.py
├── regenerate_epoch_pngs.py

Table generators:
├── make_minimal_tables.py
├── generate_architecture_table.py
├── make_method_tables.py

Figure generators:
├── make_testacc_plots_simplecnn.py
├── make_loss_plots_simplecnn.py
├── make_figures.py

Paper:
└── paper_figures/paper-draft.tex
    ├── \input{simplecnn_arch_content.tex}
    ├── \input{sradam_grouping_content.tex}
    ├── \input{experimental_figures.tex}
    └── \input{minimal-tables-content.tex}
```

---

## Citation and Attribution

If you use this code, please cite:

```bibtex
@article{arashi2024sradam,
  title={Shrinkage Estimation in High-Dimensional Deep Learning: A Stein-Rule Approach to Stochastic Optimization},
  author={Arashi, M.},
  journal={Submitted},
  year={2024}
}
```

---

## License

This code is provided for research and reproducibility purposes.
