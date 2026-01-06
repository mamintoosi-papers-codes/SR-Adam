# Minimal Reproducibility Checklist

## For Full Reproducibility (Re-running All Experiments)

Use one of these methods:

### **Windows Users:**
```bash
cd c:\git\mamintoosi-papers-codes\SR-Adam
reproduce_all.bat
```

**What happens:**
- Runs all noise levels [0.0, 0.05, 0.1] **automatically** in `main.py`
- Tests all datasets (CIFAR10, CIFAR100)
- Tests all optimizers (SGD, Momentum, Adam, SR-Adam)
- 5 independent runs per configuration
- Time: ~2-4 hours on GPU

### **Linux/Mac Users:**
```bash
cd /path/to/SR-Adam
bash reproduce_all.sh
```

### **Single Manual Command (all noise levels tested automatically):**
```bash
conda activate pth
python main.py --dataset ALL --model simplecnn --optimizers ALL --num_epochs 20 --num_runs 5 --batch_size 512
```

### **Detailed Step-by-Step:**
See `REPRODUCE.md` for full documentation

---

## For Quick Table/Figure Regeneration (Without Re-running Experiments)

If experiments already exist in `results/`, regenerate tables and figures only:

```bash
# Activate environment
conda activate pth

# Regenerate all outputs
python tools/regenerate_aggregates.py
python tools/regenerate_summary_statistics.py
python tools/regenerate_epoch_pngs.py
python tools/make_minimal_tables.py
python tools/generate_architecture_table.py
python tools/make_testacc_plots_simplecnn.py
python tools/make_loss_plots_simplecnn.py
python tools/make_figures.py

# Compile paper
cd paper
pdflatex -interaction=nonstopmode paper-draft.tex
pdflatex -interaction=nonstopmode paper-draft.tex
```

---

## Key Information

⚠️ **IMPORTANT: Noise Levels Are Automatic**
- Do NOT use `--noise` argument
- `main.py` automatically processes [0.0, 0.05, 0.1] in each execution
- All noise levels tested in a single command

✅ **Reproducibility Guarantees:**
- Fixed random seeds (42+)
- Deterministic PyTorch operations
- CUDA seedable backend

---

## Expected Outputs

### Experiments Folder Structure
```
results/
├── CIFAR10/simplecnn/noise_0.{0,05,1}/{SGD,Momentum,Adam,SR-Adam}/
│   ├── run_1.csv to run_5.csv
│   └── run_1_meta.json to run_5_meta.json
├── CIFAR100/simplecnn/noise_0.{0,05,1}/{SGD,Momentum,Adam,SR-Adam}/
│   ├── run_1.csv to run_5.csv
│   └── run_1_meta.json to run_5_meta.json
└── summary_statistics.csv (root-level)
```

### Publication Outputs
```
paper/
├── minimal-tables.tex (standalone, compilable)
├── minimal-tables-content.tex (for \input{})
├── simplecnn_arch.tex
├── simplecnn_arch_content.tex
├── sradam_grouping_content.tex
├── experimental_figures.tex
├── cifar10_noise0.{0,05,1}_acc_mean_std.pdf
├── cifar10_noise0.{0,05,1}_loss_mean_std.pdf
├── cifar100_noise0.{0,05,1}_acc_mean_std.pdf
├── cifar100_noise0.{0,05,1}_loss_mean_std.pdf
└── paper-draft.pdf (final compiled paper)
```

---

## Typical Runtime

- **Full pipeline (experiments + tables + paper):** 24 hours (GPU-dependent)
- **Regenerate tables/figures:** 5-10 minutes
- **Paper compilation:** 30 seconds

---

## Reproducibility Notes

### Version Pinning
If you need exact reproducibility across dates:

```bash
pip install torch==2.0.0 torchvision==0.15.0
pip freeze > requirements.txt
```

### Random Seeds
All experiments use 5 different random seeds (1-5):
```python
# In main.py
for seed in [1, 2, 3, 4, 5]:
    set_seeds(seed)
    # Run training
```

### Hardware Variation
- Results may vary slightly on different GPU models/drivers
- CPU-only mode is supported but **very slow**
- Batch size affects stochasticity; default is 2048

---

## Verification Checklist

After running the full pipeline:

- [ ] `results/CIFAR10/simplecnn/noise_0.0/SGD/` contains 5 `run_*.csv` files
- [ ] `results/summary_statistics.csv` has 24 rows (2 datasets × 3 noises × 4 methods)
- [ ] `paper/paper-draft.pdf` exists and opens
- [ ] Paper contains all 4 figures (CIFAR10/100 × Acc/Loss)
- [ ] Paper contains all 5 tables (architecture, grouping, best acc/loss, final acc/loss)
- [ ] No "undefined reference" warnings in LaTeX compilation

---

## Troubleshooting

### Q: How do I re-run only CIFAR10 experiments?
```bash
conda activate pth
python main.py --dataset CIFAR10
```
Note: All noise levels [0.0, 0.05, 0.1] run automatically


### Q: How do I change noise levels?
Edit the grid in `main.py` line ~180:
```python
NOISE_LEVELS = [0.0, 0.05, 0.1]  # Modify here
```

### Q: Can I run experiments on CPU?
```bash
# Set CUDA_VISIBLE_DEVICES=""
set CUDA_VISIBLE_DEVICES=
python main.py --dataset CIFAR10 --noise 0.0
# WARNING: This will take 10-20x longer
```

### Q: How do I revert to the original results?
```bash
git checkout results/  # Restore from version control
python tools/regenerate_aggregates.py
python tools/make_minimal_tables.py
```


---

## Citation

If you publish results using this pipeline, cite:

```bibtex
@misc{sradam2024code,
  title={SR-Adam: Stein-Rule Optimization for Deep Learning},
  author={Arashi, M.},
  year={2024},
  howpublished={\url{https://github.com/mamintoosi-papers-codes/SR-Adam}}
}
```

## Advanced: Filtering and Custom Analysis

### Running Experiments with Multiple Parameters

You can now specify multiple values for `--batch_size`, `--noise`, and `--optimizers`:

```bash
# Multiple batch sizes (ablation study)
python main.py \
  --dataset CIFAR10 \
  --noise 0.05 \
  --batch_size "256|512|2048" \
  --optimizers "adam|sradam" \
  --num_runs 3

# Multiple noise levels
python main.py \
  --dataset CIFAR10 \
  --noise "0.0|0.05" \
  --batch_size 512 \
  --optimizers "adam|sradam"

# Full grid search
python main.py \
  --dataset ALL \
  --noise ALL \
  --batch_size "512|2048" \
  --optimizers ALL
```
