# Quick Reference: Batch Ablation & Statistical Testing

## One-Liner Commands

### Default Run (No Ablation)
```bash
python main.py --dataset CIFAR10 --num_epochs 200 --num_runs 5
```

### With Batch Ablation
```bash
python main.py --dataset CIFAR10 --num_epochs 200 --num_runs 5 --batch_ablation
```

### With All Optimizer Variants
```bash
python main.py --dataset CIFAR10 --batch_ablation --optimizers "Adam|SR-Adam|SR-Adam-All-Weights"
```

### Both Datasets + All Ablations (Full Paper)
```bash
python main.py --dataset ALL --batch_ablation --num_runs 5
```

### Resume Interrupted Run
```bash
python main.py --dataset CIFAR10 --batch_ablation --num_runs 5
# Run same command again to continue
```

### Clean and Restart
```bash
python main.py --dataset CIFAR10 --batch_ablation --clean_previous --num_runs 5
```

---

## Batch Sizes Tested (with `--batch_ablation`)

| Optimizer | Batch Sizes |
|-----------|------------|
| SGD | 128 (default only) |
| Momentum | 128 (default only) |
| Adam | 128, 256, 512, 2048 |
| SR-Adam | 128, 256, 512, 2048 |
| SR-Adam-All-Weights | 128 (default only) |

---

## Output Files

### Per-Run Metrics
```
results/
  CIFAR10/
    simplecnn/
      noise_0.0/
        Adam/
          batch_size_256/
            run_1_metrics.csv     ← Final test acc, train loss per epoch
            run_1_metrics.json    ← Metadata + ablation info
            run_2_metrics.csv
            ...
```

### Aggregated Summary (with p-values)
```
results/
  CIFAR10/
    simplecnn/
      noise_0.0/
        Adam/
          batch_size_256/
            aggregated_summary.json  ← Mean/Std + statistical tests
            aggregated_results.xlsx  ← Formatted table
            test_acc_epoch_mean_std.png  ← Plot
```

---

## Reading the Results

### aggregated_summary.json Structure
```json
{
  "final_mean": 0.823,          // Average final test accuracy
  "final_std": 0.012,           // Std dev of final test accuracy
  "best_mean": 0.845,           // Average best test accuracy
  "best_std": 0.008,            // Std dev of best test accuracy
  "num_runs": 5,                // Number of runs
  "statistical_tests": [
    {
      "optimizer_1": "Adam",
      "optimizer_2": "SR-Adam",
      "t_test_pvalue": 0.0234,  // p-value from paired t-test
      "wilcoxon_pvalue": 0.0625 // p-value from Wilcoxon test
    }
  ]
}
```

### Interpreting P-Values
- **p < 0.01**: Highly significant difference (99% confidence)
- **p < 0.05**: Significant difference (95% confidence)  ← Industry standard
- **p >= 0.05**: Not significantly different

---

## Common Tasks

### Find Best Batch Size for Adam
```bash
python main.py --dataset CIFAR10 --batch_ablation --num_runs 5 --optimizers "Adam"
# Compare results in: results/CIFAR10/simplecnn/noise_0.0/Adam/batch_size_*/aggregated_summary.json
```

### Compare SR-Adam Variants (Conv-Only vs All-Weights)
```bash
python main.py --dataset CIFAR10 --num_runs 5 --optimizers "SR-Adam|SR-Adam-All-Weights"
# Check p-values in aggregated_summary.json
```

### Get Statistical Significance for Paper
```bash
python main.py --dataset ALL --batch_ablation --num_runs 10
# Larger --num_runs = smaller p-values (better statistical power)
```

### Quick Test (Before Full Run)
```bash
python main.py --dataset CIFAR10 --num_epochs 5 --num_runs 1 --batch_ablation
# Takes ~5 min, verifies setup works
```

---

## File Locations

| File | Purpose |
|------|---------|
| `main.py` | Main entry point (orchestrates experiments) |
| `tools/ablation_utils.py` | Batch ablation & result reuse logic |
| `tools/stat_testing.py` | Statistical significance testing |
| `src/utils.py` | Metrics saving & aggregation |
| `ABLATION_STUDIES.md` | Detailed guide (read for advanced topics) |
| `IMPLEMENTATION_SUMMARY.md` | Technical implementation notes |

---

## Troubleshooting

### Results Not Being Reused
```bash
# Check if CSV exists:
ls results/CIFAR10/simplecnn/noise_0.0/Adam/run_1_metrics.csv

# If missing, force re-run:
python main.py --clean_previous --dataset CIFAR10 --num_runs 1
```

### Batch Ablation Not Running
```bash
# Verify flag is working:
python main.py --help | grep batch_ablation

# Run with verbosity:
python main.py --dataset CIFAR10 --num_epochs 1 --num_runs 1 --batch_ablation
```

### Out of Memory with Large Batch Sizes
```bash
# Option 1: Reduce batch sizes in tools/ablation_utils.py line 28
# Option 2: Run fewer runs at once:
python main.py --dataset CIFAR10 --batch_ablation --num_runs 1

# Option 3: Run optimizers separately:
python main.py --dataset CIFAR10 --batch_ablation --optimizers "Adam"
python main.py --dataset CIFAR10 --batch_ablation --optimizers "SR-Adam"
```

---

## Dataset Notes

- **CIFAR10**: 50K training + 10K test images, 10 classes
- **CIFAR100**: 50K training + 10K test images, 100 classes
- **Noise**: Controlled label noise, configurable via `--noise` parameter
- **Default**: noise_std=0.0 (clean labels)

---

## Performance Expectations

| Configuration | Est. Time | GPU Memory |
|---|---|---|
| 1 optimizer, 1 run, 50 epochs | 5 min | 2GB |
| 1 optimizer, 5 runs, 50 epochs | 25 min | 2GB |
| With batch ablation (2 ops, 3 sizes, 5 runs) | 2-3 hours | 2GB |
| Full paper (5 ops, batch ablation, 5 runs) | 15-20 hours | 2GB |

---

## Paper-Ready Commands

### For Methodology Section
```bash
python main.py --dataset CIFAR10 --batch_ablation --num_runs 5 --optimizers "Adam|SR-Adam"
```
**Output**: Comparison table for batch size impact on Adam/SR-Adam

### For Results Section
```bash
python main.py --dataset ALL --batch_ablation --num_runs 10 --optimizers "SGD|Momentum|Adam|SR-Adam|SR-Adam-All-Weights"
```
**Output**: Comprehensive results with p-values for all comparisons

### For Supplementary Material
```bash
python main.py --dataset CIFAR10 --batch_ablation --noise 0.1 --num_runs 5
```
**Output**: Results under label noise for robustness analysis

---

## Next Steps After Running

1. **Check Results**: `cat results/CIFAR10/simplecnn/noise_0.0/Adam/batch_size_256/aggregated_summary.json`
2. **Plot Results**: Use existing plotting scripts with aggregated files
3. **Statistical Analysis**: Extract p-values for paper tables
4. **Visualize**: Use `aggregated_results.xlsx` for formatting

---

**For detailed information, see `ABLATION_STUDIES.md`**

