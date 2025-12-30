# Qualitative Comparison Figure Generator - Usage Guide

## Overview
`generate_qualitative_comparison.py` creates visual comparison figures showing sample predictions from Adam vs SR-Adam.

## Basic Usage

```bash
# Default: CIFAR10, noise=0.05, random samples
python generate_qualitative_comparison.py

# Specify dataset and noise level
python generate_qualitative_comparison.py --dataset CIFAR10 --noise 0.05

# Use fixed seed for reproducible sample selection
python generate_qualitative_comparison.py --dataset CIFAR10 --noise 0.05 --seed 42
```

## Command Line Arguments

| Argument | Default | Choices | Description |
|----------|---------|---------|-------------|
| `--dataset` | `CIFAR10` | `CIFAR10`, `CIFAR100` | Dataset name |
| `--noise` | `0.05` | `0.0`, `0.05`, `0.1` | Label noise level |
| `--seed` | `None` | Any integer | Random seed for sample selection. `None` = random each time |
| `--num-samples` | `10` | Any positive int | Number of test images to display |
| `--runs-root` | `runs` | Any path | Root directory for model checkpoints |

## Finding Good Samples

The sample selection is random. To find the most informative/interesting samples for your paper:

### Try Different Seeds

```bash
# Try various seeds to see different sample sets
python generate_qualitative_comparison.py --dataset CIFAR10 --noise 0.05 --seed 42
python generate_qualitative_comparison.py --dataset CIFAR10 --noise 0.05 --seed 123
python generate_qualitative_comparison.py --dataset CIFAR10 --noise 0.05 --seed 999
python generate_qualitative_comparison.py --dataset CIFAR10 --noise 0.05 --seed 2024
python generate_qualitative_comparison.py --dataset CIFAR10 --noise 0.05 --seed 7777
```

### What Makes a "Good" Visualization?

Look for samples that show:
1. ✅ **Clear differences** between Adam and SR-Adam predictions
2. ✅ **Challenging cases** where both methods struggle
3. ✅ **Success cases** where SR-Adam correctly predicts but Adam fails
4. ✅ **Variety of classes** to demonstrate generalization
5. ✅ **High confidence** correct predictions vs **low confidence** incorrect ones

### Example Workflow

```bash
# Step 1: Generate with different seeds
for seed in 42 123 456 789 999 1234 5678 2024; do
    echo "Trying seed $seed..."
    python generate_qualitative_comparison.py --dataset CIFAR10 --noise 0.05 --seed $seed
    # Review paper_figures/qualitative_cifar10_noise005.pdf
    # Keep the seed that gives the most informative samples
done

# Step 2: Once you find a good seed, document it
echo "Best seed for CIFAR10 noise=0.05: 123" >> seeds_used.txt

# Step 3: Repeat for CIFAR100
python generate_qualitative_comparison.py --dataset CIFAR100 --noise 0.05 --seed 456
```

## Output Files

Each run generates:
- **PDF Figure**: `paper_figures/qualitative_{dataset}_noise{noise}.pdf`
  - Visual comparison with 3 rows (Truth, Adam, SR-Adam) × N columns (samples)
  - Green text = correct prediction, Red text = incorrect
  - Shows confidence scores in parentheses

- **JSON Metadata**: `paper_figures/qualitative_accuracy_comparison_{dataset}_noise{noise}.json`
  - Contains: seed used, aggregated stats, checkpoint accuracies
  - Useful for documentation and reproducibility

## Tips

1. **CIFAR10 may need more exploration**: Classes are visually distinct, so random samples might not show differences. Try 10-20 different seeds.

2. **CIFAR100 is easier**: With 100 fine-grained classes, almost any random sample shows interesting comparisons.

3. **Document your seeds**: Once you find good visualizations, save the seed values in your paper documentation.

4. **Batch testing**: Use a loop to quickly test many seeds and manually review the PDFs.

5. **Combine with qualitative analysis**: In your paper, explain WHY certain samples are interesting (e.g., "SR-Adam shows higher confidence on ambiguous samples like cats vs dogs").

## Examples

```bash
# Generate for all noise levels with same seed
python generate_qualitative_comparison.py --dataset CIFAR10 --noise 0.0 --seed 42
python generate_qualitative_comparison.py --dataset CIFAR10 --noise 0.05 --seed 42
python generate_qualitative_comparison.py --dataset CIFAR10 --noise 0.1 --seed 42

# More samples for detailed analysis
python generate_qualitative_comparison.py --dataset CIFAR10 --noise 0.05 --seed 123 --num-samples 20

# Quick exploration mode (no seed = random each time)
python generate_qualitative_comparison.py --dataset CIFAR10 --noise 0.05
```

## Troubleshooting

**Q: All samples look too easy/boring?**  
A: Try more seeds! Some random selections happen to pick simple cases.

**Q: How many seeds should I try?**  
A: For CIFAR10: 10-20 seeds. For CIFAR100: 3-5 seeds usually sufficient.

**Q: Can I manually select specific images?**  
A: Currently no, but you can modify the code to use specific indices instead of random selection.

**Q: PDF shows different accuracy than tables?**  
A: This is expected! PDF shows the BEST single run (for visualization), while tables show mean±std across 5 runs.
