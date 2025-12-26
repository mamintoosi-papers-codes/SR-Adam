You are assisting in refactoring a research codebase for reproducible optimizer experiments.

Goal:
Run controlled experiments comparing Adam vs SR-Adam (Conv-only) under input noise,
with multiple runs and proper statistical aggregation, suitable for a conference paper.

Required experimental protocol:
- Rename all 'SR-Adam (Conv-only)' to 'SR-Adam'
- Datasets: CIFAR10 and CIFAR100
- Models:
  - CIFAR10: SimpleCNN
  - CIFAR100: ResNet (existing implementation)
- Noise levels: {0.0, 0.05, 0.1}
- Optimizers: Adam, SR-Adam
- Runs per setting: 5 (different random seeds)
- Batch size: keep existing defaults (512 unless already changed)
- Epochs: keep current value

Key requirements for code changes:

1. main.py
   - Loop over runs inside each (dataset, noise, optimizer) configuration
   - Automatically set seed = base_seed + run_id
   - Store metrics for *each run separately*
   - Do NOT overwrite previous run results

2. Result saving (utils.py or new helper):
   - Save run times
   - Save per-run metrics to CSV:
     results/{dataset}/{model}/noise_{noise}/{optimizer}/run_{k}.csv
   - After all runs:
     - Compute mean and std across runs for:
       * final test accuracy
       * best test accuracy
     - Save an aggregated summary CSV and Excel file
   - Ensure previous behavior (single-run overwrite) is fixed

3. Plotting:
   - Generate plots with mean curve and shaded std region (±1 std)
   - At minimum:
     * Test Accuracy vs Epoch (Adam vs SR-Adam)
     * Test Accuracy vs Noise (aggregated over runs)

4. Reproducibility:
   - Ensure each run logs:
     dataset, model, optimizer, noise level, seed
   - Save a config.json summarizing all experiment settings

5. Scope constraints:
   - Do NOT add new noise values beyond {0, 0.05, 0.1}
   - Do NOT add new optimizers
   - Do NOT change optimizer math
   - Only refactor for experimental rigor and result aggregation

Please update all relevant files (Readme, main.py, utils.py, training loop if needed)
so that running a single command executes the full experiment grid and
produces publication-ready results.
