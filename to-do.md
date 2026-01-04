
> **Context:**
> This repository already contains a working experimental pipeline for comparing optimizers (SGD, Momentum, Adam, SR-Adam) on CIFAR10/CIFAR100, with multiple noise levels and multiple independent runs.
> The code already saves per-run CSV/JSON results and aggregated summaries.
>
> **Important constraint:**
> Keep the repository clean.
> **Do NOT** introduce extra logging files, markdown reports, debug dumps, or new documentation artifacts.
> Only modify existing Python code where strictly necessary.

---

## 🎯 Objective

Make **minimal, controlled code changes** to support:

1. Lightweight ablation studies:

   * SR-Adam conv-only vs SR-Adam all-weights
   * Small batch-size sensitivity study
2. Reuse of previously computed results (no redundant training)
3. Statistical significance testing (paper-grade)
4. Clean, reproducible, paper-ready aggregated outputs

---

## ✅ Hard constraints (must follow)

* ❌ Do NOT create new log files, markdown files, or report files
* ❌ Do NOT add verbose printouts or debug dumps
* ❌ Do NOT restructure directories
* ❌ Do NOT refactor the training loop or optimizer logic
* ❌ Do NOT add new datasets or noise levels

---

## 🔧 Required changes

### 1. Optimizer ablation (minimal)

* Keep existing optimizer names unchanged.
* Add **exactly one** additional optimizer option:

  * `"SR-Adam-All-Weights"`
* This must reuse the existing SR-Adam implementation.
* The only difference:

  * Conv-only: Stein applied only to Conv layers
  * All-weights: Stein applied to all parameters
* Do NOT duplicate optimizer code.

---

### 2. Batch-size ablation (minimal & controlled)

* Add a **small, fixed batch-size grid**, e.g.:

  ```python
  batch_size_list = [256, 512, 2048]
  ```
* Run batch-size ablation **only for**:

  * Adam
  * SR-Adam (conv-only)
* Do NOT combine batch-size ablation with:

  * multiple noise levels
  * multiple models
* Clearly tag these runs in metadata:

  ```json
  "ablation": "batch_size"
  ```

---

### 3. Result reuse (critical)

* Before training any run:

  * Check whether the corresponding `run_k.csv` and `run_k_meta.json` already exist.
  * If they exist:

    * Load results
    * Skip training
* Respect the existing `--clean_previous` flag to override this behavior.
* This logic should be simple and local (no new caching system).

---

### 4. Statistical significance testing

* Add a **minimal statistical testing utility** :

  * Paired t-test
  * Wilcoxon signed-rank test
* Apply tests only to:

  * Adam vs SR-Adam (conv-only)
  * Adam vs SR-Adam (all-weights)
* Use final test accuracy across runs.
* Save results:

  * In the existing aggregated summary outputs only
  * No new standalone files

---

### 5. Aggregated outputs (paper-ready, minimal)

* Extend current aggregation to include:

  * mean ± std
  * best accuracy
  * p-values (t-test + Wilcoxon)
  * batch size (if ablation run)
* Output formats must remain:

  * CSV
  * Excel
  * JSON (already used)
* Do NOT add new plotting code.

---

## 🧪 Ablation study scope (explicit)

Include **only** these ablations:

1. SR-Adam conv-only vs SR-Adam all-weights
2. Batch size sensitivity (small vs large) for Adam and SR-Adam conv-only

Exclude:

* More noise levels
* More batch sizes
* Hyperparameter sweeps
* Architectural changes

---

## ✅ Expected result

After these changes:

* The code can reproduce all existing experiments
* New ablation results integrate seamlessly with old ones
* No clutter is added to the repository
* Outputs are directly usable in:

  * Experimental Results
  * Ablation Study
  * Statistical Significance subsection of a paper

---

### ⚠️ Priority

Minimalism, reproducibility, and clarity are higher priority than extensibility or performance.

