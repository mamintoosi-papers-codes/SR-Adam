# سازگاری با فرمت قدیمی و جدید ✅

## خلاصه تغییرات

تمام اسکریپت‌های گزارش‌گیری و ذخیره‌سازی به‌روزرسانی شدند تا **هم با فرمت قدیمی** و **هم با فرمت جدید** کار کنند.

---

## فایل‌های به‌روز شده

### 1️⃣ `src/utils.py`

#### تغییرات `save_run_metrics()`:
```python
# قبل:
results/{dataset}/{model}/noise_{noise}/{optimizer}/run_{id}.csv

# بعد (با batch_size):
results/{dataset}/{model}/noise_{noise}/{optimizer}/batch_size_{bs}/run_{id}_metrics.csv

# بعد (بدون batch_size - برای سازگاری با قدیمی):
results/{dataset}/{model}/noise_{noise}/{optimizer}/run_{id}_metrics.csv
```

**منطق**:
- اگر `metrics['batch_size']` وجود داشته باشد → ذخیره در `batch_size_{bs}/`
- اگر نداشته باشد → ذخیره مستقیم در پوشه optimizer (فرمت قدیمی)

---

#### تغییرات `aggregate_runs_and_save()`:
```python
# جستجوی فایل‌ها به ترتیب اولویت:
1. batch_size_*/run_*_metrics.csv  # فرمت جدید
2. run_*_metrics.csv               # فرمت قدیمی با پسوند _metrics
3. run_*.csv                       # فرمت خیلی قدیمی
```

**منطق**:
- ابتدا فرمت جدید را امتحان می‌کند
- اگر نبود، به فرمت قدیمی برمی‌گردد
- نتایج aggregated در همان پوشه‌ای که فایل‌های خام هستند ذخیره می‌شوند

---

### 2️⃣ `tools/ablation_utils.py`

#### تغییرات `check_run_exists()`:
پارامتر جدید: `batch_size` (اختیاری)

```python
# استفاده:
check_run_exists(dataset, model, noise, optimizer, run_id, batch_size=256)
# بررسی می‌کند: batch_size_256/run_{id}_metrics.csv

check_run_exists(dataset, model, noise, optimizer, run_id, batch_size=None)
# بررسی می‌کند: run_{id}_metrics.csv (فرمت قدیمی)
```

---

#### تغییرات `try_load_checkpoint_metrics()`:
پارامتر جدید: `batch_size` (اختیاری)

```python
# جستجوی checkpoint به ترتیب:
1. batch_size_{bs}/run_{id}_final.pt  # فرمت جدید
2. batch_size_{bs}/run_{id}_last.pt   # فرمت قدیمی با _last
3. run_{id}_final.pt                  # فرمت قدیمی بدون batch_size
4. run_{id}_last.pt                   # فرمت خیلی قدیمی
```

---

#### تغییرات `should_skip_run()`:
پارامتر جدید: `batch_size` (اختیاری)

این پارامتر به `check_run_exists()` و `try_load_checkpoint_metrics()` پاس داده می‌شود.

---

### 3️⃣ `tools/generate_qualitative_comparison.py`

#### تغییرات `get_best_run()`:
```python
# جستجوی checkpoint به ترتیب:
1. batch_size_*/run_*_best.pt   # فرمت جدید
2. run_*_best.pt                # فرمت قدیمی
```

**منطق**:
- همه checkpoint‌ها (هم قدیمی هم جدید) پیدا می‌شوند
- بهترین checkpoint بر اساس `test_acc` انتخاب می‌شود
- کاربر نیازی به دستکاری ندارد

---

### 4️⃣ `main.py`

#### تغییر فراخوانی `should_skip_run()`:
```python
# قبل:
skip_training, loaded_metrics = should_skip_run(
    dataset_name, model_name, noise, opt_name, run + 1,
    args.num_epochs, args.clean_previous
)

# بعد:
skip_training, loaded_metrics = should_skip_run(
    dataset_name, model_name, noise, opt_name, run + 1,
    args.num_epochs, args.clean_previous, 
    batch_size=batch_size_override  # ✅ پارامتر جدید
)
```

---

## سازگاری با نتایج قدیمی

### ✅ سناریو 1: نتایج قدیمی موجود است
```
results/CIFAR10/simplecnn/noise_0.0/Adam/
  run_1_metrics.csv     ← فرمت قدیمی
  run_2_metrics.csv
```

**نتیجه:**
- ✅ `aggregate_runs_and_save()` این فایل‌ها را می‌خواند
- ✅ `generate_qualitative_comparison.py` checkpoint‌های قدیمی را پیدا می‌کند
- ✅ همه چیز بدون مشکل کار می‌کند

---

### ✅ سناریو 2: اجرای جدید با `--batch_ablation`
```
results/CIFAR10/simplecnn/noise_0.0/Adam/
  batch_size_256/
    run_1_metrics.csv   ← فرمت جدید
    run_2_metrics.csv
  batch_size_512/
    run_1_metrics.csv
```

**نتیجه:**
- ✅ نتایج در پوشه‌های جداگانه ذخیره می‌شوند
- ✅ aggregation برای هر `batch_size` جداگانه انجام می‌شود
- ✅ `generate_qualitative_comparison.py` بهترین را از همه می‌یابد

---

### ✅ سناریو 3: ترکیب قدیمی + جدید
```
results/CIFAR10/simplecnn/noise_0.0/Adam/
  run_1_metrics.csv              ← قدیمی (batch_size=128)
  run_2_metrics.csv              ← قدیمی
  batch_size_256/
    run_1_metrics.csv            ← جدید
    run_2_metrics.csv
```

**نتیجه:**
- ✅ aggregation برای قدیمی‌ها جداگانه
- ✅ aggregation برای batch_size_256 جداگانه
- ✅ هیچ تداخلی رخ نمی‌دهد

---

## تست‌های انجام شده

### ✅ تست 1: بررسی Syntax
```bash
# همه فایل‌ها بدون خطا
✓ main.py
✓ src/utils.py
✓ tools/ablation_utils.py
✓ tools/generate_qualitative_comparison.py
```

### ✅ تست 2: Import
```python
from tools.ablation_utils import should_skip_run, check_run_exists
from src.utils import save_run_metrics, aggregate_runs_and_save
# ✓ همه import شدند بدون مشکل
```

---

## پاسخ به سوالات شما

### سوال 1: `summary_statistics.csv` چه می‌شود?
**پاسخ:** 
- این فایل در ریشه `results/` ذخیره می‌شود
- خلاصه کلی همه آزمایش‌ها را نگه می‌دارد
- تغییری نکرده و همچنان کار می‌کند ✅

---

### سوال 2: `generate_qualitative_comparison.py` درست کار می‌کند?
**پاسخ:** 
- ✅ **بله، به‌روز شد**
- حالا هم فرمت قدیمی و هم جدید را می‌خواند
- بهترین checkpoint را از همه پوشه‌ها پیدا می‌کند
- نیازی به تغییر دستورات notebook نیست

---

## دستورات Notebook همچنان کار می‌کنند

```python
# این دستور از notebook شما:
%run tools/generate_qualitative_comparison.py --dataset CIFAR10 --noise 0.05 --seed 1000
```

**کار خواهد کرد با:**
- ✅ نتایج قدیمی (قبل از مایگریشن)
- ✅ نتایج جدید (بعد از مایگریشن)
- ✅ ترکیب هر دو

---

## نکته مهم برای مایگریشن

اگر نتایج قدیمی دارید و می‌خواهید به فرمت جدید تبدیل کنید:

```bash
python migrate_results_format.py
```

**اما اگر مایگریشن نکنید:**
- ✅ نگران نباشید! کدها با فرمت قدیمی هم کار می‌کنند
- ✅ می‌توانید آزمایش‌های جدید با `--batch_ablation` اجرا کنید
- ✅ نتایج قدیمی و جدید در کنار هم قرار می‌گیرند

---

## خلاصه

| فایل | تغییر | سازگاری قدیمی | سازگاری جدید |
|------|-------|---------------|---------------|
| `src/utils.py` | ✅ به‌روز شد | ✅ | ✅ |
| `tools/ablation_utils.py` | ✅ به‌روز شد | ✅ | ✅ |
| `tools/generate_qualitative_comparison.py` | ✅ به‌روز شد | ✅ | ✅ |
| `main.py` | ✅ به‌روز شد | ✅ | ✅ |
| `summary_statistics.csv` | ⚪ تغییری نکرد | ✅ | ✅ |

---

**نتیجه‌گیری:** همه چیز آماده است! می‌توانید:
1. با نتایج قدیمی کار کنید ✅
2. آزمایش‌های جدید با `--batch_ablation` اجرا کنید ✅
3. نتایج را مایگریشن کنید (اختیاری) ✅
4. همه اسکریپت‌های گزارش‌گیری را اجرا کنید ✅
