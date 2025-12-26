# CIFAR-100 Results Analysis & Improvement Roadmap

## Current Results Summary

```
CIFAR-100 (20 epochs, SimpleCNN, batch_size=512)

Adam:                      40.19 ± 0.60%
SR-Adam (Conv-only):       41.50 ± 1.20%
                           ────────────
Improvement:               +1.31 percentage points ✅
Variance Ratio:            1.20 / 0.60 = 2.0x higher ⚠️
```

### Key Observations

1. **SR-Adam Still Improves** (+1.31%) ✅
   - Maintains advantage over Adam
   - Validates Conv-only strategy on 100 classes

2. **Higher Variance Issue** ⚠️
   - SR-Adam: 1.20 (2x higher than Adam)
   - Suggests hyperparameter sensitivity
   - Indicates need for tuning on 100-class problem

3. **Absolute Performance** 🔴
   - 40-41% on CIFAR-100 is below state-of-the-art
   - SimpleCNN too simple for 100 classes
   - Need stronger baseline architecture

---

## Problem Analysis

### Why Performance Lags

| Factor | Impact | Severity |
|--------|--------|----------|
| **Simple model** | Only 2 conv layers, ~250K params | 🔴 High |
| **Limited epochs** | 20 epochs → underfitting | 🟡 Medium |
| **Fixed learning rate** | No scheduling → poor convergence | 🟡 Medium |
| **Basic augmentation** | Only HFlip + RandCrop | 🟡 Medium |
| **Large batch size** | 512 → poor gradient quality? | 🟡 Medium |
| **SR-Adam tuning** | Hyperparams optimized for CIFAR-10 | 🟡 Medium |

---

## Improvement Strategy (Priority Order)

### 🔥 **Priority 1: Increase Model Capacity** (Highest Impact)

**Problem:** SimpleCNN was designed for CIFAR-10 (10 classes)  
**Solution:** Use ResNet-18 or upgrade SimpleCNN

#### Option A: Use ResNet-18 (Recommended)

```python
import torchvision.models as models

# Load pre-trained ResNet-18
model = models.resnet18(pretrained=False)

# Adjust final layer for CIFAR-100
model.fc = nn.Linear(512, 100)

# Expected improvement: +15-20 percentage points
```

**Why ResNet-18:**
- Standard benchmark for CIFAR
- ~11M parameters (44x larger than SimpleCNN)
- Built-in skip connections prevent vanishing gradients
- Well-studied hyperparameters available

#### Option B: Enhanced SimpleCNN (Minimal Changes)

```python
class SimpleCNNEnhanced(nn.Module):
    """Improved CNN with more capacity"""
    def __init__(self, num_classes=100):
        super().__init__()
        # More filters: 32→64→128
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        
        # Larger FC layers
        self.fc1 = nn.Linear(256 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # 32x32 → 16x16
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # 16x16 → 8x8
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # 8x8 → 4x4
        x = x.view(x.size(0), -1)  # Flatten
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x
```

**Expected improvement: +5-8 percentage points**

---

### 🔥 **Priority 2: Increase Training Epochs**

**Problem:** 20 epochs insufficient for convergence  
**Solution:** Increase to 50-100 epochs

```bash
# Test different epoch counts
python main.py --dataset CIFAR100 --num_epochs 50 --num_runs 3 --optimizers "adam|sradam"
python main.py --dataset CIFAR100 --num_epochs 100 --num_runs 3 --optimizers "adam|sradam"
```

**Expected improvement: +3-5 percentage points**

---

### 🔥 **Priority 3: Learning Rate Scheduling**

**Problem:** Constant learning rate → poor convergence  
**Solution:** Implement cosine annealing or step decay

```python
# Add to training.py - in train_model() function

def get_scheduler(optimizer, num_epochs):
    """Cosine annealing schedule"""
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=num_epochs
    )

# In training loop:
scheduler = get_scheduler(optimizer, num_epochs)

for epoch in range(num_epochs):
    # ... training ...
    scheduler.step()
```

**Or step decay:**

```python
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, 
    step_size=30,      # Drop LR every 30 epochs
    gamma=0.1          # Multiply by 0.1
)
```

**Expected improvement: +5-8 percentage points**

---

### 🟡 **Priority 4: Data Augmentation**

**Problem:** Basic augmentation (HFlip, RandCrop)  
**Solution:** Add stronger augmentation

```python
# In data.py - enhance train_transform_list

train_transform_list = [
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomRotation(15),          # ← NEW
    transforms.ColorJitter(
        brightness=0.2,                     # ← NEW
        contrast=0.2, 
        saturation=0.2
    ),
    transforms.RandomAffine(
        degrees=0,
        translate=(0.1, 0.1)               # ← NEW
    ),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
    transforms.RandomErasing(p=0.5),        # ← NEW (CutOut)
]
```

**Or use AutoAugment:**

```python
from torchvision.transforms import AutoAugment, AutoAugmentPolicy

train_transform_list = [
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    AutoAugment(policy=AutoAugmentPolicy.CIFAR10),  # ← Automatic
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
]
```

**Expected improvement: +2-4 percentage points**

---

### 🟡 **Priority 5: Tune SR-Adam Hyperparameters**

**Problem:** Parameters tuned for CIFAR-10 (10 classes)  
**Solution:** Re-tune for CIFAR-100

```python
# Current settings (CIFAR-10 optimized)
warmup_steps=5
shrink_clip=(0.1, 1.0)

# Try for CIFAR-100:
# More conservative shrinkage due to 100 classes
warmup_steps=10        # Longer warmup
shrink_clip=(0.2, 1.0) # Tighter clip (less aggressive)
```

**Grid search template:**

```bash
for warmup in 5 10 20; do
  for clip_lo in 0.1 0.2 0.3; do
    python main.py \
      --dataset CIFAR100 \
      --num_epochs 50 \
      --num_runs 3 \
      --sradam_warmup $warmup \
      --sradam_clip_lo $clip_lo
  done
done
```

**Expected improvement: +1-2 percentage points**

---

### 🟡 **Priority 6: Batch Size Tuning**

**Problem:** Batch size 512 might be too large  
**Solution:** Test smaller batches for better gradient

```bash
# Test different batch sizes
for bs in 128 256 512; do
  python main.py \
    --dataset CIFAR100 \
    --batch_size $bs \
    --num_runs 3 \
    --num_epochs 50
done
```

**Trade-offs:**
- Batch 128: Better gradient quality, slower convergence
- Batch 256: Good balance (recommended)
- Batch 512: Fast but noisy

**Expected improvement: +1-3 percentage points**

---

## Recommended Quick Wins (Do First)

### Combo 1: Minimal Changes (~+8-12%)

```bash
# 1. Use ResNet-18
# 2. Increase epochs to 50
# 3. Add learning rate scheduling

python main.py \
  --dataset CIFAR100 \
  --model resnet18 \
  --num_epochs 50 \
  --use_scheduler cosine \
  --num_runs 5 \
  --optimizers "adam|sradam"
```

**Expected result:** 48-52% accuracy

### Combo 2: Maximum Improvement (~+15-20%)

```bash
# 1. ResNet-18
# 2. 100 epochs with cosine annealing
# 3. AutoAugment
# 4. Batch size 256
# 5. Tune SR-Adam params

python main.py \
  --dataset CIFAR100 \
  --model resnet18 \
  --batch_size 256 \
  --num_epochs 100 \
  --use_scheduler cosine \
  --use_augment auto \
  --num_runs 5 \
  --optimizers "adam|sradam"
```

**Expected result:** 55-60% accuracy

---

## Implementation Priority

| Step | Action | Time | Impact | Difficulty |
|------|--------|------|--------|------------|
| 1 | Add ResNet-18 option | 30 min | +15% | Easy |
| 2 | Increase epochs to 50 | 5 min | +3% | Trivial |
| 3 | Add LR scheduling | 30 min | +5% | Easy |
| 4 | Enhanced augmentation | 45 min | +3% | Medium |
| 5 | Hyperparameter search | 2 hours | +2% | Hard |
| 6 | Batch size tuning | 2 hours | +2% | Hard |

**Total time for steps 1-3: ~1 hour → +23% improvement**

---

## Code Changes Needed

### 1. Modify main.py to support model selection

```python
parser.add_argument("--model", type=str, default="simplecnn",
                    choices=["simplecnn", "resnet18"])

# In main():
if args.model == "resnet18":
    from torchvision.models import resnet18
    model = resnet18(pretrained=False)
    model.fc = nn.Linear(512, num_classes)
else:
    model = SimpleCNN(num_classes=num_classes)
```

### 2. Add learning rate scheduling

```python
def get_scheduler(optimizer, args):
    if args.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.num_epochs
        )
    elif args.scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=30, gamma=0.1
        )
    return None

# In train_model():
scheduler = get_scheduler(optimizer, args)
for epoch in range(num_epochs):
    # training...
    if scheduler:
        scheduler.step()
```

### 3. Enhanced data augmentation

```python
def get_data_loaders(dataset_name, batch_size, augment="standard"):
    if augment == "auto":
        train_transform_list.append(
            AutoAugment(policy=AutoAugmentPolicy.CIFAR10)
        )
    elif augment == "strong":
        train_transform_list.extend([
            transforms.RandomRotation(15),
            transforms.ColorJitter(0.2, 0.2, 0.2),
            transforms.RandomErasing(p=0.5),
        ])
```

---

## Variance Issue (SR-Adam)

The higher variance (1.20 vs 0.60) suggests:

1. **SR-Adam parameters need tuning for 100 classes**
   - More conservative shrinkage
   - Longer warmup period
   - Different clip bounds

2. **Batch effects**
   - Larger model needs proper gradient normalization
   - Consider gradient clipping: `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)`

3. **Recommendation**
   ```python
   # For CIFAR-100, be more conservative
   SRAdamAdaptiveLocal(
       [...],
       lr=1e-3,
       warmup_steps=10,        # ← Increased
       shrink_clip=(0.2, 1.0), # ← More conservative
   )
   ```

---

## Suggested Experiments

### Week 1: Quick Wins
```bash
# Test 1: ResNet-18 + 50 epochs
python main.py --dataset CIFAR100 --model resnet18 --num_epochs 50

# Test 2: With LR scheduling
python main.py --dataset CIFAR100 --model resnet18 --num_epochs 50 --scheduler cosine

# Test 3: With better augmentation
python main.py --dataset CIFAR100 --model resnet18 --num_epochs 50 --augment auto
```

### Week 2: Hyperparameter Search
```bash
# Test different warmup and clip settings for SR-Adam
for warmup in 5 10 15 20; do
  for clip in 0.1 0.2 0.3; do
    python main.py --dataset CIFAR100 --sradam_warmup $warmup --sradam_clip_lo $clip
  done
done
```

### Week 3: Batch Size & Scheduler
```bash
# Test batch sizes
for bs in 128 256 512; do
  python main.py --dataset CIFAR100 --model resnet18 --batch_size $bs --num_epochs 100
done
```

---

## Expected Progress

```
Current:        40-41% (SimpleCNN, 20 epochs)
After Step 1:   55-56% (ResNet-18, 50 epochs, LR schedule)
After Steps 2-3: 58-60% (+ better augmentation + tuning)
Final Goal:      65-70% (+ batch tuning + deep optimization)
```

---

## Summary: What to Do Next

### Immediate Actions (This Week)
1. ✅ Add ResNet-18 support to main.py
2. ✅ Implement cosine annealing scheduler
3. ✅ Run: `python main.py --model resnet18 --num_epochs 50`

### Short Term (Next Week)
4. ✅ Add AutoAugment
5. ✅ Tune SR-Adam hyperparameters (warmup, clip)
6. ✅ Grid search batch sizes

### Medium Term (Later)
7. ✅ Try deeper models (ResNet-50)
8. ✅ Try Vision Transformer (ViT)
9. ✅ Publish results

---

**Estimated effort for +20% improvement: 3-4 hours of coding + tuning**

Would you like me to implement any of these improvements?
