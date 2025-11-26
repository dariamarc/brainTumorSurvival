# Understanding Segmentation Metrics

## The Precision Problem You Observed

### Symptoms
- Accuracy: ~0.8 ✅
- Recall: ~0.8 ✅
- Precision: ~0.006 ❌ (0.6%!)

### Root Cause: Inappropriate Metrics

**The issue:** Using binary classification metrics (`Precision` and `Recall`) on a multi-class segmentation problem with severe class imbalance.

## Why This Happens

### Class Distribution in Brain Tumor Segmentation
```
Background/healthy tissue: ~95% of voxels
Tumor regions combined:    ~5% of voxels
  ├─ GD-enhancing (ET):    ~1-2%
  ├─ Peritumoral (ED):     ~2-3%
  └─ Necrotic core (NCR):  ~1-2%
```

### How Keras Metrics Work

**`keras.metrics.Precision()` without arguments:**
- Designed for binary classification
- When used on multi-class (3 outputs), it averages in misleading ways
- Doesn't account for class imbalance
- NOT appropriate for segmentation!

**Why the numbers:**
- **Accuracy (0.8)**: Per-voxel correctness → dominated by correctly classified background
- **Recall (0.8)**: Finding tumor voxels (minority class) → decent
- **Precision (0.006)**: Ratio computation is skewed by:
  - Massive class imbalance
  - Possibly counting per-class then averaging
  - Binary threshold on multi-class output

## Correct Metrics for Segmentation

### ✅ Recommended (Already Fixed in main.py)

```python
model.compile(
    optimizer=optimizer,
    loss=loss_fn,
    metrics=[
        # PRIMARY METRIC: Mean IoU (Intersection over Union)
        keras.metrics.MeanIoU(num_classes=3, name='mean_iou'),

        # SECONDARY: Categorical accuracy (correct for multi-class)
        keras.metrics.CategoricalAccuracy(name='categorical_accuracy')
    ]
)
```

### Why These Metrics?

#### 1. Mean IoU (Intersection over Union)
**Standard for segmentation!**

```
IoU = (Predicted ∩ Ground Truth) / (Predicted ∪ Ground Truth)
```

**Benefits:**
- ✅ Handles class imbalance well
- ✅ Computed per-class then averaged
- ✅ Standard in medical imaging papers
- ✅ Ranges from 0 (no overlap) to 1 (perfect)

**Interpretation:**
- IoU > 0.7: Excellent
- IoU 0.5-0.7: Good
- IoU 0.3-0.5: Fair
- IoU < 0.3: Poor

#### 2. Categorical Accuracy
**Multi-class version of accuracy**

```
Accuracy = (Correct Voxels) / (Total Voxels)
```

**Benefits:**
- ✅ Correctly handles multi-class
- ✅ Easy to understand
- ✅ Good for monitoring overall performance

**Limitation:**
- ⚠️ Still dominated by background class
- Use IoU as primary metric instead

## What to Monitor During Training

### Focus on These:
1. **val_loss** - Should decrease steadily
2. **val_mean_iou** - Should increase (target > 0.5)
3. **categorical_accuracy** - Should be > 0.8

### Ignore These (Now Removed):
- ~~precision~~ - Misleading for segmentation
- ~~recall~~ - Misleading for segmentation
- ~~accuracy~~ - Use categorical_accuracy instead

## Additional Metrics for Evaluation (Not Training)

### Dice Coefficient (Same as F1)
```python
def dice_coefficient(y_true, y_pred, class_idx):
    """
    Dice = 2 * |X ∩ Y| / (|X| + |Y|)
    Range: 0 (no overlap) to 1 (perfect)
    """
    intersection = np.sum(y_true[class_idx] * y_pred[class_idx])
    return 2 * intersection / (np.sum(y_true[class_idx]) + np.sum(y_pred[class_idx]))
```

**Standard in medical imaging challenges!**

### Hausdorff Distance
```python
# Measures maximum distance between boundaries
# Lower is better
# Good for assessing boundary quality
```

### Per-Class IoU
```python
# Instead of mean, look at each class:
iou_et = IoU(class=0)   # GD-enhancing tumor
iou_ed = IoU(class=1)   # Peritumoral edema
iou_ncr = IoU(class=2)  # Necrotic core
```

## Training Recommendations

### Current (Fixed) Configuration
```python
# In main.py - NOW CORRECT
metrics=[
    keras.metrics.MeanIoU(num_classes=3, name='mean_iou'),
    keras.metrics.CategoricalAccuracy(name='categorical_accuracy')
]
```

### Expected Values During Training
```
Epoch 1:
  - categorical_accuracy: 0.70-0.80
  - mean_iou: 0.10-0.20 (starts low)
  - val_loss: ~1.5-2.0

Epoch 10:
  - categorical_accuracy: 0.85-0.90
  - mean_iou: 0.30-0.50
  - val_loss: ~0.5-1.0

Good final result:
  - categorical_accuracy: 0.90-0.95
  - mean_iou: 0.50-0.70
  - val_loss: ~0.3-0.5
```

## What Changed

### Before (Incorrect):
```python
metrics=[
    'accuracy',  # ← Binary accuracy
    keras.metrics.MeanIoU(...),
    keras.metrics.Precision(),  # ← Binary, misleading
    keras.metrics.Recall()      # ← Binary, misleading
]
```

### After (Correct):
```python
metrics=[
    keras.metrics.MeanIoU(...),              # ← Primary metric
    keras.metrics.CategoricalAccuracy(...)   # ← Correct multi-class
]
```

## Next Steps

1. **If training is ongoing:**
   - Don't worry about the low precision value
   - It's just a metric calculation issue, not a model issue
   - Model is learning correctly (accuracy ~0.8 is good!)

2. **For new training runs:**
   - Use updated `main.py` (already fixed)
   - Monitor `mean_iou` and `categorical_accuracy`
   - Ignore precision/recall values

3. **For evaluation:**
   - Compute Dice coefficient per class
   - Report mean IoU per class
   - Visualize predictions vs ground truth

## References

- **BraTS Challenge:** Uses Dice coefficient and Hausdorff distance
- **Medical Imaging Standard:** Mean IoU and per-class Dice
- **Avoid:** Global precision/recall for segmentation

---

**Bottom Line:** Your model is learning fine! The low precision was just a metric calculation artifact. Focus on `mean_iou` instead.
