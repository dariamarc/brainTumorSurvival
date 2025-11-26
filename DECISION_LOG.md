# Decision Log - Brain Tumor Segmentation with MProtoNet3D

This document tracks important architectural decisions and their rationale for the brain tumor segmentation project.

---

## Decision 1: Fix Prototype Learning (2024-11-25)

### Problem Identified
The prototypes in the model were being computed but not used in the forward pass or loss calculation. This resulted in:
- Prototypes remaining at random initialization
- No gradients flowing back to update prototypes
- Model essentially functioning as a standard U-Net without prototype learning
- Visualizations showing black/empty prototype patches

**Evidence:**
- `model.py` lines 181-188: Prototypes computed but stored in unused variable
- `model.py` line 191: Decoder uses `f_processed` directly, bypassing prototypes
- Prototype visualization showed black images (random/uninformative features)

### Solution Options

#### Option 1: Use Prototypes in Decoder (SELECTED - IMPLEMENTED - FIXED)

**Approach:**
Integrate prototype similarities directly into the decoder path so they influence the output and receive gradients.

**Implementation (REVISED):**
```python
# In __init__:
self.prototype_to_features = layers.Conv3D(
    128, kernel_size=1,
    activation='relu',
    name='prototype_to_features'
)

# In call():
# After computing prototype similarities
prototype_features = self.prototype_to_features(prototype_voxel_similarities)

# CRITICAL FIX: Combine with original features (don't replace!)
combined_features = layers.add([f_processed, prototype_features])
up = combined_features  # Use combined features
```

**⚠️ Critical Fix (2024-11-25):**
Initial implementation replaced original features entirely with prototype features, causing severe information bottleneck:
- Accuracy dropped to 0.01 (1%)
- Precision at 0.01, Recall at 0.8 (predicting everything as one class)
- Problem: 128 features → 21 prototypes → 128 features loses information

**Solution:** Use element-wise addition to combine both:
- Preserves original feature information
- Adds prototype information on top
- Allows gradual learning: model can rely on original features initially
- Prototypes contribute more as they learn meaningful patterns

**Pros:**
- ✅ Simple integration with existing architecture
- ✅ Prototypes directly influence output
- ✅ Automatic gradient flow through backpropagation
- ✅ Minimal architectural changes
- ✅ Compatible with existing training pipeline

**Cons:**
- ❌ May not enforce prototype diversity
- ❌ No explicit clustering/separation constraints
- ❌ Prototypes might learn redundant features

**Expected Outcome:**
- Prototypes will learn to represent discriminative patterns
- Each prototype should capture features relevant to segmentation
- Gradient updates will optimize prototypes for loss minimization

---

#### Option 2: Add Prototype-Specific Losses (ALTERNATIVE - NOT YET IMPLEMENTED)

**Approach:**
Add auxiliary losses inspired by ProtoPNet to enforce prototype properties while keeping the current architecture.

**Proposed Implementation:**

1. **Cluster Loss** - Encourages each training patch to be close to at least one prototype of its class:
```python
def cluster_loss(prototype_distances, labels, prototype_class_identity):
    """
    For each patch, find minimum distance to a prototype of the correct class.
    Encourages prototypes to cluster around class-specific features.
    """
    # For each class, get distances to prototypes of that class
    # Minimize the minimum distance
    pass
```

2. **Separation Loss** - Encourages prototypes to be far from patches of other classes:
```python
def separation_loss(prototype_distances, labels, prototype_class_identity):
    """
    For each patch, maximize distance to prototypes of incorrect classes.
    Encourages class-specific prototypes.
    """
    pass
```

3. **L1 Regularization** - Encourages prototype sparsity:
```python
l1_loss = tf.reduce_mean(tf.abs(model.prototype_vectors))
```

4. **Diversity Loss** - Encourages prototypes within same class to be different:
```python
def diversity_loss(prototype_vectors, num_classes):
    """
    Minimize similarity between prototypes of the same class.
    Encourages diverse feature learning.
    """
    pass
```

**Total Loss:**
```python
total_loss = (
    segmentation_loss +
    lambda_cluster * cluster_loss +
    lambda_separation * separation_loss +
    lambda_l1 * l1_loss +
    lambda_diversity * diversity_loss
)
```

**Pros:**
- ✅ Explicit control over prototype properties
- ✅ Enforces interpretability (class-specific prototypes)
- ✅ Encourages prototype diversity
- ✅ Follows proven ProtoPNet methodology
- ✅ Better theoretical grounding

**Cons:**
- ❌ More complex implementation
- ❌ Additional hyperparameters to tune (lambdas)
- ❌ Requires custom training loop modifications
- ❌ Slower training (more loss computations)
- ❌ May need careful balancing of loss weights

**When to Consider:**
- If Option 1 results in redundant/uninformative prototypes
- If better interpretability is needed
- If prototypes don't cluster well by class
- For publication/research requiring rigorous methodology

---

## Decision 2: Data Preprocessing (2024-11-25)

### Problem
Google Colab training was not feasible due to:
- Large data size (240×240×155 volumes)
- Memory constraints
- Time limits (12-hour sessions)
- Slow training speed

### Solution
Implemented data preprocessing pipeline to downsample volumes from 240×240×155 to 160×160×96.

**Benefits:**
- ~51% reduction in data size
- ~2x faster training per epoch
- Larger batch sizes possible (4 instead of 2)
- Better compatibility with Colab resources
- Minimal quality loss (160×160 still captures important features)

**Implementation:**
- `preprocess_data.py`: Downsampling script using scipy.ndimage.zoom
- Center-cropping depth dimension (155 → 96 slices)
- Bilinear interpolation for images, nearest-neighbor for masks
- Preserved all 4 modalities and 3 mask channels

**Status:** ✅ Implemented and deployed

---

## Decision 3: Training Platform Strategy (2024-11-25)

### Problem
Need reliable, cost-effective platform for training 3D segmentation models.

### Solution
Multi-platform approach documented in `TRAINING_PLATFORMS.md`:

**Primary (Free Experimentation):**
- Kaggle Notebooks: 30 hours/week GPU, 12-hour sessions

**Production (Long Runs):**
- Lambda Labs: $0.50-1.10/hour for A100 GPUs
- AWS SageMaker Spot: ~$0.20-0.50/hour with spot instances

**Rationale:**
- Use free tier for validation and short experiments
- Use paid platforms for full training runs
- Avoid Google Colab due to reliability issues

**Status:** ✅ Documented, ready for deployment

---

## Future Decisions to Track

### 1. Model Evaluation Strategy
- Which metrics to prioritize (Dice, IoU, HD95)?
- How to handle class imbalance in evaluation?
- Cross-validation strategy

### 2. Prototype Interpretation
- How many prototypes per class is optimal? (currently 7)
- Should we prune/merge similar prototypes?
- How to present prototypes to medical professionals?

### 3. Deployment Considerations
- Real-time inference requirements?
- Model quantization/compression?
- Integration with clinical workflows?

---

## Changelog

| Date | Decision | Implemented By | Status |
|------|----------|----------------|--------|
| 2024-11-25 | Fix prototype learning (Option 1) | Claude | ✅ Complete |
| 2024-11-25 | CRITICAL FIX: Combine features instead of replace | Claude | ✅ Complete |
| 2024-11-25 | Data preprocessing pipeline | Claude | ✅ Complete |
| 2024-11-25 | Multi-platform training strategy | Claude | ✅ Documented |

---

## Notes

- Keep this log updated with major architectural decisions
- Document both chosen and rejected alternatives
- Include rationale for future reference
- Update status as implementations progress
