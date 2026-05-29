# Brain Tumor Segmentation — Experiment Summary

## Project Goal

Train a 3D brain tumour segmentation model on BraTS2020 that is both accurate and interpretable. Interpretability is achieved through **prototype learning**: the model learns a small set of reference feature vectors (prototypes), one set per tumour class, and makes predictions based on similarity to those prototypes rather than opaque activations. The three tumour classes are NCR (necrotic core), ED (oedema), and ET (enhancing tumour).

All experiments run on SageMaker `ml.g4dn.xlarge` (NVIDIA T4, 16 GB) with volumes of shape `(128, 192, 160, 4)` and `batch_size=1`.

---

## Architecture: UNet3DProto

All steps share the same base architecture, incrementally extended:

```
Input (1, 128, 192, 160, 4)
    ↓
Encoder:  3 × [Conv3D → LayerNorm → ReLU] × 2 + MaxPool3D(1,2,2)
          channels: 16 → 32 → 64
    ↓
Bottleneck:  [Conv3D → LayerNorm → ReLU] × 2 → Dropout(0.2)
             channels: 128
    ↓                          ← prototype layer inserted here (Steps 2–3)
Decoder:  3 × [ConvTranspose3D + concat skip + ConvBlock]
          channels: 64 → 32 → 16
    ↓
Output Conv3D(4, 1×1×1) → logits (1, 128, 192, 160, 4)
```

Key architectural choices:
- **1×2×2 pooling** (not 2×2×2): preserves depth dimension, critical for 3D brain volumes
- **LayerNorm** instead of BatchNorm: stable at batch_size=1
- **1.36M parameters** (base_channels=16)

---

## Step 1 — Plain UNet3D Baseline

**Objective:** Establish a segmentation baseline with no prototype machinery. Define pass criteria for subsequent steps.

**Loss:** Hybrid Dice + volume-weighted cross-entropy (α=0.5 each)

**Epochs:** 50 (peaked at epoch 45, patience=10 did not trigger)

### Results

| Metric | Value |
|--------|-------|
| Mean Dice | **0.704** |
| NCR Dice | 0.630 |
| ED Dice | 0.740 |
| ET Dice | 0.742 |
| WT Dice | 0.878 |

Pass criteria: WT > 0.65 ✓ — Mean > 0.40 ✓

### Interpretation

A clean, strong baseline for a model of this size. NCR is the weakest class at 0.630 — expected, since the necrotic core is the smallest and most heterogeneous region. The hybrid loss (Dice + CE) helped the model localise tumours early in training: CE penalises every voxel individually, forcing the model to find the tumour region rather than over-predicting background. WT at 0.878 is close to saturation; future gains must come primarily from NCR.

---

## Step 2 — Prototype Bottleneck, No Prototype Losses

**Objective:** Validate that adding a prototype layer at the bottleneck does not hurt segmentation. Confirm the mechanism works before adding the training losses that make prototypes meaningful.

**Changes from Step 1:**
- 9 prototype vectors added (3 per tumour class, background excluded): `tf.Variable` of shape `(9, 128, 1, 1, 1)`
- Similarity maps → `prototype_to_features` (1×1×1 Conv3D) → element-wise add to bottleneck
- `prototype_to_features` kernel **initialised to zeros**: prototype contribution starts at zero, matching Step 1 behaviour at epoch 0
- Loss: pure Dice only (CE removed to isolate prototype effect)
- Initialised from Step 1 backbone weights; prototype layers freshly initialised

**Epochs:** 29 (early stopping, best at epoch ~15)

### Results

| Metric | Value | vs Step 1 |
|--------|-------|-----------|
| Mean Dice | **0.722** | +0.018 |
| NCR Dice | ~0.636 | +0.006 |
| ED Dice | ~0.753 | +0.013 |
| ET Dice | ~0.744 | +0.002 |
| WT Dice | 0.861 | −0.017 |

Pass criterion (within 5pp of Step 1): **PASS**

**Prototype activation ratios at convergence:**

| Class | Ratio | Interpretation |
|-------|-------|----------------|
| NCR | 0.93–0.94 | Below 1 — prototypes fire more outside NCR than inside |
| ED | 1.04–1.05 | Marginally class-specific |
| ET | 0.99 | Neutral — effectively random |

### Interpretation

The +0.018 Mean Dice improvement over Step 1 is real, but comes from the backbone continuing to fine-tune under the pure Dice loss — not from the prototypes. The prototype ratios were stuck at virtually the same values from epoch 1 to epoch 29, confirming the prototypes were passengers throughout training.

Why did prototypes not learn? The zero-initialised `prototype_to_features` kernel produces near-zero output throughout training. The gradient flowing back through the prototype path is orders of magnitude smaller than the gradient through the main bottleneck. Without an explicit loss that directly penalises the prototype vectors based on their class alignment, there is no force to move them.

This step answered its question: adding the prototype layer does not break the model. It is safe to introduce the prototype learning losses.

---

## Step 3 — Prototype Learning Losses

**Objective:** Force the prototype vectors to become genuinely class-specific using three additional losses: clustering (pull prototypes toward their class), separation (push prototypes away from other classes), and diversity (spread same-class prototypes apart).

**Changes from Step 2:**
- Three new loss terms:
  - `L_clst`: for each prototype p, minimise its minimum L2 distance to any voxel in class(p) at bottleneck resolution — distances normalised by `sqrt(128)`
  - `L_sep`: for each prototype p, maximise its minimum L2 distance to any voxel outside class(p)
  - `L_div`: minimise positive cosine similarity between same-class prototypes
- All runs initialised from Step 2 best checkpoint

### Loss weight exploration

Four runs were performed to find the right `clst_weight`. All share `sep_weight=0.1`, `div_weight=0.1`.

| Run | clst_weight | Mean Dice | vs Step 2 | NCR ratio | ET ratio | sep active? | Notes |
|-----|-------------|-----------|-----------|-----------|----------|-------------|-------|
| v1 | 0.8 (unnorm) | 0.7179 | −0.0041 | 4.81 | 5.75 | No (0.007 flat) | Pre-normalisation |
| v2 | 0.01 (norm) | ~0.718 | −0.004 | 0.89 | 0.93 | Yes, but ratios < 1 | clst too weak |
| v3 | 0.05 (norm) | ~0.712 | −0.010 | declining | declining | Working wrong direction | Prototypes drifting to neutral space |
| **v4** | **0.2 (norm)** | **0.7195** | **−0.0025** | **4.74** | **5.99** | No (0.020) | **Best result** |

**Final proto ratios (v4):**

| Class | Step 2 | v4 final |
|-------|--------|----------|
| NCR | 0.93 | **4.74** |
| ED | 1.05 | **3.54** |
| ET | 0.99 | **5.99** |

### Interpretation

V4 (`clst_weight=0.2`) is the best Step 3 result: Mean Dice 0.7195 (only −0.0025 vs Step 2) with proto ratios of 4.74 / 3.54 / 5.99 — prototypes fire 4–6× more strongly inside their class than outside.

The loss weight exploration revealed a consistent pattern: **whenever `clst` is strong enough to move prototypes into class-specific regions, `sep` becomes inactive** — because once prototypes are deeply embedded in class feature space, they are already far from other-class voxels and the sep gradient vanishes. Conversely, when `clst` is weak enough for sep to activate (v2/v3), prototypes never reach class-specific regions and ratios stay below 1.0.

This means sep inactivity in v4 is not a weakness — it is a sign that prototypes are already well-separated by class. The diversity loss (`div`) also converges to zero incidentally, as clustering pulls each prototype to a different local class minimum. Both sep and div serve as verification signals rather than active training forces.

### Effect of Prototype Count

To assess whether more prototypes improve coverage of NCR's intra-class variability, a second Step 3 run used 5 prototypes per class (15 total) with identical loss weights (λc=0.2, λs=0.1, λd=0.1), initialised from the same Step 2 checkpoint.

| Config | Mean Dice | Gap vs Step 2 | NCR ratio | ED ratio | ET ratio |
|--------|-----------|---------------|-----------|----------|----------|
| 3 protos/class | 0.7195 | −0.0025 | 4.74 | 3.54 | 5.99 |
| **5 protos/class** | **0.7206** | **−0.0014** | **4.40** | **3.07** | **5.33** |

The 5-prototype model achieves higher Mean Dice and a smaller gap to the Step 2 baseline, while all ratios remain well above 1.0. The slight decrease in per-prototype ratios is expected: with more prototypes, each covers a narrower portion of its class region, so individual ratios are lower even though collective class coverage is greater.

**5 protos/class is the final Step 3 result.** The interpretability claim stands: class-specific prototypes with negligible Dice cost.

---

## Comparative Summary

| | Step 1 | Step 2 | Step 3 (5 proto/class, final) |
|---|---|---|---|
| **Architecture** | Plain UNet3D | + prototype bottleneck | + prototype losses |
| **Loss** | Dice + CE | Dice only | Dice + 0.2·clst − 0.1·sep + 0.1·div |
| **Initialisation** | Random | Step 1 weights | Step 2 weights |
| **Protos per class** | — | 3 | **5** |
| **Mean Dice** | 0.704 | **0.722** | **0.7206** |
| **WT Dice** | 0.878 | 0.861 | 0.875 |
| **ET Dice** | 0.742 | ~0.744 | 0.773 |
| **NCR proto ratio** | — | 0.93 | **4.40** |
| **ED proto ratio** | — | 1.04 | **3.07** |
| **ET proto ratio** | — | 0.99 | **5.33** |
| **Prototypes active?** | — | No | **Yes** |
| **Sep loss effective?** | — | — | Inactive (prototypes already well-separated) |
| **Div loss effective?** | — | — | Incidentally yes |

---

## Comparison with State-of-the-Art

Results from the Step 3 final model (5 prototypes/class) contextualised against published methods on BraTS2020. SOTA values are from official challenge evaluations on a held-out test set; UNet3DProto results are on a custom 80/20 patient-level validation split of the 369 training volumes — the comparison is indicative, not strictly controlled.

| Method | Params | WT Dice | TC Dice | ET Dice | Interpretable? |
|--------|--------|---------|---------|---------|----------------|
| nnU-Net | ~30M | ~0.910 | ~0.860 | ~0.780 | No |
| TransBTS | ~33M | ~0.903 | ~0.836 | ~0.788 | No |
| SwinUNETR | ~62M | ~0.922 | ~0.862 | ~0.828 | No |
| **UNet3DProto** | **1.36M** | **0.875** | n/a | **0.773** | **Yes** |

TC Dice is not reported for UNet3DProto as combining NCR and ET predictions was not computed in the current evaluation. UNet3DProto operates at 1–2 orders of magnitude fewer parameters than all reference methods. The gap in WT and ET Dice reflects both reduced model capacity and the absence of explicit multi-scale feature extraction. Closing this gap while preserving prototype-based interpretability is the primary objective of the PrototypeSegNet3D (Architecture 2) experiments.

---

## Next Steps

### 1. Cross-Validation and Augmentation for UNet3DProto

The POC results reported above were obtained on a single fixed 80/20 split and are therefore split-sensitive. The improved `unet_step*/` scripts implement 5-fold cross-validation (C=32 base channels, ~5.4M parameters) with online augmentation (random axis flips, per-modality intensity jitter). Step 1 is currently running on SageMaker; Steps 2 and 3 follow with fold-matched checkpoint initialisation.

### 2. Reintroduce Cross-Entropy

Steps 2 and 3 use pure Dice loss. Step 1's hybrid Dice + CE helped the model localise tumours early by penalising individual voxel misclassifications. Reintroducing CE at a small weight (e.g. `loss_alpha=0.3`) alongside the prototype losses for the cross-validated runs should recover boundary accuracy lost under pure Dice, particularly for thin ED boundaries and small ET regions.

### 3. Architecture 2 — PrototypeSegNet3D (ResNet + ASPP)

Architecture 2 (17.4M parameters) uses a ResNet3D backbone with explicit multi-scale context via ASPP (dilation rates 2, 4, 8), a single prototype per tumour class, and a fully transparent 4×3 interpretable classifier whose weight matrix directly maps prototype similarities to class logits. Training follows three sequential experiments:

1. **Experiment A** — ResNet3D + ASPP3D baseline, no prototypes, Dice loss only. **Completed** (results not yet recorded).
2. **Experiment B1** — Full PrototypeSegNet3D, segmentation loss only, initialised from Exp A.
3. **Experiment B2** — Full loss (Dice + clst + sep), initialised from B1.

The planned analysis will directly compare Architecture 1 (skip-connection multi-scale, 1.36M) against Architecture 2 (ASPP multi-scale, 17.4M) at each matching step (Step 1 vs Exp A, Step 2 vs Exp B1, Step 3 vs Exp B2).
