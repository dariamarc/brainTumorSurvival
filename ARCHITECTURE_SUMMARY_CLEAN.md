# Brain Tumor Segmentation Architectures

## Architecture 1: U-Net with Prototype Learning

### Network Architecture

**Input:** 160 x 240 x 240 x 4
**Output:** 160 x 240 x 240 x 3

**Encoder:**
- Conv3D(32, 3x3x3) + ReLU
- MaxPool3D(1x2x2) → (160, 120, 120, 32)
- Conv3D(64, 3x3x3) + ReLU
- MaxPool3D(1x2x2) → (160, 60, 60, 64)
- Conv3D(128, 3x3x3) + ReLU
- MaxPool3D(1x2x2) → (160, 30, 30, 128)
- Conv3D(128, 3x3x3) + ReLU → (160, 30, 30, 128)

**Bottleneck Processing:**
- Conv3D(128, 1x1x1) + BatchNorm + ReLU
- Conv3D(128, 1x1x1) + BatchNorm

**Prototype Layer:**
- Learnable prototypes: (num_prototypes, 128)
- L2 distance computation: d_ij = sqrt(sum((x_i - p_j)^2))
- Logarithmic activation: s_ij = log((d_ij + 1) / (d_ij + epsilon))
- Projection: Conv3D(128, 1x1x1) + ReLU
- Feature combination: combined = f_processed + prototype_features

**Decoder:**
- Conv3DTranspose(128, (1,2,2)) + Conv3D(128, 3x3x3) x2 → (160, 60, 60, 128)
- Concatenate with encoder level 3 → (160, 60, 60, 256)
- Conv3DTranspose(64, (1,2,2)) + Conv3D(64, 3x3x3) x2 → (160, 120, 120, 64)
- Concatenate with encoder level 2 → (160, 120, 120, 128)
- Conv3DTranspose(32, (1,2,2)) + Conv3D(32, 3x3x3) x2 → (160, 240, 240, 32)
- Concatenate with encoder level 1 → (160, 240, 240, 64)
- Conv3DTranspose(16, (1,2,2)) + Conv3D(16, 3x3x3) x2 → (160, 240, 240, 16)
- Conv3D(16, 3x3x3) + ReLU + BatchNorm → (160, 240, 240, 16)
- Conv3D(3, 1x1x1) → (160, 240, 240, 3)

### Training

**Configuration:**
- Optimizer: Adam, learning rate = 1e-4
- Batch size: 4
- Epochs: 100-200
- Loss function: Hybrid Segmentation Loss

**Training loop per iteration:**
1. Forward propagation through entire network
2. Hybrid loss computation
3. Backpropagation through all layers
4. Parameter updates via Adam optimizer

### Loss Function

**Hybrid Segmentation Loss:**
L_hybrid = L_wCross + 100 × L_mDSC

**Component 1 - Volume-Size Weighted Cross Entropy:**
L_wCross = Σ_i -η_ℓ(xi) × log p(yi = ℓ|xi)
where η_ℓ(xi) = 1 - |X_ℓ| / |X|

**Component 2 - Multi-class Dice Coefficient:**
L_mDSC = -Σ_c [(2/N) × Σ_i(G_c^i × P_c^i)] / [(1/N) × Σ_i(G_c^i)^2 + (1/N) × Σ_i(P_c^i)^2]

### Evaluation Metrics

**Per-Class Dice Coefficient:**
Dice_class = (2 × |GT ∩ Pred|) / (|GT| + |Pred|)
- Dice_gd_enhancing
- Dice_edema
- Dice_necrotic

**Mean Dice Score:**
Dice_mean = (Dice_gd_enhancing + Dice_edema + Dice_necrotic) / 3

**Whole Tumor Dice:**
Dice_whole_tumor = (2 × |GT_tumor ∩ Pred_tumor|) / (|GT_tumor| + |Pred_tumor|)

---

## Architecture 2: ResNet with ASPP and Prototype Classification

### Network Architecture

**Input:** 160 x 192 x 128 x 4
**Output:** 160 x 192 x 128 x 4

**ResNet3D Backbone:**
- Initial: Conv3D(64, 7x7x7, stride=2) + BatchNorm + ReLU → (80, 96, 64, 64)
- Stage 1: 2 ResidualBlocks(64, stride=1) → (80, 96, 64, 64)
- Stage 2: 2 ResidualBlocks(128, stride=2) → (40, 48, 32, 128)
- Stage 3: 2 ResidualBlocks(256, stride=2) → (20, 24, 16, 256)
- Stage 4: 2 ResidualBlocks(512, stride=1) → (20, 24, 16, 512)

**Residual Block:**
- Conv3D(filters, 3x3x3, stride=s) + BatchNorm + ReLU
- Conv3D(filters, 3x3x3) + BatchNorm
- Skip connection [if stride > 1: Conv3D(filters, 1x1x1, stride=s) + BatchNorm]
- Element-wise addition + ReLU

**ASPP3D Module (Input: 20, 24, 16, 512):**
Five parallel branches:
- Branch 1: Conv3D(256, 1x1x1) + BatchNorm + ReLU
- Branch 2: Conv3D(256, 3x3x3, dilation=2) + BatchNorm + ReLU
- Branch 3: Conv3D(256, 3x3x3, dilation=4) + BatchNorm + ReLU
- Branch 4: Conv3D(256, 3x3x3, dilation=8) + BatchNorm + ReLU
- Branch 5: GlobalAveragePooling + Conv3D(256, 1x1x1) + LayerNorm + ReLU + TrilinearUpsample

Fusion: Concatenate all branches → Conv3D(256, 1x1x1) + BatchNorm + ReLU
Output: (20, 24, 16, 256)

**Prototype Layer:**
- Learnable prototypes: (n_prototypes=3, 256)
- L2 distance: d_ij = sqrt(sum((f_i - p_j)^2))
- Similarity activation: log((d + 1) / (d + epsilon))
- Output: (20, 24, 16, 3)

**Trilinear Upsampling:**
Sequential bilinear interpolation of H, W, D dimensions
Output: (160, 192, 128, 3)

**Interpretable Classifier:**
- Learned weight matrix: (3, 4) [3 prototypes → 4 classes]
- Per-voxel classification based on prototype activations
- Output: (160, 192, 128, 4)

### Training

**Phase 1: Warm-up (Epochs 1-50)**
- Frozen: ResNet3D backbone
- Trainable: ASPP, prototype layer, classifier
- Optimizer: Adam, learning rate = 1e-3
- Batch size: 4
- Loss: Hybrid Segmentation Loss
- Process: Extract backbone features → forward through ASPP → Prototypes → Classifier → backpropagate through trainable components

**Phase 2: Joint Fine-tuning (Epochs 51-250)**
- Frozen: None
- Trainable: All components
- Optimizer: Differential learning rates
  - Backbone: 1e-4
  - ASPP, prototypes, classifier: 5e-4
- Batch size: 4
- Loss: Hybrid Segmentation Loss
- Process: Complete forward propagation → loss computation → backpropagation through all layers → separate parameter updates

**Phase 3: Prototype Projection (Optional)**
Project learned prototypes onto actual training patches for interpretability enhancement

### Loss Function

**Hybrid Segmentation Loss:**
L_hybrid = L_wCross + 100 × L_mDSC

**Component 1 - Volume-Size Weighted Cross Entropy:**
L_wCross = Σ_i -η_ℓ(xi) × log p(yi = ℓ|xi)
where η_ℓ(xi) = 1 - |X_ℓ| / |X|

**Component 2 - Multi-class Dice Coefficient:**
L_mDSC = -Σ_c [(2/N) × Σ_i(G_c^i × P_c^i)] / [(1/N) × Σ_i(G_c^i)^2 + (1/N) × Σ_i(P_c^i)^2]

### Evaluation Metrics

**Per-Class Dice Coefficient:**
Dice_class = (2 × |GT ∩ Pred|) / (|GT| + |Pred|)
- Dice_gd_enhancing
- Dice_edema
- Dice_necrotic

**Mean Dice Score:**
Dice_mean = (Dice_gd_enhancing + Dice_edema + Dice_necrotic) / 3

**Whole Tumor Dice:**
Dice_whole_tumor = (2 × |GT_tumor ∩ Pred_tumor|) / (|GT_tumor| + |Pred_tumor|)
