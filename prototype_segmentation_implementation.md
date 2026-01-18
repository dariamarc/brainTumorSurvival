# Prototype-Based 3D Brain Tumor Segmentation for BraTS2020

## Overview

This document outlines the implementation of an intrinsically explainable neural network based on prototypes for brain tumor segmentation using the BraTS2020 dataset. The architecture uses a CNN + ASPP backbone, followed by a prototype layer and fully connected layer for final segmentation.

## Dataset Context

- **Input**: 3D MRI volumes with 4 modalities (FLAIR, T1, T1ce, T2)
- **Output**: Segmented tumor with 3 tumor region classes
- **Classes**:
  1. GD-enhancing tumor
  2. Peritumoral edema
  3. Necrotic and non-enhancing tumor core
  4. Background (non-tumor)

## Architecture Design

### High-Level Architecture

```
Input (4-channel 3D MRI: FLAIR, T1, T1ce, T2)
    ↓
3D CNN Backbone (feature extraction)
    ↓
3D ASPP (multi-scale context)
    ↓
Prototype Layer (3 prototypes → 3 similarity maps)
    ↓
1×1×1 Conv (fully connected per-voxel)
    ↓
Output (4 classes: background + 3 tumor regions)
```

### Main Network Architecture

```python
class PrototypeSegNet3D(nn.Module):
    def __init__(self, n_prototypes=3, prototype_dim=256, 
                 prototype_shape=(1, 1, 1)):
        super().__init__()
        
        # 1. Backbone: 3D ResNet or similar
        self.backbone = ResNet3D(in_channels=4, base_channels=64)
        # Output: [B, 512, D/8, H/8, W/8]
        
        # 2. ASPP for multi-scale context
        self.aspp = ASPP3D(in_channels=512, out_channels=256)
        # Output: [B, 256, D/8, H/8, W/8]
        
        # 3. Prototype Layer
        self.prototype_vectors = nn.Parameter(
            torch.randn(n_prototypes, prototype_dim, *prototype_shape)
        )
        # Shape: [3, 256, 1, 1, 1] for point prototypes
        # or [3, 256, 3, 3, 3] for patch prototypes
        
        # 4. Final classification layer (1x1x1 conv = FC per voxel)
        self.classifier = nn.Conv3d(
            in_channels=n_prototypes,  # 3 similarity maps
            out_channels=4,             # 4 output classes
            kernel_size=1
        )
        
    def compute_similarities(self, features):
        """
        features: [B, 256, D, H, W]
        prototypes: [3, 256, 1, 1, 1]
        returns: [B, 3, D, H, W]
        """
        B, C, D, H, W = features.shape
        P = self.prototype_vectors.shape[0]
        
        # Reshape for broadcasting
        features_flat = features.view(B, C, -1)  # [B, 256, D*H*W]
        prototypes = self.prototype_vectors.view(P, C, -1)  # [3, 256, 1]
        
        # Compute L2 distances
        # features: [B, 1, 256, D*H*W]
        # prototypes: [1, 3, 256, 1]
        features_expanded = features_flat.unsqueeze(1)  # [B, 1, 256, D*H*W]
        prototypes_expanded = prototypes.unsqueeze(0).unsqueeze(-1)  # [1, 3, 256, 1]
        
        distances = torch.sum(
            (features_expanded - prototypes_expanded) ** 2, 
            dim=2
        )  # [B, 3, D*H*W]
        
        # Convert distance to similarity
        similarities = torch.log((distances + 1) / (distances + 1e-4))
        
        # Reshape back to spatial
        similarities = similarities.view(B, P, D, H, W)
        
        return similarities
    
    def forward(self, x):
        # x: [B, 4, D, H, W]
        
        # Extract features
        features = self.backbone(x)  # [B, 512, D/8, H/8, W/8]
        features = self.aspp(features)  # [B, 256, D/8, H/8, W/8]
        
        # Compute prototype similarities
        similarities = self.compute_similarities(features)  # [B, 3, D/8, H/8, W/8]
        
        # Upsample to original resolution (if needed)
        similarities_upsampled = F.interpolate(
            similarities, 
            size=x.shape[2:],  # (D, H, W)
            mode='trilinear',
            align_corners=False
        )  # [B, 3, D, H, W]
        
        # Final classification
        output = self.classifier(similarities_upsampled)  # [B, 4, D, H, W]
        
        return output, similarities_upsampled
```

### 3D ASPP Module

```python
class ASPP3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        # Multiple dilated convolutions
        self.conv1 = nn.Conv3d(in_channels, out_channels, 1)
        self.conv2 = nn.Conv3d(in_channels, out_channels, 3, 
                               padding=6, dilation=6)
        self.conv3 = nn.Conv3d(in_channels, out_channels, 3, 
                               padding=12, dilation=12)
        self.conv4 = nn.Conv3d(in_channels, out_channels, 3, 
                               padding=18, dilation=18)
        
        # Global pooling branch
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(in_channels, out_channels, 1)
        )
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Conv3d(out_channels * 5, out_channels, 1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        size = x.shape[2:]
        
        # Parallel branches
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        x5 = F.interpolate(self.global_pool(x), size=size, 
                          mode='trilinear', align_corners=False)
        
        # Concatenate and fuse
        x = torch.cat([x1, x2, x3, x4, x5], dim=1)
        x = self.fusion(x)
        
        return x
```

## Training Strategy

### Three-Phase Training

```python
# Phase 1: Warm-up (freeze prototypes)
# Train backbone + ASPP + classifier with random prototype init
for epoch in range(warmup_epochs):
    for batch in dataloader:
        output, similarities = model(batch['image'])
        
        # Only segmentation loss
        loss = dice_loss(output, batch['mask']) + ce_loss(output, batch['mask'])
        
        # Don't update prototypes
        optimizer_backbone.zero_grad()
        loss.backward()
        optimizer_backbone.step()

# Phase 2: Joint training (train everything)
for epoch in range(main_epochs):
    for batch in dataloader:
        output, similarities = model(batch['image'])
        
        # Multi-component loss
        seg_loss = dice_loss(output, batch['mask']) + ce_loss(output, batch['mask'])
        
        # Prototype purity: each prototype should activate on ONE class
        purity_loss = compute_purity_loss(similarities, batch['mask'])
        
        # Clustering: features should be close to their assigned prototype
        cluster_loss = compute_clustering_loss(features, prototypes, batch['mask'])
        
        # Separation: prototypes should be diverse
        sep_loss = compute_separation_loss(prototypes)
        
        total_loss = seg_loss + λ1*purity_loss + λ2*cluster_loss + λ3*sep_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

# Phase 3 (optional): Prototype projection
# Project prototypes onto actual training patches for interpretability
project_prototypes_onto_training_data(model, dataloader)
```

## Loss Functions

### 1. Prototype Purity Loss

```python
def compute_purity_loss(similarities, masks):
    """
    Encourage each prototype to activate strongly on only one class
    similarities: [B, 3, D, H, W]
    masks: [B, D, H, W] with values in {0, 1, 2, 3}
    """
    loss = 0
    for proto_idx in range(3):
        target_class = proto_idx + 1  # map proto 0→class 1, etc.
        
        # Where should this prototype activate?
        target_mask = (masks == target_class).float()
        
        # High similarity where mask matches
        loss += -torch.mean(similarities[:, proto_idx] * target_mask)
        
        # Low similarity elsewhere
        non_target_mask = (masks != target_class).float()
        loss += torch.mean(similarities[:, proto_idx] * non_target_mask)
    
    return loss / 3
```

### 2. Clustering Loss

```python
def compute_clustering_loss(features, prototypes, masks):
    """
    Features within same class should be close to their prototype
    """
    loss = 0
    for class_idx in range(1, 4):  # 3 tumor classes
        proto_idx = class_idx - 1
        
        # Get features from this class
        class_mask = (masks == class_idx)
        if class_mask.sum() == 0:
            continue
            
        class_features = features[class_mask.unsqueeze(1).expand_as(features)]
        
        # Distance to assigned prototype
        distances = torch.norm(
            class_features - prototypes[proto_idx].flatten(), 
            dim=-1
        )
        loss += torch.mean(distances)
    
    return loss / 3
```

### 3. Separation Loss

```python
def compute_separation_loss(prototypes):
    """
    Prototypes should be far apart from each other
    """
    # Pairwise distances between prototypes
    P = prototypes.shape[0]
    prototypes_flat = prototypes.view(P, -1)
    
    distances = torch.cdist(prototypes_flat, prototypes_flat)
    
    # Maximize minimum distance (encourage separation)
    # Use negative log to make it a loss (minimize)
    return -torch.log(torch.min(distances[distances > 0]) + 1e-4)
```

## Prototype Initialization

### Data-Driven Initialization

```python
def initialize_prototypes(model, dataloader, n_samples_per_class=100):
    """
    Initialize prototypes from actual training data
    """
    model.eval()
    
    # Collect features for each class
    class_features = {1: [], 2: [], 3: []}
    
    with torch.no_grad():
        for batch in dataloader:
            features = model.backbone(batch['image'])
            features = model.aspp(features)
            
            masks = batch['mask']
            
            # Sample features from each class
            for class_idx in [1, 2, 3]:
                class_mask = (masks == class_idx)
                if class_mask.sum() > 0:
                    # Extract random features from this class
                    indices = torch.where(class_mask)
                    sampled = random.sample(range(len(indices[0])), 
                                          min(10, len(indices[0])))
                    
                    for idx in sampled:
                        feat = features[
                            indices[0][idx], :,
                            indices[1][idx], 
                            indices[2][idx],
                            indices[3][idx]
                        ]
                        class_features[class_idx].append(feat)
                
                if len(class_features[class_idx]) >= n_samples_per_class:
                    break
    
    # Cluster and set prototypes
    for class_idx in [1, 2, 3]:
        proto_idx = class_idx - 1
        features_tensor = torch.stack(class_features[class_idx])
        
        # Use k-means or just mean
        prototype = torch.mean(features_tensor, dim=0)
        
        model.prototype_vectors.data[proto_idx] = prototype.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
```

## Implementation Considerations

### BraTS2020-Specific Parameters

- **Input dimensions**: 4 channels (FLAIR, T1, T1ce, T2)
- **Patch size for training**: 128×128×128 or 96×96×96 (memory constraints)
- **Feature dimension at prototype layer**: 256 channels
- **Number of prototypes**: 3 (one per tumor class)
- **Prototype shape**: Start with 1×1×1 (point prototypes)
- **Output classes**: 4 (background + 3 tumor regions)

### Memory Optimization

1. **Patch-based training**: Use 128³ or 96³ crops instead of full volumes
2. **Mixed precision training**: Use torch.cuda.amp for FP16
3. **Gradient checkpointing**: For deeper backbones if needed
4. **Batch size**: Start with 1-2 per GPU due to 3D volume size

### Handling Class Imbalance

- Use weighted Dice loss (tumor regions are small vs background)
- Consider focal loss for hard examples
- Online hard example mining during training

### Hyperparameters

**Loss weights (to tune):**
- λ1 (purity loss): 0.1 - 1.0
- λ2 (clustering loss): 0.1 - 1.0
- λ3 (separation loss): 0.01 - 0.1

**Training schedule:**
- Phase 1 (warm-up): 20-50 epochs
- Phase 2 (joint training): 100-200 epochs
- Learning rate: 1e-4 (Adam/AdamW)
- LR schedule: Cosine annealing or ReduceLROnPlateau

## Advantages of This Architecture

1. **Intrinsic interpretability**: Similarity maps show exactly where each prototype activates
2. **Simpler than U-Net**: No complex decoder, just 1×1×1 conv
3. **Multi-scale context**: ASPP captures tumors of varying sizes
4. **Proven approach**: This pattern works well in 2D segmentation
5. **Class-specific explanations**: Each prototype corresponds to one tumor type

## Expected Outputs

For each input volume, the network produces:
1. **Final segmentation**: [B, 4, D, H, W] - 4-class probability map
2. **Similarity maps**: [B, 3, D, H, W] - activation of each prototype
   - Map 0: GD-enhancing tumor activation
   - Map 1: Peritumoral edema activation
   - Map 2: Necrotic core activation

The similarity maps provide direct visual explanation of which regions activated each tumor-type prototype.

## Next Steps

1. Implement ResNet3D backbone (or use pretrained if available)
2. Implement ASPP3D module
3. Implement main PrototypeSegNet3D class
4. Implement all loss functions
5. Create BraTS2020 dataloader with proper preprocessing
6. Implement prototype initialization from training data
7. Set up three-phase training pipeline
8. Add visualization tools for prototype activations
9. Experiment with hyperparameters (loss weights, prototype shapes)
10. Evaluate on BraTS2020 validation set

## Potential Variations to Explore

- **Number of prototypes**: Try 5-10 per class instead of 1
- **Prototype shapes**: Experiment with 3×3×3 or 5×5×5 patches
- **Similarity metrics**: Try cosine similarity instead of L2 distance
- **Backbone**: Compare ResNet3D vs EfficientNet3D vs custom CNN
- **ASPP dilation rates**: Tune for optimal multi-scale capture
