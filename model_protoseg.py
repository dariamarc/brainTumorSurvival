"""
ProtoSeg 3D Model for Brain Tumor Segmentation
Based on: "ProtoSeg: Interpretable Semantic Segmentation with Prototypical Parts" (WACV 2023)

Architecture:
- Custom 3D Encoder (kept from original implementation)
- ASPP 3D module (adapted from ProtoSeg paper)
- Prototype Layer (with L2/Cosine distance)
- Fully Connected Layer for classification (replaces U-Net decoder)
- Bilinear upsampling to restore resolution
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


class ASPP_3D(keras.layers.Layer):
    """
    Atrous Spatial Pyramid Pooling in 3D.
    Adapted from DeepLabv2/v3 for 3D volumetric data.

    Uses multiple parallel atrous (dilated) convolutions with different rates
    to capture multi-scale context.
    """
    def __init__(self, in_channels, out_channels=128, rates=[1, 2, 4, 8], **kwargs):
        super(ASPP_3D, self).__init__(**kwargs)
        self.out_channels = out_channels
        self.rates = rates

        # Branch 1: 1x1x1 convolution
        self.aspp1 = layers.Conv3D(
            out_channels,
            kernel_size=1,
            padding='same',
            activation='relu',
            kernel_initializer='he_normal',
            name='aspp_1x1'
        )

        # Branch 2: 3x3x3 conv with dilation rate 2
        self.aspp2 = layers.Conv3D(
            out_channels,
            kernel_size=3,
            padding='same',
            dilation_rate=rates[1],
            activation='relu',
            kernel_initializer='he_normal',
            name='aspp_rate2'
        )

        # Branch 3: 3x3x3 conv with dilation rate 4
        self.aspp3 = layers.Conv3D(
            out_channels,
            kernel_size=3,
            padding='same',
            dilation_rate=rates[2],
            activation='relu',
            kernel_initializer='he_normal',
            name='aspp_rate4'
        )

        # Branch 4: 3x3x3 conv with dilation rate 8
        self.aspp4 = layers.Conv3D(
            out_channels,
            kernel_size=3,
            padding='same',
            dilation_rate=rates[3],
            activation='relu',
            kernel_initializer='he_normal',
            name='aspp_rate6'
        )

        # Branch 5: Global Average Pooling + 1x1x1 conv
        self.global_avg_pool = layers.GlobalAveragePooling3D(keepdims=True)
        self.aspp5 = layers.Conv3D(
            out_channels,
            kernel_size=1,
            activation='relu',
            kernel_initializer='he_normal',
            name='aspp_global'
        )

        # Combine all branches
        self.concat = layers.Concatenate(axis=-1)

        # 1x1x1 conv to reduce concatenated features
        self.project = layers.Conv3D(
            out_channels,
            kernel_size=1,
            activation='relu',
            kernel_initializer='he_normal',
            name='aspp_project'
        )

        self.bn = layers.BatchNormalization()
        self.dropout = layers.Dropout(0.5)

    def call(self, inputs, training=False):
        # Get input shape for upsampling global features
        shape = tf.shape(inputs)

        # Apply all ASPP branches
        branch1 = self.aspp1(inputs)
        branch2 = self.aspp2(inputs)
        branch3 = self.aspp3(inputs)
        branch4 = self.aspp4(inputs)

        # Global pooling branch
        branch5 = self.global_avg_pool(inputs)
        branch5 = self.aspp5(branch5)
        # Upsample to match input spatial dimensions
        branch5 = tf.image.resize(
            branch5[:, 0, :, :, :],  # Remove D dimension for resize
            size=[shape[2], shape[3]],
            method='bilinear'
        )
        branch5 = tf.expand_dims(branch5, axis=1)  # Add D dimension back
        branch5 = tf.tile(branch5, [1, shape[1], 1, 1, 1])  # Repeat along D

        # Concatenate all branches
        concatenated = self.concat([branch1, branch2, branch3, branch4, branch5])

        # Project to output channels
        output = self.project(concatenated)
        output = self.bn(output, training=training)
        output = self.dropout(output, training=training)

        return output


def build_3d_encoder(model_name='resnet50_ri'):
    """
    Custom 3D encoder with isotropic pooling for isotropic data.
    Returns only the final bottleneck features (no skip connections needed).
    """
    if model_name == 'resnet50_ri':
        input_tensor = keras.Input(shape=(None, None, None, 4))  # (D, H, W, C_in)

        # Encoder Block 1 - Isotropic pooling
        conv1 = layers.Conv3D(32, 3, activation='relu', padding='same',
                             kernel_initializer='he_normal')(input_tensor)
        pool1 = layers.MaxPool3D(pool_size=(2, 2, 2))(conv1)  # (D/2, H/2, W/2, 32)

        # Encoder Block 2 - Isotropic pooling
        conv2 = layers.Conv3D(64, 3, activation='relu', padding='same',
                             kernel_initializer='he_normal')(pool1)
        pool2 = layers.MaxPool3D(pool_size=(2, 2, 2))(conv2)  # (D/4, H/4, W/4, 64)

        # Encoder Block 3 - Isotropic pooling
        conv3 = layers.Conv3D(128, 3, activation='relu', padding='same',
                             kernel_initializer='he_normal')(pool2)
        pool3 = layers.MaxPool3D(pool_size=(2, 2, 2))(conv3)  # (D/8, H/8, W/8, 128)

        # Encoder Block 4 (Bottleneck)
        conv4 = layers.Conv3D(128, 3, activation='relu', padding='same',
                             kernel_initializer='he_normal')(pool3)
        # Final: (D/8, H/8, W/8, 128)

        # Return only bottleneck (no skip connections for ProtoSeg)
        return keras.Model(inputs=input_tensor, outputs=conv4, name='encoder_3d')
    else:
        raise ValueError(f"Unknown feature extractor: {model_name}")


class ProtoSeg3D(keras.Model):
    """
    ProtoSeg adapted for 3D brain tumor segmentation.

    Architecture:
    1. 3D Encoder (custom)
    2. ASPP 3D module
    3. Prototype Layer (computes similarity to M prototypes)
    4. Fully Connected Layer (M prototypes → C classes)
    5. Bilinear interpolation to original resolution

    Key differences from original implementation:
    - No U-Net decoder
    - No skip connections
    - Direct prototype → class mapping via FC layer
    - Upsampling via interpolation (not learned)
    """

    def __init__(self,
                 in_size=(128, 160, 192, 4),
                 num_classes=4,
                 num_prototypes_per_class=7,
                 features='resnet50_ri',
                 prototype_dim=128,
                 f_dist='l2',
                 prototype_activation_function='log',
                 aspp_out_channels=128,
                 **kwargs):
        super(ProtoSeg3D, self).__init__(**kwargs)

        self.input_shape_keras = in_size
        self.num_classes = num_classes
        self.num_prototypes_per_class = num_prototypes_per_class
        self.num_prototypes = num_classes * num_prototypes_per_class
        self.prototype_dim = prototype_dim
        self.f_dist = f_dist
        self.prototype_activation_function = prototype_activation_function
        self.epsilon = 1e-4

        # Prototype class identity: which prototype belongs to which class
        # Shape: (num_prototypes, num_classes)
        # Each row is one-hot encoded class assignment
        self.prototype_class_identity = tf.constant(
            np.repeat(np.eye(num_classes), num_prototypes_per_class, axis=0),
            dtype=tf.float32,
            name='prototype_class_identity'
        )

        print(f"\n{'='*80}")
        print(f"ProtoSeg3D Architecture")
        print(f"{'='*80}")
        print(f"Input shape: {in_size}")
        print(f"Number of classes: {num_classes}")
        print(f"Prototypes per class: {num_prototypes_per_class}")
        print(f"Total prototypes: {self.num_prototypes}")
        print(f"Prototype dimension: {prototype_dim}")
        print(f"Distance metric: {f_dist}")
        print(f"{'='*80}\n")

        # --- ENCODER ---
        self.encoder = build_3d_encoder(features)

        # Test encoder output shape
        dummy_input = tf.zeros((1,) + self.input_shape_keras, dtype=tf.float32)
        encoder_output = self.encoder(dummy_input)
        print(f"Encoder output shape: {encoder_output.shape}")

        encoder_out_channels = encoder_output.shape[-1]

        # --- ASPP 3D MODULE ---
        self.aspp = ASPP_3D(
            in_channels=encoder_out_channels,
            out_channels=aspp_out_channels,
            rates=[1, 2, 4, 8]
        )

        # Test ASPP output
        aspp_output = self.aspp(encoder_output)
        print(f"ASPP output shape: {aspp_output.shape}")

        # --- PROJECTION TO PROTOTYPE DIMENSION ---
        # Map ASPP features to prototype dimension
        self.feature_projection = layers.Conv3D(
            prototype_dim,
            kernel_size=1,
            activation='relu',
            kernel_initializer='he_normal',
            bias_initializer='zeros',
            name='feature_projection'
        )

        # --- PROTOTYPE LAYER ---
        # Prototypes: (num_prototypes, prototype_dim, 1, 1, 1)
        # Each prototype is a point in feature space
        self.prototype_vectors = tf.Variable(
            tf.random.normal((self.num_prototypes, prototype_dim, 1, 1, 1), stddev=0.1),
            trainable=True,
            name='prototype_vectors'
        )

        print(f"Prototype vectors shape: {self.prototype_vectors.shape}")

        # --- FULLY CONNECTED LAYER ---
        # Maps prototype similarities to class probabilities
        # Input: (num_prototypes,) per spatial point
        # Output: (num_classes,) per spatial point
        self.fc_layer = layers.Dense(
            num_classes,
            use_bias=True,
            kernel_initializer='zeros',
            bias_initializer='zeros',
            name='prototype_to_class'
        )

        # Initialize FC weights as in ProtoSeg paper (Section 3.1):
        # w_h^(c,j) = 1 if p_j ∈ P_c (prototype j belongs to class c)
        # w_h^(c,j) = -0.5 otherwise
        self._initialize_fc_weights()

        print(f"\nModel initialized successfully!")
        print(f"{'='*80}\n")

    def _initialize_fc_weights(self):
        """
        Initialize FC layer weights as in ProtoSeg paper.
        Encourages prototypes to activate strongly for their assigned class.
        """
        # Build the layer first by calling it
        dummy_input = tf.zeros((1, self.num_prototypes))
        _ = self.fc_layer(dummy_input)

        # Initialize weights: (num_prototypes, num_classes)
        initial_weights = np.zeros((self.num_prototypes, self.num_classes))

        for c in range(self.num_classes):
            for p in range(self.num_prototypes):
                if self.prototype_class_identity[p, c] == 1:
                    # Prototype belongs to this class
                    initial_weights[p, c] = 1.0
                else:
                    # Prototype doesn't belong to this class
                    initial_weights[p, c] = -0.5

        # Set weights
        self.fc_layer.set_weights([
            initial_weights.astype(np.float32),
            np.zeros(self.num_classes, dtype=np.float32)
        ])

        print(f"FC layer weights initialized (ProtoSeg strategy):")
        print(f"  Positive weight (same class): 1.0")
        print(f"  Negative weight (other class): -0.5")

    def l2_convolution_3D(self, x):
        """
        Compute L2 distances between feature map points and prototypes.
        Uses convolution for efficiency.

        Args:
            x: Feature map (B, D, H, W, C)
        Returns:
            distances: (B, D, H, W, num_prototypes)
        """
        # Reshape prototypes for convolution: (1, 1, 1, C, M)
        proto_filters = tf.transpose(self.prototype_vectors, perm=[2, 3, 4, 1, 0])

        # Compute dot product via convolution
        dot_product = tf.nn.conv3d(
            x,
            filters=proto_filters,
            strides=(1, 1, 1, 1, 1),
            padding='SAME'
        )

        # ||x||^2
        x2 = tf.reduce_sum(tf.square(x), axis=-1, keepdims=True)

        # ||p||^2
        p2 = tf.reduce_sum(
            tf.square(self.prototype_vectors),
            axis=[1, 2, 3, 4],
            keepdims=True
        )
        p2 = tf.transpose(p2, perm=[1, 2, 3, 4, 0])

        # L2 distance: ||x - p||^2 = ||x||^2 - 2(x·p) + ||p||^2
        distances = x2 - 2 * dot_product + p2
        distances = tf.maximum(distances, self.epsilon)
        distances = tf.sqrt(distances)

        return distances

    def cosine_convolution_3D(self, x):
        """
        Compute cosine similarity between feature map points and prototypes.

        Args:
            x: Feature map (B, D, H, W, C)
        Returns:
            similarities: (B, D, H, W, num_prototypes)
        """
        # Reshape prototypes for convolution
        proto_filters = tf.transpose(self.prototype_vectors, perm=[2, 3, 4, 1, 0])

        # Dot product
        dot_product = tf.nn.conv3d(
            x,
            filters=proto_filters,
            strides=(1, 1, 1, 1, 1),
            padding='SAME'
        )

        # ||x||
        x_norm = tf.norm(x, axis=-1, keepdims=True)

        # ||p||
        p_norm = tf.norm(
            self.prototype_vectors,
            axis=[1, 2, 3, 4],
            keepdims=True
        )
        p_norm = tf.transpose(p_norm, perm=[1, 2, 3, 4, 0])

        # Cosine similarity: (x·p) / (||x|| * ||p||)
        similarities = dot_product / (x_norm * p_norm + self.epsilon)

        return similarities

    def distance_2_similarity(self, distances):
        """
        Convert distances to similarities using activation function.
        From ProtoSeg paper Equation 1.

        Args:
            distances: (B, D, H, W, M)
        Returns:
            similarities: (B, D, H, W, M)
        """
        if self.prototype_activation_function == 'log':
            # log((d^2 + 1) / (d^2 + ε))
            return tf.math.log((distances + 1.0) / (distances + self.epsilon))
        elif self.prototype_activation_function == 'linear':
            return -distances
        elif self.prototype_activation_function == 'exp':
            return tf.exp(-distances)
        else:
            raise NotImplementedError(
                f"Unknown activation: {self.prototype_activation_function}"
            )

    def call(self, inputs, training=False):
        """
        Forward pass.

        Args:
            inputs: (B, D, H, W, 4) - 4 MRI modalities
        Returns:
            logits: (B, D, H, W, num_classes) - class logits
        """
        # Store original shape for final upsampling
        original_shape = tf.shape(inputs)

        # 1. ENCODER: Extract features (isotropic downsampling)
        # (B, D, H, W, 4) → (B, D/8, H/8, W/8, 128)
        encoder_features = self.encoder(inputs, training=training)

        # 2. ASPP: Multi-scale context
        # (B, D/8, H/8, W/8, 128) → (B, D/8, H/8, W/8, 128)
        aspp_features = self.aspp(encoder_features, training=training)

        # 3. PROJECT to prototype dimension
        # (B, D/8, H/8, W/8, 128) → (B, D/8, H/8, W/8, prototype_dim)
        projected_features = self.feature_projection(aspp_features, training=training)

        # 4. PROTOTYPE LAYER: Compute similarities
        # (B, D/8, H/8, W/8, prototype_dim) → (B, D/8, H/8, W/8, num_prototypes)
        if self.f_dist == 'l2':
            prototype_distances = self.l2_convolution_3D(projected_features)
            prototype_similarities = self.distance_2_similarity(prototype_distances)
        elif self.f_dist == 'cosine':
            prototype_similarities = self.cosine_convolution_3D(projected_features)
        else:
            raise NotImplementedError(f"Unknown distance metric: {self.f_dist}")

        # 5. FULLY CONNECTED LAYER: Prototype similarities → Class logits
        # Process each spatial point independently
        # (B, D/8, H/8, W/8, num_prototypes) → (B, D/8, H/8, W/8, num_classes)

        # Get feature map shape
        feat_shape = tf.shape(prototype_similarities)
        batch_size = feat_shape[0]
        d_feat = feat_shape[1]
        h_feat = feat_shape[2]
        w_feat = feat_shape[3]

        # Reshape to (B*D*H*W, num_prototypes)
        similarities_flat = tf.reshape(
            prototype_similarities,
            [-1, self.num_prototypes]
        )

        # Apply FC layer: (B*D*H*W, num_prototypes) → (B*D*H*W, num_classes)
        logits_flat = self.fc_layer(similarities_flat)

        # Reshape back: (B*D*H*W, num_classes) → (B, D, H/8, W/8, num_classes)
        logits_low_res = tf.reshape(
            logits_flat,
            [batch_size, d_feat, h_feat, w_feat, self.num_classes]
        )

        # 6. UPSAMPLE to original resolution using TRILINEAR INTERPOLATION
        # This is NOT learned (different from U-Net decoder)
        # (B, D/8, H/8, W/8, num_classes) → (B, D, H, W, num_classes)

        # Since we now use isotropic pooling, we need to upsample D, H, and W
        # Strategy: First upsample H and W, then upsample D

        # Step 1: Upsample H and W using tf.image.resize
        # Permute to (B, num_classes, D, H, W) for easier processing
        logits_permuted = tf.transpose(logits_low_res, [0, 4, 1, 2, 3])

        # Reshape to (B*num_classes*D, H, W, 1) for tf.image.resize
        logits_reshaped = tf.reshape(
            logits_permuted,
            [batch_size * self.num_classes * d_feat, h_feat, w_feat, 1]
        )

        # Resize H and W
        logits_hw_upsampled = tf.image.resize(
            logits_reshaped,
            size=[original_shape[2], original_shape[3]],
            method='bilinear'
        )

        # Reshape back: (B*num_classes*D, H, W, 1) → (B, num_classes, D, H, W)
        logits_hw_upsampled = tf.reshape(
            logits_hw_upsampled,
            [batch_size, self.num_classes, d_feat, original_shape[2], original_shape[3]]
        )

        # Step 2: Upsample D dimension
        # Permute to (B, num_classes, H, W, D) so D is last
        logits_permuted2 = tf.transpose(logits_hw_upsampled, [0, 1, 3, 4, 2])

        # Reshape to (B*num_classes*H*W, D, 1) for tf.image.resize
        logits_reshaped2 = tf.reshape(
            logits_permuted2,
            [batch_size * self.num_classes * original_shape[2] * original_shape[3], d_feat, 1]
        )

        # Add channel dimension for resize
        logits_reshaped2 = tf.expand_dims(logits_reshaped2, axis=-1)

        # Resize D dimension
        logits_d_upsampled = tf.image.resize(
            logits_reshaped2,
            size=[original_shape[1], 1],
            method='bilinear'
        )

        # Remove extra dimension and reshape
        logits_d_upsampled = tf.squeeze(logits_d_upsampled, axis=-1)
        logits_d_upsampled = tf.reshape(
            logits_d_upsampled,
            [batch_size, self.num_classes, original_shape[2], original_shape[3], original_shape[1]]
        )

        # Permute back: (B, num_classes, H, W, D) → (B, D, H, W, num_classes)
        output_logits = tf.transpose(logits_d_upsampled, [0, 4, 2, 3, 1])

        return output_logits

    def get_prototype_info(self):
        """
        Get information about prototypes for visualization/analysis.

        Returns:
            dict with prototype information
        """
        return {
            'num_prototypes': self.num_prototypes,
            'num_classes': self.num_classes,
            'prototypes_per_class': self.num_prototypes_per_class,
            'prototype_dim': self.prototype_dim,
            'prototype_vectors': self.prototype_vectors.numpy(),
            'prototype_class_identity': self.prototype_class_identity.numpy(),
            'fc_weights': self.fc_layer.get_weights()[0],
            'fc_bias': self.fc_layer.get_weights()[1]
        }

    def train_step(self, data):
        """
        Custom training step with diversity loss.

        Computes: L = L_CE + λ_J * L_J
        Where:
        - L_CE: Cross-entropy loss for segmentation
        - L_J: Diversity loss for prototypes

        Args:
            data: (images, labels) tuple

        Returns:
            dict of metrics
        """
        x, y = data

        with tf.GradientTape() as tape:
            # Forward pass to get logits
            y_pred = self(x, training=True)

            # Also need to get prototype activations for diversity loss
            # Re-compute forward pass components (inefficient but works for now)
            encoder_features = self.encoder(x, training=True)
            aspp_features = self.aspp(encoder_features, training=True)
            projected_features = self.feature_projection(aspp_features, training=True)

            # Compute prototype distances (needed for diversity loss)
            if self.f_dist == 'l2':
                prototype_distances = self.l2_convolution_3D(projected_features)
            elif self.f_dist == 'cosine':
                prototype_distances = self.cosine_convolution_3D(projected_features)
            else:
                raise NotImplementedError(f"Unknown distance metric: {self.f_dist}")

            # Compute cross-entropy loss
            ce_loss = self.compiled_loss(y, y_pred)

            # Compute diversity loss if enabled
            if hasattr(self, 'use_diversity_loss') and self.use_diversity_loss:
                from losses_protoseg import compute_diversity_loss

                diversity_loss = compute_diversity_loss(
                    y,
                    prototype_distances,
                    self.prototype_class_identity,
                    lambda_j=self.diversity_lambda
                )

                # Total loss
                total_loss = ce_loss + diversity_loss
            else:
                diversity_loss = 0.0
                total_loss = ce_loss

        # Compute gradients and update weights
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update metrics
        self.compiled_metrics.update_state(y, y_pred, sample_weight=None)

        # Return metrics dict
        metrics = {m.name: m.result() for m in self.metrics}
        metrics['loss'] = total_loss
        if hasattr(self, 'use_diversity_loss') and self.use_diversity_loss:
            metrics['ce_loss'] = ce_loss
            metrics['diversity_loss'] = diversity_loss

        return metrics

    def enable_diversity_loss(self, lambda_j=0.25):
        """
        Enable diversity loss for training.

        Args:
            lambda_j: Weight of diversity loss (default: 0.25 as in paper)
        """
        self.use_diversity_loss = True
        self.diversity_lambda = lambda_j
        print(f"✓ Diversity loss enabled (λ_J = {lambda_j})")

    def disable_diversity_loss(self):
        """Disable diversity loss for training."""
        self.use_diversity_loss = False
        print("✓ Diversity loss disabled")

    # ===================================================================
    # Multi-step training protocol methods
    # ===================================================================

    def freeze_encoder(self):
        """Freeze encoder layers (for warmup phase)."""
        self.encoder.trainable = False
        print("✓ Encoder frozen")

    def unfreeze_encoder(self):
        """Unfreeze encoder layers (for joint training)."""
        self.encoder.trainable = True
        print("✓ Encoder unfrozen")

    def freeze_aspp(self):
        """Freeze ASPP module."""
        self.aspp.trainable = False
        print("✓ ASPP frozen")

    def unfreeze_aspp(self):
        """Unfreeze ASPP module."""
        self.aspp.trainable = True
        print("✓ ASPP unfrozen")

    def freeze_fc_layer(self):
        """Freeze FC classification layer (for warmup and joint training)."""
        self.fc_layer.trainable = False
        print("✓ FC layer frozen")

    def unfreeze_fc_layer(self):
        """Unfreeze FC layer (for fine-tuning phases)."""
        self.fc_layer.trainable = True
        print("✓ FC layer unfrozen")

    def freeze_prototypes(self):
        """Freeze prototype vectors."""
        # Note: Variables can't be set trainable directly, need to exclude from trainable_variables
        self.prototype_vectors._trainable = False
        print("✓ Prototypes frozen")

    def unfreeze_prototypes(self):
        """Unfreeze prototype vectors."""
        self.prototype_vectors._trainable = True
        print("✓ Prototypes unfrozen")

    def freeze_feature_projection(self):
        """Freeze feature projection layer."""
        self.feature_projection.trainable = False
        print("✓ Feature projection frozen")

    def unfreeze_feature_projection(self):
        """Unfreeze feature projection layer."""
        self.feature_projection.trainable = True
        print("✓ Feature projection unfrozen")

    def setup_warmup_phase(self, rebuild_optimizer=True):
        """
        Setup for warmup phase:
        - Freeze encoder
        - Freeze FC layer
        - Train ASPP and prototypes only
        """
        print("\n" + "="*80)
        print("Setting up WARMUP PHASE")
        print("="*80)
        self.freeze_encoder()
        self.freeze_fc_layer()
        self.unfreeze_aspp()
        self.unfreeze_feature_projection()
        self.unfreeze_prototypes()
        print("Training: ASPP + Feature Projection + Prototypes")
        print("="*80 + "\n")

        if rebuild_optimizer and hasattr(self, 'optimizer'):
            print("⚠ Note: Optimizer needs to be rebuilt after changing trainable variables")

    def setup_joint_training_phase(self, rebuild_optimizer=True):
        """
        Setup for joint training phase:
        - Train everything except FC layer
        """
        print("\n" + "="*80)
        print("Setting up JOINT TRAINING PHASE")
        print("="*80)
        self.unfreeze_encoder()
        self.unfreeze_aspp()
        self.unfreeze_feature_projection()
        self.unfreeze_prototypes()
        self.freeze_fc_layer()
        print("Training: Encoder + ASPP + Feature Projection + Prototypes")
        print("Frozen: FC layer")
        print("="*80 + "\n")

        if rebuild_optimizer and hasattr(self, 'optimizer'):
            print("⚠ Note: Optimizer needs to be rebuilt after changing trainable variables")

    def setup_finetuning_phase(self, rebuild_optimizer=True):
        """
        Setup for fine-tuning phase:
        - Freeze everything except FC layer
        """
        print("\n" + "="*80)
        print("Setting up FINE-TUNING PHASE")
        print("="*80)
        self.freeze_encoder()
        self.freeze_aspp()
        self.freeze_feature_projection()
        self.freeze_prototypes()
        self.unfreeze_fc_layer()
        print("Training: FC layer only")
        print("Frozen: Encoder + ASPP + Feature Projection + Prototypes")
        print("="*80 + "\n")

        if rebuild_optimizer and hasattr(self, 'optimizer'):
            print("⚠ Note: Optimizer needs to be rebuilt after changing trainable variables")

    def print_trainable_status(self):
        """Print which components are trainable."""
        print("\nTrainable status:")
        print(f"  Encoder: {self.encoder.trainable}")
        print(f"  ASPP: {self.aspp.trainable}")
        print(f"  Feature projection: {self.feature_projection.trainable}")
        print(f"  Prototypes: {self.prototype_vectors._trainable if hasattr(self.prototype_vectors, '_trainable') else self.prototype_vectors.trainable}")
        print(f"  FC layer: {self.fc_layer.trainable}")
        print()
