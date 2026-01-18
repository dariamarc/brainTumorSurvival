import tensorflow as tf
from tensorflow import keras


class ClusteringLoss(keras.losses.Loss):
    """
    Clustering Loss.

    Features from the same class should cluster around their assigned prototype.
    Computes L2 distance between class features and their prototype.
    Encourages backbone to produce features that align with prototypes.

    Used in Phase 2 for feature-prototype coherence.
    """

    def __init__(self, n_prototypes=3, n_classes=4, **kwargs):
        super(ClusteringLoss, self).__init__(**kwargs)
        self.n_prototypes = n_prototypes
        self.n_classes = n_classes
        self.epsilon = 1e-6

    def call(self, inputs, prototypes):
        """
        Compute Clustering loss.

        Args:
            inputs: tuple of (features, masks)
                - features: (B, D, H, W, C) feature maps from ASPP
                - masks: (B, D, H, W, n_classes) one-hot masks
            prototypes: (n_prototypes, C, 1, 1, 1) prototype vectors

        Returns:
            Clustering loss scalar
        """
        features, masks = inputs

        loss = 0.0
        count = 0

        # Flatten prototypes to (n_prototypes, C)
        prototypes_flat = tf.reshape(prototypes, [self.n_prototypes, -1])

        for class_idx in range(1, self.n_classes):  # Skip background (class 0)
            proto_idx = class_idx - 1

            # Get mask for this class
            class_mask = masks[..., class_idx]  # (B, D, H, W)

            # Count voxels in this class
            mask_sum = tf.reduce_sum(class_mask)

            # Skip if no voxels belong to this class
            if mask_sum < 1.0:
                continue

            # Expand mask for broadcasting with features
            class_mask_expanded = tf.expand_dims(class_mask, axis=-1)  # (B, D, H, W, 1)

            # Masked features
            masked_features = features * class_mask_expanded  # (B, D, H, W, C)

            # Compute mean feature for this class
            mean_feature = tf.reduce_sum(masked_features, axis=[0, 1, 2, 3]) / (mask_sum + self.epsilon)
            # (C,)

            # Distance to assigned prototype
            distance = tf.norm(mean_feature - prototypes_flat[proto_idx])
            loss += distance
            count += 1

        if count > 0:
            loss = loss / tf.cast(count, tf.float32)

        return loss

    def get_config(self):
        config = super().get_config()
        config.update({
            'n_prototypes': self.n_prototypes,
            'n_classes': self.n_classes
        })
        return config
