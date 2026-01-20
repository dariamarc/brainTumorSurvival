import tensorflow as tf


class ActivationConsistencyLoss:
    """
    Activation Consistency Loss.

    Within same class regions, prototype activations should be consistent.
    Measures variance of activations within each class.
    Encourages stable, predictable prototype responses.

    Helps network produce reliable explanations.
    Used in Phase 3 after prototype projection.
    """

    def __init__(self, n_prototypes=3, n_classes=4):
        self.n_prototypes = n_prototypes
        self.n_classes = n_classes
        self.epsilon = 1e-6

    def __call__(self, masks, similarities):
        """
        Compute Activation Consistency loss.

        Args:
            masks: (B, D, H, W, n_classes) one-hot masks
            similarities: (B, D, H, W, n_prototypes) prototype similarity maps

        Returns:
            Activation consistency loss scalar
        """
        loss = 0.0
        count = 0

        for class_idx in range(1, self.n_classes):  # Skip background
            proto_idx = class_idx - 1

            # Get mask for this class
            class_mask = masks[..., class_idx]  # (B, D, H, W)

            # Count voxels in this class
            mask_sum = tf.reduce_sum(class_mask)

            # Skip if too few voxels
            if mask_sum < 2.0:
                continue

            # Get similarity for the corresponding prototype
            proto_sim = similarities[..., proto_idx]  # (B, D, H, W)

            # Get activations within this class region
            masked_activations = proto_sim * class_mask

            # Compute mean activation within class
            mean_activation = tf.reduce_sum(masked_activations) / (mask_sum + self.epsilon)

            # Compute variance within class
            squared_diff = tf.square(proto_sim - mean_activation) * class_mask
            variance = tf.reduce_sum(squared_diff) / (mask_sum + self.epsilon)

            loss += variance
            count += 1

        if count > 0:
            loss = loss / tf.cast(count, tf.float32)

        return loss
