import tensorflow as tf
from tensorflow import keras


class PrototypeLoss(keras.losses.Loss):
    """
    Combined loss for prototype-based segmentation.

    Components:
        - Segmentation loss (Dice + CrossEntropy)
        - Purity loss (each prototype activates on one class)
        - Clustering loss (features close to assigned prototype)
        - Separation loss (prototypes far apart)
        - Interpretability loss (classifier weights maintain structure)
    """

    def __init__(self,
                 lambda_purity=0.5,
                 lambda_cluster=0.5,
                 lambda_separation=0.05,
                 lambda_interpretability=0.1,
                 sparsity_weight=0.01,
                 dice_weight=0.7,
                 ce_weight=0.3,
                 n_classes=4,
                 n_prototypes=3,
                 **kwargs):
        super(PrototypeLoss, self).__init__(**kwargs)
        self.lambda_purity = lambda_purity
        self.lambda_cluster = lambda_cluster
        self.lambda_separation = lambda_separation
        self.lambda_interpretability = lambda_interpretability
        self.sparsity_weight = sparsity_weight
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.n_classes = n_classes
        self.n_prototypes = n_prototypes
        self.epsilon = 1e-6

        # Build target structure for interpretability loss
        # Class 1 → proto 0, Class 2 → proto 1, Class 3 → proto 2
        target = tf.zeros((n_classes, n_prototypes))
        target = tf.tensor_scatter_nd_update(
            target,
            [[1, 0], [2, 1], [3, 2]],
            [1.0, 1.0, 1.0]
        )
        self.interpretability_target = target

    def call(self, y_true, y_pred):
        """
        Basic segmentation loss (Dice + CE).
        Use compute_total_loss() for full prototype loss.
        """
        return self.segmentation_loss(y_true, y_pred)

    def segmentation_loss(self, y_true, y_pred):
        """Combined Dice + CrossEntropy loss."""
        dice = self.dice_loss(y_true, y_pred)
        ce = self.cross_entropy_loss(y_true, y_pred)
        return self.dice_weight * dice + self.ce_weight * ce

    def dice_loss(self, y_true, y_pred):
        """Soft Dice loss for segmentation."""
        y_pred_soft = tf.nn.softmax(y_pred, axis=-1)

        # Flatten spatial dimensions
        y_true_flat = tf.reshape(y_true, [-1, self.n_classes])
        y_pred_flat = tf.reshape(y_pred_soft, [-1, self.n_classes])

        # Compute Dice per class
        intersection = tf.reduce_sum(y_true_flat * y_pred_flat, axis=0)
        union = tf.reduce_sum(y_true_flat, axis=0) + tf.reduce_sum(y_pred_flat, axis=0)

        dice_per_class = (2.0 * intersection + self.epsilon) / (union + self.epsilon)

        return 1.0 - tf.reduce_mean(dice_per_class)

    def cross_entropy_loss(self, y_true, y_pred):
        """Categorical cross-entropy loss."""
        return tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
        )

    def purity_loss(self, similarities, masks):
        """
        Encourage each prototype to activate strongly on only one class.

        Args:
            similarities: (B, D, H, W, n_prototypes) similarity maps
            masks: (B, D, H, W, n_classes) one-hot masks

        Returns:
            Purity loss scalar
        """
        loss = 0.0

        for proto_idx in range(self.n_prototypes):
            target_class = proto_idx + 1  # proto 0 → class 1, etc.

            # Get similarity for this prototype
            proto_sim = similarities[..., proto_idx]  # (B, D, H, W)

            # Target mask (where this prototype should activate)
            target_mask = masks[..., target_class]  # (B, D, H, W)

            # Non-target mask
            non_target_mask = 1.0 - target_mask

            # High similarity where mask matches (maximize)
            loss -= tf.reduce_mean(proto_sim * target_mask)

            # Low similarity elsewhere (minimize)
            loss += tf.reduce_mean(proto_sim * non_target_mask)

        return loss / self.n_prototypes

    def clustering_loss(self, features, prototypes, masks):
        """
        Features within same class should be close to their assigned prototype.

        Args:
            features: (B, D, H, W, C) feature maps from ASPP
            prototypes: (n_prototypes, C, 1, 1, 1) prototype vectors
            masks: (B, D, H, W, n_classes) one-hot masks

        Returns:
            Clustering loss scalar
        """
        loss = 0.0
        count = 0

        # Flatten prototypes to (n_prototypes, C)
        prototypes_flat = tf.reshape(prototypes, [self.n_prototypes, -1])

        for class_idx in range(1, self.n_classes):  # Skip background (class 0)
            proto_idx = class_idx - 1

            # Get mask for this class
            class_mask = masks[..., class_idx]  # (B, D, H, W)

            # Check if any voxels belong to this class
            mask_sum = tf.reduce_sum(class_mask)
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
            loss = loss / count

        return loss

    def separation_loss(self, prototypes):
        """
        Prototypes should be far apart from each other.

        Args:
            prototypes: (n_prototypes, C, 1, 1, 1) prototype vectors

        Returns:
            Separation loss scalar (minimize to maximize distances)
        """
        # Flatten prototypes to (n_prototypes, C)
        prototypes_flat = tf.reshape(prototypes, [self.n_prototypes, -1])

        # Compute pairwise distances
        # Using broadcasting: (P, 1, C) - (1, P, C) -> (P, P, C)
        diff = tf.expand_dims(prototypes_flat, 1) - tf.expand_dims(prototypes_flat, 0)
        distances = tf.norm(diff, axis=-1)  # (P, P)

        # Get non-diagonal elements (pairwise distances)
        mask = 1.0 - tf.eye(self.n_prototypes)
        pairwise_distances = distances * mask

        # Find minimum non-zero distance
        # Replace zeros with large value for min computation
        large_value = 1e6
        pairwise_distances_for_min = tf.where(
            pairwise_distances > 0,
            pairwise_distances,
            large_value * tf.ones_like(pairwise_distances)
        )
        min_distance = tf.reduce_min(pairwise_distances_for_min)

        # Maximize minimum distance (return negative log)
        return -tf.math.log(min_distance + self.epsilon)

    def interpretability_loss(self, classifier_weights):
        """
        Encourage classifier weights to maintain diagonal-like structure.

        Args:
            classifier_weights: (n_classes, n_prototypes) weight matrix

        Returns:
            Interpretability loss scalar
        """
        # L2 loss to target structure
        l2_loss = tf.reduce_mean(tf.square(classifier_weights - self.interpretability_target))

        # L1 sparsity regularization
        l1_loss = self.sparsity_weight * tf.reduce_sum(tf.abs(classifier_weights))

        return l2_loss + l1_loss

    def compute_total_loss(self, y_true, y_pred, similarities, features,
                           prototypes, classifier_weights):
        """
        Compute full prototype loss with all components.

        Args:
            y_true: (B, D, H, W, n_classes) ground truth masks
            y_pred: (B, D, H, W, n_classes) predicted logits
            similarities: (B, D, H, W, n_prototypes) similarity maps
            features: (B, D, H, W, C) features from ASPP
            prototypes: (n_prototypes, C, 1, 1, 1) prototype vectors
            classifier_weights: (n_classes, n_prototypes) classifier weights

        Returns:
            total_loss: Combined loss scalar
            loss_dict: Dictionary with individual loss components
        """
        # Segmentation loss
        seg_loss = self.segmentation_loss(y_true, y_pred)

        # Prototype-specific losses
        purity = self.purity_loss(similarities, y_true)
        cluster = self.clustering_loss(features, prototypes, y_true)
        separation = self.separation_loss(prototypes)
        interpretability = self.interpretability_loss(classifier_weights)

        # Total loss
        total_loss = (
            seg_loss +
            self.lambda_purity * purity +
            self.lambda_cluster * cluster +
            self.lambda_separation * separation +
            self.lambda_interpretability * interpretability
        )

        loss_dict = {
            'total_loss': total_loss,
            'seg_loss': seg_loss,
            'purity_loss': purity,
            'cluster_loss': cluster,
            'separation_loss': separation,
            'interpretability_loss': interpretability
        }

        return total_loss, loss_dict


def create_prototype_loss(lambda_purity=0.5,
                          lambda_cluster=0.5,
                          lambda_separation=0.05,
                          lambda_interpretability=0.1,
                          sparsity_weight=0.01,
                          dice_weight=0.7,
                          ce_weight=0.3,
                          n_classes=4,
                          n_prototypes=3):
    """
    Factory function to create PrototypeLoss.

    Args:
        lambda_purity: Weight for purity loss (default 0.5)
        lambda_cluster: Weight for clustering loss (default 0.5)
        lambda_separation: Weight for separation loss (default 0.05)
        lambda_interpretability: Weight for interpretability loss (default 0.1)
        sparsity_weight: L1 sparsity weight for classifier (default 0.01)
        dice_weight: Weight for Dice loss component (default 0.7)
        ce_weight: Weight for CrossEntropy component (default 0.3)
        n_classes: Number of classes (default 4)
        n_prototypes: Number of prototypes (default 3)

    Returns:
        PrototypeLoss instance
    """
    return PrototypeLoss(
        lambda_purity=lambda_purity,
        lambda_cluster=lambda_cluster,
        lambda_separation=lambda_separation,
        lambda_interpretability=lambda_interpretability,
        sparsity_weight=sparsity_weight,
        dice_weight=dice_weight,
        ce_weight=ce_weight,
        n_classes=n_classes,
        n_prototypes=n_prototypes
    )
