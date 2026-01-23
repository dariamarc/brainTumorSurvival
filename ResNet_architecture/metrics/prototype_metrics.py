"""
Prototype interpretability metrics.

Computes purity ratios and per-prototype class activations.
"""

import tensorflow as tf
import numpy as np
from typing import Dict, Any


class PrototypeMetrics:
    """Compute prototype interpretability metrics."""

    def __init__(self, n_prototypes: int = 3, n_classes: int = 4):
        """
        Args:
            n_prototypes: Number of prototypes (default 3)
            n_classes: Number of classes including background (default 4)
        """
        self.n_prototypes = n_prototypes
        self.n_classes = n_classes
        # Mapping: proto 0 -> class 1, proto 1 -> class 2, proto 2 -> class 3
        self.proto_to_class = {i: i + 1 for i in range(n_prototypes)}
        self.epsilon = 1e-6

    def compute_purity_ratios(
        self,
        masks: tf.Tensor,
        similarities: tf.Tensor
    ) -> Dict[str, float]:
        """
        Compute purity ratio for each prototype.

        Purity ratio = (mean activation on target class) / (mean activation on other tumor classes)
        Higher values indicate the prototype is more class-specific.

        Args:
            masks: (B, D, H, W, n_classes) one-hot ground truth
            similarities: (B, D, H, W, n_prototypes) similarity maps

        Returns:
            Dict with keys: 'purity_proto_0', 'purity_proto_1',
                           'purity_proto_2', 'purity_mean'
        """
        purity = {}

        for proto_idx in range(self.n_prototypes):
            target_class = self.proto_to_class[proto_idx]

            # Get masks
            target_mask = masks[..., target_class]  # (B, D, H, W)

            # Other tumor classes (excluding background and target)
            other_tumor_mask = tf.zeros_like(target_mask)
            for c in range(1, self.n_classes):
                if c != target_class:
                    other_tumor_mask = other_tumor_mask + masks[..., c]

            # Get prototype similarities
            proto_sim = similarities[..., proto_idx]  # (B, D, H, W)

            # Compute mean activation on target class region
            target_voxels = tf.reduce_sum(target_mask)
            if target_voxels > 0:
                target_activation = tf.reduce_sum(proto_sim * target_mask) / (target_voxels + self.epsilon)
            else:
                target_activation = tf.constant(0.0)

            # Compute mean activation on other tumor class regions
            other_voxels = tf.reduce_sum(other_tumor_mask)
            if other_voxels > 0:
                other_activation = tf.reduce_sum(proto_sim * other_tumor_mask) / (other_voxels + self.epsilon)
            else:
                other_activation = tf.constant(self.epsilon)

            # Purity ratio
            ratio = float((target_activation / (other_activation + self.epsilon)).numpy())
            purity[f'purity_proto_{proto_idx}'] = ratio

        # Mean purity
        purity['purity_mean'] = float(np.mean([
            purity[f'purity_proto_{i}'] for i in range(self.n_prototypes)
        ]))

        return purity

    def compute_per_prototype_class_activations(
        self,
        masks: tf.Tensor,
        similarities: tf.Tensor
    ) -> np.ndarray:
        """
        Compute mean activation of each prototype on each class.

        Args:
            masks: (B, D, H, W, n_classes) one-hot ground truth
            similarities: (B, D, H, W, n_prototypes) similarity maps

        Returns:
            (n_prototypes, n_classes) matrix of mean activations
        """
        activation_matrix = np.zeros((self.n_prototypes, self.n_classes))

        for proto_idx in range(self.n_prototypes):
            proto_sim = similarities[..., proto_idx]  # (B, D, H, W)

            for class_idx in range(self.n_classes):
                class_mask = masks[..., class_idx]  # (B, D, H, W)

                # Count voxels in this class
                num_voxels = tf.reduce_sum(class_mask)

                if num_voxels > 0:
                    mean_activation = tf.reduce_sum(proto_sim * class_mask) / num_voxels
                    activation_matrix[proto_idx, class_idx] = float(mean_activation.numpy())

        return activation_matrix

    def compute_all(
        self,
        masks: tf.Tensor,
        similarities: tf.Tensor
    ) -> Dict[str, Any]:
        """
        Compute all prototype metrics.

        Args:
            masks: One-hot ground truth
            similarities: Similarity maps

        Returns:
            Dict with all prototype metrics:
            - purity_proto_0, purity_proto_1, purity_proto_2, purity_mean
            - proto_class_activations: (3, 4) matrix
        """
        metrics = self.compute_purity_ratios(masks, similarities)
        metrics['proto_class_activations'] = self.compute_per_prototype_class_activations(
            masks, similarities
        ).tolist()  # Convert to list for JSON serialization

        return metrics
