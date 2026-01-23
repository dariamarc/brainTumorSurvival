"""
Prototype utilization metrics.

Computes activation statistics and coverage for each prototype.
"""

import tensorflow as tf
import numpy as np
from typing import Dict


class UtilizationMetrics:
    """Compute prototype utilization metrics."""

    def __init__(self, n_prototypes: int = 3, activation_threshold: float = 0.5):
        """
        Args:
            n_prototypes: Number of prototypes
            activation_threshold: Threshold for considering a voxel "activated"
        """
        self.n_prototypes = n_prototypes
        self.activation_threshold = activation_threshold

    def compute_activation_statistics(
        self,
        similarities: tf.Tensor
    ) -> Dict[str, float]:
        """
        Compute max and mean activation per prototype.

        Args:
            similarities: (B, D, H, W, n_prototypes) similarity maps

        Returns:
            Dict with keys: 'max_act_proto_{i}', 'mean_act_proto_{i}'
        """
        stats = {}

        for proto_idx in range(self.n_prototypes):
            proto_sim = similarities[..., proto_idx]  # (B, D, H, W)

            max_activation = tf.reduce_max(proto_sim)
            mean_activation = tf.reduce_mean(proto_sim)

            stats[f'max_act_proto_{proto_idx}'] = float(max_activation.numpy())
            stats[f'mean_act_proto_{proto_idx}'] = float(mean_activation.numpy())

        return stats

    def compute_activation_coverage(
        self,
        similarities: tf.Tensor
    ) -> Dict[str, float]:
        """
        Compute percentage of voxels above activation threshold per prototype.

        Args:
            similarities: (B, D, H, W, n_prototypes) similarity maps

        Returns:
            Dict with keys: 'coverage_proto_{i}' (as percentages 0-100)
        """
        coverage = {}
        total_voxels = tf.cast(tf.size(similarities[..., 0]), tf.float32)

        for proto_idx in range(self.n_prototypes):
            proto_sim = similarities[..., proto_idx]  # (B, D, H, W)

            # Count voxels above threshold
            activated_voxels = tf.reduce_sum(
                tf.cast(proto_sim > self.activation_threshold, tf.float32)
            )

            # Percentage
            coverage_pct = (activated_voxels / total_voxels) * 100.0
            coverage[f'coverage_proto_{proto_idx}'] = float(coverage_pct.numpy())

        return coverage

    def compute_all(
        self,
        similarities: tf.Tensor
    ) -> Dict[str, float]:
        """
        Compute all utilization metrics.

        Args:
            similarities: Similarity maps

        Returns:
            Dict with all utilization metrics:
            - max_act_proto_0, max_act_proto_1, max_act_proto_2
            - mean_act_proto_0, mean_act_proto_1, mean_act_proto_2
            - coverage_proto_0, coverage_proto_1, coverage_proto_2
        """
        metrics = self.compute_activation_statistics(similarities)
        metrics.update(self.compute_activation_coverage(similarities))

        return metrics
