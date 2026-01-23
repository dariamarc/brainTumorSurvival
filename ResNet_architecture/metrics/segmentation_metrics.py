"""
Segmentation metrics for brain tumor segmentation.

Computes Dice scores per class and whole tumor metrics.
"""

import tensorflow as tf
import numpy as np
from typing import Dict


class SegmentationMetrics:
    """Compute segmentation quality metrics for 3D brain tumor segmentation."""

    CLASS_NAMES = {
        1: 'gd_enhancing',  # GD-enhancing tumor
        2: 'edema',         # Peritumoral edema
        3: 'necrotic'       # Necrotic/non-enhancing tumor core
    }

    def __init__(self, num_classes: int = 4, smooth: float = 1e-6):
        """
        Args:
            num_classes: Number of segmentation classes (default 4: bg + 3 tumor)
            smooth: Smoothing factor for Dice computation to avoid division by zero
        """
        self.num_classes = num_classes
        self.smooth = smooth

    def compute_dice_per_class(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor
    ) -> Dict[str, float]:
        """
        Compute Dice coefficient for each tumor class.

        Args:
            y_true: (B, D, H, W, num_classes) one-hot ground truth
            y_pred: (B, D, H, W, num_classes) prediction logits

        Returns:
            Dict with keys: 'dice_gd_enhancing', 'dice_edema', 'dice_necrotic'
        """
        y_pred_soft = tf.nn.softmax(y_pred, axis=-1)
        dice_scores = {}

        for class_idx, name in self.CLASS_NAMES.items():
            y_true_class = y_true[..., class_idx]
            y_pred_class = y_pred_soft[..., class_idx]

            intersection = tf.reduce_sum(y_true_class * y_pred_class)
            union = tf.reduce_sum(y_true_class) + tf.reduce_sum(y_pred_class)

            dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
            dice_scores[f'dice_{name}'] = float(dice.numpy())

        return dice_scores

    def compute_dice_mean(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor
    ) -> float:
        """
        Compute mean Dice across tumor classes (excluding background).

        Args:
            y_true: One-hot ground truth
            y_pred: Prediction logits

        Returns:
            Mean Dice score
        """
        dice_per_class = self.compute_dice_per_class(y_true, y_pred)
        return float(np.mean(list(dice_per_class.values())))

    def compute_whole_tumor_dice(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor
    ) -> float:
        """
        Compute Dice for whole tumor (union of all tumor classes vs background).

        Args:
            y_true: One-hot ground truth
            y_pred: Prediction logits

        Returns:
            Whole tumor Dice coefficient
        """
        y_pred_soft = tf.nn.softmax(y_pred, axis=-1)

        # Whole tumor = sum of all non-background classes
        y_true_tumor = tf.reduce_sum(y_true[..., 1:], axis=-1)  # (B, D, H, W)
        y_pred_tumor = tf.reduce_sum(y_pred_soft[..., 1:], axis=-1)  # (B, D, H, W)

        # Clip to [0, 1] in case of overlaps
        y_true_tumor = tf.clip_by_value(y_true_tumor, 0.0, 1.0)
        y_pred_tumor = tf.clip_by_value(y_pred_tumor, 0.0, 1.0)

        intersection = tf.reduce_sum(y_true_tumor * y_pred_tumor)
        union = tf.reduce_sum(y_true_tumor) + tf.reduce_sum(y_pred_tumor)

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return float(dice.numpy())

    def compute_all(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor
    ) -> Dict[str, float]:
        """
        Compute all segmentation metrics.

        Args:
            y_true: One-hot ground truth
            y_pred: Prediction logits

        Returns:
            Dict with all 5 dice metrics:
            - dice_gd_enhancing
            - dice_edema
            - dice_necrotic
            - dice_mean
            - dice_whole_tumor
        """
        metrics = self.compute_dice_per_class(y_true, y_pred)
        metrics['dice_mean'] = float(np.mean([
            metrics['dice_gd_enhancing'],
            metrics['dice_edema'],
            metrics['dice_necrotic']
        ]))
        metrics['dice_whole_tumor'] = self.compute_whole_tumor_dice(y_true, y_pred)

        return metrics
