import tensorflow as tf
from tensorflow import keras


class HybridSegmentationLoss(keras.losses.Loss):
    """
    Hybrid Segmentation Loss: L_hybrid = L_wCross + α × L_mDSC

    Combines two complementary loss functions for brain tumor segmentation:

    1. Volume Size Weighted Cross Entropy (wCross) - Eq. 2:
       - Weights each voxel by η_ℓ = 1 - |X_ℓ|/|X|
       - Minor classes (smaller volume) get larger weights
       - Preserves complex boundary details but may introduce noise

    2. Multi-class Dice Similarity Coefficient (mDSC) - Eq. 3:
       - Self-normalized per-class Dice loss
       - 1/N factor in denominator suppresses prediction noise
       - Generates compact, clear segmentation but may lose branchy details

    The hybrid combination (Eq. 4) produces "compact but details-enhanced" results.

    Reference: Volume-weighted cross entropy and multi-class DSC for medical
    image segmentation with class imbalance.
    """

    def __init__(self, alpha_mdsc=100.0, n_classes=4, epsilon=1e-6, **kwargs):
        """
        Args:
            alpha_mdsc: Weight for mDSC component (default 100.0 as per paper)
            n_classes: Number of segmentation classes (default 4)
            epsilon: Small constant for numerical stability
        """
        super(HybridSegmentationLoss, self).__init__(**kwargs)
        self.alpha_mdsc = alpha_mdsc
        self.n_classes = n_classes
        self.epsilon = epsilon

    def call(self, y_true, y_pred):
        """
        Compute hybrid loss: L_hybrid = L_wCross + α × L_mDSC

        Args:
            y_true: (B, D, H, W, n_classes) one-hot ground truth
            y_pred: (B, D, H, W, n_classes) predicted logits

        Returns:
            Hybrid loss scalar
        """
        wcross = self.weighted_cross_entropy_loss(y_true, y_pred)
        mdsc = self.multiclass_dice_loss(y_true, y_pred)
        return wcross + self.alpha_mdsc * mdsc

    def weighted_cross_entropy_loss(self, y_true, y_pred):
        """
        Volume Size Weighted Cross Entropy (wCross) - Eq. 2

        L_wCross = Σ -η_ℓ(xi) × log p(yi = ℓ(xi)|xi; W)
        where η_ℓ(xi) = 1 - |X_ℓ(xi)| / |X|

        Minor classes get larger weights, helping with class imbalance.
        Preserves complex boundary details.

        Args:
            y_true: (B, D, H, W, n_classes) one-hot ground truth
            y_pred: (B, D, H, W, n_classes) predicted logits

        Returns:
            Weighted cross entropy loss scalar
        """
        y_pred_soft = tf.nn.softmax(y_pred, axis=-1)

        # Flatten spatial dimensions: (B, D, H, W, C) -> (N, C)
        y_true_flat = tf.reshape(y_true, [-1, self.n_classes])
        y_pred_flat = tf.reshape(y_pred_soft, [-1, self.n_classes])

        # Total number of voxels |X|
        total_voxels = tf.cast(tf.shape(y_true_flat)[0], tf.float32)

        # Volume size per class |X_ℓ| = number of voxels belonging to each class
        class_volumes = tf.reduce_sum(y_true_flat, axis=0)  # (C,)

        # Compute per-class weights: η_ℓ = 1 - |X_ℓ| / |X|
        # Minor classes (smaller volume) get larger weights
        class_weights = 1.0 - (class_volumes / (total_voxels + self.epsilon))  # (C,)

        # Get the weight for each voxel based on its true class
        # y_true_flat is one-hot, so multiply and sum to get the weight for each voxel
        voxel_weights = tf.reduce_sum(y_true_flat * class_weights, axis=-1)  # (N,)

        # Compute cross entropy per voxel: -log p(y=ℓ|x)
        # Clip predictions to avoid log(0)
        y_pred_clipped = tf.clip_by_value(y_pred_flat, self.epsilon, 1.0 - self.epsilon)
        ce_per_voxel = -tf.reduce_sum(y_true_flat * tf.math.log(y_pred_clipped), axis=-1)  # (N,)

        # Weighted cross entropy
        weighted_ce = voxel_weights * ce_per_voxel

        return tf.reduce_mean(weighted_ce)

    def multiclass_dice_loss(self, y_true, y_pred):
        """
        Multi-class Dice Similarity Coefficient (mDSC) loss - Eq. 3

        L_mDSC = -Σ_c [ (2/N) × Σ_i(G_c^i × P_c^i) ] / [ (1/N)×Σ_i(G_c^i)² + (1/N)×Σ_i(P_c^i)² ]

        The 1/N in denominator suppresses prediction noise.
        Self-normalized per class, generates compact and clear segmentation.

        Args:
            y_true: (B, D, H, W, n_classes) one-hot ground truth
            y_pred: (B, D, H, W, n_classes) predicted logits

        Returns:
            Negative mDSC loss scalar (minimizing this maximizes Dice)
        """
        y_pred_soft = tf.nn.softmax(y_pred, axis=-1)

        # Flatten spatial dimensions: (B, D, H, W, C) -> (N, C)
        y_true_flat = tf.reshape(y_true, [-1, self.n_classes])
        y_pred_flat = tf.reshape(y_pred_soft, [-1, self.n_classes])

        # Total number of voxels N = w × h × d (per batch element, but we flatten all)
        N = tf.cast(tf.shape(y_true_flat)[0], tf.float32)

        # Compute mDSC for each class
        # Numerator: (2/N) × Σ_i(G_c^i × P_c^i)
        intersection = tf.reduce_sum(y_true_flat * y_pred_flat, axis=0)  # (C,)
        numerator = (2.0 / N) * intersection

        # Denominator: (1/N)×Σ_i(G_c^i × G_c^i) + (1/N)×Σ_i(P_c^i × P_c^i)
        # Since G is one-hot, G² = G
        sum_g_squared = tf.reduce_sum(y_true_flat * y_true_flat, axis=0) / N  # (C,)
        sum_p_squared = tf.reduce_sum(y_pred_flat * y_pred_flat, axis=0) / N  # (C,)
        denominator = sum_g_squared + sum_p_squared + self.epsilon

        # DSC per class
        dsc_per_class = numerator / denominator  # (C,)

        # Sum over all classes (as in Eq. 3, negative sum means we minimize -DSC = maximize DSC)
        mdsc = tf.reduce_sum(dsc_per_class)

        # Return negative mDSC as loss (we want to maximize DSC, so minimize -DSC)
        return -mdsc

    def get_config(self):
        config = super().get_config()
        config.update({
            'alpha_mdsc': self.alpha_mdsc,
            'n_classes': self.n_classes,
            'epsilon': self.epsilon
        })
        return config


def create_hybrid_loss(alpha_mdsc=100.0, n_classes=4):
    """
    Factory function to create HybridSegmentationLoss.

    Args:
        alpha_mdsc: Weight for mDSC in hybrid loss (default 100.0, as per paper)
        n_classes: Number of classes (default 4)

    Returns:
        HybridSegmentationLoss instance
    """
    return HybridSegmentationLoss(
        alpha_mdsc=alpha_mdsc,
        n_classes=n_classes
    )
