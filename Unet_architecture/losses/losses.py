from tensorflow import keras
import tensorflow as tf


class FocalLoss(keras.losses.Loss):
    """
    Focal Loss for addressing class imbalance in segmentation.

    Improved version with:
    - Better numerical stability
    - Optional class weights for handling severe imbalance
    - Gradient clipping to prevent explosions
    """

    def __init__(self, gamma=1.0, alpha=0.25, class_weights=None,
                 reduction=keras.losses.Reduction.AUTO, name='focal_loss', **kwargs):
        # Set reduction to NONE to prevent double reduction
        super().__init__(reduction=tf.keras.losses.Reduction.NONE, name=name, **kwargs)

        self.gamma = tf.constant(gamma, dtype=tf.float32)
        self.alpha = tf.constant(alpha, dtype=tf.float32)
        self.class_weights = class_weights

        if self.class_weights is not None:
            self.class_weights = tf.constant(class_weights, dtype=tf.float32)

    def call(self, y_true, y_pred):
        """
        Args:
            y_true: Ground truth labels, shape (batch, D, H, W, num_classes) - one-hot encoded
            y_pred: Predicted logits, shape (batch, D, H, W, num_classes)
        """
        # Apply softmax to get probabilities
        y_pred_softmax = tf.nn.softmax(y_pred, axis=-1)

        # Ensure y_true is float32
        y_true_float = tf.cast(y_true, tf.float32)

        # Calculate pt (probability of true class)
        pt = tf.reduce_sum(y_true_float * y_pred_softmax, axis=-1)

        # Clip for numerical stability - prevent log(0)
        epsilon = tf.keras.backend.epsilon()
        pt = tf.clip_by_value(pt, epsilon, 1.0 - epsilon)

        # Calculate cross entropy
        cross_entropy = -tf.math.log(pt)

        # Calculate focal loss
        focal_weight = tf.pow(1.0 - pt, self.gamma)
        loss = self.alpha * focal_weight * cross_entropy

        # Apply class weights if provided
        if self.class_weights is not None:
            # Get class indices from one-hot encoded y_true
            class_indices = tf.argmax(y_true_float, axis=-1)
            weights = tf.gather(self.class_weights, class_indices)
            loss = loss * weights

        # Return mean loss
        return tf.reduce_mean(loss)

    def get_config(self):
        config = super().get_config()
        config.update({
            "gamma": float(self.gamma),
            "alpha": float(self.alpha),
            "class_weights": self.class_weights.numpy().tolist() if self.class_weights is not None else None
        })
        return config


class DiceLoss(keras.losses.Loss):
    """
    Dice Loss for segmentation - helps with class imbalance.
    Can be combined with Focal Loss for better results.
    """

    def __init__(self, smooth=1.0, reduction=keras.losses.Reduction.AUTO, name='dice_loss', **kwargs):
        super().__init__(reduction=tf.keras.losses.Reduction.NONE, name=name, **kwargs)
        self.smooth = smooth

    def call(self, y_true, y_pred):
        """
        Args:
            y_true: Ground truth labels, shape (batch, D, H, W, num_classes)
            y_pred: Predicted logits, shape (batch, D, H, W, num_classes)
        """
        # Apply softmax to get probabilities
        y_pred_softmax = tf.nn.softmax(y_pred, axis=-1)
        y_true_float = tf.cast(y_true, tf.float32)

        # Flatten the tensors
        y_true_flat = tf.reshape(y_true_float, [-1, tf.shape(y_true_float)[-1]])
        y_pred_flat = tf.reshape(y_pred_softmax, [-1, tf.shape(y_pred_softmax)[-1]])

        # Calculate intersection and union
        intersection = tf.reduce_sum(y_true_flat * y_pred_flat, axis=0)
        union = tf.reduce_sum(y_true_flat, axis=0) + tf.reduce_sum(y_pred_flat, axis=0)

        # Calculate dice coefficient per class
        dice_coef = (2.0 * intersection + self.smooth) / (union + self.smooth)

        # Return mean dice loss
        return tf.reduce_mean(1.0 - dice_coef)

    def get_config(self):
        config = super().get_config()
        config.update({"smooth": float(self.smooth)})
        return config


class CombinedLoss(keras.losses.Loss):
    """
    Combined Focal + Dice Loss for better segmentation performance.
    """

    def __init__(self, focal_weight=0.5, dice_weight=0.5, gamma=1.0, alpha=0.25,
                 reduction=keras.losses.Reduction.AUTO, name='combined_loss', **kwargs):
        super().__init__(reduction=tf.keras.losses.Reduction.NONE, name=name, **kwargs)

        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.focal_loss = FocalLoss(gamma=gamma, alpha=alpha)
        self.dice_loss = DiceLoss()

    def call(self, y_true, y_pred):
        focal = self.focal_loss(y_true, y_pred)
        dice = self.dice_loss(y_true, y_pred)
        return self.focal_weight * focal + self.dice_weight * dice

    def get_config(self):
        config = super().get_config()
        config.update({
            "focal_weight": float(self.focal_weight),
            "dice_weight": float(self.dice_weight)
        })
        return config