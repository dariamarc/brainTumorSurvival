import tensorflow as tf
from tensorflow import keras


class DiceLoss(keras.losses.Loss):
    """
    Weighted Soft Dice loss for segmentation.

    Measures overlap between predicted and ground truth segmentation.
    Works well for imbalanced classes common in medical imaging.

    Class weights allow ignoring background (weight=0) and focusing on tumor classes.
    """

    def __init__(self, smooth=1e-6, class_weights=None, num_classes=4, **kwargs):
        """
        Args:
            smooth: Smoothing factor to avoid division by zero
            class_weights: List of weights per class. Default [0, 1, 1, 1]
                          (0 for background, 1 for tumor classes)
            num_classes: Number of classes (default 4)
        """
        super(DiceLoss, self).__init__(**kwargs)
        self.smooth = smooth
        self.num_classes = num_classes

        # Default: ignore background (0), weight tumor classes equally (1)
        if class_weights is None:
            self.class_weights = [0.0] + [1.0] * (num_classes - 1)
        else:
            self.class_weights = class_weights

        self.class_weights_tensor = tf.constant(self.class_weights, dtype=tf.float32)

    def call(self, y_true, y_pred):
        """
        Compute weighted Dice loss.

        Args:
            y_true: (B, D, H, W, n_classes) one-hot ground truth
            y_pred: (B, D, H, W, n_classes) predicted logits

        Returns:
            Weighted Dice loss scalar
        """
        y_pred_soft = tf.nn.softmax(y_pred, axis=-1)

        # Flatten spatial dimensions
        y_true_flat = tf.reshape(y_true, [-1, tf.shape(y_true)[-1]])
        y_pred_flat = tf.reshape(y_pred_soft, [-1, tf.shape(y_pred_soft)[-1]])

        # Compute Dice per class
        intersection = tf.reduce_sum(y_true_flat * y_pred_flat, axis=0)
        union = tf.reduce_sum(y_true_flat, axis=0) + tf.reduce_sum(y_pred_flat, axis=0)

        dice_per_class = (2.0 * intersection + self.smooth) / (union + self.smooth)

        # Apply class weights
        weighted_dice = dice_per_class * self.class_weights_tensor

        # Compute weighted mean (only over classes with non-zero weight)
        weight_sum = tf.reduce_sum(self.class_weights_tensor)
        weighted_mean_dice = tf.reduce_sum(weighted_dice) / (weight_sum + self.smooth)

        return 1.0 - weighted_mean_dice

    def get_config(self):
        config = super().get_config()
        config.update({
            'smooth': self.smooth,
            'class_weights': self.class_weights,
            'num_classes': self.num_classes
        })
        return config
