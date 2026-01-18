import tensorflow as tf
from tensorflow import keras


class DiceLoss(keras.losses.Loss):
    """
    Soft Dice loss for segmentation.

    Measures overlap between predicted and ground truth segmentation.
    Works well for imbalanced classes common in medical imaging.
    """

    def __init__(self, smooth=1e-6, **kwargs):
        super(DiceLoss, self).__init__(**kwargs)
        self.smooth = smooth

    def call(self, y_true, y_pred):
        """
        Compute Dice loss.

        Args:
            y_true: (B, D, H, W, n_classes) one-hot ground truth
            y_pred: (B, D, H, W, n_classes) predicted logits

        Returns:
            Dice loss scalar
        """
        y_pred_soft = tf.nn.softmax(y_pred, axis=-1)

        # Flatten spatial dimensions
        y_true_flat = tf.reshape(y_true, [-1, tf.shape(y_true)[-1]])
        y_pred_flat = tf.reshape(y_pred_soft, [-1, tf.shape(y_pred_soft)[-1]])

        # Compute Dice per class
        intersection = tf.reduce_sum(y_true_flat * y_pred_flat, axis=0)
        union = tf.reduce_sum(y_true_flat, axis=0) + tf.reduce_sum(y_pred_flat, axis=0)

        dice_per_class = (2.0 * intersection + self.smooth) / (union + self.smooth)

        return 1.0 - tf.reduce_mean(dice_per_class)

    def get_config(self):
        config = super().get_config()
        config.update({'smooth': self.smooth})
        return config
