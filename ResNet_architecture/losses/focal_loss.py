import tensorflow as tf
from tensorflow import keras


class FocalLoss(keras.losses.Loss):
    """
    Focal Loss for handling class imbalance.

    Down-weights well-classified examples and focuses on hard negatives.
    Particularly useful for tumor segmentation where tumor regions are small.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, gamma=2.0, alpha=0.25, **kwargs):
        super(FocalLoss, self).__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = 1e-7

    def call(self, y_true, y_pred):
        """
        Compute Focal loss.

        Args:
            y_true: (B, D, H, W, n_classes) one-hot ground truth
            y_pred: (B, D, H, W, n_classes) predicted logits

        Returns:
            Focal loss scalar
        """
        y_pred_soft = tf.nn.softmax(y_pred, axis=-1)
        y_pred_soft = tf.clip_by_value(y_pred_soft, self.epsilon, 1.0 - self.epsilon)

        # Compute cross entropy
        cross_entropy = -y_true * tf.math.log(y_pred_soft)

        # Compute focal weight
        p_t = tf.reduce_sum(y_true * y_pred_soft, axis=-1, keepdims=True)
        focal_weight = tf.pow(1.0 - p_t, self.gamma)

        # Apply focal weight and alpha
        focal_loss = self.alpha * focal_weight * cross_entropy

        return tf.reduce_mean(tf.reduce_sum(focal_loss, axis=-1))

    def get_config(self):
        config = super().get_config()
        config.update({
            'gamma': self.gamma,
            'alpha': self.alpha
        })
        return config
