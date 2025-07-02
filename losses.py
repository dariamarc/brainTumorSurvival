from tensorflow import keras
import tensorflow as tf

# --- FocalLoss (assuming it's already defined as per previous corrections) ---
class FocalLoss(keras.losses.Loss):
    def __init__(self, gamma=2.0, alpha=0.25, reduction=keras.losses.Reduction.AUTO, name='focal_loss', **kwargs):
        # FIX: Change reduction to a recognized value.
        # Since you are already doing tf.reduce_mean(loss) in call,
        # set this to NONE to prevent double reduction.
        super().__init__(reduction=tf.keras.losses.Reduction.NONE, name=name, **kwargs)  # <- Change this line

        self.gamma = tf.constant(gamma, dtype=tf.float32)
        self.alpha = tf.constant(alpha, dtype=tf.float32)

    def call(self, y_true, y_pred):
        y_pred_softmax = tf.nn.softmax(y_pred, axis=-1)
        y_true_float = tf.cast(y_true, tf.float32)
        pt = tf.reduce_sum(y_true_float * y_pred_softmax, axis=-1)
        pt = tf.clip_by_value(pt, tf.keras.backend.epsilon(), 1. - tf.keras.backend.epsilon())
        cross_entropy = -tf.math.log(pt)
        loss = self.alpha * tf.pow(1. - pt, self.gamma) * cross_entropy

        # You are already taking the mean here.
        return tf.reduce_mean(loss)

    def get_config(self):
        config = super().get_config()
        config.update({
            "gamma": float(self.gamma),
            "alpha": float(self.alpha),
        })
        return config
