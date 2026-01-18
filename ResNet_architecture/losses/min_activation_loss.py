import tensorflow as tf
from tensorflow import keras


class MinActivationLoss(keras.losses.Loss):
    """
    Minimum Activation Loss.

    Prevents "dead" prototypes that never activate.
    Each prototype must activate somewhere in the training data.
    Penalizes prototypes whose maximum activation is below a threshold.

    Ensures all prototypes are utilized during training.
    """

    def __init__(self, n_prototypes=3, activation_threshold=0.5, **kwargs):
        super(MinActivationLoss, self).__init__(**kwargs)
        self.n_prototypes = n_prototypes
        self.activation_threshold = activation_threshold

    def call(self, y_true, similarities):
        """
        Compute Minimum Activation loss.

        Args:
            y_true: Unused (required by Keras loss interface)
            similarities: (B, D, H, W, n_prototypes) prototype similarity maps

        Returns:
            Minimum activation loss scalar
        """
        loss = 0.0

        for proto_idx in range(self.n_prototypes):
            # Get similarity map for this prototype
            proto_sim = similarities[..., proto_idx]  # (B, D, H, W)

            # Find maximum activation for this prototype
            max_activation = tf.reduce_max(proto_sim)

            # Penalize if max activation is below threshold
            # ReLU ensures we only penalize when below threshold
            penalty = tf.nn.relu(self.activation_threshold - max_activation)

            loss += penalty

        return loss / self.n_prototypes

    def get_config(self):
        config = super().get_config()
        config.update({
            'n_prototypes': self.n_prototypes,
            'activation_threshold': self.activation_threshold
        })
        return config
