import tensorflow as tf
from tensorflow import keras


class PurityLoss(keras.losses.Loss):
    """
    Prototype Purity Loss.

    Encourages each prototype to activate strongly on only ONE tumor class.
    - High activation where prototype's target class exists
    - Low activation elsewhere

    Critical for establishing class-specific prototypes.
    """

    def __init__(self, n_prototypes=3, **kwargs):
        super(PurityLoss, self).__init__(**kwargs)
        self.n_prototypes = n_prototypes

    def call(self, y_true, similarities):
        """
        Compute Purity loss.

        Args:
            y_true: (B, D, H, W, n_classes) one-hot ground truth masks
            similarities: (B, D, H, W, n_prototypes) prototype similarity maps

        Returns:
            Purity loss scalar
        """
        loss = 0.0

        for proto_idx in range(self.n_prototypes):
            target_class = proto_idx + 1  # proto 0 → class 1, etc.

            # Get similarity for this prototype
            proto_sim = similarities[..., proto_idx]  # (B, D, H, W)

            # Target mask (where this prototype should activate)
            target_mask = y_true[..., target_class]  # (B, D, H, W)

            # Non-target mask
            non_target_mask = 1.0 - target_mask

            # High similarity where mask matches (maximize → negative)
            loss -= tf.reduce_mean(proto_sim * target_mask)

            # Low similarity elsewhere (minimize → positive)
            loss += tf.reduce_mean(proto_sim * non_target_mask)

        return loss / self.n_prototypes

    def get_config(self):
        config = super().get_config()
        config.update({'n_prototypes': self.n_prototypes})
        return config
