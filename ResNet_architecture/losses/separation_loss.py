import tensorflow as tf
from tensorflow import keras


class SeparationLoss(keras.losses.Loss):
    """
    Prototype Separation Loss.

    Prototypes should be far apart in feature space.
    Maximizes pairwise distances between prototypes.
    Stronger than diversity loss - enforces geometric separation.

    Prevents prototypes from drifting together during joint training.
    Used in Phase 2.
    """

    def __init__(self, n_prototypes=3, **kwargs):
        super(SeparationLoss, self).__init__(**kwargs)
        self.n_prototypes = n_prototypes
        self.epsilon = 1e-6

    def call(self, y_true, prototypes):
        """
        Compute Separation loss.

        Args:
            y_true: Unused (required by Keras loss interface)
            prototypes: (n_prototypes, C, 1, 1, 1) prototype vectors

        Returns:
            Separation loss scalar (minimize to maximize distances)
        """
        # Flatten prototypes to (n_prototypes, C)
        prototypes_flat = tf.reshape(prototypes, [self.n_prototypes, -1])

        # Compute pairwise distances
        # Using broadcasting: (P, 1, C) - (1, P, C) -> (P, P, C)
        diff = tf.expand_dims(prototypes_flat, 1) - tf.expand_dims(prototypes_flat, 0)
        distances = tf.norm(diff, axis=-1)  # (P, P)

        # Get non-diagonal elements (pairwise distances)
        mask = 1.0 - tf.eye(self.n_prototypes)
        pairwise_distances = distances * mask

        # Find minimum non-zero distance
        # Replace zeros with large value for min computation
        large_value = 1e6
        pairwise_distances_for_min = tf.where(
            pairwise_distances > 0,
            pairwise_distances,
            large_value * tf.ones_like(pairwise_distances)
        )
        min_distance = tf.reduce_min(pairwise_distances_for_min)

        # Maximize minimum distance (return negative log)
        return -tf.math.log(min_distance + self.epsilon)

    def get_config(self):
        config = super().get_config()
        config.update({'n_prototypes': self.n_prototypes})
        return config
