import tensorflow as tf
from tensorflow import keras


class DiversityLoss(keras.losses.Loss):
    """
    Prototype Diversity Loss.

    Prevents mode collapse by ensuring prototypes are different from each other.
    Measures cosine similarity between prototype pairs and penalizes high similarity.

    Used in Phase 1 to establish diverse initial prototypes.
    """

    def __init__(self, n_prototypes=3, **kwargs):
        super(DiversityLoss, self).__init__(**kwargs)
        self.n_prototypes = n_prototypes
        self.epsilon = 1e-8

    def call(self, y_true, prototypes):
        """
        Compute Diversity loss.

        Args:
            y_true: Unused (required by Keras loss interface)
            prototypes: (n_prototypes, C, 1, 1, 1) prototype vectors

        Returns:
            Diversity loss scalar
        """
        # Flatten prototypes to (n_prototypes, C)
        prototypes_flat = tf.reshape(prototypes, [self.n_prototypes, -1])

        # Normalize for cosine similarity
        prototypes_norm = tf.nn.l2_normalize(prototypes_flat, axis=-1)

        # Compute pairwise cosine similarity matrix
        similarity_matrix = tf.matmul(prototypes_norm, prototypes_norm, transpose_b=True)
        # (n_prototypes, n_prototypes)

        # Create mask to exclude diagonal (self-similarity)
        mask = 1.0 - tf.eye(self.n_prototypes)

        # Get off-diagonal similarities
        off_diagonal_sim = similarity_matrix * mask

        # Penalize high similarity (want prototypes to be different)
        # Mean of absolute similarities (penalize both positive and negative correlation)
        loss = tf.reduce_sum(tf.abs(off_diagonal_sim)) / (self.n_prototypes * (self.n_prototypes - 1))

        return loss

    def get_config(self):
        config = super().get_config()
        config.update({'n_prototypes': self.n_prototypes})
        return config
