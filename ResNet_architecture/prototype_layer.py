import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class PrototypeLayer(layers.Layer):
    """
    Prototype layer for interpretable segmentation.

    Computes similarity between input features and learnable prototype vectors.
    Each prototype corresponds to one tumor class.

    Input: (B, D, H, W, feature_dim) - e.g., (B, 20, 24, 16, 256)
    Output: (B, D, H, W, n_prototypes) - e.g., (B, 20, 24, 16, 3)
    """

    def __init__(self, n_prototypes=3, prototype_dim=256,
                 distance_type='l2', activation_function='log',
                 epsilon=1e-4, **kwargs):
        super(PrototypeLayer, self).__init__(**kwargs)
        self.n_prototypes = n_prototypes
        self.prototype_dim = prototype_dim
        self.distance_type = distance_type
        self.activation_function = activation_function
        self.epsilon = epsilon

    def build(self, input_shape):
        # Learnable prototype vectors: (n_prototypes, prototype_dim, 1, 1, 1)
        self.prototype_vectors = self.add_weight(
            name='prototype_vectors',
            shape=(self.n_prototypes, self.prototype_dim, 1, 1, 1),
            initializer='glorot_uniform',
            trainable=True
        )
        super(PrototypeLayer, self).build(input_shape)

    def _l2_distance(self, features):
        """
        Compute L2 distance between features and prototypes.

        features: (B, D, H, W, C)
        prototypes: (P, C, 1, 1, 1)
        returns: (B, D, H, W, P)
        """
        # Reshape prototypes to (C, P) for convolution as filters
        # TensorFlow conv3d expects filters as (kD, kH, kW, in_channels, out_channels)
        proto_filters = tf.transpose(self.prototype_vectors, perm=[2, 3, 4, 1, 0])
        # proto_filters: (1, 1, 1, C, P)

        # Compute dot product: features @ prototypes
        dot_product = tf.nn.conv3d(features, filters=proto_filters,
                                   strides=[1, 1, 1, 1, 1], padding='SAME')
        # dot_product: (B, D, H, W, P)

        # Compute ||features||^2 for each voxel
        features_sq = tf.reduce_sum(tf.square(features), axis=-1, keepdims=True)
        # features_sq: (B, D, H, W, 1)

        # Compute ||prototype||^2 for each prototype
        proto_sq = tf.reduce_sum(tf.square(self.prototype_vectors), axis=[1, 2, 3, 4])
        proto_sq = tf.reshape(proto_sq, [1, 1, 1, 1, self.n_prototypes])
        # proto_sq: (1, 1, 1, 1, P)

        # L2 distance: ||f - p||^2 = ||f||^2 - 2*f·p + ||p||^2
        distances = features_sq - 2 * dot_product + proto_sq
        distances = tf.maximum(distances, self.epsilon)
        distances = tf.sqrt(distances)

        return distances

    def _cosine_similarity(self, features):
        """
        Compute cosine similarity between features and prototypes.

        features: (B, D, H, W, C)
        prototypes: (P, C, 1, 1, 1)
        returns: (B, D, H, W, P)
        """
        proto_filters = tf.transpose(self.prototype_vectors, perm=[2, 3, 4, 1, 0])

        dot_product = tf.nn.conv3d(features, filters=proto_filters,
                                   strides=[1, 1, 1, 1, 1], padding='SAME')

        features_norm = tf.norm(features, axis=-1, keepdims=True)
        proto_norm = tf.norm(self.prototype_vectors, axis=[1, 2, 3, 4])
        proto_norm = tf.reshape(proto_norm, [1, 1, 1, 1, self.n_prototypes])

        similarity = dot_product / (features_norm * proto_norm + self.epsilon)

        return similarity

    def _distance_to_similarity(self, distances):
        """Convert L2 distances to similarity scores."""
        if self.activation_function == 'log':
            return tf.math.log((distances + 1.0) / (distances + self.epsilon))
        elif self.activation_function == 'linear':
            return -distances
        elif self.activation_function == 'exp':
            return tf.exp(-distances)
        else:
            raise ValueError(f"Unknown activation function: {self.activation_function}")

    def call(self, features, training=False):
        """
        Forward pass.

        Args:
            features: (B, D, H, W, C) feature maps from ASPP
            training: whether in training mode

        Returns:
            similarities: (B, D, H, W, n_prototypes) similarity maps
        """
        if self.distance_type == 'l2':
            distances = self._l2_distance(features)
            similarities = self._distance_to_similarity(distances)
        elif self.distance_type == 'cosine':
            similarities = self._cosine_similarity(features)
        else:
            raise ValueError(f"Unknown distance type: {self.distance_type}")

        return similarities

    def get_prototypes(self):
        """Return the prototype vectors."""
        return self.prototype_vectors

    def set_prototypes(self, new_prototypes):
        """Set prototype vectors (e.g., from data-driven initialization)."""
        self.prototype_vectors.assign(new_prototypes)

    def get_config(self):
        config = super().get_config()
        config.update({
            'n_prototypes': self.n_prototypes,
            'prototype_dim': self.prototype_dim,
            'distance_type': self.distance_type,
            'activation_function': self.activation_function,
            'epsilon': self.epsilon
        })
        return config


def create_prototype_layer(n_prototypes=3, prototype_dim=256,
                           distance_type='l2', activation_function='log'):
    """
    Factory function to create PrototypeLayer.

    Args:
        n_prototypes: Number of prototypes (default 3, one per tumor class)
        prototype_dim: Dimension of each prototype (default 256, matching ASPP output)
        distance_type: 'l2' or 'cosine'
        activation_function: 'log', 'linear', or 'exp'

    Returns:
        PrototypeLayer
    """
    return PrototypeLayer(
        n_prototypes=n_prototypes,
        prototype_dim=prototype_dim,
        distance_type=distance_type,
        activation_function=activation_function
    )
