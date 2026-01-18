import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class InterpretableClassifier(layers.Layer):
    """
    Interpretable classifier that preserves prototype-class relationships.

    Uses a constrained weight matrix where each class is primarily associated
    with one prototype, making the decision process transparent.

    Initialization:
        - Class 1 (GD-enhancing) → mainly prototype 0
        - Class 2 (Edema) → mainly prototype 1
        - Class 3 (Necrotic) → mainly prototype 2
        - Class 0 (Background) → negatively correlated with all prototypes

    Input: (B, D, H, W, n_prototypes)
    Output: (B, D, H, W, n_classes)
    """

    def __init__(self, n_prototypes=3, n_classes=4, **kwargs):
        super(InterpretableClassifier, self).__init__(**kwargs)
        self.n_prototypes = n_prototypes
        self.n_classes = n_classes

    def build(self, input_shape):
        # Learnable weights: (n_classes, n_prototypes)
        self.weights_matrix = self.add_weight(
            name='weights_matrix',
            shape=(self.n_classes, self.n_prototypes),
            initializer='zeros',
            trainable=True
        )

        # Bias: (n_classes,)
        self.bias = self.add_weight(
            name='bias',
            shape=(self.n_classes,),
            initializer='zeros',
            trainable=True
        )

        # Initialize with identity-like structure
        initial_weights = tf.zeros((self.n_classes, self.n_prototypes))

        # Tumor classes: each linked to one prototype
        # Class 1 (GD-enhancing) → prototype 0
        # Class 2 (Edema) → prototype 1
        # Class 3 (Necrotic) → prototype 2
        indices = [
            [1, 0],  # Class 1 → proto 0
            [2, 1],  # Class 2 → proto 1
            [3, 2],  # Class 3 → proto 2
        ]
        updates = [1.0, 1.0, 1.0]
        initial_weights = tf.tensor_scatter_nd_update(
            initial_weights, indices, updates
        )

        # Background class: negatively correlated with all prototypes
        background_weights = tf.constant([[-0.3, -0.3, -0.3]], dtype=tf.float32)
        initial_weights = tf.tensor_scatter_nd_update(
            initial_weights,
            [[0, 0], [0, 1], [0, 2]],
            [-0.3, -0.3, -0.3]
        )

        self.weights_matrix.assign(initial_weights)

        super(InterpretableClassifier, self).build(input_shape)

    def call(self, similarities, training=False):
        """
        Forward pass.

        Args:
            similarities: (B, D, H, W, n_prototypes) similarity maps

        Returns:
            logits: (B, D, H, W, n_classes) class logits
        """
        input_shape = tf.shape(similarities)
        batch_size = input_shape[0]
        D, H, W = input_shape[1], input_shape[2], input_shape[3]

        # Reshape for matrix multiply: (B*D*H*W, n_prototypes)
        sim_flat = tf.reshape(similarities, [-1, self.n_prototypes])

        # Apply learned linear transformation: (B*D*H*W, n_classes)
        # sim_flat @ weights^T + bias
        logits_flat = tf.matmul(sim_flat, self.weights_matrix, transpose_b=True) + self.bias

        # Reshape back to (B, D, H, W, n_classes)
        logits = tf.reshape(logits_flat, [batch_size, D, H, W, self.n_classes])

        return logits

    def get_prototype_contributions(self):
        """
        For interpretability: which prototypes contribute to each class?

        Returns:
            Softmax-normalized weights showing prototype contributions per class.
            Shape: (n_classes, n_prototypes)
        """
        return tf.nn.softmax(self.weights_matrix, axis=1)

    def get_weights_matrix(self):
        """Return raw weights matrix for inspection."""
        return self.weights_matrix

    def get_config(self):
        config = super().get_config()
        config.update({
            'n_prototypes': self.n_prototypes,
            'n_classes': self.n_classes
        })
        return config


def create_interpretable_classifier(n_prototypes=3, n_classes=4):
    """
    Factory function to create InterpretableClassifier.

    Args:
        n_prototypes: Number of prototypes (default 3)
        n_classes: Number of output classes (default 4)

    Returns:
        InterpretableClassifier layer
    """
    return InterpretableClassifier(n_prototypes=n_prototypes, n_classes=n_classes)
