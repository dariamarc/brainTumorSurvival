"""
ProtoSeg Loss Functions

Implements diversity loss based on Jeffrey's divergence as described in:
"ProtoSeg: Interpretable Semantic Segmentation with Prototypical Parts" (WACV 2023)

Key components:
1. Jeffrey's Divergence: Symmetrized KL divergence
2. Jeffrey's Similarity: Measures how similar prototype activations are
3. Diversity Loss: Encourages prototypes from same class to activate on different regions
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np


@tf.function
def compute_diversity_loss(y_true, prototype_activations, prototype_class_identity,
                          lambda_j=0.25, epsilon=1e-10):
    """
    Compute diversity loss based on Jeffrey's divergence.

    Encourages prototypes from the same class to activate on different spatial regions.
    This prevents redundant prototypes and improves interpretability.

    From ProtoSeg paper, Equations 2-6.

    Args:
        y_true: Ground truth labels (B, D, H, W, num_classes) - one-hot encoded
        prototype_activations: Prototype distances (B, D, H, W, num_prototypes)
        prototype_class_identity: Class assignment (num_prototypes, num_classes)
        lambda_j: Weight of diversity loss (default: 0.25 as in paper)
        epsilon: Small constant for numerical stability

    Returns:
        diversity_loss: Scalar loss value
    """

    # Get shapes
    batch_size = tf.shape(y_true)[0]
    num_classes = tf.shape(y_true)[-1]

    # Convert to float32
    y_true = tf.cast(y_true, tf.float32)
    prototype_activations = tf.cast(prototype_activations, tf.float32)
    prototype_class_identity = tf.cast(prototype_class_identity, tf.float32)

    # IMPORTANT: Downsample y_true to match prototype_activations spatial resolution
    # y_true: (B, D, H, W, C) - full resolution
    # prototype_activations: (B, D', H', W', M) - reduced resolution
    activation_shape = tf.shape(prototype_activations)
    y_true_shape = tf.shape(y_true)

    # Helper function to downsample y_true
    def downsample_ytrue():
        """Downsample y_true to match activation spatial dimensions."""
        y_true_shape = tf.shape(y_true)
        b, d, h, w, c = y_true_shape[0], y_true_shape[1], y_true_shape[2], y_true_shape[3], y_true_shape[4]

        # Reshape to treat depth slices as batch items
        y_true_reshaped = tf.reshape(y_true, [b * d, h, w, c])

        # Downsample spatially
        y_true_resized = tf.image.resize(y_true_reshaped,
                                         size=[activation_shape[2], activation_shape[3]],
                                         method='nearest')

        # Reshape back to original rank
        new_h, new_w = activation_shape[2], activation_shape[3]
        y_true_downsampled = tf.reshape(y_true_resized, [b, d, new_h, new_w, c])

        return y_true_downsampled

    # Check if downsampling is needed using tf.cond
    shapes_match = tf.reduce_all([
        tf.equal(activation_shape[1], y_true_shape[1]),
        tf.equal(activation_shape[2], y_true_shape[2]),
        tf.equal(activation_shape[3], y_true_shape[3])
    ])

    y_true = tf.cond(
        shapes_match,
        lambda: y_true,
        downsample_ytrue
    )

    total_diversity_loss = tf.constant(0.0, dtype=tf.float32)
    num_valid_pairs = tf.constant(0, dtype=tf.int32)

    # Loop over each class using tf.while_loop
    c = tf.constant(0)
    cond_c = lambda c, *_: tf.less(c, num_classes)

    def body_c(c, total_loss, valid_pairs):
        class_mask = prototype_class_identity[:, c] > 0.5
        class_proto_indices = tf.where(class_mask)
        num_class_protos = tf.shape(class_proto_indices)[0]

        # Conditional execution for classes with at least 2 prototypes
        def has_enough_protos():
            # Loop over each image in batch
            b = tf.constant(0)
            cond_b = lambda b, *_: tf.less(b, batch_size)

            def body_b(b, class_loss, class_valid_pairs):
                gt_class_mask = y_true[b, :, :, :, c]
                class_locations = tf.where(gt_class_mask > 0.5)
                num_points = tf.shape(class_locations)[0]

                # Conditional execution for images with class points
                def has_points():
                    # Inner loop for prototype vectors
                    p_idx = tf.constant(0)
                    proto_vectors_ta = tf.TensorArray(tf.float32, size=num_class_protos)

                    cond_p = lambda p_idx, *_: tf.less(p_idx, num_class_protos)

                    def body_p(p_idx, ta):
                        proto_id = class_proto_indices[p_idx, 0]
                        proto_id = tf.cast(proto_id, tf.int32)
                        activation_map = prototype_activations[b, :, :, :, proto_id]
                        point_activations = tf.gather_nd(activation_map, class_locations)
                        distances_squared = tf.square(point_activations)
                        v = tf.nn.softmax(distances_squared)
                        ta = ta.write(p_idx, v)
                        return p_idx + 1, ta

                    _, proto_vectors_final_ta = tf.while_loop(cond_p, body_p, [p_idx, proto_vectors_ta])
                    proto_vectors = proto_vectors_final_ta.stack()

                    # Compute Jeffrey's similarity
                    diversity_loss_per_class = _jeffreys_similarity(proto_vectors, epsilon)
                    return diversity_loss_per_class, tf.constant(1, dtype=tf.int32)

                def no_points():
                    return tf.constant(0.0, dtype=tf.float32), tf.constant(0, dtype=tf.int32)

                loss_per_b, pairs_per_b = tf.cond(tf.greater(num_points, 0), has_points, no_points)
                return b + 1, class_loss + loss_per_b, class_valid_pairs + pairs_per_b

            # Initial values for batch loop
            b_init = tf.constant(0)
            class_loss_init = tf.constant(0.0, dtype=tf.float32)
            class_valid_pairs_init = tf.constant(0, dtype=tf.int32)
            _, final_class_loss, final_class_pairs = tf.while_loop(cond_b, body_b, [b_init, class_loss_init, class_valid_pairs_init])
            return final_class_loss, final_class_pairs

        def not_enough_protos():
            return tf.constant(0.0, dtype=tf.float32), tf.constant(0, dtype=tf.int32)

        class_loss, class_pairs = tf.cond(tf.greater_equal(num_class_protos, 2), has_enough_protos, not_enough_protos)
        return c + 1, total_loss + class_loss, valid_pairs + class_pairs

    # Initial values for class loop
    c_init = tf.constant(0)
    total_loss_init = tf.constant(0.0, dtype=tf.float32)
    valid_pairs_init = tf.constant(0, dtype=tf.int32)
    _, total_diversity_loss, num_valid_pairs = tf.while_loop(cond_c, body_c, [c_init, total_loss_init, valid_pairs_init])

    # Final averaging
    avg_diversity_loss = tf.cond(
        tf.greater(num_valid_pairs, 0),
        lambda: total_diversity_loss / tf.cast(num_valid_pairs, tf.float32),
        lambda: tf.constant(0.0, dtype=tf.float32)
    )

    return lambda_j * avg_diversity_loss


def _jeffreys_similarity(distributions, epsilon=1e-10):
    """
    TensorFlow-native implementation of Jeffrey's similarity.
    """
    n = tf.shape(distributions)[0]

    def has_pairs():
        # Create all pairs of indices
        i_indices, j_indices = tf.meshgrid(tf.range(n), tf.range(n), indexing='ij')
        i_flat = tf.reshape(i_indices, [-1])
        j_flat = tf.reshape(j_indices, [-1])
        
        # Filter for unique pairs where i < j
        pair_mask = tf.less(i_flat, j_flat)
        valid_i = tf.boolean_mask(i_flat, pair_mask)
        valid_j = tf.boolean_mask(j_flat, pair_mask)
        
        num_pairs = tf.shape(valid_i)[0]

        # Gather distributions for all pairs
        p_dist = tf.gather(distributions, valid_i)
        q_dist = tf.gather(distributions, valid_j)

        # Add epsilon and normalize
        p = (p_dist + epsilon) / tf.reduce_sum(p_dist + epsilon, axis=1, keepdims=True)
        q = (q_dist + epsilon) / tf.reduce_sum(q_dist + epsilon, axis=1, keepdims=True)

        # KL Divergence for all pairs
        kl_pq = tf.reduce_sum(p * tf.math.log(p / q), axis=1)
        kl_qp = tf.reduce_sum(q * tf.math.log(q / p), axis=1)

        # Jeffrey's Divergence
        dj = 0.5 * (kl_pq + kl_qp)
        
        # Similarity
        similarity = tf.exp(-dj)
        
        # Average similarity
        avg_similarity = tf.reduce_mean(similarity)
        return avg_similarity

    def no_pairs():
        return tf.constant(0.0, dtype=tf.float32)

    return tf.cond(tf.greater_equal(n, 2), has_pairs, no_pairs)
