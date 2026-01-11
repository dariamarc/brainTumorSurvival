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


@tf.function
def compute_diversity_loss(y_true, prototype_similarities, prototype_class_identity,
                          lambda_j=0.25, epsilon=1e-10):
    """
    Compute diversity loss based on Jeffrey's divergence.

    Encourages prototypes from the same class to activate on different spatial regions
    WITHIN LOCATIONS WHERE THAT CLASS APPEARS in the ground truth.

    From ProtoSeg paper, Equations 2-6.

    Args:
        y_true: Ground truth labels (B, D, H, W, num_classes) - one-hot encoded
        prototype_similarities: Prototype activations (B, D, H, W, num_prototypes)
                               Higher value = stronger activation
                               Already converted from distances using activation function
        prototype_class_identity: Class assignment (num_prototypes, num_classes)
        lambda_j: Weight of diversity loss (default: 0.25 as in paper)
        epsilon: Small constant for numerical stability

    Returns:
        diversity_loss: Scalar loss value
    """

    # Get shapes
    batch_size = tf.shape(y_true)[0]
    num_classes = tf.shape(y_true)[-1]
    num_prototypes = tf.shape(prototype_similarities)[-1]

    # Convert to float32
    y_true = tf.cast(y_true, tf.float32)
    prototype_similarities = tf.cast(prototype_similarities, tf.float32)
    prototype_class_identity = tf.cast(prototype_class_identity, tf.float32)

    # Downsample y_true to match prototype_similarities spatial resolution
    activation_shape = tf.shape(prototype_similarities)
    y_true_shape = tf.shape(y_true)

    def downsample_ytrue():
        """Downsample y_true to match similarity map spatial dimensions."""
        b, d, h, w, c = y_true_shape[0], y_true_shape[1], y_true_shape[2], y_true_shape[3], y_true_shape[4]

        # Reshape to treat depth slices as batch items
        y_true_reshaped = tf.reshape(y_true, [b * d, h, w, c])

        # Downsample spatially (H, W)
        y_true_resized = tf.image.resize(y_true_reshaped,
                                         size=[activation_shape[2], activation_shape[3]],
                                         method='nearest')

        # Reshape back to 5D
        new_h, new_w = activation_shape[2], activation_shape[3]
        y_true_downsampled = tf.reshape(y_true_resized, [b, d, new_h, new_w, c])

        # Now downsample depth dimension
        # Reshape to (B, C, D, H, W) for easier processing
        y_true_perm = tf.transpose(y_true_downsampled, [0, 4, 1, 2, 3])
        y_true_depth_reshape = tf.reshape(y_true_perm, [b * c, d, new_h * new_w])
        y_true_depth_reshape = tf.expand_dims(y_true_depth_reshape, axis=-1)

        # Resize depth
        y_true_depth_resized = tf.image.resize(y_true_depth_reshape,
                                              size=[activation_shape[1], new_h * new_w],
                                              method='nearest')

        y_true_depth_resized = tf.squeeze(y_true_depth_resized, axis=-1)
        new_d = activation_shape[1]
        y_true_final = tf.reshape(y_true_depth_resized, [b, c, new_d, new_h, new_w])
        y_true_final = tf.transpose(y_true_final, [0, 2, 3, 4, 1])

        return y_true_final

    # Check if downsampling is needed
    shapes_match = tf.reduce_all([
        tf.equal(activation_shape[1], y_true_shape[1]),
        tf.equal(activation_shape[2], y_true_shape[2]),
        tf.equal(activation_shape[3], y_true_shape[3])
    ])

    y_true = tf.cond(shapes_match, lambda: y_true, downsample_ytrue)

    # Accumulate loss across all classes and batches
    total_diversity_loss = tf.constant(0.0, dtype=tf.float32)
    num_valid_comparisons = tf.constant(0, dtype=tf.int32)

    # Loop over each class
    def class_loop_body(c, total_loss, valid_count):
        """Process one class."""
        # Find prototypes belonging to this class
        class_mask = prototype_class_identity[:, c] > 0.5
        class_proto_indices = tf.where(class_mask)
        num_class_protos = tf.shape(class_proto_indices)[0]

        def process_class():
            """Process class if it has at least 2 prototypes."""
            # Loop over batch
            def batch_loop_body(b, class_loss, class_count):
                """Process one batch item."""
                # Get ground truth mask for this class
                gt_class_mask = y_true[b, :, :, :, c]  # (D, H, W)
                class_locations = tf.where(gt_class_mask > 0.5)
                num_points = tf.shape(class_locations)[0]

                def compute_divergence():
                    """Compute Jeffrey's divergence if class appears in this sample."""
                    # Extract prototype distances at class locations
                    # For each prototype of this class, get its distances at class locations

                    # Use TensorArray for graph-compatible accumulation
                    proto_vectors_ta = tf.TensorArray(tf.float32, size=num_class_protos)

                    def proto_loop_body(p_idx, ta):
                        """Extract similarities for one prototype."""
                        proto_id = tf.cast(class_proto_indices[p_idx, 0], tf.int32)
                        similarity_map = prototype_similarities[b, :, :, :, proto_id]

                        # Get similarities at class locations
                        point_similarities = tf.gather_nd(similarity_map, class_locations)

                        # Convert to probability distribution (as in ProtoSeg paper)
                        # Square the similarities, then softmax
                        similarities_squared = tf.square(point_similarities)
                        v = tf.nn.softmax(similarities_squared)

                        ta = ta.write(p_idx, v)
                        return p_idx + 1, ta

                    # Run proto loop
                    _, proto_vectors_final = tf.while_loop(
                        lambda p_idx, _: tf.less(p_idx, num_class_protos),
                        proto_loop_body,
                        [tf.constant(0), proto_vectors_ta]
                    )

                    proto_vectors = proto_vectors_final.stack()  # (num_class_protos, num_points)

                    # Compute Jeffrey's similarity between all pairs
                    similarity = _jeffreys_similarity(proto_vectors, epsilon)

                    return similarity, tf.constant(1, dtype=tf.int32)

                def skip_sample():
                    """Skip if class doesn't appear in this sample."""
                    return tf.constant(0.0, dtype=tf.float32), tf.constant(0, dtype=tf.int32)

                loss_b, count_b = tf.cond(tf.greater(num_points, 0), compute_divergence, skip_sample)
                return b + 1, class_loss + loss_b, class_count + count_b

            # Run batch loop
            _, final_class_loss, final_class_count = tf.while_loop(
                lambda b, *_: tf.less(b, batch_size),
                batch_loop_body,
                [tf.constant(0), tf.constant(0.0, dtype=tf.float32), tf.constant(0, dtype=tf.int32)]
            )

            return final_class_loss, final_class_count

        def skip_class():
            """Skip class if it has fewer than 2 prototypes."""
            return tf.constant(0.0, dtype=tf.float32), tf.constant(0, dtype=tf.int32)

        class_loss, class_count = tf.cond(
            tf.greater_equal(num_class_protos, 2),
            process_class,
            skip_class
        )

        return c + 1, total_loss + class_loss, valid_count + class_count

    # Run class loop
    _, total_diversity_loss, num_valid_comparisons = tf.while_loop(
        lambda c, *_: tf.less(c, num_classes),
        class_loop_body,
        [tf.constant(0), total_diversity_loss, num_valid_comparisons]
    )

    # Average and weight
    avg_diversity_loss = tf.cond(
        tf.greater(num_valid_comparisons, 0),
        lambda: total_diversity_loss / tf.cast(num_valid_comparisons, tf.float32),
        lambda: tf.constant(0.0, dtype=tf.float32)
    )

    return lambda_j * avg_diversity_loss


def _jeffreys_similarity(distributions, epsilon=1e-10):
    """
    Compute average Jeffrey's similarity between all pairs of distributions.

    Args:
        distributions: (n, num_points) - n probability distributions
        epsilon: numerical stability constant

    Returns:
        Average similarity across all pairs
    """
    n = tf.shape(distributions)[0]

    def has_pairs():
        """Compute if we have at least 2 distributions."""
        # Create all pairs of indices
        i_indices, j_indices = tf.meshgrid(tf.range(n), tf.range(n), indexing='ij')
        i_flat = tf.reshape(i_indices, [-1])
        j_flat = tf.reshape(j_indices, [-1])

        # Filter for unique pairs where i < j
        pair_mask = tf.less(i_flat, j_flat)
        valid_i = tf.boolean_mask(i_flat, pair_mask)
        valid_j = tf.boolean_mask(j_flat, pair_mask)

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

        # Average similarity (this is what we want to minimize)
        avg_similarity = tf.reduce_mean(similarity)
        return avg_similarity

    def no_pairs():
        """Return 0 if we have fewer than 2 distributions."""
        return tf.constant(0.0, dtype=tf.float32)

    return tf.cond(tf.greater_equal(n, 2), has_pairs, no_pairs)
