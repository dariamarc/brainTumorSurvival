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
        # Use tf.map_fn to process each sample in batch
        def process_sample(sample):
            # sample: (D, H, W, C)
            # Use tf.map_fn to process each channel
            def process_channel(channel):
                # channel: (D, H, W)
                channel_expanded = tf.expand_dims(channel, axis=-1)  # (D, H, W, 1)

                # Use tf.map_fn to process each depth slice
                def process_slice(slice_3d):
                    # slice_3d: (H, W, 1)
                    slice_batch = tf.expand_dims(slice_3d, axis=0)  # (1, H, W, 1)
                    # Resize using nearest neighbor
                    slice_resized = tf.image.resize(
                        slice_batch,
                        size=[activation_shape[2], activation_shape[3]],
                        method='nearest'
                    )
                    return slice_resized[0, :, :, 0]  # (H', W')

                # Process all depth slices
                channel_downsampled = tf.map_fn(
                    process_slice,
                    channel_expanded,
                    fn_output_signature=tf.TensorSpec(shape=[None, None], dtype=tf.float32)
                )
                return channel_downsampled  # (D, H', W')

            # Process all channels
            sample_downsampled = tf.map_fn(
                process_channel,
                tf.transpose(sample, [3, 0, 1, 2]),  # (C, D, H, W)
                fn_output_signature=tf.TensorSpec(shape=[None, None, None], dtype=tf.float32)
            )
            return tf.transpose(sample_downsampled, [1, 2, 3, 0])  # (D, H', W', C)

        # Process all samples in batch
        y_true_downsampled = tf.map_fn(
            process_sample,
            y_true,
            fn_output_signature=tf.TensorSpec(shape=[None, None, None, None], dtype=tf.float32)
        )
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

    total_diversity_loss = 0.0
    num_valid_pairs = 0

    # For each class
    for c in range(num_classes):
        # Get indices of prototypes belonging to this class
        class_mask = prototype_class_identity[:, c] > 0.5
        class_proto_indices = tf.where(class_mask)

        num_class_protos = tf.shape(class_proto_indices)[0]

        # Need at least 2 prototypes to compute pairwise similarity
        if num_class_protos < 2:
            continue

        # For each image in batch
        for b in range(batch_size):
            # Get ground truth mask for this class: (D, H, W)
            gt_class_mask = y_true[b, :, :, :, c]

            # Get spatial locations where this class is present
            class_locations = tf.where(gt_class_mask > 0.5)
            num_points = tf.shape(class_locations)[0]

            # Skip if no points of this class in this image
            if num_points == 0:
                continue

            # Compute prototype-class-image distance vectors
            proto_vectors = []

            for p_idx in range(num_class_protos):
                proto_id = class_proto_indices[p_idx, 0]

                # Get activation map for this prototype: (D, H, W)
                activation_map = prototype_activations[b, :, :, :, proto_id]

                # Extract activations at locations where class c is present
                point_activations = tf.gather_nd(activation_map, class_locations)

                # Convert distances to probabilities using softmax
                distances_squared = tf.square(point_activations)
                v = tf.nn.softmax(distances_squared)

                proto_vectors.append(v)

            # Compute Jeffrey's similarity between all pairs
            diversity_loss_per_class = _jeffreys_similarity(proto_vectors, epsilon)
            total_diversity_loss += diversity_loss_per_class
            num_valid_pairs += 1

    # Average over all valid class-image pairs
    if num_valid_pairs > 0:
        avg_diversity_loss = total_diversity_loss / tf.cast(num_valid_pairs, tf.float32)
    else:
        avg_diversity_loss = 0.0

    return lambda_j * avg_diversity_loss


def _jeffreys_similarity(distributions, epsilon=1e-10):
    """
    Compute Jeffrey's similarity: 1/C(n,2) * sum_{i<j} exp(-D_J(U_i, U_j))

    From ProtoSeg paper Equation 3.

    Args:
        distributions: List of probability distributions (each is a 1D tensor)
        epsilon: Small constant for numerical stability

    Returns:
        similarity: Scalar value in [0, 1]
            - 0 if distributions have disjoint supports (perfect diversity)
            - 1 if all distributions are identical (no diversity)
    """
    n = len(distributions)
    if n < 2:
        return 0.0

    total_similarity = 0.0
    num_pairs = 0

    # Compute pairwise Jeffrey's divergence
    for i in range(n):
        for j in range(i + 1, n):
            # Get distributions
            p = distributions[i] + epsilon
            q = distributions[j] + epsilon

            # Normalize to ensure they sum to 1
            p = p / tf.reduce_sum(p)
            q = q / tf.reduce_sum(q)

            # KL divergence: sum(p * log(p/q))
            kl_pq = tf.reduce_sum(p * tf.math.log(p / q))
            kl_qp = tf.reduce_sum(q * tf.math.log(q / p))

            # Jeffrey's divergence (Equation 2)
            dj = 0.5 * kl_pq + 0.5 * kl_qp

            # Convert to similarity
            similarity = tf.exp(-dj)

            total_similarity += similarity
            num_pairs += 1

    # Average similarity over all pairs
    avg_similarity = total_similarity / tf.cast(num_pairs, tf.float32)

    return avg_similarity


if __name__ == "__main__":
    """Test diversity loss computation"""
    print("\n" + "="*80)
    print("Testing ProtoSeg Diversity Loss")
    print("="*80 + "\n")

    # Create dummy data
    batch_size = 2
    D, H, W = 96, 20, 20  # Reduced spatial dims (after encoder)
    num_classes = 4
    num_prototypes = 28  # 7 per class

    # Ground truth (one-hot encoded)
    y_true = tf.random.uniform((batch_size, D, H, W, num_classes))
    y_true = tf.one_hot(tf.argmax(y_true, axis=-1), num_classes)

    # Prototype activations (distances)
    prototype_activations = tf.random.uniform((batch_size, D, H, W, num_prototypes))

    # Prototype class identity (7 prototypes per class)
    prototype_class_identity = np.repeat(np.eye(num_classes), 7, axis=0)
    prototype_class_identity = tf.constant(prototype_class_identity, dtype=tf.float32)

    print(f"Input shapes:")
    print(f"  y_true: {y_true.shape}")
    print(f"  prototype_activations: {prototype_activations.shape}")
    print(f"  prototype_class_identity: {prototype_class_identity.shape}")

    # Test diversity loss
    print("\n" + "-"*80)
    print("Testing Diversity Loss")
    print("-"*80)

    diversity = compute_diversity_loss(
        y_true,
        prototype_activations,
        prototype_class_identity,
        lambda_j=0.25
    )

    print(f"Diversity loss: {diversity.numpy():.6f}")
    print(f"Expected: Small positive value (encourages diversity)")

    # Test Jeffrey's divergence computation
    print("\n" + "-"*80)
    print("Testing Jeffrey's Divergence")
    print("-"*80)

    # Create two identical distributions (should have high similarity)
    dist1 = tf.constant([0.5, 0.3, 0.2])
    dist2 = tf.constant([0.5, 0.3, 0.2])

    sim_identical = _jeffreys_similarity([dist1, dist2])
    print(f"Similarity of identical distributions: {sim_identical.numpy():.6f}")
    print(f"  Expected: Close to 1.0 (high similarity = bad for diversity)")

    # Create two very different distributions (should have low similarity)
    dist3 = tf.constant([0.9, 0.05, 0.05])
    dist4 = tf.constant([0.05, 0.05, 0.9])

    sim_different = _jeffreys_similarity([dist3, dist4])
    print(f"Similarity of different distributions: {sim_different.numpy():.6f}")
    print(f"  Expected: Close to 0.0 (low similarity = good for diversity)")

    # Test with three distributions
    print("\n" + "-"*80)
    print("Testing with Multiple Prototypes")
    print("-"*80)

    dist_a = tf.constant([0.7, 0.2, 0.1])
    dist_b = tf.constant([0.1, 0.7, 0.2])
    dist_c = tf.constant([0.2, 0.1, 0.7])

    sim_diverse = _jeffreys_similarity([dist_a, dist_b, dist_c])
    print(f"Similarity of 3 diverse distributions: {sim_diverse.numpy():.6f}")
    print(f"  Expected: Low value (prototypes activate on different regions)")

    print("\n" + "="*80)
    print("✓ All tests completed successfully!")
    print("="*80 + "\n")

    print("Key insights:")
    print("  - Diversity loss encourages prototypes to activate on different regions")
    print("  - Jeffrey's similarity: 0 = diverse, 1 = identical")
    print("  - We MINIMIZE diversity loss to encourage prototype diversity")
    print("  - λ_J = 0.25 is recommended (from ProtoSeg paper)")
