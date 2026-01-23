import tensorflow as tf
import numpy as np
from tqdm import tqdm


class PrototypeProjector:
    """
    Projects learned prototypes onto actual training data patches.

    Finds the nearest training patch to each learned prototype and replaces
    the abstract prototype with a concrete, visualizable feature vector.

    This creates interpretable prototypes that correspond to real brain regions.
    """

    def __init__(self, model, n_prototypes=3, n_classes=4):
        self.model = model
        self.n_prototypes = n_prototypes
        self.n_classes = n_classes

    def project_prototypes(self, data_generator, max_batches=None):
        """
        Project prototypes onto nearest training data features.

        Args:
            data_generator: Data generator yielding (images, masks) batches
            max_batches: Optional limit on batches to process (None = all)

        Returns:
            projection_info: Dictionary containing:
                - 'projected_prototypes': New prototype vectors from real data
                - 'projection_metadata': Location info for each prototype
                - 'distances': Distance from original to projected prototype
        """
        # Get current learned prototypes
        learned_prototypes = self.model.get_prototypes().numpy()
        # Shape: (n_prototypes, C, 1, 1, 1)
        learned_prototypes_flat = learned_prototypes.reshape(self.n_prototypes, -1)

        # Initialize storage for best matches
        best_features = [None] * self.n_prototypes
        best_distances = [float('inf')] * self.n_prototypes
        best_metadata = [None] * self.n_prototypes

        # Process batches
        n_batches = len(data_generator) if max_batches is None else min(max_batches, len(data_generator))

        print(f"Projecting prototypes using {n_batches} batches...")

        for batch_idx in tqdm(range(n_batches), desc="Processing batches"):
            images, masks = data_generator[batch_idx]

            # Get ASPP features (at reduced resolution)
            features = self.model.get_features(images, training=False)
            # Shape: (B, D', H', W', C) e.g., (B, 20, 24, 16, 256)

            # Downsample masks to match feature resolution
            masks_downsampled = self._downsample_masks(masks, features.shape[1:4])
            # Shape: (B, D', H', W', n_classes)

            # Find nearest features for each prototype
            self._update_best_matches(
                features=features.numpy(),
                masks=masks_downsampled.numpy(),
                learned_prototypes_flat=learned_prototypes_flat,
                best_features=best_features,
                best_distances=best_distances,
                best_metadata=best_metadata,
                batch_idx=batch_idx
            )

        # Build projected prototypes tensor
        projected_prototypes = np.stack(best_features, axis=0)
        # Reshape to (n_prototypes, C, 1, 1, 1)
        C = learned_prototypes_flat.shape[1]
        projected_prototypes = projected_prototypes.reshape(self.n_prototypes, C, 1, 1, 1)

        projection_info = {
            'projected_prototypes': projected_prototypes,
            'projection_metadata': best_metadata,
            'distances': best_distances,
            'original_prototypes': learned_prototypes
        }

        # Print summary
        self._print_projection_summary(projection_info)

        return projection_info

    def _downsample_masks(self, masks, target_shape):
        """
        Downsample masks to match feature resolution.

        Args:
            masks: (B, D, H, W, n_classes) one-hot masks
            target_shape: (D', H', W') target spatial dimensions

        Returns:
            Downsampled masks (B, D', H', W', n_classes)
        """
        # Convert one-hot to class indices for downsampling
        mask_indices = tf.argmax(masks, axis=-1)  # (B, D, H, W)

        # Resize using nearest neighbor to preserve class labels
        batch_size = tf.shape(masks)[0]
        n_classes = masks.shape[-1]

        # Process each sample in batch
        downsampled_list = []

        for b in range(masks.shape[0]):
            mask_3d = mask_indices[b]  # (D, H, W)

            # Downsample D dimension
            mask_3d = tf.cast(mask_3d, tf.float32)

            # Reshape for 2D resize: (D, H*W)
            D, H, W = mask_3d.shape
            mask_2d = tf.reshape(mask_3d, [D, H * W])
            mask_2d = tf.expand_dims(mask_2d, axis=-1)  # (D, H*W, 1)

            # Resize D dimension
            mask_2d = tf.image.resize(mask_2d, [target_shape[0], H * W], method='nearest')
            mask_3d = tf.reshape(mask_2d, [target_shape[0], H, W])

            # Resize H, W dimensions
            mask_3d = tf.expand_dims(mask_3d, axis=-1)  # (D', H, W, 1)
            mask_resized = tf.image.resize(
                tf.reshape(mask_3d, [target_shape[0], H, W, 1]),
                [target_shape[1], target_shape[2]],
                method='nearest'
            )
            mask_resized = tf.squeeze(mask_resized, axis=-1)  # (D', H', W')

            # Convert back to one-hot
            mask_onehot = tf.one_hot(tf.cast(mask_resized, tf.int32), depth=n_classes)
            downsampled_list.append(mask_onehot)

        return tf.stack(downsampled_list, axis=0)

    def _update_best_matches(self, features, masks, learned_prototypes_flat,
                             best_features, best_distances, best_metadata, batch_idx):
        """
        Update best matches for each prototype based on current batch.

        Args:
            features: (B, D', H', W', C) numpy array
            masks: (B, D', H', W', n_classes) numpy array
            learned_prototypes_flat: (n_prototypes, C) numpy array
            best_features: List of best feature vectors found so far
            best_distances: List of best distances found so far
            best_metadata: List of metadata for best matches
            batch_idx: Current batch index
        """
        B, D, H, W, C = features.shape

        for proto_idx in range(self.n_prototypes):
            target_class = proto_idx + 1  # proto 0 → class 1, etc.

            # Get the learned prototype
            prototype = learned_prototypes_flat[proto_idx]

            # Find voxels belonging to target class
            class_mask = masks[..., target_class]  # (B, D', H', W')

            # Get indices where mask is active
            indices = np.where(class_mask > 0.5)

            if len(indices[0]) == 0:
                continue  # No voxels of this class in batch

            # Extract features at these locations
            class_features = features[indices]  # (N, C)

            # Compute distances to prototype
            distances = np.linalg.norm(class_features - prototype, axis=1)

            # Find minimum distance in this batch
            min_idx = np.argmin(distances)
            min_distance = distances[min_idx]

            # Update if better than current best
            if min_distance < best_distances[proto_idx]:
                best_distances[proto_idx] = min_distance
                best_features[proto_idx] = class_features[min_idx]
                best_metadata[proto_idx] = {
                    'batch_idx': batch_idx,
                    'sample_idx': int(indices[0][min_idx]),
                    'voxel_coords': {
                        'd': int(indices[1][min_idx]),
                        'h': int(indices[2][min_idx]),
                        'w': int(indices[3][min_idx])
                    },
                    'target_class': target_class,
                    'distance': float(min_distance)
                }

    def _print_projection_summary(self, projection_info):
        """Print summary of projection results."""
        print("\n" + "=" * 50)
        print("Prototype Projection Summary")
        print("=" * 50)

        class_names = ['Background', 'GD-enhancing', 'Edema', 'Necrotic']

        for proto_idx in range(self.n_prototypes):
            metadata = projection_info['projection_metadata'][proto_idx]
            distance = projection_info['distances'][proto_idx]
            target_class = proto_idx + 1

            print(f"\nPrototype {proto_idx} → {class_names[target_class]}:")
            if metadata is not None:
                print(f"  Batch: {metadata['batch_idx']}, Sample: {metadata['sample_idx']}")
                print(f"  Voxel: (d={metadata['voxel_coords']['d']}, "
                      f"h={metadata['voxel_coords']['h']}, "
                      f"w={metadata['voxel_coords']['w']})")
                print(f"  Distance to learned prototype: {distance:.4f}")
            else:
                print(f"  WARNING: No matching voxels found!")

        print("=" * 50)

    def apply_projection(self, projection_info):
        """
        Apply projected prototypes to the model.

        Args:
            projection_info: Dictionary from project_prototypes()
        """
        projected = projection_info['projected_prototypes']
        self.model.set_prototypes(tf.constant(projected, dtype=tf.float32))
        print("Projected prototypes applied to model.")

    def _convert_to_serializable(self, obj):
        """Recursively convert numpy types to Python types for JSON serialization."""
        if isinstance(obj, dict):
            return {k: self._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(v) for v in obj]
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    def save_projection(self, projection_info, save_path):
        """
        Save projection info to disk.

        Args:
            projection_info: Dictionary from project_prototypes()
            save_path: Path to save (without extension)
        """
        # Save prototypes as numpy
        np.save(f"{save_path}_prototypes.npy", projection_info['projected_prototypes'])
        np.save(f"{save_path}_original_prototypes.npy", projection_info['original_prototypes'])

        # Save metadata as JSON-compatible dict (convert numpy types)
        import json
        metadata = {
            'projection_metadata': projection_info['projection_metadata'],
            'distances': projection_info['distances']
        }
        serializable_metadata = self._convert_to_serializable(metadata)
        with open(f"{save_path}_metadata.json", 'w') as f:
            json.dump(serializable_metadata, f, indent=2)

        print(f"Projection saved to {save_path}_*.npy/json")

    def load_projection(self, load_path):
        """
        Load projection info from disk.

        Args:
            load_path: Path to load (without extension)

        Returns:
            projection_info: Dictionary with projection data
        """
        import json

        projected_prototypes = np.load(f"{load_path}_prototypes.npy")
        original_prototypes = np.load(f"{load_path}_original_prototypes.npy")

        with open(f"{load_path}_metadata.json", 'r') as f:
            metadata = json.load(f)

        projection_info = {
            'projected_prototypes': projected_prototypes,
            'original_prototypes': original_prototypes,
            'projection_metadata': metadata['projection_metadata'],
            'distances': metadata['distances']
        }

        return projection_info


def project_and_apply(model, data_generator, save_path=None, max_batches=None):
    """
    Convenience function to project prototypes and apply to model.

    Args:
        model: PrototypeSegNet3D model
        data_generator: Data generator
        save_path: Optional path to save projection info
        max_batches: Optional limit on batches to process

    Returns:
        projection_info: Dictionary with projection data
    """
    projector = PrototypeProjector(model)
    projection_info = projector.project_prototypes(data_generator, max_batches)
    projector.apply_projection(projection_info)

    if save_path:
        projector.save_projection(projection_info, save_path)

    return projection_info
