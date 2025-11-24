"""
Visualize learned prototypes from trained MProtoNet3D model.

This script:
1. Loads the trained model
2. Extracts prototype vectors
3. Finds closest patches in training data to each prototype
4. Visualizes prototypes and their activations

Usage:
    python visualize_prototypes.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import h5py
from tqdm import tqdm

from model import MProtoNet3D_Segmentation_Keras
from losses import CombinedLoss
from data_generator import MRIDataGenerator

# ============================================================================
# CONFIGURATION
# ============================================================================
MODEL_PATH = "best_model.keras"
DATA_PATH = "preprocessed_data"
NUM_VOLUMES = 369
NUM_SLICES = 96
BATCH_SIZE = 1
NUM_SAMPLES_TO_CHECK = 50  # Number of volumes to search for closest patches
OUTPUT_DIR = "prototype_visualizations"
# ============================================================================


def load_trained_model(model_path):
    """Load trained model with custom objects."""
    print(f"Loading model from: {model_path}")
    model = keras.models.load_model(
        model_path,
        custom_objects={'CombinedLoss': CombinedLoss}
    )
    print(f"✓ Model loaded successfully!")
    return model


def extract_prototypes(model):
    """Extract prototype vectors from model."""
    prototype_vectors = model.prototype_vectors.numpy()
    print(f"Prototype vectors shape: {prototype_vectors.shape}")
    print(f"Number of prototypes: {prototype_vectors.shape[0]}")
    print(f"Feature dimensions: {prototype_vectors.shape[1]}")
    return prototype_vectors


def get_prototype_class_assignment(model):
    """Get which prototypes belong to which class."""
    num_prototypes = model.num_prototypes
    num_classes = model.num_classes
    prototypes_per_class = num_prototypes // num_classes

    assignments = {}
    for i in range(num_prototypes):
        class_idx = i // prototypes_per_class
        assignments[i] = class_idx

    return assignments


def extract_features_from_data(model, data_generator, num_samples):
    """
    Extract features from data samples and find closest matches to prototypes.

    Returns:
        Dictionary mapping prototype_idx -> (volume_idx, slice_idx, spatial_location, activation)
    """
    print(f"\nSearching through {num_samples} volumes for closest prototype matches...")

    # Get intermediate layer (after add_ons) for feature extraction
    feature_extractor = keras.Model(
        inputs=model.input,
        outputs=model.add_ons.output
    )

    prototype_vectors = model.prototype_vectors.numpy()
    num_prototypes = prototype_vectors.shape[0]

    # Track best matches for each prototype
    best_matches = {i: {
        'activation': -np.inf,
        'volume_idx': None,
        'slice_idx': None,
        'location': None,
        'image_patch': None,
        'mask_patch': None
    } for i in range(num_prototypes)}

    # Iterate through data
    for sample_idx in tqdm(range(min(num_samples, len(data_generator)))):
        batch_x, batch_y = data_generator[sample_idx]

        # Extract features
        features = feature_extractor.predict(batch_x, verbose=0)  # (B, D, H, W, C)

        # Compute prototype activations
        if model.f_dist == 'l2':
            distances = model.l2_convolution_3D(features).numpy()
            similarities = model.distance_2_similarity(distances).numpy()
        else:
            similarities = model.cosine_convolution_3D(features).numpy()

        # For each prototype, check if this is the best activation
        for proto_idx in range(num_prototypes):
            proto_similarities = similarities[0, :, :, :, proto_idx]  # (D, H, W)
            max_activation = np.max(proto_similarities)

            if max_activation > best_matches[proto_idx]['activation']:
                # Found better match
                max_loc = np.unravel_index(np.argmax(proto_similarities), proto_similarities.shape)
                d, h, w = max_loc

                # Get the corresponding image and mask patches
                # Extract a small patch around the location
                patch_size = 16
                h_start = max(0, h - patch_size // 2)
                h_end = min(batch_x.shape[2], h + patch_size // 2)
                w_start = max(0, w - patch_size // 2)
                w_end = min(batch_x.shape[3], w + patch_size // 2)

                image_patch = batch_x[0, d, h_start:h_end, w_start:w_end, :]
                if len(batch_y.shape) == 5:
                    mask_patch = batch_y[0, d, h_start:h_end, w_start:w_end, :]
                else:
                    mask_patch = batch_y[0, d, h_start:h_end, w_start:w_end]

                best_matches[proto_idx] = {
                    'activation': max_activation,
                    'volume_idx': data_generator.volume_ids[data_generator.indices[sample_idx]],
                    'slice_idx': d,
                    'location': (h, w),
                    'image_patch': image_patch,
                    'mask_patch': mask_patch
                }

    return best_matches


def visualize_prototypes(model, best_matches, output_dir):
    """Visualize prototypes with their closest matching patches."""
    os.makedirs(output_dir, exist_ok=True)

    num_prototypes = model.num_prototypes
    num_classes = model.num_classes
    prototypes_per_class = num_prototypes // num_classes

    class_names = ['GD-enhancing tumor (ET)', 'Peritumoral edema (ED)', 'Necrotic core (NCR/NET)']
    modality_names = ['T1', 'T1ce', 'T2', 'FLAIR']

    # Visualize prototypes class by class
    for class_idx in range(num_classes):
        print(f"\nVisualizing prototypes for class {class_idx}: {class_names[class_idx]}")

        start_proto = class_idx * prototypes_per_class
        end_proto = start_proto + prototypes_per_class

        # Create figure with prototypes for this class
        fig, axes = plt.subplots(prototypes_per_class, 5, figsize=(25, 5 * prototypes_per_class))

        for local_idx, proto_idx in enumerate(range(start_proto, end_proto)):
            match = best_matches[proto_idx]

            # Column 0: Prototype info
            axes[local_idx, 0].text(0.5, 0.5,
                f"Prototype {proto_idx}\n"
                f"Class: {class_idx}\n"
                f"Activation: {match['activation']:.4f}\n"
                f"Volume: {match['volume_idx']}\n"
                f"Slice: {match['slice_idx']}\n"
                f"Location: {match['location']}",
                ha='center', va='center', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            axes[local_idx, 0].axis('off')

            # Columns 1-4: Show all 4 modalities
            if match['image_patch'] is not None:
                for mod_idx in range(4):
                    axes[local_idx, mod_idx + 1].imshow(match['image_patch'][:, :, mod_idx], cmap='gray')
                    axes[local_idx, mod_idx + 1].set_title(f"{modality_names[mod_idx]}", fontsize=10)
                    axes[local_idx, mod_idx + 1].axis('off')

        plt.suptitle(f"Learned Prototypes for Class {class_idx}: {class_names[class_idx]}",
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/prototypes_class_{class_idx}.png", dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {output_dir}/prototypes_class_{class_idx}.png")
        plt.close()


def visualize_prototype_activation_maps(model, best_matches, output_dir):
    """Visualize activation maps for each prototype."""
    print("\nGenerating prototype activation heatmaps...")

    num_prototypes = model.num_prototypes
    num_classes = model.num_classes
    prototypes_per_class = num_prototypes // num_classes

    class_names = ['GD-enhancing tumor', 'Peritumoral edema', 'Necrotic core']

    for class_idx in range(num_classes):
        start_proto = class_idx * prototypes_per_class
        end_proto = start_proto + prototypes_per_class

        fig, axes = plt.subplots(2, prototypes_per_class, figsize=(prototypes_per_class * 4, 8))

        for local_idx, proto_idx in enumerate(range(start_proto, end_proto)):
            match = best_matches[proto_idx]

            if match['image_patch'] is not None:
                # Show T1ce image (most informative)
                axes[0, local_idx].imshow(match['image_patch'][:, :, 1], cmap='gray')
                axes[0, local_idx].set_title(f"Proto {proto_idx}\nT1ce", fontsize=10)
                axes[0, local_idx].axis('off')

                # Show mask overlay
                if len(match['mask_patch'].shape) == 3:
                    # Multi-channel mask
                    mask_vis = np.argmax(match['mask_patch'], axis=-1)
                else:
                    mask_vis = match['mask_patch']

                axes[1, local_idx].imshow(match['image_patch'][:, :, 1], cmap='gray')
                axes[1, local_idx].imshow(mask_vis, cmap='jet', alpha=0.5)
                axes[1, local_idx].set_title(f"With Mask\nAct: {match['activation']:.3f}", fontsize=10)
                axes[1, local_idx].axis('off')

        plt.suptitle(f"Prototype Activation Maps - {class_names[class_idx]}",
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/activation_maps_class_{class_idx}.png", dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {output_dir}/activation_maps_class_{class_idx}.png")
        plt.close()


def visualize_prototype_statistics(model, best_matches, output_dir):
    """Visualize statistics about learned prototypes."""
    print("\nGenerating prototype statistics...")

    prototype_vectors = model.prototype_vectors.numpy()
    num_prototypes = prototype_vectors.shape[0]

    # Compute prototype norms and similarities
    proto_norms = np.linalg.norm(prototype_vectors.reshape(num_prototypes, -1), axis=1)

    # Compute pairwise similarities between prototypes
    proto_flat = prototype_vectors.reshape(num_prototypes, -1)
    similarities = np.dot(proto_flat, proto_flat.T) / (
        np.outer(proto_norms, proto_norms) + 1e-8
    )

    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Prototype norms
    axes[0].bar(range(num_prototypes), proto_norms)
    axes[0].set_xlabel('Prototype Index')
    axes[0].set_ylabel('L2 Norm')
    axes[0].set_title('Prototype Vector Magnitudes')
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Activation strengths
    activations = [best_matches[i]['activation'] for i in range(num_prototypes)]
    axes[1].bar(range(num_prototypes), activations)
    axes[1].set_xlabel('Prototype Index')
    axes[1].set_ylabel('Max Activation')
    axes[1].set_title('Best Activation Strength per Prototype')
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Similarity matrix
    im = axes[2].imshow(similarities, cmap='coolwarm', vmin=-1, vmax=1)
    axes[2].set_xlabel('Prototype Index')
    axes[2].set_ylabel('Prototype Index')
    axes[2].set_title('Prototype Similarity Matrix')
    plt.colorbar(im, ax=axes[2])

    plt.tight_layout()
    plt.savefig(f"{output_dir}/prototype_statistics.png", dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_dir}/prototype_statistics.png")
    plt.close()


def main():
    print("=" * 70)
    print("PROTOTYPE VISUALIZATION FOR MPROTONET3D")
    print("=" * 70)

    # Load model
    model = load_trained_model(MODEL_PATH)

    # Extract prototypes
    prototypes = extract_prototypes(model)

    # Get class assignments
    assignments = get_prototype_class_assignment(model)
    print(f"\nPrototype class assignments:")
    for proto_idx, class_idx in assignments.items():
        print(f"  Prototype {proto_idx} -> Class {class_idx}")

    # Create data generator
    print(f"\nCreating data generator...")
    data_generator = MRIDataGenerator(
        DATA_PATH,
        batch_size=BATCH_SIZE,
        num_slices=NUM_SLICES,
        num_volumes=NUM_VOLUMES,
        split_ratio=0.2,
        subset='train',
        shuffle=False,
        random_state=42
    )

    # Find closest patches
    best_matches = extract_features_from_data(model, data_generator, NUM_SAMPLES_TO_CHECK)

    # Create visualizations
    print(f"\nCreating visualizations...")
    visualize_prototypes(model, best_matches, OUTPUT_DIR)
    visualize_prototype_activation_maps(model, best_matches, OUTPUT_DIR)
    visualize_prototype_statistics(model, best_matches, OUTPUT_DIR)

    print("\n" + "=" * 70)
    print("✓ VISUALIZATION COMPLETE!")
    print("=" * 70)
    print(f"All visualizations saved to: {OUTPUT_DIR}/")
    print(f"  - prototypes_class_0.png (GD-enhancing tumor prototypes)")
    print(f"  - prototypes_class_1.png (Peritumoral edema prototypes)")
    print(f"  - prototypes_class_2.png (Necrotic core prototypes)")
    print(f"  - activation_maps_class_*.png (Activation heatmaps)")
    print(f"  - prototype_statistics.png (Statistics and similarities)")
    print("=" * 70)


if __name__ == '__main__':
    main()
