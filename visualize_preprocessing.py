"""
Visualization script to compare original and preprocessed brain tumor MRI data.
Shows the same volume and slice from both datasets side by side.

Usage:
    1. Update the configuration variables below
    2. Run: python visualize_preprocessing.py
"""

import os
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid PyCharm issues
import matplotlib.pyplot as plt

# ============================================================================
# CONFIGURATION - UPDATE THESE PATHS AND PARAMETERS
# ============================================================================
ORIGINAL_DIR = "archive/BraTS2020_training_data/content/data"
PREPROCESSED_DIR = "preprocessed_data"
VOLUME_INDEX = 2
ORIGINAL_SLICE_INDEX = 65
ORIGINAL_NUM_SLICES = 155
PREPROCESSED_NUM_SLICES = 96
SHOW_PLOTS = False  # Set to True to display plots (may not work in PyCharm), False to just save
# ============================================================================


def load_volume_from_slices(folder_path, volume_index, num_slices):
    """
    Load a complete volume from individual H5 slice files.

    Args:
        folder_path: directory containing H5 files
        volume_index: which volume to load
        num_slices: number of slices in the volume

    Returns:
        dict with 'image' and 'mask' arrays
    """
    image_slices = []
    mask_slices = []

    for slice_idx in range(num_slices):
        file_path = os.path.join(folder_path, f"volume_{volume_index}_slice_{slice_idx}.h5")

        if not os.path.exists(file_path):
            print(f"Warning: Missing file {file_path}")
            continue

        with h5py.File(file_path, 'r') as f:
            image_slices.append(f['image'][:])
            mask_slices.append(f['mask'][:])

    if not image_slices:
        raise ValueError(f"No slices found for volume {volume_index}")

    # Stack along depth dimension
    volume_image = np.stack(image_slices, axis=0)  # (D, H, W, C)
    volume_mask = np.stack(mask_slices, axis=0)    # (D, H, W) or (D, H, W, num_classes)

    return {'image': volume_image, 'mask': volume_mask}


def plot_comparison(original_data, preprocessed_data, original_slice_idx, preprocessed_slice_idx, volume_index):
    """
    Plot comparison of original vs preprocessed data.

    Args:
        original_data: dict with 'image' and 'mask' from original data
        preprocessed_data: dict with 'image' and 'mask' from preprocessed data
        original_slice_idx: slice index in original data
        preprocessed_slice_idx: corresponding slice index in preprocessed data
        volume_index: volume number for title
    """
    modalities = ["T1", "T1 with contrast", "T2", "T2 FLAIR"]

    # Create figure with 2 rows (original and preprocessed) and 4 columns (modalities)
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    # Original data - modalities
    for i in range(4):
        original_slice = original_data['image'][original_slice_idx, :, :, i]
        axes[0, i].imshow(original_slice, cmap='gray')
        axes[0, i].set_title(f'Original: {modalities[i]}\nSlice {original_slice_idx} ({original_data["image"].shape[1]}x{original_data["image"].shape[2]})',
                            fontsize=12, fontweight='bold')
        axes[0, i].axis('off')

    # Preprocessed data - modalities
    for i in range(4):
        preprocessed_slice = preprocessed_data['image'][preprocessed_slice_idx, :, :, i]
        axes[1, i].imshow(preprocessed_slice, cmap='gray')
        axes[1, i].set_title(f'Preprocessed: {modalities[i]}\nSlice {preprocessed_slice_idx} ({preprocessed_data["image"].shape[1]}x{preprocessed_data["image"].shape[2]})',
                            fontsize=12, fontweight='bold')
        axes[1, i].axis('off')

    plt.suptitle(f'Volume {volume_index} - Original vs Preprocessed Comparison',
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(f'comparison_volume_{volume_index}_modalities.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: comparison_volume_{volume_index}_modalities.png")
    if SHOW_PLOTS:
        plt.show()
    plt.close()

    # Plot mask comparison if mask has channels
    original_mask = original_data['mask']
    preprocessed_mask = preprocessed_data['mask']

    # Check if masks have channels
    if len(original_mask.shape) == 4 and original_mask.shape[-1] == 3:
        plot_mask_comparison(original_mask, preprocessed_mask,
                           original_slice_idx, preprocessed_slice_idx, volume_index)


def plot_mask_comparison(original_mask, preprocessed_mask, original_slice_idx, preprocessed_slice_idx, volume_index):
    """
    Plot comparison of mask channels.

    Args:
        original_mask: original mask array (D, H, W, num_classes)
        preprocessed_mask: preprocessed mask array (D, H, W, num_classes)
        original_slice_idx: slice index in original data
        preprocessed_slice_idx: slice index in preprocessed data
        volume_index: volume number for title
    """
    mask_labels = ["GD-enhancing tumor (ET)", "Peritumoral edema (ED)", "Necrotic/non-enhancing core (NCR/NET)"]
    num_channels = original_mask.shape[-1]

    # Create figure with 2 rows (original and preprocessed) and num_channels columns
    fig, axes = plt.subplots(2, num_channels, figsize=(6 * num_channels, 10))

    # Original masks
    for i in range(num_channels):
        original_slice = original_mask[original_slice_idx, :, :, i]
        axes[0, i].imshow(original_slice, cmap='gray')
        axes[0, i].set_title(f'Original: {mask_labels[i]}\nSlice {original_slice_idx} ({original_mask.shape[1]}x{original_mask.shape[2]})',
                           fontsize=12, fontweight='bold')
        axes[0, i].axis('off')

    # Preprocessed masks
    for i in range(num_channels):
        preprocessed_slice = preprocessed_mask[preprocessed_slice_idx, :, :, i]
        axes[1, i].imshow(preprocessed_slice, cmap='gray')
        axes[1, i].set_title(f'Preprocessed: {mask_labels[i]}\nSlice {preprocessed_slice_idx} ({preprocessed_mask.shape[1]}x{preprocessed_mask.shape[2]})',
                           fontsize=12, fontweight='bold')
        axes[1, i].axis('off')

    plt.suptitle(f'Volume {volume_index} - Mask Comparison (Original vs Preprocessed)',
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(f'comparison_volume_{volume_index}_masks.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: comparison_volume_{volume_index}_masks.png")
    if SHOW_PLOTS:
        plt.show()
    plt.close()


def main():
    print("=" * 70)
    print("BRAIN TUMOR DATA COMPARISON VISUALIZATION")
    print("=" * 70)
    print(f"Original data directory: {ORIGINAL_DIR}")
    print(f"Preprocessed data directory: {PREPROCESSED_DIR}")
    print(f"Volume index: {VOLUME_INDEX}")
    print(f"Original slice index: {ORIGINAL_SLICE_INDEX}")
    print("=" * 70)

    # Load original data
    print("\nLoading original data...")
    original_data = load_volume_from_slices(ORIGINAL_DIR, VOLUME_INDEX, ORIGINAL_NUM_SLICES)
    print(f"✓ Original image shape: {original_data['image'].shape}")
    print(f"✓ Original mask shape: {original_data['mask'].shape}")

    # Calculate corresponding slice in preprocessed data
    # Original: 155 slices, we take middle portion for preprocessed (96 slices)
    # If original used slices 29-125 (center 96 slices from 155)
    # Then original slice 65 maps to preprocessed slice: 65 - 29 = 36
    slice_start = (ORIGINAL_NUM_SLICES - PREPROCESSED_NUM_SLICES) // 2
    preprocessed_slice_idx = ORIGINAL_SLICE_INDEX - slice_start

    # Ensure the slice index is valid
    if preprocessed_slice_idx < 0 or preprocessed_slice_idx >= PREPROCESSED_NUM_SLICES:
        print(f"\nWarning: Original slice {ORIGINAL_SLICE_INDEX} is outside the preprocessed range.")
        print(f"Preprocessed data contains slices {slice_start} to {slice_start + PREPROCESSED_NUM_SLICES - 1} from original.")
        print(f"Using middle slice of preprocessed data instead.")
        preprocessed_slice_idx = PREPROCESSED_NUM_SLICES // 2
        print(f"Adjusted to preprocessed slice: {preprocessed_slice_idx}")
    else:
        print(f"\nOriginal slice {ORIGINAL_SLICE_INDEX} corresponds to preprocessed slice {preprocessed_slice_idx}")

    # Load preprocessed data
    print("\nLoading preprocessed data...")
    preprocessed_data = load_volume_from_slices(PREPROCESSED_DIR, VOLUME_INDEX, PREPROCESSED_NUM_SLICES)
    print(f"✓ Preprocessed image shape: {preprocessed_data['image'].shape}")
    print(f"✓ Preprocessed mask shape: {preprocessed_data['mask'].shape}")

    # Plot comparison
    print("\nGenerating comparison plots...")
    plot_comparison(original_data, preprocessed_data, ORIGINAL_SLICE_INDEX, preprocessed_slice_idx, VOLUME_INDEX)

    print("\n" + "=" * 70)
    print("✓ VISUALIZATION COMPLETE!")
    print("=" * 70)
    print(f"Images saved in current directory:")
    print(f"  - comparison_volume_{VOLUME_INDEX}_modalities.png")
    print(f"  - comparison_volume_{VOLUME_INDEX}_masks.png")
    print("\nYou can open these images to view the comparison.")
    print("=" * 70)


if __name__ == '__main__':
    main()
