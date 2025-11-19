"""
Data Preprocessing Script for Brain Tumor MRI
Downsamples volumes from 240x240x155 to 160x160x96 for faster training.

Usage:
    python preprocess_data.py --input_dir /path/to/original/data --output_dir /path/to/preprocessed/data
"""

import os
import argparse
import h5py
import numpy as np
from scipy.ndimage import zoom
from tqdm import tqdm


def downsample_volume(volume, target_shape):
    """
    Downsample a volume to target shape using trilinear interpolation.

    Args:
        volume: numpy array of shape (D, H, W) or (D, H, W, C)
        target_shape: tuple (target_D, target_H, target_W)

    Returns:
        downsampled volume
    """
    current_shape = volume.shape

    if len(current_shape) == 4:  # (D, H, W, C) - image with channels
        D, H, W, C = current_shape
        target_D, target_H, target_W = target_shape

        # Calculate zoom factors for each dimension (excluding channels)
        zoom_factors = (target_D / D, target_H / H, target_W / W, 1)
    else:  # (D, H, W) - mask
        D, H, W = current_shape
        target_D, target_H, target_W = target_shape

        # Calculate zoom factors
        zoom_factors = (target_D / D, target_H / H, target_W / W)

    # Use order=1 (linear) for images, order=0 (nearest) for masks
    if len(current_shape) == 4:
        downsampled = zoom(volume, zoom_factors, order=1, mode='nearest')
    else:
        downsampled = zoom(volume, zoom_factors, order=0, mode='nearest')

    return downsampled


def center_crop_depth(volume, target_depth):
    """
    Center crop the depth dimension.

    Args:
        volume: numpy array with shape (D, H, W) or (D, H, W, C)
        target_depth: target depth after cropping

    Returns:
        cropped volume
    """
    current_depth = volume.shape[0]

    if current_depth <= target_depth:
        # If already smaller or equal, return as is
        return volume

    # Calculate crop indices
    start = (current_depth - target_depth) // 2
    end = start + target_depth

    # Crop depth dimension
    if len(volume.shape) == 4:  # Image with channels
        return volume[start:end, :, :, :]
    else:  # Mask
        return volume[start:end, :, :]


def preprocess_slice_file(input_path, output_path, target_spatial=(160, 160)):
    """
    Preprocess a single H5 slice file.

    Args:
        input_path: path to original H5 file
        output_path: path to save preprocessed H5 file
        target_spatial: tuple (target_H, target_W)
    """
    with h5py.File(input_path, 'r') as f_in:
        # Load image (H, W, C) and mask (H, W) or (H, W, num_classes)
        image = f_in['image'][:]
        mask = f_in['mask'][:]

    # Handle mask shape variations
    # Mask might be (H, W) or (H, W, num_classes)
    if len(mask.shape) == 3:
        # If mask has 3 dimensions (H, W, C), we need to handle it differently
        mask_has_channels = True
        H, W, num_classes = mask.shape
    elif len(mask.shape) == 2:
        # Normal case: (H, W)
        mask_has_channels = False
        H, W = mask.shape
    else:
        raise ValueError(f"Unexpected mask shape in {input_path}: {mask.shape}")

    # Verify image shape
    if len(image.shape) != 3:
        raise ValueError(f"Unexpected image shape in {input_path}: {image.shape}")

    target_H, target_W = target_spatial

    # Calculate zoom factors for spatial dimensions
    zoom_factor_H = target_H / H
    zoom_factor_W = target_W / W

    # Downsample image (H, W, C)
    zoom_factors_image = (zoom_factor_H, zoom_factor_W, 1)
    downsampled_image = zoom(image, zoom_factors_image, order=1, mode='nearest')

    # Downsample mask - handle both 2D and 3D cases
    if mask_has_channels:
        # Mask is (H, W, num_classes)
        zoom_factors_mask = (zoom_factor_H, zoom_factor_W, 1)
        downsampled_mask = zoom(mask, zoom_factors_mask, order=0, mode='nearest')
    else:
        # Mask is (H, W)
        zoom_factors_mask = (zoom_factor_H, zoom_factor_W)
        downsampled_mask = zoom(mask, zoom_factors_mask, order=0, mode='nearest')

    # Save preprocessed data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with h5py.File(output_path, 'w') as f_out:
        f_out.create_dataset('image', data=downsampled_image, compression='gzip')
        f_out.create_dataset('mask', data=downsampled_mask, compression='gzip')


def preprocess_dataset(input_dir, output_dir, num_volumes=369, num_slices=155,
                       target_spatial=(160, 160), target_slices=96):
    """
    Preprocess entire dataset.

    Args:
        input_dir: directory containing original H5 files
        output_dir: directory to save preprocessed H5 files
        num_volumes: total number of volumes (default 369)
        num_slices: number of slices per volume in original data (default 155)
        target_spatial: tuple (target_H, target_W) for spatial downsampling
        target_slices: target number of slices to keep (will center crop)
    """
    print("=" * 70)
    print("BRAIN TUMOR MRI DATA PREPROCESSING")
    print("=" * 70)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Original dimensions: ({num_slices}, 240, 240, 4)")
    print(f"Target dimensions: ({target_slices}, {target_spatial[0]}, {target_spatial[1]}, 4)")
    print(f"Total volumes: {num_volumes}")
    print(f"Processing volumes 1 to {num_volumes}")
    print("=" * 70)

    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    os.makedirs(output_dir, exist_ok=True)

    # Calculate which slices to keep (center crop)
    slice_start = (num_slices - target_slices) // 2
    slice_end = slice_start + target_slices
    slices_to_keep = list(range(slice_start, slice_end))

    print(f"\nKeeping slices {slice_start} to {slice_end-1} (center {target_slices} slices)")
    print("\nStarting preprocessing...")
    print("-" * 70)

    total_files = num_volumes * target_slices
    processed_count = 0

    with tqdm(total=total_files, desc="Processing slices") as pbar:
        for volume_id in range(1, num_volumes + 1):  # Volumes start from 1
            for new_slice_idx, original_slice_idx in enumerate(slices_to_keep):
                # Original file path
                input_filename = f"volume_{volume_id}_slice_{original_slice_idx}.h5"
                input_path = os.path.join(input_dir, input_filename)

                # Output file path (renumber slices to 0-95)
                output_filename = f"volume_{volume_id}_slice_{new_slice_idx}.h5"
                output_path = os.path.join(output_dir, output_filename)

                if not os.path.exists(input_path):
                    print(f"\nWarning: Missing file {input_path}, skipping...")
                    pbar.update(1)
                    continue

                # Preprocess and save
                try:
                    preprocess_slice_file(input_path, output_path, target_spatial)
                    processed_count += 1
                except Exception as e:
                    print(f"\nError processing {input_filename}: {e}")

                pbar.update(1)

    print("\n" + "=" * 70)
    print("PREPROCESSING COMPLETE!")
    print("=" * 70)
    print(f"✓ Processed {processed_count} / {total_files} files")
    print(f"✓ Output saved to: {output_dir}")

    # Calculate size reduction
    original_size = 240 * 240 * num_slices
    new_size = target_spatial[0] * target_spatial[1] * target_slices
    reduction = (1 - new_size / original_size) * 100

    print(f"\n📊 Data Reduction:")
    print(f"  - Original volume size: {num_slices} × 240 × 240 = {original_size:,} voxels")
    print(f"  - New volume size: {target_slices} × {target_spatial[0]} × {target_spatial[1]} = {new_size:,} voxels")
    print(f"  - Reduction: {reduction:.1f}%")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description='Preprocess brain tumor MRI data')
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Directory containing original H5 files')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save preprocessed H5 files')
    parser.add_argument('--num_volumes', type=int, default=369,
                       help='Total number of volumes (default: 369)')
    parser.add_argument('--num_slices', type=int, default=155,
                       help='Number of slices per volume in original data (default: 155)')
    parser.add_argument('--target_height', type=int, default=160,
                       help='Target height after downsampling (default: 160)')
    parser.add_argument('--target_width', type=int, default=160,
                       help='Target width after downsampling (default: 160)')
    parser.add_argument('--target_slices', type=int, default=96,
                       help='Target number of slices (will center crop, default: 96)')

    args = parser.parse_args()

    preprocess_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        num_volumes=args.num_volumes,
        num_slices=args.num_slices,
        target_spatial=(args.target_height, args.target_width),
        target_slices=args.target_slices
    )


if __name__ == '__main__':
    main()
