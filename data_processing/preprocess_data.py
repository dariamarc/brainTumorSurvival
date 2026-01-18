"""
Data Preprocessing Script for Brain Tumor MRI
Uses CENTER CROPPING ONLY (no resizing) to preserve original resolution and avoid blurring small labels.
Crops volumes from 240×240×155 to 192×160×128 (H×W×D).

Usage:
    python preprocess_data.py --input_dir /path/to/original/data --output_dir /path/to/preprocessed/data
"""

import os
import argparse
import h5py
import numpy as np
from tqdm import tqdm


def center_crop_2d(array, target_height, target_width):
    """
    Center crop a 2D array or 2D slice.

    Args:
        array: numpy array of shape (H, W) or (H, W, C)
        target_height: target height after cropping
        target_width: target width after cropping

    Returns:
        cropped array
    """
    current_height, current_width = array.shape[0], array.shape[1]

    if current_height < target_height or current_width < target_width:
        raise ValueError(f"Cannot crop from ({current_height}, {current_width}) to ({target_height}, {target_width})")

    # Calculate crop indices for height
    start_h = (current_height - target_height) // 2
    end_h = start_h + target_height

    # Calculate crop indices for width
    start_w = (current_width - target_width) // 2
    end_w = start_w + target_width

    # Crop
    if len(array.shape) == 3:  # (H, W, C)
        return array[start_h:end_h, start_w:end_w, :]
    else:  # (H, W)
        return array[start_h:end_h, start_w:end_w]


def preprocess_slice_file(input_path, output_path, target_spatial=(192, 160)):
    """
    Preprocess a single H5 slice file using CENTER CROPPING ONLY.

    Args:
        input_path: path to original H5 file
        output_path: path to save preprocessed H5 file
        target_spatial: tuple (target_H, target_W)
    """
    with h5py.File(input_path, 'r') as f_in:
        # Load image (H, W, C) and mask (H, W) or (H, W, num_classes)
        image = f_in['image'][:]
        mask = f_in['mask'][:]

    # Verify shapes
    if len(image.shape) != 3:
        raise ValueError(f"Unexpected image shape in {input_path}: {image.shape}")

    target_H, target_W = target_spatial

    # Center crop image (H, W, C)
    cropped_image = center_crop_2d(image, target_H, target_W)

    # Center crop mask - handle both 2D and 3D cases
    cropped_mask = center_crop_2d(mask, target_H, target_W)

    # Save preprocessed data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with h5py.File(output_path, 'w') as f_out:
        f_out.create_dataset('image', data=cropped_image, compression='gzip')
        f_out.create_dataset('mask', data=cropped_mask, compression='gzip')


def preprocess_dataset(input_dir, output_dir, num_volumes=369, num_slices=155,
                       target_spatial=(160, 192), target_slices=128):
    """
    Preprocess entire dataset using CENTER CROPPING ONLY (no resizing).

    Args:
        input_dir: directory containing original H5 files
        output_dir: directory to save preprocessed H5 files
        num_volumes: total number of volumes (default 369)
        num_slices: number of slices per volume in original data (default 155)
        target_spatial: tuple (target_H, target_W) for center cropping (default 192×160)
        target_slices: target number of slices to keep via center crop (default 128)
    """
    print("=" * 80)
    print("BRAIN TUMOR MRI DATA PREPROCESSING - CENTER CROP ONLY")
    print("=" * 80)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Original dimensions: ({num_slices}, 240, 240, 4) [D×H×W×C]")
    print(f"Target dimensions: ({target_slices}, {target_spatial[0]}, {target_spatial[1]}, 4) [D×H×W×C]")
    print(f"Total volumes: {num_volumes}")
    print(f"Processing volumes 1 to {num_volumes}")
    print(f"\n✓ Method: CENTER CROPPING ONLY (preserves original resolution)")
    print(f"✓ No resizing/interpolation - avoids blurring small labels")
    print("=" * 80)

    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    os.makedirs(output_dir, exist_ok=True)

    # Calculate which slices to keep (center crop in depth dimension)
    if target_slices > num_slices:
        raise ValueError(f"Cannot crop to {target_slices} slices from {num_slices} slices")

    slice_start = (num_slices - target_slices) // 2
    slice_end = slice_start + target_slices
    slices_to_keep = list(range(slice_start, slice_end))

    print(f"\n📏 Cropping details:")
    print(f"  - Depth: Keeping slices {slice_start} to {slice_end-1} (center {target_slices} slices)")
    print(f"  - Height: Cropping from 240 to {target_spatial[0]} (center crop)")
    print(f"  - Width: Cropping from 240 to {target_spatial[1]} (center crop)")
    print("\nStarting preprocessing...")
    print("-" * 80)

    total_files = num_volumes * target_slices
    processed_count = 0

    with tqdm(total=total_files, desc="Processing slices") as pbar:
        for volume_id in range(1, num_volumes + 1):  # Volumes start from 1
            for new_slice_idx, original_slice_idx in enumerate(slices_to_keep):
                # Original file path
                input_filename = f"volume_{volume_id}_slice_{original_slice_idx}.h5"
                input_path = os.path.join(input_dir, input_filename)

                # Output file path (renumber slices to 0-127 for target_slices=128)
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

    print("\n" + "=" * 80)
    print("PREPROCESSING COMPLETE!")
    print("=" * 80)
    print(f"✓ Processed {processed_count} / {total_files} files")
    print(f"✓ Output saved to: {output_dir}")

    # Calculate size reduction
    original_size = 240 * 240 * num_slices
    new_size = target_spatial[0] * target_spatial[1] * target_slices
    reduction = (1 - new_size / original_size) * 100

    print(f"\n📊 Data Statistics:")
    print(f"  - Original volume size: {num_slices} × 240 × 240 = {original_size:,} voxels")
    print(f"  - Cropped volume size: {target_slices} × {target_spatial[0]} × {target_spatial[1]} = {new_size:,} voxels")
    print(f"  - Size reduction: {reduction:.1f}%")
    print(f"  - Voxels retained: {100-reduction:.1f}%")
    print(f"\n✓ Original resolution preserved (no interpolation)")
    print(f"✓ Small tumor features intact")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Preprocess brain tumor MRI data using center cropping only')
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Directory containing original H5 files')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save preprocessed H5 files')
    parser.add_argument('--num_volumes', type=int, default=369,
                       help='Total number of volumes (default: 369)')
    parser.add_argument('--num_slices', type=int, default=155,
                       help='Number of slices per volume in original data (default: 155)')
    parser.add_argument('--target_height', type=int, default=160,
                       help='Target height after center cropping (default: 160)')
    parser.add_argument('--target_width', type=int, default=192,
                       help='Target width after center cropping (default: 192)')
    parser.add_argument('--target_slices', type=int, default=128,
                       help='Target number of slices via center crop (default: 128)')

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
