import os
import numpy as np
from tensorflow.keras.utils import Sequence
import h5py
import tensorflow as tf


class MRIDataGenerator(Sequence):
    def __init__(self, folder_path, batch_size=1, num_slices=155, num_volumes=369, split_ratio=0.2, subset='train',
                 shuffle=True, random_state=42):
        self.folder_path = folder_path
        self.batch_size = batch_size
        self.num_slices = num_slices
        self.num_volumes = num_volumes  # Total expected volumes (0 to 368)
        self.split_ratio = split_ratio
        self.shuffle = shuffle
        self.random_state = random_state

        print(f"MRIDataGenerator: Initializing for H5 files from: {self.folder_path}")
        if not os.path.exists(self.folder_path):
            raise FileNotFoundError(f"Error: Folder path does not exist: {self.folder_path}")

        # Instead of listing folders, identify unique volume IDs
        # We expect volume_0_slice_0.h5 to volume_368_slice_154.h5
        self.volume_ids = list(range(self.num_volumes))  # Simply generate expected volume IDs

        if not self.volume_ids:
            raise ValueError(
                f"No volume IDs found. Expected {self.num_volumes} volumes. Please check data organization.")

        print(f"MRIDataGenerator: Found {len(self.volume_ids)} unique volume IDs (0 to {self.num_volumes - 1}).")

        self.train_indices, self.val_indices = self._split_data()

        if subset == 'train':
            self.indices = self.train_indices
        elif subset == 'val':
            self.indices = self.val_indices
        else:
            raise ValueError("Subset must be 'train' or 'val'")

        print(
            f"MRIDataGenerator: {subset} subset has {len(self.indices)} volumes (each containing {self.num_slices} slices).")
        if len(self.indices) == 0:
            print(
                f"WARNING: The '{subset}' subset indices are empty. This will lead to a 'PyDataset has length 0' error.")

        self.on_epoch_end()

    def _split_data(self):
        np.random.seed(self.random_state)
        num_volumes_found = len(self.volume_ids)
        split_idx = int(num_volumes_found * (1 - self.split_ratio))

        shuffled_indices = np.random.permutation(num_volumes_found)
        train_indices = shuffled_indices[:split_idx]
        val_indices = shuffled_indices[split_idx:]

        return train_indices, val_indices

    def __len__(self):
        # Returns number of batches, which is number of volumes / batch_size
        return int(np.floor(len(self.indices) / self.batch_size))

    def __getitem__(self, idx):
        # Get the batch of volume IDs
        batch_volume_indices_in_split = self.indices[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_volume_ids = [self.volume_ids[i] for i in batch_volume_indices_in_split]

        batch_images = []
        batch_masks = []

        for current_volume_id in batch_volume_ids:
            volume_image_slices = []
            volume_mask_slices = []

            for s in range(self.num_slices):
                file_name = f"volume_{current_volume_id}_slice_{s}.h5"
                file_path = os.path.join(self.folder_path, file_name)

                if not os.path.exists(file_path):
                    # This is a critical error if a slice is missing for a volume
                    raise FileNotFoundError(f"Missing H5 slice file: {file_path}. Data integrity issue.")

                with h5py.File(file_path, 'r') as f:
                    # Load image (H, W, 4) and mask (H, W)
                    image_slice = f['image'][:]
                    mask_slice = f['mask'][:]

                    volume_image_slices.append(image_slice)
                    volume_mask_slices.append(mask_slice)

            # Stack slices to form 3D/4D volumes
            # Resulting image shape: (D, H, W, C)
            full_volume_image = np.stack(volume_image_slices, axis=0)
            # Resulting mask shape: (D, H, W)
            full_volume_mask = np.stack(volume_mask_slices, axis=0)

            # Normalization (per-volume)
            image_data_min = np.min(full_volume_image)
            image_data_max = np.max(full_volume_image)
            # Avoid division by zero if all values are identical
            if image_data_max - image_data_min > 1e-8:
                full_volume_image = (full_volume_image - image_data_min) / (image_data_max - image_data_min)
            else:
                full_volume_image = np.zeros_like(full_volume_image,
                                                  dtype=np.float32)  # Or handle as appropriate if image is uniform

            # FIX: Proper one-hot encoding for 3D masks
            # Ensure mask values are integers and in correct range [0, 1, 2]
            full_volume_mask = full_volume_mask.astype(np.int32)

            # Debug: Print mask info to understand the data
            print(f"Mask shape before one-hot: {full_volume_mask.shape}")
            print(f"Unique mask values: {np.unique(full_volume_mask)}")

            # Apply one-hot encoding: (D, H, W) -> (D, H, W, 3)
            mask_one_hot = tf.keras.utils.to_categorical(full_volume_mask, num_classes=3)

            print(f"Mask shape after one-hot: {mask_one_hot.shape}")

            batch_images.append(full_volume_image)
            batch_masks.append(mask_one_hot)

        # Convert lists of volumes to NumPy arrays for the batch
        # batch_images_arr shape: (Batch_Size, D, H, W, C)
        # batch_masks_arr shape: (Batch_Size, D, H, W, Num_Classes)
        batch_images_arr = np.array(batch_images, dtype=np.float32)
        batch_masks_arr = np.array(batch_masks, dtype=np.float32)  # ensure mask is float for Keras loss

        print(f"Final batch shapes - Images: {batch_images_arr.shape}, Masks: {batch_masks_arr.shape}")

        return batch_images_arr, batch_masks_arr

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
