#!/usr/bin/env python3
"""
Step 1 SageMaker Training Script — Plain 3D U-Net Baseline (no prototypes)

Pass criteria (log these after training completes):
  - Whole-tumor Dice > 0.65 on validation
  - Mean class Dice (NCR + ED + ET) / 3 > 0.40

Class ordering in generator output:
  ch0 = Background, ch1 = NCR/NET, ch2 = ED, ch3 = ET
"""

import os
import sys
import json
import argparse
import logging
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow import keras


logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger(__name__)


# ── Data generator ────────────────────────────────────────────────────────────

class MRIDataGenerator:
    """Loads preprocessed H5 slice files and assembles 3D volumes."""

    def __init__(self, folder_path, volume_ids, num_slices=128, batch_size=1, shuffle=True):
        self.folder_path = folder_path
        self.volume_ids = list(volume_ids)
        self.num_slices = num_slices
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.volume_ids))
        if shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return len(self.volume_ids) // self.batch_size

    def __getitem__(self, idx):
        import h5py
        vol_id = self.volume_ids[self.indices[idx * self.batch_size]]

        img_slices, mask_slices = [], []
        for s in range(self.num_slices):
            fpath = os.path.join(self.folder_path, f'volume_{vol_id}_slice_{s}.h5')
            with h5py.File(fpath, 'r') as f:
                img_slices.append(f['image'][:].astype(np.float32))
                mask_slices.append(f['mask'][:])

        img  = np.stack(img_slices,  axis=0)  # (D, H, W, 4)
        mask = np.stack(mask_slices, axis=0)  # (D, H, W, 3) — [NCR, ED, ET]

        # Per-volume min-max normalisation
        vmin, vmax = img.min(), img.max()
        if vmax - vmin > 1e-8:
            img = (img - vmin) / (vmax - vmin)

        # Add background channel: [BG, NCR, ED, ET]
        mask = mask.astype(np.float32)
        bg = (mask.sum(axis=-1, keepdims=True) == 0).astype(np.float32)
        mask = np.concatenate([bg, mask], axis=-1)  # (D, H, W, 4)

        return img[np.newaxis], mask[np.newaxis]  # (1, D, H, W, 4)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


# ── Loss ──────────────────────────────────────────────────────────────────────

def dice_loss(y_true, y_pred, smooth=1e-6):
    """
    Soft multi-class Dice loss computed over tumor classes only (skip background).
    y_true: one-hot (B, D, H, W, 4)
    y_pred: logits (B, D, H, W, 4)
    """
    probs = tf.nn.softmax(y_pred, axis=-1)
    # Only tumor classes: channels 1, 2, 3
    y_true_t = y_true[..., 1:]   # (B, D, H, W, 3)
    probs_t  = probs[...,  1:]

    axes = [1, 2, 3]  # sum over D, H, W
    intersection = tf.reduce_sum(y_true_t * probs_t, axis=axes)   # (B, 3)
    denom        = tf.reduce_sum(y_true_t + probs_t,  axis=axes)   # (B, 3)
    dice_per_class = (2.0 * intersection + smooth) / (denom + smooth)
    return 1.0 - tf.reduce_mean(dice_per_class)


def weighted_ce_loss(y_true, y_pred, class_weights):
    """Volume-weighted cross-entropy."""
    weights = tf.constant(class_weights, dtype=tf.float32)
    ce = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
    # Weight each voxel by its class weight
    voxel_weights = tf.reduce_sum(y_true * weights, axis=-1)
    return tf.reduce_mean(ce * voxel_weights)


def combined_loss(y_true, y_pred, class_weights, alpha=0.5):
    return alpha * dice_loss(y_true, y_pred) + (1.0 - alpha) * weighted_ce_loss(y_true, y_pred, class_weights)


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_dice_scores(y_true, y_pred_logits, smooth=1e-6):
    """
    Returns dict with per-class and composite Dice scores.
    Class ordering: ch0=BG, ch1=NCR, ch2=ED, ch3=ET
    """
    probs = tf.nn.softmax(y_pred_logits, axis=-1).numpy()
    pred_hard = (probs == probs.max(axis=-1, keepdims=True)).astype(np.float32)

    scores = {}
    class_names = ['bg', 'ncr', 'ed', 'et']
    for ch, name in enumerate(class_names):
        if name == 'bg':
            continue
        inter = (y_true[..., ch] * pred_hard[..., ch]).sum()
        denom = y_true[..., ch].sum() + pred_hard[..., ch].sum()
        scores[f'dice_{name}'] = float((2.0 * inter + smooth) / (denom + smooth))

    # Whole-tumor: union of NCR + ED + ET
    wt_true = y_true[..., 1:].max(axis=-1)
    wt_pred = pred_hard[..., 1:].max(axis=-1)
    inter = (wt_true * wt_pred).sum()
    denom = wt_true.sum() + wt_pred.sum()
    scores['dice_wt'] = float((2.0 * inter + smooth) / (denom + smooth))

    scores['dice_mean'] = float(np.mean([scores['dice_ncr'], scores['dice_ed'], scores['dice_et']]))
    return scores


# ── Training ──────────────────────────────────────────────────────────────────

def run_validation(model, val_gen):
    all_scores = []
    for i in range(len(val_gen)):
        imgs, masks = val_gen[i]
        logits = model(imgs, training=False)
        scores = compute_dice_scores(masks[0], logits[0])
        all_scores.append(scores)

    avg = {}
    for key in all_scores[0]:
        avg[key] = float(np.mean([s[key] for s in all_scores]))
    return avg


def train(args):
    log.info("=" * 60)
    log.info("STEP 1: Plain 3D U-Net Baseline")
    log.info("=" * 60)

    # GPU setup
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    log.info(f"GPUs: {gpus}")
    log.info(f"Mixed precision policy: {tf.keras.mixed_precision.global_policy().name}")

    # Data path
    data_path = args.data_dir
    h5_files = [f for f in os.listdir(data_path) if f.endswith('.h5')]
    log.info(f"Found {len(h5_files)} H5 files in {data_path}")

    # Train/val split — deterministic
    np.random.seed(42)
    all_ids = list(range(1, args.num_volumes + 1))
    shuffled = np.random.permutation(all_ids)
    n_val = int(len(all_ids) * args.split_ratio)
    val_ids   = shuffled[:n_val].tolist()
    train_ids = shuffled[n_val:].tolist()
    log.info(f"Train volumes: {len(train_ids)}  |  Val volumes: {len(val_ids)}")

    train_gen = MRIDataGenerator(data_path, train_ids, num_slices=args.num_slices,
                                  batch_size=1, shuffle=True)
    val_gen   = MRIDataGenerator(data_path, val_ids,   num_slices=args.num_slices,
                                  batch_size=1, shuffle=False)

    # Model
    log.info("Building UNet3D...")
    from unet3d import UNet3D
    model = UNet3D(n_classes=args.num_classes, base_channels=args.base_channels)
    dummy = tf.zeros((1, args.num_slices, args.height, args.width, args.channels))
    _ = model(dummy, training=False)
    model.summary(print_fn=log.info)
    log.info(f"Trainable params: {model.count_params():,}")

    optimizer = keras.optimizers.Adam(learning_rate=args.lr)

    # Class weights: inverse of approximate class frequency (from Step 0)
    # BG ~90%, NCR ~2%, ED ~5%, ET ~1.5%  → higher weight for rarer classes
    class_weights = [0.1, 2.0, 1.0, 2.5]
    log.info(f"Class weights [BG, NCR, ED, ET]: {class_weights}")

    # Training state
    best_val_mean_dice = -1.0
    patience_counter = 0
    history = []

    log.info(f"Starting training for up to {args.epochs} epochs (patience={args.patience})")
    log.info("=" * 60)

    for epoch in range(1, args.epochs + 1):
        epoch_start = datetime.now()
        train_losses = []

        # ── Training loop ──────────────────────────────────────
        for batch_idx in range(len(train_gen)):
            imgs, masks = train_gen[batch_idx]
            imgs  = tf.constant(imgs,  dtype=tf.float32)
            masks = tf.constant(masks, dtype=tf.float32)

            with tf.GradientTape() as tape:
                logits = model(imgs, training=True)
                loss   = combined_loss(masks, logits, class_weights, alpha=args.loss_alpha)

            grads = tape.gradient(loss, model.trainable_variables)
            grads, _ = tf.clip_by_global_norm(grads, 1.0)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            train_losses.append(float(loss))

        train_gen.on_epoch_end()

        avg_train_loss = float(np.mean(train_losses))

        # ── Validation ─────────────────────────────────────────
        val_scores = run_validation(model, val_gen)
        elapsed = (datetime.now() - epoch_start).seconds

        log.info(
            f"Epoch {epoch:>3d}/{args.epochs} | "
            f"loss={avg_train_loss:.4f} | "
            f"NCR={val_scores['dice_ncr']:.3f} "
            f"ED={val_scores['dice_ed']:.3f} "
            f"ET={val_scores['dice_et']:.3f} | "
            f"WT={val_scores['dice_wt']:.3f} "
            f"Mean={val_scores['dice_mean']:.3f} | "
            f"{elapsed}s"
        )

        entry = {'epoch': epoch, 'train_loss': avg_train_loss, **val_scores}
        history.append(entry)

        # ── Checkpoint on improvement ───────────────────────────
        if val_scores['dice_mean'] > best_val_mean_dice:
            best_val_mean_dice = val_scores['dice_mean']
            patience_counter = 0
            ckpt_path = os.path.join(args.model_dir, 'best_model.weights.h5')
            model.save_weights(ckpt_path)
            log.info(f"  ✓ New best mean Dice={best_val_mean_dice:.3f} — checkpoint saved")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                log.info(f"Early stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
                break

    # ── Final summary ──────────────────────────────────────────
    log.info("=" * 60)
    log.info("TRAINING COMPLETE")
    log.info(f"Best val mean Dice: {best_val_mean_dice:.4f}")
    log.info(f"Pass criteria:")
    log.info(f"  Whole-tumor Dice > 0.65: {'PASS' if history[-1]['dice_wt'] > 0.65 else 'FAIL'} ({history[-1]['dice_wt']:.3f})")
    log.info(f"  Mean class Dice  > 0.40: {'PASS' if best_val_mean_dice > 0.40 else 'FAIL'} ({best_val_mean_dice:.3f})")
    log.info("=" * 60)

    # Save history
    history_path = os.path.join(args.output_dir, 'history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    log.info(f"History saved: {history_path}")

    # Save final weights
    model.save_weights(os.path.join(args.model_dir, 'final_model.weights.h5'))


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser()
    # SageMaker paths
    parser.add_argument('--model-dir',  default=os.environ.get('SM_MODEL_DIR',  '/opt/ml/model'))
    parser.add_argument('--data-dir',   default=os.environ.get('SM_CHANNEL_TRAINING', '/opt/ml/input/data/training'))
    parser.add_argument('--output-dir', default=os.environ.get('SM_OUTPUT_DATA_DIR',  '/opt/ml/output/data'))
    # Volume dimensions (actual stored shape: H=192, W=160)
    parser.add_argument('--num-slices', type=int, default=128)
    parser.add_argument('--height',     type=int, default=192)
    parser.add_argument('--width',      type=int, default=160)
    parser.add_argument('--channels',   type=int, default=4)
    parser.add_argument('--num-classes',type=int, default=4)
    # Dataset
    parser.add_argument('--num-volumes',type=int, default=369)
    parser.add_argument('--split-ratio',type=float, default=0.2)
    # Model
    parser.add_argument('--base-channels', type=int, default=16,
                        help='Base channels for UNet (16=~1.3M params, 32=~5.4M params). '
                             'Use 16 on ml.g4dn.xlarge, 32 on ml.g5.xlarge or larger.')
    # Training
    parser.add_argument('--epochs',     type=int,   default=50)
    parser.add_argument('--patience',   type=int,   default=10)
    parser.add_argument('--lr',         type=float, default=1e-4)
    parser.add_argument('--loss-alpha', type=float, default=0.5,
                        help='Weight for Dice loss (1-alpha for CE loss)')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    os.makedirs(args.model_dir,  exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    train(args)
