#!/usr/bin/env python3
"""
Step 2 SageMaker Training Script — UNet3D + Prototype Bottleneck

Changes vs Step 1:
  - Prototype layer added at bottleneck (9 prototypes, 3 per tumor class)
  - Pure Dice loss only (no cross-entropy)
  - Prototype activation ratio logged per epoch

Pass criterion:
  Mean Dice within 5 absolute percentage points of Step 1 final baseline.
  This confirms the prototype layer doesn't interfere with segmentation.
  If it drops more than 5pp, check prototype_to_features magnitude vs bottleneck.
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


# ── Data generator (identical to Step 1) ──────────────────────────────────────

class MRIDataGenerator:
    def __init__(self, folder_path, volume_ids, num_slices=128, batch_size=1, shuffle=True):
        self.folder_path = folder_path
        self.volume_ids  = list(volume_ids)
        self.num_slices  = num_slices
        self.batch_size  = batch_size
        self.shuffle     = shuffle
        self.indices     = np.arange(len(self.volume_ids))
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

        img  = np.stack(img_slices,  axis=0)   # (D, H, W, 4)
        mask = np.stack(mask_slices, axis=0)   # (D, H, W, 3)

        vmin, vmax = img.min(), img.max()
        if vmax - vmin > 1e-8:
            img = (img - vmin) / (vmax - vmin)

        mask = mask.astype(np.float32)
        bg   = (mask.sum(axis=-1, keepdims=True) == 0).astype(np.float32)
        mask = np.concatenate([bg, mask], axis=-1)   # (D, H, W, 4)

        return img[np.newaxis], mask[np.newaxis]     # (1, D, H, W, 4)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


# ── Loss — pure Dice only ──────────────────────────────────────────────────────

def dice_loss(y_true, y_pred, smooth=1e-6):
    """Soft multi-class Dice over tumor classes only (skip background)."""
    probs    = tf.nn.softmax(y_pred, axis=-1)
    y_true_t = y_true[..., 1:]
    probs_t  = probs[...,  1:]
    axes     = [1, 2, 3]
    inter    = tf.reduce_sum(y_true_t * probs_t, axis=axes)
    denom    = tf.reduce_sum(y_true_t + probs_t, axis=axes)
    return 1.0 - tf.reduce_mean((2.0 * inter + smooth) / (denom + smooth))


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_dice_scores(y_true, y_pred_logits, smooth=1e-6):
    probs     = tf.nn.softmax(y_pred_logits, axis=-1).numpy()
    pred_hard = (probs == probs.max(axis=-1, keepdims=True)).astype(np.float32)

    scores = {}
    for ch, name in enumerate(['bg', 'ncr', 'ed', 'et']):
        if name == 'bg':
            continue
        inter = (y_true[..., ch] * pred_hard[..., ch]).sum()
        denom = y_true[..., ch].sum() + pred_hard[..., ch].sum()
        scores[f'dice_{name}'] = float((2.0 * inter + smooth) / (denom + smooth))

    wt_true = y_true[..., 1:].max(axis=-1)
    wt_pred = pred_hard[..., 1:].max(axis=-1)
    inter   = (wt_true * wt_pred).sum()
    denom   = wt_true.sum() + wt_pred.sum()
    scores['dice_wt']   = float((2.0 * inter + smooth) / (denom + smooth))
    scores['dice_mean'] = float(np.mean([scores['dice_ncr'],
                                          scores['dice_ed'],
                                          scores['dice_et']]))
    return scores


# ── Prototype activation ratio ─────────────────────────────────────────────────

def compute_activation_ratios(similarities_np, mask_np, protos_per_class,
                               pool_factor=8, epsilon=1e-6):
    """
    For each prototype k (assigned to tumor class c_k), compute:
        ratio_k = mean_similarity_inside_c_k / mean_similarity_outside_c_k

    A ratio > 1 means the prototype fires more strongly within its own class.
    Collapsing prototypes produce ratios ≈ 1.

    similarities_np : (D, H', W', P)  — bottleneck resolution
    mask_np         : (D, H,  W,  4)  — full resolution one-hot [BG, NCR, ED, ET]
    pool_factor     : spatial downscale applied by 3 encoder pooling ops (2^3 = 8)
    """
    D, H, W, _ = mask_np.shape
    H_s = H // pool_factor
    W_s = W // pool_factor

    # Downsample mask to bottleneck resolution via average pooling, then argmax
    mask_b = mask_np[np.newaxis]   # (1, D, H, W, 4)
    mask_t = tf.constant(mask_b, dtype=tf.float32)
    mask_small = tf.nn.avg_pool3d(
        mask_t, ksize=[1, 1, pool_factor, pool_factor, 1],
        strides=[1, 1, pool_factor, pool_factor, 1], padding='VALID'
    ).numpy()[0]                   # (D, H', W', 4)
    class_map = np.argmax(mask_small, axis=-1)  # (D, H', W')  values 0-3

    ratios = {}
    for p in range(protos_per_class * 3):
        tumor_class = p // protos_per_class + 1   # 1=NCR, 2=ED, 3=ET
        class_name  = {1: 'ncr', 2: 'ed', 3: 'et'}[tumor_class]
        sim         = similarities_np[..., p]     # (D, H', W')

        inside  = class_map == tumor_class
        outside = ~inside

        mean_in  = sim[inside].mean()  if inside.any()  else 0.0
        mean_out = sim[outside].mean() if outside.any() else epsilon
        ratio    = float(mean_in / (mean_out + epsilon))
        ratios[f'proto{p}_{class_name}'] = ratio

    return ratios


# ── Validation ────────────────────────────────────────────────────────────────

def run_validation(model, val_gen, protos_per_class, log_proto_ratios=True):
    all_scores  = []
    all_ratios  = []

    for i in range(len(val_gen)):
        imgs, masks = val_gen[i]
        imgs_t = tf.constant(imgs, dtype=tf.float32)

        if log_proto_ratios:
            logits, sims = model.forward_with_similarities(imgs_t)
            sims_np = sims.numpy()[0]   # (D, H', W', P)
            ratios  = compute_activation_ratios(sims_np, masks[0], protos_per_class)
            all_ratios.append(ratios)
        else:
            logits = model(imgs_t, training=False)

        scores = compute_dice_scores(masks[0], logits[0])
        all_scores.append(scores)

    avg_scores = {k: float(np.mean([s[k] for s in all_scores]))
                  for k in all_scores[0]}

    avg_ratios = {}
    if all_ratios:
        avg_ratios = {k: float(np.mean([r[k] for r in all_ratios]))
                      for k in all_ratios[0]}

    return avg_scores, avg_ratios


# ── Step 1 weight transfer ────────────────────────────────────────────────────

def fetch_step1_weights(s3_uri):
    """
    Download and extract model.tar.gz from S3.
    Returns the local path to best_model.weights.h5.
    """
    import boto3, tarfile
    s3_path  = s3_uri.replace('s3://', '')
    bucket, key = s3_path.split('/', 1)

    local_dir = '/tmp/step1_model'
    os.makedirs(local_dir, exist_ok=True)
    local_tar = os.path.join(local_dir, 'model.tar.gz')

    log.info(f"Downloading Step 1 model: s3://{bucket}/{key}")
    boto3.client('s3').download_file(bucket, key, local_tar)

    with tarfile.open(local_tar, 'r:gz') as tar:
        tar.extractall(local_dir)
    log.info(f"Extracted to {local_dir}: {os.listdir(local_dir)}")

    weights_path = os.path.join(local_dir, 'best_model.weights.h5')
    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"best_model.weights.h5 not found in {local_dir} after extraction. "
            f"Contents: {os.listdir(local_dir)}")
    return weights_path


def load_step1_weights(proto_model, weights_path, args):
    """
    Build a UNet3D, load Step 1 weights into it, then copy the shared
    layers to the prototype model.  The prototype-specific layers
    (prototype_vectors, prototype_to_features) keep their own initialization.
    """
    from unet3d_proto import UNet3D
    step1 = UNet3D(n_classes=args.num_classes, base_channels=args.base_channels)
    dummy = tf.zeros((1, args.num_slices, args.height, args.width, args.channels))
    _     = step1(dummy, training=False)
    step1.load_weights(weights_path)
    log.info(f"Step 1 weights loaded from: {weights_path}")

    shared = ['enc1', 'enc2', 'enc3', 'bottleneck', 'dec3', 'dec2', 'dec1', 'out_conv']
    for name in shared:
        src_weights = getattr(step1, name).get_weights()
        getattr(proto_model, name).set_weights(src_weights)

    log.info(f"Transferred {len(shared)} layer groups from Step 1 → UNet3DProto")
    log.info("Prototype layers (prototype_vectors, prototype_to_features) "
             "kept at fresh initialization.")


# ── Training ──────────────────────────────────────────────────────────────────

def train(args):
    log.info("=" * 60)
    log.info("STEP 2: UNet3D + Prototype Bottleneck")
    log.info("=" * 60)

    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    log.info(f"GPUs: {gpus}")

    data_path = args.data_dir
    h5_files  = [f for f in os.listdir(data_path) if f.endswith('.h5')]
    log.info(f"Found {len(h5_files)} H5 files in {data_path}")

    np.random.seed(42)
    all_ids  = list(range(1, args.num_volumes + 1))
    shuffled = np.random.permutation(all_ids)
    n_val    = int(len(all_ids) * args.split_ratio)
    val_ids  = shuffled[:n_val].tolist()
    train_ids = shuffled[n_val:].tolist()
    log.info(f"Train volumes: {len(train_ids)}  |  Val volumes: {len(val_ids)}")

    train_gen = MRIDataGenerator(data_path, train_ids, num_slices=args.num_slices,
                                  batch_size=1, shuffle=True)
    val_gen   = MRIDataGenerator(data_path, val_ids,   num_slices=args.num_slices,
                                  batch_size=1, shuffle=False)

    log.info("Building UNet3DProto...")
    from unet3d_proto import UNet3DProto
    model = UNet3DProto(n_classes=args.num_classes,
                        base_channels=args.base_channels,
                        protos_per_class=args.protos_per_class)
    dummy = tf.zeros((1, args.num_slices, args.height, args.width, args.channels))
    _     = model(dummy, training=False)
    model.summary(print_fn=log.info)
    log.info(f"Trainable params: {model.count_params():,}")
    log.info(f"Prototypes: {model.num_prototypes} total "
             f"({args.protos_per_class} per tumor class)")

    if args.step1_weights:
        load_step1_weights(model, args.step1_weights, args)
    else:
        log.info("No Step 1 weights provided — training from scratch.")

    optimizer = keras.optimizers.Adam(learning_rate=args.lr)

    best_val_mean_dice = -1.0
    patience_counter   = 0
    history            = []

    log.info(f"Loss: pure Dice (no cross-entropy)")
    log.info(f"Starting training for up to {args.epochs} epochs "
             f"(patience={args.patience})")
    log.info("=" * 60)

    for epoch in range(1, args.epochs + 1):
        epoch_start  = datetime.now()
        train_losses = []

        for batch_idx in range(len(train_gen)):
            imgs, masks = train_gen[batch_idx]
            imgs_t  = tf.constant(imgs,  dtype=tf.float32)
            masks_t = tf.constant(masks, dtype=tf.float32)

            with tf.GradientTape() as tape:
                logits = model(imgs_t, training=True)
                loss   = dice_loss(masks_t, logits)

            grads = tape.gradient(loss, model.trainable_variables)
            grads, _ = tf.clip_by_global_norm(grads, 1.0)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            train_losses.append(float(loss))

        train_gen.on_epoch_end()
        avg_loss = float(np.mean(train_losses))

        val_scores, proto_ratios = run_validation(
            model, val_gen, args.protos_per_class, log_proto_ratios=True
        )
        elapsed = (datetime.now() - epoch_start).seconds

        log.info(
            f"Epoch {epoch:>3d}/{args.epochs} | "
            f"loss={avg_loss:.4f} | "
            f"NCR={val_scores['dice_ncr']:.3f} "
            f"ED={val_scores['dice_ed']:.3f} "
            f"ET={val_scores['dice_et']:.3f} | "
            f"WT={val_scores['dice_wt']:.3f} "
            f"Mean={val_scores['dice_mean']:.3f} | "
            f"{elapsed}s"
        )

        # Log prototype activation ratios (compact: one line)
        ratio_str = '  '.join(f"{k}={v:.2f}" for k, v in sorted(proto_ratios.items()))
        log.info(f"  Proto ratios  | {ratio_str}")

        entry = {'epoch': epoch, 'train_loss': avg_loss,
                 **val_scores, **{f'ratio_{k}': v for k, v in proto_ratios.items()}}
        history.append(entry)

        if val_scores['dice_mean'] > best_val_mean_dice:
            best_val_mean_dice = val_scores['dice_mean']
            patience_counter   = 0
            ckpt_path = os.path.join(args.model_dir, 'best_model.weights.h5')
            model.save_weights(ckpt_path)
            log.info(f"  ✓ New best mean Dice={best_val_mean_dice:.3f} — checkpoint saved")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                log.info(f"Early stopping at epoch {epoch} "
                         f"(no improvement for {args.patience} epochs)")
                break

    # ── Final summary ──────────────────────────────────────────────────────────
    last = history[-1]
    step1_baseline = args.step1_baseline_dice

    log.info("=" * 60)
    log.info("TRAINING COMPLETE")
    log.info(f"Best val mean Dice : {best_val_mean_dice:.4f}")
    log.info(f"Step 1 baseline    : {step1_baseline:.4f}")
    log.info(f"Difference         : {best_val_mean_dice - step1_baseline:+.4f}")
    within_5pp = best_val_mean_dice >= step1_baseline - 0.05
    log.info(f"Pass criterion (within 5pp of baseline): "
             f"{'PASS' if within_5pp else 'FAIL'}")
    log.info(f"WT Dice > 0.65 : "
             f"{'PASS' if last['dice_wt'] > 0.65 else 'FAIL'} "
             f"({last['dice_wt']:.3f})")
    log.info("=" * 60)

    history_path = os.path.join(args.output_dir, 'history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    log.info(f"History saved: {history_path}")

    model.save_weights(os.path.join(args.model_dir, 'final_model.weights.h5'))


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir',  default=os.environ.get('SM_MODEL_DIR',  '/opt/ml/model'))
    parser.add_argument('--data-dir',   default=os.environ.get('SM_CHANNEL_TRAINING', '/opt/ml/input/data/training'))
    parser.add_argument('--output-dir', default=os.environ.get('SM_OUTPUT_DATA_DIR',  '/opt/ml/output/data'))
    parser.add_argument('--num-slices', type=int, default=128)
    parser.add_argument('--height',     type=int, default=192)
    parser.add_argument('--width',      type=int, default=160)
    parser.add_argument('--channels',   type=int, default=4)
    parser.add_argument('--num-classes',type=int, default=4)
    parser.add_argument('--num-volumes',type=int, default=369)
    parser.add_argument('--split-ratio',type=float, default=0.2)
    parser.add_argument('--base-channels',   type=int,   default=16)
    parser.add_argument('--protos-per-class', type=int,   default=3,
                        help='Prototypes per tumor class (NCR/ED/ET). Total = 3x this.')
    parser.add_argument('--epochs',     type=int,   default=50)
    parser.add_argument('--patience',   type=int,   default=10)
    parser.add_argument('--lr',         type=float, default=1e-4)
    parser.add_argument('--step1-baseline-dice', type=float, default=0.659,
                        help='Final mean Dice from Step 1 — used for pass/fail report.')
    parser.add_argument('--step1-weights', type=str, default=None,
                        help='Local path to Step 1 best_model.weights.h5.')
    parser.add_argument('--step1-model-s3', type=str, default=None,
                        help='S3 URI of Step 1 model.tar.gz. Downloaded and extracted '
                             'automatically if --step1-weights is not set.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    os.makedirs(args.model_dir,  exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    # Resolve Step 1 weights: explicit path > S3 URI download
    if args.step1_weights is None and args.step1_model_s3:
        args.step1_weights = fetch_step1_weights(args.step1_model_s3)

    train(args)
