#!/usr/bin/env python3
"""
Step 3 SageMaker Training Script — UNet3DProto + Prototype Learning Losses

Adds three prototype losses on top of Step 2:

  L_total = L_dice
          + clst_weight * L_clustering   (pull each proto toward nearest in-class voxel)
          - sep_weight  * L_separation   (push each proto away from nearest out-of-class voxel)
          + div_weight  * L_diversity    (spread same-class protos apart via cosine similarity)

Initializes from Step 2 best checkpoint (identical architecture — direct load).

Pass criterion:
  Mean Dice ≥ Step 2 baseline AND proto ratios climbing above 1.2 for all classes.
  NCR ratio > 1.0 is the key signal: Step 2 left NCR ratios at ~0.91.
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

POOL_FACTOR = 8   # 3 encoder stages, each (1,2,2) pool → H/8, W/8


# ── Data generator ────────────────────────────────────────────────────────────

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
        img  = np.stack(img_slices,  axis=0)
        mask = np.stack(mask_slices, axis=0)
        vmin, vmax = img.min(), img.max()
        if vmax - vmin > 1e-8:
            img = (img - vmin) / (vmax - vmin)
        mask = mask.astype(np.float32)
        bg   = (mask.sum(axis=-1, keepdims=True) == 0).astype(np.float32)
        mask = np.concatenate([bg, mask], axis=-1)
        return img[np.newaxis], mask[np.newaxis]

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


# ── Losses ────────────────────────────────────────────────────────────────────

def dice_loss(y_true, y_pred, smooth=1e-6):
    probs    = tf.nn.softmax(y_pred, axis=-1)
    y_true_t = y_true[..., 1:]
    probs_t  = probs[...,  1:]
    axes     = [1, 2, 3]
    inter    = tf.reduce_sum(y_true_t * probs_t, axis=axes)
    denom    = tf.reduce_sum(y_true_t + probs_t, axis=axes)
    return 1.0 - tf.reduce_mean((2.0 * inter + smooth) / (denom + smooth))


def downsample_mask(mask_np):
    """
    (1, D, H, W, 4) numpy → (1, D, H/8, W/8) int TF tensor at bottleneck resolution.
    Average-pools the one-hot mask then takes argmax.
    """
    t = tf.constant(mask_np, dtype=tf.float32)
    small = tf.nn.avg_pool3d(t,
                              ksize=[1, 1, POOL_FACTOR, POOL_FACTOR, 1],
                              strides=[1, 1, POOL_FACTOR, POOL_FACTOR, 1],
                              padding='VALID')           # (1, D, H/8, W/8, 4)
    return tf.cast(tf.argmax(small, axis=-1), tf.int32)  # (1, D, H/8, W/8)


def proto_clustering_loss(distances, class_map, num_prototypes, protos_per_class, proto_dim):
    """
    For each prototype p (class c_p), minimize its minimum distance to any
    voxel in class c_p.  Pulls each prototype into the region of its class.

    Distances are normalized by sqrt(proto_dim) so the loss is scale-invariant
    and comparable to the Dice loss regardless of feature space dimensionality.
    Raw L2 distances in 128-dim space reach ~600; normalized they become ~53,
    putting clst_weight=0.01 in the same order of magnitude as Dice (~0.3).

    distances:  (B, D, H', W', P)
    class_map:  (B, D, H', W')  int labels 0–3
    proto_dim:  bottleneck channel count (used for normalization)
    """
    losses = []
    for p in range(num_prototypes):
        c      = p // protos_per_class + 1
        dist_p = distances[..., p]
        in_c   = tf.cast(tf.equal(class_map, c), tf.float32)
        masked = dist_p * in_c + (1.0 - in_c) * 1e6
        losses.append(tf.minimum(tf.reduce_min(masked), 1e4))
    return tf.reduce_mean(tf.stack(losses)) / tf.sqrt(float(proto_dim))


def proto_separation_loss(distances, class_map, num_prototypes, protos_per_class, proto_dim):
    """
    For each prototype p (class c_p), compute its minimum distance to any voxel
    NOT in class c_p.  We want to MAXIMIZE this, so the caller subtracts it.

    Normalized by sqrt(proto_dim) for the same reason as clustering loss.
    Returns a positive value.
    """
    losses = []
    for p in range(num_prototypes):
        c      = p // protos_per_class + 1
        dist_p = distances[..., p]
        out_c  = tf.cast(tf.not_equal(class_map, c), tf.float32)
        masked = dist_p * out_c + (1.0 - out_c) * 1e6
        losses.append(tf.minimum(tf.reduce_min(masked), 1e4))
    return tf.reduce_mean(tf.stack(losses)) / tf.sqrt(float(proto_dim))


def proto_diversity_loss(prototype_vectors, protos_per_class):
    """
    Penalize cosine similarity between same-class prototypes.
    Encourages each class's prototypes to capture different sub-patterns.

    prototype_vectors: tf.Variable (P, C, 1, 1, 1)
    """
    if protos_per_class < 2:
        return tf.constant(0.0)

    P    = protos_per_class * 3
    pvec = tf.reshape(prototype_vectors, [P, -1])   # (P, C)
    pvec = tf.nn.l2_normalize(pvec, axis=1)          # unit vectors

    total = tf.constant(0.0)
    n_pairs = float(protos_per_class * (protos_per_class - 1))
    for c in range(3):
        i  = c * protos_per_class
        cp = pvec[i: i + protos_per_class]                        # (K, C)
        gram = tf.matmul(cp, cp, transpose_b=True)                # (K, K)
        off  = gram * (1.0 - tf.eye(protos_per_class))
        total = total + tf.reduce_sum(tf.maximum(off, 0.0)) / n_pairs
    return total / 3.0


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
                                          scores['dice_ed'], scores['dice_et']]))
    return scores


def compute_activation_ratios(similarities_np, mask_np, protos_per_class, epsilon=1e-6):
    mask_b = mask_np[np.newaxis]
    mask_t = tf.constant(mask_b, dtype=tf.float32)
    mask_small = tf.nn.avg_pool3d(
        mask_t, ksize=[1, 1, POOL_FACTOR, POOL_FACTOR, 1],
        strides=[1, 1, POOL_FACTOR, POOL_FACTOR, 1], padding='VALID'
    ).numpy()[0]
    class_map = np.argmax(mask_small, axis=-1)
    ratios = {}
    for p in range(protos_per_class * 3):
        tc   = p // protos_per_class + 1
        name = {1: 'ncr', 2: 'ed', 3: 'et'}[tc]
        sim  = similarities_np[..., p]
        inside  = class_map == tc
        outside = ~inside
        mean_in  = sim[inside].mean()  if inside.any()  else 0.0
        mean_out = sim[outside].mean() if outside.any() else epsilon
        ratios[f'proto{p}_{name}'] = float(mean_in / (mean_out + epsilon))
    return ratios


# ── Validation ────────────────────────────────────────────────────────────────

def run_validation(model, val_gen, protos_per_class):
    all_scores, all_ratios = [], []
    for i in range(len(val_gen)):
        imgs, masks = val_gen[i]
        imgs_t      = tf.constant(imgs, dtype=tf.float32)
        logits, sims = model.forward_with_similarities(imgs_t)
        sims_np      = sims.numpy()[0]
        all_scores.append(compute_dice_scores(masks[0], logits[0]))
        all_ratios.append(compute_activation_ratios(sims_np, masks[0], protos_per_class))

    avg_scores = {k: float(np.mean([s[k] for s in all_scores])) for k in all_scores[0]}
    avg_ratios = {k: float(np.mean([r[k] for r in all_ratios])) for k in all_ratios[0]}
    return avg_scores, avg_ratios


# ── Weight loading ────────────────────────────────────────────────────────────

def fetch_weights(s3_uri, local_dir):
    """Download and extract model.tar.gz from S3; return path to best_model.weights.h5."""
    import boto3, tarfile
    s3_path      = s3_uri.replace('s3://', '')
    bucket, key  = s3_path.split('/', 1)
    os.makedirs(local_dir, exist_ok=True)
    local_tar    = os.path.join(local_dir, 'model.tar.gz')
    log.info(f"Downloading: s3://{bucket}/{key}")
    boto3.client('s3').download_file(bucket, key, local_tar)
    with tarfile.open(local_tar, 'r:gz') as tar:
        tar.extractall(local_dir)
    log.info(f"Extracted: {os.listdir(local_dir)}")
    path = os.path.join(local_dir, 'best_model.weights.h5')
    if not os.path.exists(path):
        raise FileNotFoundError(f"best_model.weights.h5 not in {local_dir}")
    return path


# ── Training ──────────────────────────────────────────────────────────────────

def train(args):
    log.info("=" * 60)
    log.info("STEP 3: UNet3DProto + Prototype Learning Losses")
    log.info("=" * 60)

    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    log.info(f"GPUs: {gpus}")

    data_path = args.data_dir
    log.info(f"Found {len([f for f in os.listdir(data_path) if f.endswith('.h5')])} H5 files")

    np.random.seed(42)
    all_ids   = list(range(1, args.num_volumes + 1))
    shuffled  = np.random.permutation(all_ids)
    n_val     = int(len(all_ids) * args.split_ratio)
    val_ids   = shuffled[:n_val].tolist()
    train_ids = shuffled[n_val:].tolist()
    log.info(f"Train: {len(train_ids)}  Val: {len(val_ids)}")

    train_gen = MRIDataGenerator(data_path, train_ids,
                                  num_slices=args.num_slices, batch_size=1, shuffle=True)
    val_gen   = MRIDataGenerator(data_path, val_ids,
                                  num_slices=args.num_slices, batch_size=1, shuffle=False)

    log.info("Building UNet3DProto...")
    from unet3d_proto import UNet3DProto
    model = UNet3DProto(n_classes=args.num_classes,
                        base_channels=args.base_channels,
                        protos_per_class=args.protos_per_class)
    dummy = tf.zeros((1, args.num_slices, args.height, args.width, args.channels))
    _     = model(dummy, training=False)
    model.summary(print_fn=log.info)
    log.info(f"Trainable params: {model.count_params():,}")

    if args.step2_weights:
        try:
            model.load_weights(args.step2_weights)
            log.info(f"Loaded Step 2 weights (strict): {args.step2_weights}")
        except (ValueError, tf.errors.InvalidArgumentError):
            # protos_per_class changed — load backbone only, prototype layers start fresh
            model.load_weights(args.step2_weights, by_name=True, skip_mismatch=True)
            log.info(f"Loaded Step 2 backbone weights (prototype layers reinitialised): {args.step2_weights}")
    else:
        log.info("No pretrained weights — training from scratch.")

    log.info(f"Loss weights: dice=1.0  clst={args.clst_weight}  "
             f"sep={args.sep_weight}  div={args.div_weight}")
    log.info("=" * 60)

    optimizer          = keras.optimizers.Adam(learning_rate=args.lr)
    best_val_mean_dice = -1.0
    patience_counter   = 0
    history            = []

    for epoch in range(1, args.epochs + 1):
        epoch_start  = datetime.now()
        epoch_losses = []

        for batch_idx in range(len(train_gen)):
            imgs, masks = train_gen[batch_idx]
            imgs_t    = tf.constant(imgs,  dtype=tf.float32)
            masks_t   = tf.constant(masks, dtype=tf.float32)
            class_map = downsample_mask(masks)   # (1, D, H/8, W/8)

            with tf.GradientTape() as tape:
                logits, distances = model.forward_train(imgs_t)

                l_dice = dice_loss(masks_t, logits)
                l_clst = proto_clustering_loss(distances, class_map,
                                               model.num_prototypes, args.protos_per_class,
                                               model.proto_dim)
                l_sep  = proto_separation_loss(distances, class_map,
                                               model.num_prototypes, args.protos_per_class,
                                               model.proto_dim)
                l_div  = proto_diversity_loss(model.prototype_vectors, args.protos_per_class)

                loss = (l_dice
                        + args.clst_weight * l_clst
                        - args.sep_weight  * l_sep   # maximize separation
                        + args.div_weight  * l_div)

            grads = tape.gradient(loss, model.trainable_variables)
            grads, _ = tf.clip_by_global_norm(grads, 1.0)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            epoch_losses.append({
                'total': float(loss),   'dice': float(l_dice),
                'clst':  float(l_clst), 'sep':  float(l_sep),
                'div':   float(l_div),
            })

        train_gen.on_epoch_end()

        avg = {k: float(np.mean([b[k] for b in epoch_losses])) for k in epoch_losses[0]}
        val_scores, proto_ratios = run_validation(model, val_gen, args.protos_per_class)
        elapsed = (datetime.now() - epoch_start).seconds

        log.info(
            f"Epoch {epoch:>3d}/{args.epochs} | "
            f"loss={avg['total']:.4f} "
            f"[dice={avg['dice']:.4f} clst={avg['clst']:.3f} "
            f"sep={avg['sep']:.3f} div={avg['div']:.3f}] | "
            f"NCR={val_scores['dice_ncr']:.3f} "
            f"ED={val_scores['dice_ed']:.3f} "
            f"ET={val_scores['dice_et']:.3f} | "
            f"WT={val_scores['dice_wt']:.3f} "
            f"Mean={val_scores['dice_mean']:.3f} | "
            f"{elapsed}s"
        )
        ratio_str = '  '.join(f"{k}={v:.2f}" for k, v in sorted(proto_ratios.items()))
        log.info(f"  Proto ratios  | {ratio_str}")

        entry = {'epoch': epoch, **avg, **val_scores,
                 **{f'ratio_{k}': v for k, v in proto_ratios.items()}}
        history.append(entry)

        if val_scores['dice_mean'] > best_val_mean_dice:
            best_val_mean_dice = val_scores['dice_mean']
            patience_counter   = 0
            model.save_weights(os.path.join(args.model_dir, 'best_model.weights.h5'))
            log.info(f"  ✓ New best mean Dice={best_val_mean_dice:.3f} — checkpoint saved")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                log.info(f"Early stopping at epoch {epoch}")
                break

    # ── Summary ───────────────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("TRAINING COMPLETE")
    log.info(f"Best val mean Dice  : {best_val_mean_dice:.4f}")
    log.info(f"Step 2 baseline     : {args.step2_baseline_dice:.4f}")
    log.info(f"Difference          : {best_val_mean_dice - args.step2_baseline_dice:+.4f}")

    # Report final proto ratios (from last epoch)
    last_ratios = {k: v for k, v in history[-1].items() if k.startswith('ratio_')}
    ncr_ratios  = [v for k, v in last_ratios.items() if 'ncr' in k]
    ed_ratios   = [v for k, v in last_ratios.items() if '_ed' in k]
    et_ratios   = [v for k, v in last_ratios.items() if '_et' in k]
    log.info(f"Final proto ratios  : "
             f"NCR={np.mean(ncr_ratios):.2f}  "
             f"ED={np.mean(ed_ratios):.2f}  "
             f"ET={np.mean(et_ratios):.2f}")
    log.info(f"  NCR > 1.0 (protos learned NCR specificity): "
             f"{'YES' if np.mean(ncr_ratios) > 1.0 else 'NO — diversity loss may need higher weight'}")
    log.info("=" * 60)

    with open(os.path.join(args.output_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    model.save_weights(os.path.join(args.model_dir, 'final_model.weights.h5'))


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model-dir',  default=os.environ.get('SM_MODEL_DIR',  '/opt/ml/model'))
    p.add_argument('--data-dir',   default=os.environ.get('SM_CHANNEL_TRAINING', '/opt/ml/input/data/training'))
    p.add_argument('--output-dir', default=os.environ.get('SM_OUTPUT_DATA_DIR',  '/opt/ml/output/data'))
    p.add_argument('--num-slices', type=int,   default=128)
    p.add_argument('--height',     type=int,   default=192)
    p.add_argument('--width',      type=int,   default=160)
    p.add_argument('--channels',   type=int,   default=4)
    p.add_argument('--num-classes',type=int,   default=4)
    p.add_argument('--num-volumes',type=int,   default=369)
    p.add_argument('--split-ratio',type=float, default=0.2)
    p.add_argument('--base-channels',    type=int,   default=16)
    p.add_argument('--protos-per-class', type=int,   default=3)
    p.add_argument('--epochs',           type=int,   default=30)
    p.add_argument('--patience',         type=int,   default=10)
    p.add_argument('--lr',               type=float, default=1e-4)
    # Prototype loss weights.
    # Distances are normalized by sqrt(proto_dim=128)≈11.3, giving clst≈53, sep≈0.2.
    # With clst_weight=0.01: clst contributes ~0.53 vs Dice ~0.30 (balanced).
    # With sep_weight=0.1:   sep contributes ~0.02 (moderate push).
    p.add_argument('--clst-weight', type=float, default=0.01,
                   help='Weight for clustering loss (pull protos toward class)')
    p.add_argument('--sep-weight',  type=float, default=0.1,
                   help='Weight for separation loss (push protos from other classes)')
    p.add_argument('--div-weight',  type=float, default=0.1,
                   help='Weight for diversity loss (spread same-class protos apart)')
    # Baselines and weight loading
    p.add_argument('--step2-baseline-dice', type=float, default=0.711,
                   help='Step 2 best mean Dice — used in final report.')
    p.add_argument('--step2-weights',   type=str, default=None,
                   help='Local path to Step 2 best_model.weights.h5.')
    p.add_argument('--step2-model-s3',  type=str, default=None,
                   help='S3 URI of Step 2 model.tar.gz. Downloaded if --step2-weights not set.')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    os.makedirs(args.model_dir,  exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.step2_weights is None and args.step2_model_s3:
        args.step2_weights = fetch_weights(args.step2_model_s3, '/tmp/step2_model')

    train(args)
