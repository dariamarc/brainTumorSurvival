#!/usr/bin/env python3
"""
Architecture 2 — Experiment A: ResNet3D + ASPP3D Baseline (no prototypes)

Improvements over initial version:
  - 5-fold cross-validation
  - Data augmentation (random D/H/W flips + per-modality intensity jitter)
  - HD95 metric alongside Dice

Architecture : Input → ResNet3D → ASPP3D → Conv3D(4) → upsample → logits
Loss         : Dice loss (tumour classes only)
Metrics      : Dice + HD95 per class (NCR, ED, ET), whole-tumour, mean
Output       : fold_{k}_best.weights.h5 × 5  +  cv_results.json
"""

import os, sys, json, argparse, logging, math
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from scipy.ndimage import distance_transform_edt, binary_erosion

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger(__name__)


# ── Data generator ────────────────────────────────────────────────────────────

class MRIDataGenerator:
    def __init__(self, folder_path, volume_ids, num_slices=128,
                 shuffle=True, augment=False, seed=42):
        self.folder_path = folder_path
        self.volume_ids  = list(volume_ids)
        self.num_slices  = num_slices
        self.shuffle     = shuffle
        self.augment     = augment
        self.rng         = np.random.default_rng(seed)
        self.indices     = np.arange(len(self.volume_ids))
        if shuffle:
            self.rng.shuffle(self.indices)

    def __len__(self):
        return len(self.volume_ids)

    def __getitem__(self, idx):
        import h5py
        vol_id = self.volume_ids[self.indices[idx]]
        imgs, masks = [], []
        for s in range(self.num_slices):
            fp = os.path.join(self.folder_path, f'volume_{vol_id}_slice_{s}.h5')
            with h5py.File(fp, 'r') as f:
                imgs.append(f['image'][:].astype(np.float32))
                masks.append(f['mask'][:])
        img  = np.stack(imgs,  axis=0)
        mask = np.stack(masks, axis=0).astype(np.float32)
        for c in range(img.shape[-1]):
            ch = img[..., c]
            nz = ch[ch > 0]
            if len(nz) > 0 and nz.std() > 1e-8:
                img[..., c] = (ch - nz.mean()) / nz.std()
        bg   = (mask.sum(-1, keepdims=True) == 0).astype(np.float32)
        mask = np.concatenate([bg, mask], axis=-1)
        if self.augment:
            img, mask = self._augment(img, mask)
        return img[None], mask[None]

    def _augment(self, img, mask):
        for axis in [0, 1, 2]:
            if self.rng.random() > 0.5:
                img  = np.flip(img,  axis=axis).copy()
                mask = np.flip(mask, axis=axis).copy()
        for m in range(img.shape[-1]):
            scale = self.rng.uniform(0.9, 1.1)
            shift = self.rng.uniform(-0.05, 0.05)
            img[..., m] = img[..., m] * scale + shift
        return img, mask

    def on_epoch_end(self):
        if self.shuffle:
            self.rng.shuffle(self.indices)


# ── K-fold ────────────────────────────────────────────────────────────────────

def make_folds(all_ids, n_folds=5, seed=42):
    rng      = np.random.default_rng(seed)
    shuffled = rng.permutation(all_ids)
    size     = len(shuffled) // n_folds
    folds    = []
    for k in range(n_folds):
        start   = k * size
        end     = start + size if k < n_folds - 1 else len(shuffled)
        val_ids = shuffled[start:end].tolist()
        trn_ids = [x for x in shuffled if x not in set(val_ids)]
        folds.append((trn_ids, val_ids))
    return folds


# ── Building blocks ───────────────────────────────────────────────────────────

class ResidualBlock3D(layers.Layer):
    def __init__(self, filters, strides=1, downsample=False, **kw):
        super().__init__(**kw)
        self.conv1    = layers.Conv3D(filters, 3, strides=strides, padding='same',
                                      use_bias=False, kernel_initializer='he_normal')
        self.ln1      = layers.LayerNormalization()
        self.conv2    = layers.Conv3D(filters, 3, padding='same',
                                      use_bias=False, kernel_initializer='he_normal')
        self.ln2      = layers.LayerNormalization()
        self.relu     = layers.ReLU()
        self.shortcut = (keras.Sequential([
            layers.Conv3D(filters, 1, strides=strides, padding='same',
                          use_bias=False, kernel_initializer='he_normal'),
            layers.LayerNormalization()
        ]) if downsample else None)

    def call(self, x, training=False):
        identity = x
        out = self.relu(self.ln1(self.conv1(x)))
        out = self.ln2(self.conv2(out))
        if self.shortcut is not None:
            identity = self.shortcut(x)
        return self.relu(out + identity)


class ResNet3D(keras.Model):
    def __init__(self, in_channels=4, base_channels=64, **kw):
        super().__init__(**kw)
        c = base_channels
        self.base_channels = base_channels
        self.conv1  = layers.Conv3D(c, 7, strides=2, padding='same',
                                    use_bias=False, kernel_initializer='he_normal',
                                    name='conv1')
        self.ln1    = layers.LayerNormalization(name='ln1')
        self.relu   = layers.ReLU(name='relu1')
        self.stage1 = self._stage(c,     2, stride=1, name='stage1')
        self.stage2 = self._stage(c * 2, 2, stride=2, name='stage2')
        self.stage3 = self._stage(c * 4, 2, stride=2, name='stage3')
        self.stage4 = self._stage(c * 8, 2, stride=1, name='stage4')

    def _stage(self, filters, n, stride, name):
        downsample = (stride != 1) or (filters != self.base_channels)
        blocks = [ResidualBlock3D(filters, strides=stride,
                                  downsample=downsample, name=f'{name}_block1')]
        for i in range(1, n):
            blocks.append(ResidualBlock3D(filters, name=f'{name}_block{i+1}'))
        return blocks

    def call(self, x, training=False):
        x = self.relu(self.ln1(self.conv1(x)))
        for b in self.stage1: x = b(x, training=training)
        for b in self.stage2: x = b(x, training=training)
        for b in self.stage3: x = b(x, training=training)
        for b in self.stage4: x = b(x, training=training)
        return x


class ASPP3D(layers.Layer):
    def __init__(self, in_channels, out_channels, dilation_rates=(2, 4, 8), **kw):
        super().__init__(**kw)
        self.out_channels = out_channels

        def _seq(name, **ckw):
            return keras.Sequential([
                layers.Conv3D(out_channels, use_bias=False,
                              kernel_initializer='he_normal', **ckw),
                layers.LayerNormalization(), layers.ReLU()
            ], name=name)

        self.conv1x1 = _seq('aspp_conv1x1', kernel_size=1, padding='same')
        self.conv_d1 = _seq(f'aspp_conv_d{dilation_rates[0]}', kernel_size=3,
                            padding='same', dilation_rate=dilation_rates[0])
        self.conv_d2 = _seq(f'aspp_conv_d{dilation_rates[1]}', kernel_size=3,
                            padding='same', dilation_rate=dilation_rates[1])
        self.conv_d3 = _seq(f'aspp_conv_d{dilation_rates[2]}', kernel_size=3,
                            padding='same', dilation_rate=dilation_rates[2])
        self.global_pool_conv = keras.Sequential([
            layers.Conv3D(out_channels, 1, padding='same',
                          use_bias=True, kernel_initializer='he_normal'),
            layers.LayerNormalization(), layers.ReLU()
        ], name='aspp_global_pool')
        self.fusion = _seq('aspp_fusion', kernel_size=1, padding='same')

    def call(self, x, training=False):
        sp = tf.shape(x)[1:4]
        x1 = self.conv1x1(x)
        x2 = self.conv_d1(x)
        x3 = self.conv_d2(x)
        x4 = self.conv_d3(x)
        x5 = tf.reduce_mean(x, axis=[1, 2, 3], keepdims=True)
        x5 = self.global_pool_conv(x5)
        x5 = tf.image.resize(
            tf.reshape(x5, [-1, 1,
                            tf.shape(x5)[1] * tf.shape(x5)[2],
                            self.out_channels]),
            [sp[0], sp[1] * sp[2]], method='bilinear')
        x5 = tf.reshape(x5, [-1, sp[0], sp[1], sp[2], self.out_channels])
        return self.fusion(layers.concatenate([x1, x2, x3, x4, x5], axis=-1))


class ResNetASPP3D(keras.Model):
    """ResNet3D + ASPP3D + classify at bottleneck + upsample logits."""

    def __init__(self, n_classes=4, base_channels=64,
                 aspp_out_channels=256, dilation_rates=(2, 4, 8), **kw):
        super().__init__(**kw)
        self.backbone = ResNet3D(in_channels=4, base_channels=base_channels,
                                 name='resnet3d_backbone')
        self.aspp     = ASPP3D(in_channels=base_channels * 8,
                               out_channels=aspp_out_channels,
                               dilation_rates=dilation_rates, name='aspp3d')
        self.out_conv = layers.Conv3D(n_classes, 1,
                                      kernel_initializer='glorot_uniform',
                                      name='out_conv')

    def call(self, inputs, training=False):
        orig = tf.shape(inputs)[1:4]
        x = self.backbone(inputs, training=training)
        x = self.aspp(x,          training=training)
        x = self.out_conv(x)
        return _trilinear_upsample(x, orig)


def _trilinear_upsample(x, target_size):
    sh = tf.shape(x)
    x = tf.reshape(x, [-1, sh[2], sh[3], sh[4]])
    x = tf.image.resize(x, [target_size[1], target_size[2]], method='bilinear')
    x = tf.reshape(x, [-1, sh[1], target_size[1], target_size[2], sh[4]])
    x = tf.transpose(x, [0, 2, 3, 1, 4])
    x = tf.reshape(x, [-1, sh[1], sh[4]])
    x = tf.expand_dims(x, 2)
    x = tf.image.resize(x, [target_size[0], 1], method='bilinear')
    x = tf.squeeze(x, 2)
    x = tf.reshape(x, [-1, target_size[1], target_size[2], target_size[0], sh[4]])
    return tf.transpose(x, [0, 3, 1, 2, 4])


# ── Loss ──────────────────────────────────────────────────────────────────────

def weighted_ce_loss(y_true, y_pred, n_classes=4, epsilon=1e-6):
    """Volume-weighted cross entropy: minor classes get weight η_ℓ = 1 - |X_ℓ|/|X|."""
    probs      = tf.nn.softmax(y_pred, axis=-1)
    y_true_f   = tf.reshape(y_true, [-1, n_classes])
    probs_f    = tf.reshape(probs,  [-1, n_classes])
    total      = tf.cast(tf.shape(y_true_f)[0], tf.float32)
    class_vol  = tf.reduce_sum(y_true_f, axis=0)
    weights    = 1.0 - class_vol / (total + epsilon)
    vox_w      = tf.reduce_sum(y_true_f * weights, axis=-1)
    ce         = -tf.reduce_sum(y_true_f * tf.math.log(tf.clip_by_value(probs_f, epsilon, 1.0)), axis=-1)
    return tf.reduce_mean(vox_w * ce)


def mdsc_loss(y_true, y_pred, n_classes=4, epsilon=1e-6):
    """Multi-class Dice loss with 1/N normalisation (suppresses noise)."""
    probs    = tf.nn.softmax(y_pred, axis=-1)
    y_true_f = tf.reshape(y_true, [-1, n_classes])
    probs_f  = tf.reshape(probs,  [-1, n_classes])
    N        = tf.cast(tf.shape(y_true_f)[0], tf.float32)
    inter    = tf.reduce_sum(y_true_f * probs_f, axis=0)
    denom    = tf.reduce_sum(y_true_f * y_true_f, axis=0) / N \
             + tf.reduce_sum(probs_f  * probs_f,  axis=0) / N + epsilon
    return -tf.reduce_sum((2.0 / N) * inter / denom)


def make_train_step(model, opt, alpha_mdsc):
    @tf.function
    def step(imgs_t, masks_t):
        with tf.GradientTape() as tape:
            logits  = model(imgs_t, training=True)
            l_ce    = weighted_ce_loss(masks_t, logits)
            l_mdsc  = mdsc_loss(masks_t, logits)
            loss    = l_ce + alpha_mdsc * l_mdsc
        grads = tape.gradient(loss, model.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, 1.0)
        opt.apply_gradients(zip(grads, model.trainable_variables))
        return loss, l_ce, l_mdsc
    return step


# ── Metrics ───────────────────────────────────────────────────────────────────

def hd95(pred, gt, spacing=(1.0, 1.0, 1.0)):
    pred, gt = pred.astype(bool), gt.astype(bool)
    if not gt.any() and not pred.any():
        return 0.0
    if not gt.any() or not pred.any():
        return np.nan
    pred_surf = pred ^ binary_erosion(pred)
    gt_surf   = gt   ^ binary_erosion(gt)
    d1 = distance_transform_edt(~gt,   sampling=spacing)[pred_surf]
    d2 = distance_transform_edt(~pred, sampling=spacing)[gt_surf]
    return float(np.percentile(np.concatenate([d1, d2]), 95))


def compute_metrics(y_true, logits, smooth=1e-6, compute_hd95=True):
    probs = tf.nn.softmax(logits, axis=-1).numpy()
    hard  = (probs == probs.max(axis=-1, keepdims=True)).astype(np.float32)
    out   = {}
    for ch, name in enumerate(['bg', 'ncr', 'ed', 'et']):
        if name == 'bg':
            continue
        gt_c, pr_c = y_true[..., ch], hard[..., ch]
        inter = (gt_c * pr_c).sum()
        denom = gt_c.sum() + pr_c.sum()
        out[f'dice_{name}'] = float((2.0 * inter + smooth) / (denom + smooth))
        out[f'hd95_{name}'] = hd95(pr_c.astype(bool), gt_c.astype(bool)) if compute_hd95 else float('nan')
    wt_gt = y_true[..., 1:].max(-1).astype(bool)
    wt_pr = hard[...,  1:].max(-1).astype(bool)
    inter = (wt_gt & wt_pr).sum()
    denom = wt_gt.sum() + wt_pr.sum()
    out['dice_wt']   = float((2.0 * inter + smooth) / (denom + smooth))
    out['hd95_wt']   = hd95(wt_pr, wt_gt) if compute_hd95 else float('nan')
    out['dice_mean'] = float(np.mean([out['dice_ncr'], out['dice_ed'], out['dice_et']]))
    hd_vals = [out['hd95_ncr'], out['hd95_ed'], out['hd95_et']]
    out['hd95_mean'] = float(np.nanmean(hd_vals)) if any(not np.isnan(v) for v in hd_vals) else float('nan')
    return out


def run_validation(val_gen, infer_fn, compute_hd95=True):
    all_m = []
    for i in range(len(val_gen)):
        imgs, masks = val_gen[i]
        logits = infer_fn(tf.constant(imgs))
        all_m.append(compute_metrics(masks[0], logits[0], compute_hd95=compute_hd95))
    return {k: float(np.nanmean([m[k] for m in all_m])) for k in all_m[0]}


# ── Single-fold training ───────────────────────────────────────────────────────

def train_fold(fold_idx, trn_ids, val_ids, args):
    log.info(f'── Fold {fold_idx + 1}/{args.n_folds} '
             f'(train={len(trn_ids)}, val={len(val_ids)}) ──')

    trn_gen = MRIDataGenerator(args.data_dir, trn_ids,
                                num_slices=args.num_slices,
                                shuffle=True, augment=True, seed=42 + fold_idx)
    val_gen = MRIDataGenerator(args.data_dir, val_ids,
                                num_slices=args.num_slices,
                                shuffle=False, augment=False)

    model = ResNetASPP3D(n_classes=args.num_classes,
                         base_channels=args.base_channels,
                         aspp_out_channels=args.aspp_channels,
                         name='resnet_aspp3d')
    dummy = tf.zeros([1, args.num_slices, args.height, args.width, 4])
    model(dummy, training=False)
    if fold_idx == 0:
        log.info(f'Parameters: {model.count_params():,}')

    opt      = keras.optimizers.Adam(learning_rate=args.lr)
    step_fn  = make_train_step(model, opt, args.alpha_mdsc)
    infer_fn = tf.function(lambda x: model(x, training=False))

    best_dice    = 0.0
    patience_ctr = 0
    best_path    = os.path.join(args.model_dir, f'fold_{fold_idx}_best.weights.h5')
    n_steps      = len(trn_gen)

    for epoch in range(1, args.epochs + 1):
        cosine_lr = args.lr_min + 0.5 * (args.lr - args.lr_min) * (
            1.0 + math.cos(math.pi * (epoch - 1) / args.epochs))
        opt.learning_rate.assign(cosine_lr)
        trn_gen.on_epoch_end()
        ep_loss = []

        for step in range(n_steps):
            imgs, masks = trn_gen[step]
            imgs_t  = tf.constant(imgs,  dtype=tf.float32)
            masks_t = tf.constant(masks, dtype=tf.float32)
            loss, l_ce, l_mdsc = step_fn(imgs_t, masks_t)
            ep_loss.append((float(loss), float(l_ce), float(l_mdsc)))

            if (step + 1) % 50 == 0 or step == 0:
                log.info(f'  [{epoch}/{args.epochs}] step {step+1}/{n_steps} | '
                         f'loss={float(loss):.4f} (ce={float(l_ce):.4f} mdsc={float(l_mdsc):.4f})')

        avg_loss, avg_ce, avg_mdsc = [np.mean([b[i] for b in ep_loss]) for i in range(3)]
        m = run_validation(val_gen, infer_fn, compute_hd95=False)
        log.info(
            f'  Epoch {epoch:3d}/{args.epochs} | lr={cosine_lr:.2e} | '
            f'loss={avg_loss:.4f} (ce={avg_ce:.4f} mdsc={avg_mdsc:.4f}) | '
            f'Dice NCR={m["dice_ncr"]:.3f} ED={m["dice_ed"]:.3f} '
            f'ET={m["dice_et"]:.3f} WT={m["dice_wt"]:.3f} Mean={m["dice_mean"]:.4f}'
        )

        if m['dice_mean'] > best_dice:
            best_dice    = m['dice_mean']
            patience_ctr = 0
            model.save_weights(best_path)
            log.info(f'  ✓ Fold {fold_idx + 1} new best: {best_dice:.4f}')
        else:
            patience_ctr += 1
            if patience_ctr >= args.patience:
                log.info(f'  Early stopping fold {fold_idx + 1} at epoch {epoch}')
                break

    model.load_weights(best_path)
    final_m = run_validation(val_gen, infer_fn, compute_hd95=True)
    log.info(
        f'  Fold {fold_idx + 1} FINAL | '
        f'Dice Mean={final_m["dice_mean"]:.4f} | '
        f'HD95 Mean={final_m["hd95_mean"]:.1f}mm'
    )
    return final_m


# ── Main ──────────────────────────────────────────────────────────────────────

def train(args):
    log.info('=' * 60)
    log.info('EXPERIMENT A — ResNetASPP3D Baseline  '
             '(base_channels=%d, %d-fold CV, augmentation)', args.base_channels, args.n_folds)
    log.info('=' * 60)

    for gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

    all_ids = list(range(1, args.num_volumes + 1))
    folds   = make_folds(all_ids, n_folds=args.n_folds)

    fold_results = []
    for k, (trn_ids, val_ids) in enumerate(folds):
        fold_results.append(train_fold(k, trn_ids, val_ids, args))

    keys = list(fold_results[0].keys())
    cv   = {}
    for key in keys:
        vals           = [r[key] for r in fold_results if not np.isnan(r[key])]
        cv[key]        = float(np.mean(vals))
        cv[f'{key}_std'] = float(np.std(vals))

    log.info('=' * 60)
    log.info('CROSS-VALIDATION COMPLETE')
    log.info(f'Mean Dice : {cv["dice_mean"]:.4f} ± {cv["dice_mean_std"]:.4f}')
    log.info(f'Mean HD95 : {cv["hd95_mean"]:.2f} ± {cv["hd95_mean_std"]:.2f} mm')
    log.info(f'  NCR  Dice={cv["dice_ncr"]:.3f}±{cv["dice_ncr_std"]:.3f}  '
             f'HD95={cv["hd95_ncr"]:.1f}±{cv["hd95_ncr_std"]:.1f}mm')
    log.info(f'  ED   Dice={cv["dice_ed"]:.3f}±{cv["dice_ed_std"]:.3f}  '
             f'HD95={cv["hd95_ed"]:.1f}±{cv["hd95_ed_std"]:.1f}mm')
    log.info(f'  ET   Dice={cv["dice_et"]:.3f}±{cv["dice_et_std"]:.3f}  '
             f'HD95={cv["hd95_et"]:.1f}±{cv["hd95_et_std"]:.1f}mm')
    log.info(f'  WT   Dice={cv["dice_wt"]:.3f}±{cv["dice_wt_std"]:.3f}  '
             f'HD95={cv["hd95_wt"]:.1f}±{cv["hd95_wt_std"]:.1f}mm')
    log.info('=' * 60)

    with open(os.path.join(args.model_dir, 'cv_results.json'), 'w') as f:
        json.dump({'per_fold': fold_results, 'cv': cv,
                   'base_channels': args.base_channels,
                   'n_folds': args.n_folds}, f, indent=2)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir',      type=str,
                   default=os.environ.get('SM_CHANNEL_TRAINING',
                                          '/opt/ml/input/data/training'))
    p.add_argument('--model-dir',     type=str,
                   default=os.environ.get('SM_MODEL_DIR', '/opt/ml/model'))
    p.add_argument('--num-volumes',   type=int,   default=369)
    p.add_argument('--n-folds',       type=int,   default=5)
    p.add_argument('--num-slices',    type=int,   default=128)
    p.add_argument('--height',        type=int,   default=192)
    p.add_argument('--width',         type=int,   default=160)
    p.add_argument('--num-classes',   type=int,   default=4)
    p.add_argument('--base-channels', type=int,   default=64)
    p.add_argument('--aspp-channels', type=int,   default=256)
    p.add_argument('--epochs',        type=int,   default=100)
    p.add_argument('--patience',      type=int,   default=25)
    p.add_argument('--lr',            type=float, default=1e-4)
    p.add_argument('--lr-min',        type=float, default=1e-6)
    p.add_argument('--alpha-mdsc',    type=float, default=100.0)
    train(p.parse_args())
