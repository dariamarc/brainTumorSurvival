#!/usr/bin/env python3
"""
Architecture 2 — Experiment B1: Full PrototypeSegNet3D, segmentation loss only

  Input → ResNet3D → ASPP3D → PrototypeLayer(3) → upsample → InterpretableClassifier → logits

Training: dice loss only (no prototype losses).
Init    : backbone + ASPP loaded from Exp A fold-matched checkpoints.
Monitors: Dice, HD95, prototype activation ratios per class.
5-fold CV + data augmentation.

Pass criterion: NCR ratio > 1.0 (prototypes learned class specificity without explicit losses).
"""

import os, sys, json, argparse, logging
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
        vmin, vmax = img.min(), img.max()
        if vmax - vmin > 1e-8:
            img = (img - vmin) / (vmax - vmin)
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
            img[..., m] = np.clip(img[..., m] * scale + shift, 0.0, 1.0)
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


# ── Model ─────────────────────────────────────────────────────────────────────

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
        dn = (stride != 1) or (filters != self.base_channels)
        blocks = [ResidualBlock3D(filters, strides=stride,
                                  downsample=dn, name=f'{name}_block1')]
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


class PrototypeLayer(layers.Layer):
    """Computes L2-based similarity maps for n_prototypes prototype vectors."""

    def __init__(self, n_prototypes=3, prototype_dim=256, epsilon=1e-4, **kw):
        super().__init__(**kw)
        self.n_prototypes  = n_prototypes
        self.prototype_dim = prototype_dim
        self.epsilon       = epsilon

    def build(self, input_shape):
        self.prototype_vectors = self.add_weight(
            name='prototype_vectors',
            shape=(self.n_prototypes, self.prototype_dim, 1, 1, 1),
            initializer='glorot_uniform', trainable=True)
        super().build(input_shape)

    def call(self, features, training=False):
        proto_f = tf.transpose(self.prototype_vectors, perm=[2, 3, 4, 1, 0])
        dot     = tf.nn.conv3d(features, filters=proto_f,
                               strides=[1, 1, 1, 1, 1], padding='SAME')
        f_sq    = tf.reduce_sum(tf.square(features), axis=-1, keepdims=True)
        p_sq    = tf.reshape(
            tf.reduce_sum(tf.square(self.prototype_vectors), axis=[1, 2, 3, 4]),
            [1, 1, 1, 1, self.n_prototypes])
        dist    = tf.sqrt(tf.maximum(f_sq - 2.0 * dot + p_sq, self.epsilon))
        return tf.math.log((dist + 1.0) / (dist + self.epsilon))   # similarity


class InterpretableClassifier(layers.Layer):
    """Learnable linear map: (B,D,H,W,P) similarities → (B,D,H,W,C) logits."""

    def __init__(self, n_prototypes=3, n_classes=4, **kw):
        super().__init__(**kw)
        self.n_prototypes = n_prototypes
        self.n_classes    = n_classes

    def build(self, input_shape):
        # Identity-like init: tumour class k+1 → prototype k, background negative
        init = np.zeros((self.n_classes, self.n_prototypes), dtype=np.float32)
        for k in range(self.n_prototypes):
            init[k + 1, k] = 1.0          # class 1,2,3 positively linked to proto 0,1,2
        for k in range(self.n_prototypes):
            init[0, k] = -0.3              # background negatively linked to all protos
        self.weights_matrix = self.add_weight(
            name='weights_matrix',
            shape=(self.n_classes, self.n_prototypes),
            initializer=keras.initializers.Constant(init), trainable=True)
        self.bias = self.add_weight(
            name='bias', shape=(self.n_classes,),
            initializer='zeros', trainable=True)
        super().build(input_shape)

    def call(self, similarities, training=False):
        sh   = tf.shape(similarities)
        flat = tf.reshape(similarities, [-1, self.n_prototypes])
        out  = tf.matmul(flat, self.weights_matrix, transpose_b=True) + self.bias
        return tf.reshape(out, [sh[0], sh[1], sh[2], sh[3], self.n_classes])

    def get_weights_matrix(self):
        return self.weights_matrix.numpy()


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


class PrototypeSegNet3D(keras.Model):
    def __init__(self, n_classes=4, n_prototypes=3,
                 base_channels=64, aspp_out_channels=256,
                 dilation_rates=(2, 4, 8), **kw):
        super().__init__(**kw)
        self.n_prototypes = n_prototypes
        self.backbone  = ResNet3D(in_channels=4, base_channels=base_channels,
                                  name='resnet3d_backbone')
        self.aspp      = ASPP3D(in_channels=base_channels * 8,
                                out_channels=aspp_out_channels,
                                dilation_rates=dilation_rates, name='aspp3d')
        self.proto_layer  = PrototypeLayer(n_prototypes=n_prototypes,
                                           prototype_dim=aspp_out_channels,
                                           name='prototype_layer')
        self.classifier   = InterpretableClassifier(n_prototypes=n_prototypes,
                                                    n_classes=n_classes,
                                                    name='interpretable_classifier')

    def call(self, inputs, training=False):
        orig = tf.shape(inputs)[1:4]
        x    = self.backbone(inputs, training=training)
        x    = self.aspp(x,          training=training)
        sims = self.proto_layer(x,   training=training)     # (B,D/8,H/8,W/8,P)
        sims = _trilinear_upsample(sims, orig)               # (B,D,H,W,P)
        return self.classifier(sims)                         # (B,D,H,W,C)

    def forward_with_similarities(self, inputs):
        """Returns (logits, similarities_full_res) for monitoring."""
        orig = tf.shape(inputs)[1:4]
        x    = self.backbone(inputs, training=False)
        x    = self.aspp(x,          training=False)
        sims = self.proto_layer(x,   training=False)
        sims_up = _trilinear_upsample(sims, orig)
        return self.classifier(sims_up), sims_up


# ── Loss ──────────────────────────────────────────────────────────────────────

def dice_loss(y_true, y_pred, smooth=1e-6):
    probs = tf.nn.softmax(y_pred, axis=-1)
    gt, pr = y_true[..., 1:], probs[..., 1:]
    axes  = [1, 2, 3]
    inter = tf.reduce_sum(gt * pr, axis=axes)
    denom = tf.reduce_sum(gt + pr, axis=axes)
    return 1.0 - tf.reduce_mean((2.0 * inter + smooth) / (denom + smooth))


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


def compute_metrics(y_true, logits, smooth=1e-6):
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
        out[f'hd95_{name}'] = hd95(pr_c.astype(bool), gt_c.astype(bool))
    wt_gt = y_true[..., 1:].max(-1).astype(bool)
    wt_pr = hard[...,  1:].max(-1).astype(bool)
    inter = (wt_gt & wt_pr).sum()
    denom = wt_gt.sum() + wt_pr.sum()
    out['dice_wt']   = float((2.0 * inter + smooth) / (denom + smooth))
    out['hd95_wt']   = hd95(wt_pr, wt_gt)
    out['dice_mean'] = float(np.mean([out['dice_ncr'], out['dice_ed'], out['dice_et']]))
    out['hd95_mean'] = float(np.nanmean([out['hd95_ncr'], out['hd95_ed'], out['hd95_et']]))
    return out


def compute_proto_ratios(sims_np, mask_np, epsilon=1e-6):
    """
    Activation ratio at full resolution (similarities already upsampled).
    sims_np : (D, H, W, 3)   mask_np : (D, H, W, 4)
    Returns dict with NCR/ED/ET ratios (in-class mean / out-class mean).
    """
    class_map = np.argmax(mask_np, axis=-1)   # (D, H, W)  values 0-3
    names     = {1: 'ncr', 2: 'ed', 3: 'et'}
    ratios    = {}
    for p in range(3):
        cls  = p + 1
        sim  = sims_np[..., p]
        in_  = class_map == cls
        out_ = ~in_
        mean_in  = sim[in_].mean()  if in_.any()  else 0.0
        mean_out = sim[out_].mean() if out_.any() else epsilon
        ratios[names[cls]] = float(mean_in / (mean_out + epsilon))
    return ratios


def run_validation(model, val_gen):
    all_m, all_r = [], []
    for i in range(len(val_gen)):
        imgs, masks = val_gen[i]
        logits, sims = model.forward_with_similarities(tf.constant(imgs))
        all_m.append(compute_metrics(masks[0], logits[0]))
        all_r.append(compute_proto_ratios(sims.numpy()[0], masks[0]))
    avg_m = {k: float(np.nanmean([m[k] for m in all_m])) for k in all_m[0]}
    avg_r = {k: float(np.mean([r[k]  for r in all_r])) for k in all_r[0]}
    return avg_m, avg_r


# ── Weight loading from Exp A ─────────────────────────────────────────────────

def load_exp_a_weights(model, exp_a_dir, fold_idx):
    """Load backbone + ASPP from Exp A; PrototypeLayer and classifier reinitialise."""
    path = os.path.join(exp_a_dir, f'fold_{fold_idx}_best.weights.h5')
    if not os.path.exists(path):
        log.warning(f'Exp A weights not found: {path}  — training from scratch')
        return
    try:
        model.load_weights(path, by_name=True, skip_mismatch=True)
        log.info(f'Loaded Exp A backbone+ASPP (fold {fold_idx}): {path}')
    except Exception as e:
        log.warning(f'Could not load Exp A weights: {e}')


def fetch_weights(s3_uri, local_dir):
    import boto3, tarfile
    s3_path     = s3_uri.replace('s3://', '')
    bucket, key = s3_path.split('/', 1)
    os.makedirs(local_dir, exist_ok=True)
    local_tar   = os.path.join(local_dir, 'model.tar.gz')
    log.info(f'Downloading: s3://{bucket}/{key}')
    boto3.client('s3').download_file(bucket, key, local_tar)
    with tarfile.open(local_tar) as t:
        t.extractall(local_dir)
    return local_dir


# ── Single-fold training ───────────────────────────────────────────────────────

def train_fold(fold_idx, trn_ids, val_ids, args, exp_a_dir):
    log.info(f'── Fold {fold_idx + 1}/{args.n_folds} '
             f'(train={len(trn_ids)}, val={len(val_ids)}) ──')

    trn_gen = MRIDataGenerator(args.data_dir, trn_ids,
                                num_slices=args.num_slices,
                                shuffle=True, augment=True, seed=42 + fold_idx)
    val_gen = MRIDataGenerator(args.data_dir, val_ids,
                                num_slices=args.num_slices,
                                shuffle=False, augment=False)

    model = PrototypeSegNet3D(n_classes=args.num_classes,
                              n_prototypes=args.n_prototypes,
                              base_channels=args.base_channels,
                              aspp_out_channels=args.aspp_channels)
    dummy = tf.zeros([1, args.num_slices, args.height, args.width, 4])
    model(dummy, training=False)
    if fold_idx == 0:
        log.info(f'Parameters: {model.count_params():,}')

    if exp_a_dir:
        load_exp_a_weights(model, exp_a_dir, fold_idx)

    opt          = keras.optimizers.Adam(learning_rate=args.lr)
    best_dice    = 0.0
    patience_ctr = 0
    best_path    = os.path.join(args.model_dir, f'fold_{fold_idx}_best.weights.h5')

    for epoch in range(1, args.epochs + 1):
        trn_gen.on_epoch_end()
        ep_loss = []

        for step in range(len(trn_gen)):
            imgs, masks = trn_gen[step]
            imgs_t  = tf.constant(imgs,  dtype=tf.float32)
            masks_t = tf.constant(masks, dtype=tf.float32)
            with tf.GradientTape() as tape:
                logits = model(imgs_t, training=True)
                loss   = dice_loss(masks_t, logits)
            grads = tape.gradient(loss, model.trainable_variables)
            grads, _ = tf.clip_by_global_norm(grads, 1.0)
            opt.apply_gradients(zip(grads, model.trainable_variables))
            ep_loss.append(float(loss))

        m, ratios = run_validation(model, val_gen)
        log.info(
            f'  Epoch {epoch:3d}/{args.epochs} | loss={np.mean(ep_loss):.4f} | '
            f'Dice NCR={m["dice_ncr"]:.3f} ED={m["dice_ed"]:.3f} '
            f'ET={m["dice_et"]:.3f} WT={m["dice_wt"]:.3f} Mean={m["dice_mean"]:.4f} | '
            f'HD95 Mean={m["hd95_mean"]:.1f}mm | '
            f'Ratios NCR={ratios["ncr"]:.2f} ED={ratios["ed"]:.2f} ET={ratios["et"]:.2f}'
        )
        # Also log the classifier weight matrix every 5 epochs
        if epoch % 5 == 0:
            W = model.classifier.get_weights_matrix()
            log.info(f'  Classifier weights (class × proto):\n{np.round(W, 3)}')

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
    final_m, final_r = run_validation(model, val_gen)
    log.info(
        f'  Fold {fold_idx + 1} FINAL | '
        f'Dice Mean={final_m["dice_mean"]:.4f} | '
        f'HD95 Mean={final_m["hd95_mean"]:.1f}mm | '
        f'Ratios NCR={final_r["ncr"]:.2f} ED={final_r["ed"]:.2f} '
        f'ET={final_r["et"]:.2f}  '
        f'(NCR>1.0: {"YES" if final_r["ncr"] > 1.0 else "NO"})'
    )
    return {**final_m, 'ratio_ncr': final_r['ncr'],
            'ratio_ed': final_r['ed'], 'ratio_et': final_r['et']}


# ── Main ──────────────────────────────────────────────────────────────────────

def train(args):
    log.info('=' * 60)
    log.info('EXPERIMENT B1 — PrototypeSegNet3D, no prototype losses  '
             '(%d-fold CV, augmentation)', args.n_folds)
    log.info('=' * 60)

    for gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

    exp_a_dir = None
    if args.exp_a_weights_s3:
        exp_a_dir = fetch_weights(args.exp_a_weights_s3, '/tmp/exp_a_weights')
    elif args.exp_a_weights_dir and os.path.isdir(args.exp_a_weights_dir):
        exp_a_dir = args.exp_a_weights_dir

    all_ids = list(range(1, args.num_volumes + 1))
    folds   = make_folds(all_ids, n_folds=args.n_folds)

    fold_results = []
    for k, (trn_ids, val_ids) in enumerate(folds):
        fold_results.append(train_fold(k, trn_ids, val_ids, args, exp_a_dir))

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
    log.info(f'Proto ratios: NCR={cv["ratio_ncr"]:.2f} '
             f'ED={cv["ratio_ed"]:.2f} ET={cv["ratio_et"]:.2f}')
    log.info(f'  NCR  Dice={cv["dice_ncr"]:.3f}±{cv["dice_ncr_std"]:.3f}')
    log.info(f'  ED   Dice={cv["dice_ed"]:.3f}±{cv["dice_ed_std"]:.3f}')
    log.info(f'  ET   Dice={cv["dice_et"]:.3f}±{cv["dice_et_std"]:.3f}')
    log.info(f'  WT   Dice={cv["dice_wt"]:.3f}±{cv["dice_wt_std"]:.3f}')
    log.info('=' * 60)

    with open(os.path.join(args.model_dir, 'cv_results.json'), 'w') as f:
        json.dump({'per_fold': fold_results, 'cv': cv,
                   'n_prototypes': args.n_prototypes,
                   'n_folds': args.n_folds}, f, indent=2)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir',          type=str,
                   default=os.environ.get('SM_CHANNEL_TRAINING',
                                          '/opt/ml/input/data/training'))
    p.add_argument('--model-dir',         type=str,
                   default=os.environ.get('SM_MODEL_DIR', '/opt/ml/model'))
    p.add_argument('--exp-a-weights-s3',  type=str, default=None)
    p.add_argument('--exp-a-weights-dir', type=str, default=None)
    p.add_argument('--num-volumes',       type=int,   default=369)
    p.add_argument('--n-folds',           type=int,   default=5)
    p.add_argument('--num-slices',        type=int,   default=128)
    p.add_argument('--height',            type=int,   default=192)
    p.add_argument('--width',             type=int,   default=160)
    p.add_argument('--num-classes',       type=int,   default=4)
    p.add_argument('--n-prototypes',      type=int,   default=3)
    p.add_argument('--base-channels',     type=int,   default=64)
    p.add_argument('--aspp-channels',     type=int,   default=256)
    p.add_argument('--epochs',            type=int,   default=50)
    p.add_argument('--patience',          type=int,   default=10)
    p.add_argument('--lr',                type=float, default=1e-4)
    train(p.parse_args())
