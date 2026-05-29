#!/usr/bin/env python3
"""
Local smoke test for the Step 1 U-Net baseline.

Runs entirely on synthetic data — no H5 files needed.
Uses small spatial dimensions so it is fast on CPU.

Usage:
    python test_local.py          # quick (default)
    python test_local.py --full   # larger dims, more steps
"""

import sys
import platform
import argparse
import traceback
import numpy as np
import tensorflow as tf

PASS = "[PASS]"
FAIL = "[FAIL]"


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


def check(label, condition, detail=""):
    status = PASS if condition else FAIL
    suffix = f"  ({detail})" if detail else ""
    print(f"  {status}  {label}{suffix}")
    return condition


parser = argparse.ArgumentParser()
parser.add_argument('--full', action='store_true', help='Larger dims / more steps')
args = parser.parse_args()

D = 16 if args.full else 8
H = 32 if args.full else 16
W = 32 if args.full else 16
C = 4
N = 4
STEPS = 4 if args.full else 2

all_passed = True

# ── 0. Environment ─────────────────────────────────────────────────────────────

section("0. Environment")
print(f"  Platform : {platform.system()} {platform.machine()}")
print(f"  Python   : {sys.version.split()[0]}")
print(f"  TF       : {tf.__version__}")
gpus = tf.config.list_physical_devices('GPU')
print(f"  GPUs     : {gpus if gpus else 'none (CPU-only)'}")

# ── 1. Model construction ──────────────────────────────────────────────────────

section("1. Model construction")

try:
    from unet3d import UNet3D
    model = UNet3D(n_classes=N, base_channels=8)
    all_passed &= check("UNet3D imported and instantiated", True)
except Exception as e:
    all_passed &= check("UNet3D imported and instantiated", False, str(e))
    traceback.print_exc()
    sys.exit(1)

# ── 2. Forward pass ────────────────────────────────────────────────────────────

section("2. Forward pass  [shapes, dtype, NaN]")

try:
    logits = model(tf.zeros((1, D, H, W, C)), training=False)

    expected = (1, D, H, W, N)
    all_passed &= check("Output shape correct",
                        tuple(logits.shape) == expected,
                        f"got {tuple(logits.shape)}, want {expected}")
    all_passed &= check("Output dtype is float32",
                        logits.dtype == tf.float32, str(logits.dtype))
    all_passed &= check("No NaN in logits",
                        not tf.reduce_any(tf.math.is_nan(logits)).numpy())
    all_passed &= check("No Inf in logits",
                        not tf.reduce_any(tf.math.is_inf(logits)).numpy())

    lo = float(tf.reduce_min(logits))
    hi = float(tf.reduce_max(logits))
    print(f"  Logits range : [{lo:.4f}, {hi:.4f}]")
    print(f"  Params       : {model.count_params():,}")
except Exception as e:
    all_passed &= check("Forward pass", False, str(e))
    traceback.print_exc()

# ── 3. Loss functions ──────────────────────────────────────────────────────────

section("3. Loss functions")

try:
    from train import dice_loss, weighted_ce_loss, combined_loss

    rng = np.random.default_rng(0)
    label_idx = rng.integers(0, N, size=(1, D, H, W)).astype(np.int32)
    y_true    = tf.cast(tf.one_hot(label_idx, N), tf.float32)
    y_pred    = tf.random.normal((1, D, H, W, N))
    cw        = [0.1, 2.0, 1.0, 2.5]

    dl = float(dice_loss(y_true, y_pred))
    all_passed &= check("Dice loss: no NaN",     not np.isnan(dl), f"{dl:.4f}")
    all_passed &= check("Dice loss in [0, 1]",   0.0 <= dl <= 1.0 + 1e-5, f"{dl:.4f}")

    ce = float(weighted_ce_loss(y_true, y_pred, cw))
    all_passed &= check("CE loss: no NaN",       not np.isnan(ce), f"{ce:.4f}")
    all_passed &= check("CE loss > 0",           ce > 0, f"{ce:.4f}")

    combo = float(combined_loss(y_true, y_pred, cw, alpha=0.5))
    all_passed &= check("Combined loss: no NaN", not np.isnan(combo), f"{combo:.4f}")
    all_passed &= check("Combined ≈ 0.5·dice + 0.5·ce",
                        abs(combo - (0.5 * dl + 0.5 * ce)) < 1e-4,
                        f"expected≈{0.5*dl+0.5*ce:.4f} got {combo:.4f}")
except Exception as e:
    all_passed &= check("Loss functions", False, str(e))
    traceback.print_exc()

# ── 4. Training steps ─────────────────────────────────────────────────────────

section(f"4. Gradient tape — {STEPS} training steps")

try:
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    cw = [0.1, 2.0, 1.0, 2.5]

    for step in range(STEPS):
        imgs      = tf.random.normal((1, D, H, W, C))
        label_idx = tf.random.uniform((1, D, H, W), 0, N, dtype=tf.int32)
        masks     = tf.cast(tf.one_hot(label_idx, N), tf.float32)

        with tf.GradientTape() as tape:
            logits = model(imgs, training=True)
            loss   = combined_loss(masks, logits, cw, alpha=0.5)

        grads = tape.gradient(loss, model.trainable_variables)
        grads, gnorm = tf.clip_by_global_norm(grads, 1.0)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        nan_grads  = sum(1 for g in grads if g is not None and
                         tf.reduce_any(tf.math.is_nan(g)).numpy())
        none_grads = sum(1 for g in grads if g is None)
        all_passed &= check(
            f"Step {step+1}: gradients healthy",
            nan_grads == 0 and none_grads == 0,
            f"loss={float(loss):.4f}  gnorm={float(gnorm):.4f}  "
            f"nan={nan_grads}  none={none_grads}"
        )
except Exception as e:
    all_passed &= check("Training steps", False, str(e))
    traceback.print_exc()

# ── 5. Metrics ────────────────────────────────────────────────────────────────

section("5. Dice metric computation")

try:
    from train import compute_dice_scores

    logits_out = model(tf.random.normal((1, D, H, W, C)), training=False)
    label_idx  = np.random.randint(0, N, (1, D, H, W))
    masks_np   = np.eye(N)[label_idx]

    scores = compute_dice_scores(masks_np[0], logits_out[0])
    expected_keys = {'dice_ncr', 'dice_ed', 'dice_et', 'dice_wt', 'dice_mean'}

    all_passed &= check("All metric keys present",
                        expected_keys.issubset(scores.keys()),
                        str(set(scores.keys())))
    all_passed &= check("All scores in [0, 1]",
                        all(0.0 <= v <= 1.0 + 1e-5 for v in scores.values()),
                        str({k: f"{v:.3f}" for k, v in scores.items()}))
    print(f"  Scores: { {k: f'{v:.3f}' for k, v in scores.items()} }")
except Exception as e:
    all_passed &= check("Dice metrics", False, str(e))
    traceback.print_exc()

# ── Summary ───────────────────────────────────────────────────────────────────

section("Summary")
if all_passed:
    print("  All checks passed — safe to deploy to SageMaker.")
else:
    print("  One or more checks FAILED — fix before deploying.")
    sys.exit(1)
