#!/usr/bin/env python3
"""
SageMaker Training Script for PrototypeSegNet3D

This script runs as a SageMaker Training Job, independent of any notebook session.
It handles:
- Loading data from S3 (automatically mounted by SageMaker)
- Three-phase training
- Saving checkpoints and metrics to S3 (via /opt/ml/model)
- Logging progress to CloudWatch
"""

import os
import sys
import json
import argparse
import logging
from datetime import datetime

# Setup logging for CloudWatch
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def setup_environment():
    """Configure TensorFlow and GPU settings."""
    import tensorflow as tf

    # Enable memory growth to prevent TF from grabbing all GPU memory
    gpus = tf.config.list_physical_devices('GPU')
    logger.info(f"GPUs available: {len(gpus)}")
    for gpu in gpus:
        logger.info(f"  {gpu}")
        tf.config.experimental.set_memory_growth(gpu, True)

    return tf


def parse_args():
    """Parse command line arguments (passed by SageMaker)."""
    parser = argparse.ArgumentParser()

    # SageMaker specific arguments
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR', '/opt/ml/model'))
    parser.add_argument('--data-dir', type=str, default=os.environ.get('SM_CHANNEL_TRAINING', '/opt/ml/input/data/training'))
    parser.add_argument('--output-dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', '/opt/ml/output/data'))

    # Training hyperparameters
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--phase1-epochs', type=int, default=50)
    parser.add_argument('--phase2-epochs', type=int, default=150)
    parser.add_argument('--phase3-epochs', type=int, default=30)
    parser.add_argument('--split-ratio', type=float, default=0.2)
    parser.add_argument('--random-state', type=int, default=42)
    parser.add_argument('--alpha-mdsc', type=float, default=100.0)

    # Model architecture
    parser.add_argument('--backbone-channels', type=int, default=64)
    parser.add_argument('--aspp-out-channels', type=int, default=256)
    parser.add_argument('--n-prototypes', type=int, default=3)
    parser.add_argument('--num-classes', type=int, default=4)

    # Volume dimensions
    parser.add_argument('--depth', type=int, default=128)
    parser.add_argument('--height', type=int, default=160)
    parser.add_argument('--width', type=int, default=192)
    parser.add_argument('--channels', type=int, default=4)

    # Data parameters
    parser.add_argument('--num-volumes', type=int, default=369)
    parser.add_argument('--num-slices', type=int, default=128)

    # Resume training
    parser.add_argument('--resume-from', type=str, default=None,
                        help='S3 path to checkpoint to resume from')

    return parser.parse_args()


def find_data_path(base_path):
    """Find the actual data path (handle nested directories from S3)."""
    if not os.path.exists(base_path):
        raise ValueError(f"Data path does not exist: {base_path}")

    # Check if h5 files are directly in base_path
    h5_files = [f for f in os.listdir(base_path) if f.endswith('.h5')]
    if h5_files:
        logger.info(f"Found {len(h5_files)} .h5 files in {base_path}")
        return base_path

    # Check subdirectories
    for subdir in os.listdir(base_path):
        subpath = os.path.join(base_path, subdir)
        if os.path.isdir(subpath):
            h5_files = [f for f in os.listdir(subpath) if f.endswith('.h5')]
            if h5_files:
                logger.info(f"Found {len(h5_files)} .h5 files in {subpath}")
                return subpath

    raise ValueError(f"No .h5 files found in {base_path} or its subdirectories")


def save_training_summary(args, history, metrics, output_dir):
    """Save training summary to JSON."""
    summary = {
        'timestamp': datetime.now().isoformat(),
        'hyperparameters': vars(args),
        'final_metrics': metrics,
        'training_history': history
    }

    summary_path = os.path.join(output_dir, 'training_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"Training summary saved to {summary_path}")


def main():
    logger.info("=" * 60)
    logger.info("Starting PrototypeSegNet3D Training")
    logger.info("=" * 60)

    # Parse arguments
    args = parse_args()
    logger.info(f"Arguments: {vars(args)}")

    # Setup TensorFlow
    tf = setup_environment()

    # Add project paths (when running in SageMaker, code is in /opt/ml/code)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, script_dir)
    sys.path.insert(0, os.path.join(script_dir, 'ResNet_architecture'))

    # Import project modules
    logger.info("Importing project modules...")
    from ResNet_architecture.prototype_segnet3d import create_prototype_segnet3d
    from ResNet_architecture.trainer import PrototypeTrainer
    from data_processing.data_generator import MRIDataGenerator

    # Find data path
    data_path = find_data_path(args.data_dir)
    logger.info(f"Using data path: {data_path}")

    # Create output directories
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_dir = os.path.join(args.model_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Input shape
    input_shape = (args.depth, args.height, args.width, args.channels)
    logger.info(f"Input shape: {input_shape}")

    # Create data generators
    logger.info("Creating data generators...")
    train_generator = MRIDataGenerator(
        data_path,
        batch_size=args.batch_size,
        num_slices=args.num_slices,
        num_volumes=args.num_volumes,
        split_ratio=args.split_ratio,
        subset='train',
        shuffle=True,
        random_state=args.random_state
    )

    val_generator = MRIDataGenerator(
        data_path,
        batch_size=args.batch_size,
        num_slices=args.num_slices,
        num_volumes=args.num_volumes,
        split_ratio=args.split_ratio,
        subset='val',
        shuffle=False,
        random_state=args.random_state
    )

    logger.info(f"Training batches: {len(train_generator)}")
    logger.info(f"Validation batches: {len(val_generator)}")

    # Build model
    logger.info("Building model...")
    model = create_prototype_segnet3d(
        input_shape=input_shape,
        num_classes=args.num_classes,
        n_prototypes=args.n_prototypes,
        backbone_channels=args.backbone_channels,
        aspp_out_channels=args.aspp_out_channels,
        dilation_rates=(2, 4, 8),
        distance_type='l2',
        activation_function='log'
    )

    # Initialize weights
    dummy_input = tf.zeros((1,) + input_shape)
    _ = model(dummy_input, training=False)
    logger.info("Model built successfully.")
    model.summary(print_fn=logger.info)

    # Load checkpoint if resuming
    if args.resume_from:
        logger.info(f"Resuming from checkpoint: {args.resume_from}")
        model.load_weights(args.resume_from)

    # Create trainer
    logger.info("Creating trainer...")
    trainer = PrototypeTrainer(
        model=model,
        train_generator=train_generator,
        val_generator=val_generator,
        checkpoint_dir=checkpoint_dir,
        log_dir=args.output_dir
    )

    # Run training
    logger.info("=" * 60)
    logger.info("PHASE 1: Warm-up Training")
    logger.info("=" * 60)
    trainer.train_phase1(epochs=args.phase1_epochs)

    # Save Phase 1 model
    phase1_path = os.path.join(args.model_dir, 'model_after_phase1.keras')
    model.save(phase1_path)
    logger.info(f"Phase 1 model saved: {phase1_path}")

    logger.info("=" * 60)
    logger.info("PHASE 2: Joint Fine-tuning")
    logger.info("=" * 60)
    trainer.train_phase2(epochs=args.phase2_epochs)

    # Save Phase 2 model
    phase2_path = os.path.join(args.model_dir, 'model_after_phase2.keras')
    model.save(phase2_path)
    logger.info(f"Phase 2 model saved: {phase2_path}")

    logger.info("=" * 60)
    logger.info("PHASE 3: Prototype Projection & Refinement")
    logger.info("=" * 60)
    trainer.train_phase3(epochs=args.phase3_epochs)

    # Save final model
    final_path = os.path.join(args.model_dir, 'model_final.keras')
    model.save(final_path)
    logger.info(f"Final model saved: {final_path}")

    # Get training history and final metrics
    history = trainer.get_full_history()
    final_metrics = {
        'dice_mean': history['dice_mean'][-1] if history['dice_mean'] else 0,
        'dice_gd_enhancing': history['dice_gd_enhancing'][-1] if history['dice_gd_enhancing'] else 0,
        'dice_edema': history['dice_edema'][-1] if history['dice_edema'] else 0,
        'dice_necrotic': history['dice_necrotic'][-1] if history['dice_necrotic'] else 0,
        'dice_whole_tumor': history['dice_whole_tumor'][-1] if history['dice_whole_tumor'] else 0,
    }

    # Save training summary
    save_training_summary(args, history, final_metrics, args.output_dir)

    # Save history as JSON
    history_path = os.path.join(args.output_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2, default=str)
    logger.info(f"Training history saved: {history_path}")

    # Print final summary
    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Final Dice Scores:")
    logger.info(f"  GD-Enhancing: {final_metrics['dice_gd_enhancing']:.4f}")
    logger.info(f"  Edema:        {final_metrics['dice_edema']:.4f}")
    logger.info(f"  Necrotic:     {final_metrics['dice_necrotic']:.4f}")
    logger.info(f"  Mean:         {final_metrics['dice_mean']:.4f}")
    logger.info(f"  Whole Tumor:  {final_metrics['dice_whole_tumor']:.4f}")
    logger.info("=" * 60)
    logger.info(f"Model saved to: {args.model_dir}")
    logger.info(f"Outputs saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
