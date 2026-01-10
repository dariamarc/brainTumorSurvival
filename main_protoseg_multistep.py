"""
Multi-Step Training Script for ProtoSeg3D

Implements the training protocol from ProtoSeg paper (Section 3.2):
1. Warmup phase: Freeze encoder, train ASPP + prototypes only
2. Joint training: Train everything except FC layer
3. Fine-tuning phase 1: Train FC layer only
4. Fine-tuning phase 2: Final FC layer tuning

Note: Prototype projection and pruning are excluded from this implementation.
"""

from data_generator import MRIDataGenerator
from tensorflow import keras
import tensorflow as tf
import numpy as np
import os
from datetime import datetime

from model_protoseg import ProtoSeg3D


def create_polynomial_lr_schedule(initial_lr, decay_steps, power=0.9):
    """Create polynomial learning rate decay schedule."""
    return keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=initial_lr,
        decay_steps=decay_steps,
        end_learning_rate=initial_lr * 0.01,
        power=power
    )


def add_l1_regularization_loss(model, lambda_l1=1e-4):
    """
    Add L1 regularization on FC layer weights to the loss.

    This is used during fine-tuning phases to encourage sparsity.
    """
    fc_weights = model.fc_layer.get_weights()[0]  # (num_prototypes, num_classes)

    # L1 loss: sum of absolute values of incorrect class connections
    # For each prototype p_j not in class c, penalize |w_h^(c,j)|
    l1_loss = 0.0

    for c in range(model.num_classes):
        for p in range(model.num_prototypes):
            if model.prototype_class_identity[p, c] == 0:
                # Prototype p doesn't belong to class c
                l1_loss += tf.abs(fc_weights[p, c])

    return lambda_l1 * l1_loss


class L1RegularizationCallback(keras.callbacks.Callback):
    """Callback to add L1 regularization during fine-tuning."""

    def __init__(self, model_ref, lambda_l1=1e-4):
        super().__init__()
        self.model_ref = model_ref
        self.lambda_l1 = lambda_l1

    def on_train_batch_begin(self, batch, logs=None):
        # This is handled in custom train_step - just for reference
        pass


def train_phase(model, train_gen, val_gen, phase_name, epochs, initial_lr,
                callbacks, steps_per_epoch=None, validation_steps=None,
                use_lr_schedule=False, lr_schedule_steps=None):
    """
    Train a single phase.

    Args:
        model: ProtoSeg3D model
        train_gen: Training data generator
        val_gen: Validation data generator
        phase_name: Name of the phase (for logging)
        epochs: Number of epochs to train
        initial_lr: Initial learning rate
        callbacks: List of callbacks
        steps_per_epoch: Steps per epoch (None = use all data)
        validation_steps: Validation steps (None = use all data)
        use_lr_schedule: Whether to use polynomial LR decay
        lr_schedule_steps: Total steps for LR schedule

    Returns:
        Training history
    """
    print("\n" + "="*80)
    print(f"STARTING {phase_name.upper()}")
    print("="*80)

    # Set learning rate
    if use_lr_schedule and lr_schedule_steps:
        lr = create_polynomial_lr_schedule(initial_lr, lr_schedule_steps)
        print(f"Learning rate: Polynomial decay from {initial_lr} (power=0.9)")
    else:
        lr = initial_lr
        print(f"Learning rate: {initial_lr} (constant)")

    # Recompile model with new learning rate
    # This is necessary because when layers are frozen/unfrozen,
    # the optimizer needs to be rebuilt
    optimizer = keras.optimizers.Adam(learning_rate=lr)

    model.compile(
        optimizer=optimizer,
        loss=model.loss,
        metrics=model.metrics
    )

    # Re-enable diversity loss if it was enabled
    if hasattr(model, 'use_diversity_loss') and model.use_diversity_loss:
        # Just set the flag, don't print again
        model.use_diversity_loss = True

    # Train
    history = model.fit(
        train_gen,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_gen,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1
    )

    print(f"\n✓ {phase_name} completed!")
    print("="*80)

    return history


if __name__ == "__main__":
    print("\n" + "="*80)
    print("ProtoSeg3D Multi-Step Training for BraTS 2020")
    print("="*80 + "\n")

    # ========== CONFIGURATION ==========
    # Data configuration
    folder_path = "preprocessed_data"
    batch_size = 2
    split_ratio = 0.2
    random_state = 42

    # Input dimensions
    D, H, W, C = 96, 160, 160, 4
    input_shape = (D, H, W, C)
    num_output_classes = 4

    # Model configuration
    num_prototypes_per_class = 7
    prototype_dim = 128

    # Multi-step training configuration (from ProtoSeg paper)
    warmup_steps = 30000  # 3×10^4
    joint_training_steps = 30000  # 3×10^4
    finetuning_steps = 2000  # Per fine-tuning phase

    # Learning rates (from ProtoSeg paper)
    warmup_lr = 2.5e-4
    joint_lr_backbone = 2.5e-5  # For encoder
    joint_lr_other = 2.5e-4  # For ASPP and prototypes
    finetuning_lr = 1e-5
    lambda_l1 = 1e-4  # L1 regularization weight

    # Diversity loss configuration
    use_diversity_loss = True
    lambda_j = 0.25

    # Convert steps to epochs (approximate)
    samples_per_epoch = 295  # 369 * (1 - 0.2) ≈ 295 training samples
    warmup_epochs = max(1, warmup_steps * batch_size // samples_per_epoch)
    joint_epochs = max(1, joint_training_steps * batch_size // samples_per_epoch)
    finetune_epochs = max(1, finetuning_steps * batch_size // samples_per_epoch)

    print("Multi-step training configuration:")
    print(f"  Warmup: {warmup_steps} steps (~{warmup_epochs} epochs)")
    print(f"  Joint training: {joint_training_steps} steps (~{joint_epochs} epochs)")
    print(f"  Fine-tuning 1: {finetuning_steps} steps (~{finetune_epochs} epochs)")
    print(f"  Fine-tuning 2: {finetuning_steps} steps (~{finetune_epochs} epochs)")
    print()

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"protoseg_multistep_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}\n")

    # ========== DATA GENERATORS ==========
    train_generator = MRIDataGenerator(
        folder_path,
        batch_size=batch_size,
        num_slices=D,
        num_volumes=369,
        split_ratio=split_ratio,
        subset='train',
        shuffle=True,
        random_state=random_state
    )

    validation_generator = MRIDataGenerator(
        folder_path,
        batch_size=batch_size,
        num_slices=D,
        num_volumes=369,
        split_ratio=split_ratio,
        subset='val',
        shuffle=False,
        random_state=random_state
    )

    # ========== CREATE MODEL ==========
    print("\n" + "="*80)
    print("Creating ProtoSeg3D model...")
    print("="*80)

    model = ProtoSeg3D(
        in_size=input_shape,
        num_classes=num_output_classes,
        num_prototypes_per_class=num_prototypes_per_class,
        prototype_dim=prototype_dim,
        features='resnet50_ri',
        f_dist='l2',
        prototype_activation_function='log',
        aspp_out_channels=256
    )

    # ========== COMPILE MODEL ==========
    loss_fn = keras.losses.CategoricalCrossentropy(from_logits=True)

    optimizer = keras.optimizers.Adam(learning_rate=warmup_lr)

    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=[
            keras.metrics.MeanIoU(num_classes=num_output_classes, name='mean_iou'),
            keras.metrics.CategoricalAccuracy(name='accuracy')
        ]
    )

    # Enable diversity loss
    if use_diversity_loss:
        model.enable_diversity_loss(lambda_j=lambda_j)
        print(f"\n✓ Diversity loss enabled (λ_J = {lambda_j})")

    # ========================================================================
    # PHASE 1: WARMUP
    # ========================================================================
    model.setup_warmup_phase()
    model.print_trainable_status()

    warmup_callbacks = [
        keras.callbacks.CSVLogger(f'{output_dir}/warmup_log.csv'),
        keras.callbacks.ModelCheckpoint(
            f'{output_dir}/warmup_best.keras',
            monitor='val_mean_iou',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.TensorBoard(log_dir=f'{output_dir}/logs_warmup')
    ]

    warmup_history = train_phase(
        model, train_generator, validation_generator,
        phase_name="Warmup Phase",
        epochs=warmup_epochs,
        initial_lr=warmup_lr,
        callbacks=warmup_callbacks,
        use_lr_schedule=False
    )

    # Save warmup checkpoint
    model.save(f'{output_dir}/after_warmup.keras')
    print(f"✓ Model saved: {output_dir}/after_warmup.keras")

    # ========================================================================
    # PHASE 2: JOINT TRAINING
    # ========================================================================
    model.setup_joint_training_phase()
    model.print_trainable_status()

    # Note: ProtoSeg uses different LRs for backbone vs ASPP/prototypes
    # TensorFlow doesn't easily support per-layer LRs in one optimizer
    # We'll use a single LR with polynomial decay as a compromise
    joint_lr = (joint_lr_backbone + joint_lr_other) / 2  # Average

    joint_callbacks = [
        keras.callbacks.CSVLogger(f'{output_dir}/joint_log.csv'),
        keras.callbacks.ModelCheckpoint(
            f'{output_dir}/joint_best.keras',
            monitor='val_mean_iou',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.TensorBoard(log_dir=f'{output_dir}/logs_joint')
    ]

    joint_history = train_phase(
        model, train_generator, validation_generator,
        phase_name="Joint Training Phase",
        epochs=joint_epochs,
        initial_lr=joint_lr,
        callbacks=joint_callbacks,
        use_lr_schedule=True,
        lr_schedule_steps=joint_training_steps
    )

    # Save joint training checkpoint
    model.save(f'{output_dir}/after_joint.keras')
    print(f"✓ Model saved: {output_dir}/after_joint.keras")

    # ========================================================================
    # NOTE: Prototype projection and pruning would go here, but are skipped
    # ========================================================================
    print("\n" + "="*80)
    print("NOTE: Skipping prototype projection and pruning phases")
    print("="*80)

    # ========================================================================
    # PHASE 3: FINE-TUNING 1
    # ========================================================================
    model.setup_finetuning_phase()
    model.print_trainable_status()

    # TODO: Add L1 regularization on FC weights during fine-tuning
    # This would require modifying the train_step to add L1 loss

    finetune1_callbacks = [
        keras.callbacks.CSVLogger(f'{output_dir}/finetune1_log.csv'),
        keras.callbacks.ModelCheckpoint(
            f'{output_dir}/finetune1_best.keras',
            monitor='val_mean_iou',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.TensorBoard(log_dir=f'{output_dir}/logs_finetune1')
    ]

    finetune1_history = train_phase(
        model, train_generator, validation_generator,
        phase_name="Fine-tuning Phase 1",
        epochs=finetune_epochs,
        initial_lr=finetuning_lr,
        callbacks=finetune1_callbacks,
        use_lr_schedule=False
    )

    # Save fine-tuning checkpoint
    model.save(f'{output_dir}/after_finetune1.keras')
    print(f"✓ Model saved: {output_dir}/after_finetune1.keras")

    # ========================================================================
    # PHASE 4: FINE-TUNING 2 (Final)
    # ========================================================================
    # Normally this would come after pruning, but we skip pruning
    # So this is just additional fine-tuning

    finetune2_callbacks = [
        keras.callbacks.CSVLogger(f'{output_dir}/finetune2_log.csv'),
        keras.callbacks.ModelCheckpoint(
            f'{output_dir}/finetune2_best.keras',
            monitor='val_mean_iou',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.TensorBoard(log_dir=f'{output_dir}/logs_finetune2')
    ]

    finetune2_history = train_phase(
        model, train_generator, validation_generator,
        phase_name="Fine-tuning Phase 2 (Final)",
        epochs=finetune_epochs,
        initial_lr=finetuning_lr,
        callbacks=finetune2_callbacks,
        use_lr_schedule=False
    )

    # Save final model
    model.save(f'{output_dir}/final_model.keras')
    print(f"✓ Final model saved: {output_dir}/final_model.keras")

    # ========================================================================
    # TRAINING COMPLETE
    # ========================================================================
    print("\n" + "="*80)
    print("MULTI-STEP TRAINING COMPLETE!")
    print("="*80)
    print(f"\nAll outputs saved to: {output_dir}/")
    print("\nCheckpoints:")
    print(f"  - After warmup: {output_dir}/after_warmup.keras")
    print(f"  - After joint training: {output_dir}/after_joint.keras")
    print(f"  - After fine-tuning 1: {output_dir}/after_finetune1.keras")
    print(f"  - Final model: {output_dir}/final_model.keras")
    print("\nBest models per phase:")
    print(f"  - Warmup best: {output_dir}/warmup_best.keras")
    print(f"  - Joint best: {output_dir}/joint_best.keras")
    print(f"  - Fine-tune 1 best: {output_dir}/finetune1_best.keras")
    print(f"  - Fine-tune 2 best: {output_dir}/finetune2_best.keras")
    print("\nTraining logs:")
    print(f"  - Warmup: {output_dir}/warmup_log.csv")
    print(f"  - Joint: {output_dir}/joint_log.csv")
    print(f"  - Fine-tune 1: {output_dir}/finetune1_log.csv")
    print(f"  - Fine-tune 2: {output_dir}/finetune2_log.csv")
    print("\nTensorBoard logs:")
    print(f"  Run: tensorboard --logdir={output_dir}")
    print("="*80)

    # Print prototype information
    proto_info = model.get_prototype_info()
    print("\n" + "="*80)
    print("FINAL PROTOTYPE INFORMATION")
    print("="*80)
    print(f"Total prototypes: {proto_info['num_prototypes']}")
    print(f"Prototypes per class: {proto_info['prototypes_per_class']}")
    print(f"Prototype dimension: {proto_info['prototype_dim']}")
    print("\nPrototype assignments:")
    for c in range(num_output_classes):
        class_protos = [i for i in range(proto_info['num_prototypes'])
                       if proto_info['prototype_class_identity'][i, c] == 1]
        print(f"  Class {c}: Prototypes {class_protos}")
    print("="*80)

    # Print final validation metrics
    print("\n" + "="*80)
    print("FINAL VALIDATION METRICS")
    print("="*80)

    # Evaluate on validation set
    final_metrics = model.evaluate(validation_generator, verbose=0)
    metric_names = [m.name for m in model.metrics] + ['loss']

    for name, value in zip(metric_names, final_metrics):
        print(f"  {name}: {value:.4f}")

    print("="*80)
