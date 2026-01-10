"""
Training script for ProtoSeg3D model.

Based on ProtoSeg paper training protocol:
- Uses cross-entropy loss (instead of Dice)
- Ground truth is downsampled to match feature map resolution during training
- Evaluation uses full resolution
"""

from data_generator import MRIDataGenerator
from tensorflow import keras
import tensorflow as tf

from model_protoseg import ProtoSeg3D

if __name__ == "__main__":
    print("\n" + "="*80)
    print("ProtoSeg3D Training for BraTS 2020")
    print("="*80 + "\n")

    # ========== DATA CONFIGURATION ==========
    folder_path = "preprocessed_data"  # Preprocessed: 160x160x96

    batch_size = 2  # Reduced from 4 due to larger memory footprint
    split_ratio = 0.2
    random_state = 42

    # PREPROCESSED DATA DIMENSIONS (160x160x96)
    D = 96   # Depth
    H = 160  # Height
    W = 160  # Width
    C = 4    # Channels: FLAIR, T1, T1ce, T2

    input_shape = (D, H, W, C)

    # Number of output classes: background, GD enhancing, peritumoral edema, non-enhancing core
    num_output_classes = 4

    # Create data generators
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

    # ========== MODEL CONFIGURATION ==========
    # ProtoSeg configuration: 7 prototypes per class (28 total for 4 classes)
    num_prototypes_per_class = 7
    prototype_dim = 128  # Dimension of prototype feature space

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

    # ========== OPTIMIZER CONFIGURATION ==========
    # ProtoSeg uses different learning rates for different components
    # For now, we'll use a single optimizer (can be extended to multi-step training later)

    initial_learning_rate = 0.0001

    # Optional: Learning rate schedule
    lr_schedule = keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=initial_learning_rate,
        decay_steps=30000,
        end_learning_rate=initial_learning_rate * 0.1,
        power=0.9
    )

    optimizer = keras.optimizers.Adam(learning_rate=initial_learning_rate)

    # ========== LOSS FUNCTION CONFIGURATION ==========
    # ProtoSeg uses Cross-Entropy loss + Diversity loss
    # Ground truth needs to be in correct format: (B, D, H, W, num_classes) one-hot encoded

    loss_fn = keras.losses.CategoricalCrossentropy(from_logits=True)

    # ========== COMPILE MODEL ==========
    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=[
            keras.metrics.MeanIoU(num_classes=num_output_classes, name='mean_iou'),
            keras.metrics.CategoricalAccuracy(name='categorical_accuracy')
        ]
    )

    # ========== ENABLE DIVERSITY LOSS ==========
    # This is the key difference from baseline training!
    # Diversity loss encourages prototypes to activate on different regions
    use_diversity_loss = True  # Set to False to disable
    lambda_j = 0.25  # Weight of diversity loss (as in ProtoSeg paper)

    if use_diversity_loss:
        model.enable_diversity_loss(lambda_j=lambda_j)
        print(f"✓ Diversity loss enabled with λ_J = {lambda_j}")
    else:
        print("⚠ Diversity loss disabled (training without prototype diversity)")

    # ========== CALLBACKS ==========
    callbacks = [
        # Reduce learning rate when validation loss plateaus
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),

        # Stop training if validation loss doesn't improve
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),

        # Save best model
        keras.callbacks.ModelCheckpoint(
            'best_model_protoseg.keras',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),

        # Log training progress to CSV
        keras.callbacks.CSVLogger('training_log_protoseg.csv', append=True),

        # TensorBoard logging
        keras.callbacks.TensorBoard(
            log_dir='./logs_protoseg',
            histogram_freq=1,
            write_graph=True
        )
    ]

    # ========== TRAINING CONFIGURATION ==========
    epochs = 100  # Early stopping will handle when to stop
    steps_per_epoch = None  # Use all data
    validation_steps = None  # Use all validation data

    print("=" * 80)
    print("PROTOSEG TRAINING CONFIGURATION")
    print("=" * 80)
    print(f"Data path: {folder_path}")
    print(f"Input shape: ({D}, {H}, {W}, {C})")
    print(f"Batch size: {batch_size}")
    print(f"Initial learning rate: {initial_learning_rate}")
    if use_diversity_loss:
        print(f"Loss function: Cross-Entropy + Diversity Loss")
        print(f"  - Cross-Entropy: Segmentation loss")
        print(f"  - Diversity Loss (λ_J = {lambda_j}): Jeffrey's divergence")
        print(f"  - Total: L = L_CE + {lambda_j} * L_J")
    else:
        print(f"Loss function: Cross-Entropy only (no diversity)")
    print(f"Number of classes: {num_output_classes}")
    print(f"Prototypes per class: {num_prototypes_per_class}")
    print(f"Total prototypes: {num_output_classes * num_prototypes_per_class}")
    print(f"Prototype dimension: {prototype_dim}")
    print(f"Max epochs: {epochs}")
    print(f"\nMetrics tracked:")
    print(f"  - Categorical Accuracy (overall voxel accuracy)")
    print(f"  - Mean IoU (Intersection over Union per class)")
    print("\nCallbacks enabled:")
    print("  ✓ ReduceLROnPlateau: Reduces LR when val_loss plateaus (patience=5)")
    print("  ✓ EarlyStopping: Stops if no improvement (patience=10)")
    print("  ✓ ModelCheckpoint: Saves best model")
    print("  ✓ CSVLogger: Logs all metrics to CSV")
    print("  ✓ TensorBoard: Logs for visualization")
    print("=" * 80)
    print("\nNOTE: This is single-phase training.")
    print("For full ProtoSeg protocol, implement multi-step training:")
    print("  1. Warmup (freeze encoder)")
    print("  2. Joint optimization")
    print("  3. Prototype projection")
    print("  4. Last layer tuning")
    print("  5. Prototype pruning")
    print("  6. Final tuning")
    print("=" * 80)
    print("\nStarting model training...\n")

    # ========== TRAIN MODEL ==========
    history = model.fit(
        train_generator,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=validation_generator,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1
    )

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print("✓ Best model saved to: best_model_protoseg.keras")
    print("✓ Training log saved to: training_log_protoseg.csv")
    print("✓ TensorBoard logs saved to: ./logs_protoseg")
    print("=" * 80)

    # Print final metrics
    print("\nFinal Training Metrics:")
    for key, value in history.history.items():
        if not key.startswith('val_'):
            print(f"  {key}: {value[-1]:.4f}")

    print("\nFinal Validation Metrics:")
    for key, value in history.history.items():
        if key.startswith('val_'):
            print(f"  {key}: {value[-1]:.4f}")

    print("\n" + "=" * 80)
    print("To load the best model:")
    print("  from model_protoseg import ProtoSeg3D")
    print("  model = keras.models.load_model('best_model_protoseg.keras',")
    print("                                   custom_objects={'ProtoSeg3D': ProtoSeg3D})")
    print("=" * 80)

    # Print prototype information
    proto_info = model.get_prototype_info()
    print("\n" + "=" * 80)
    print("PROTOTYPE INFORMATION")
    print("=" * 80)
    print(f"Total prototypes: {proto_info['num_prototypes']}")
    print(f"Prototypes per class: {proto_info['prototypes_per_class']}")
    print(f"Prototype dimension: {proto_info['prototype_dim']}")
    print("\nPrototype assignments:")
    for c in range(num_output_classes):
        class_protos = [i for i in range(proto_info['num_prototypes'])
                       if proto_info['prototype_class_identity'][i, c] == 1]
        print(f"  Class {c}: Prototypes {class_protos}")
    print("=" * 80)
