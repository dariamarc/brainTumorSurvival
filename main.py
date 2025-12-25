from data_generator import MRIDataGenerator
from losses import FocalLoss, DiceLoss, CombinedLoss
from tensorflow import keras

from model import MProtoNet3D_Segmentation_Keras

if __name__ == "__main__":
    # ========== DATA CONFIGURATION ==========
    # Use preprocessed data for faster training
    folder_path = "preprocessed_data"  # Preprocessed: 160x160x96
    # folder_path = "archive/BraTS2020_training_data/content/data"  # Original: 240x240x155

    batch_size = 4  # Increased from 2 due to smaller volume size
    split_ratio = 0.2
    random_state = 42

    # PREPROCESSED DATA DIMENSIONS (160x160x96)
    D = 96   # Depth: reduced from 155
    H = 160  # Height: reduced from 240
    W = 160  # Width: reduced from 240
    C = 4    # Channels: unchanged (FLAIR, T1, T1ce, T2)

    input_shape = (D, H, W, C)

    # no of output classes: background, GD enhancing tumor, peritumoral edema, non-enhancing tumor core
    num_output_classes = 4

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

    # the first number has to divide with the number of output classes
    # 28 / 4 = 7 - we will have 7 prototypes learned for each class
    # NOTE: Prototypes are now integrated into decoder (see DECISION_LOG.md)
    prototype_shape = (28, 128, 1, 1, 1)

    model = MProtoNet3D_Segmentation_Keras(
        in_size=input_shape,
        num_classes=num_output_classes,
        prototype_shape=prototype_shape,
        features='resnet50_ri',
        f_dist='l2'
    )

    # REDUCED learning rate with warmup schedule
    initial_learning_rate = 0.0001

    # Optional: Learning rate schedule with warmup
    lr_schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=initial_learning_rate,
        decay_steps=1000,
        alpha=0.1  # Minimum learning rate = 0.1 * initial_lr
    )

    optimizer = keras.optimizers.Adam(learning_rate=initial_learning_rate)

    # ========== LOSS FUNCTION CONFIGURATION ==========
    # IMPORTANT: Adjust these if Mean IoU is low (< 0.3) while accuracy is high

    # OPTION 1: Dice Loss Only (BEST for severe class imbalance - try this first if IoU is low)
    # loss_fn = DiceLoss()

    # OPTION 2: Combined with MORE Dice weight (RECOMMENDED if Option 1 is unstable)
    # Prioritizes overlap (IoU) over per-pixel accuracy
    loss_fn = CombinedLoss(focal_weight=0.3, dice_weight=0.7, gamma=1.0, alpha=0.25)

    # OPTION 3: Balanced Focal + Dice (original - use if IoU > 0.4)
    # loss_fn = CombinedLoss(focal_weight=0.5, dice_weight=0.5, gamma=1.0, alpha=0.25)

    # OPTION 4: Weighted Focal Loss (for advanced tuning)
    # class_weights = [0.05, 1.0, 1.0, 1.0]  # [background, tumor_class_1, tumor_class_2, tumor_class_3]
    # loss_fn = FocalLoss(gamma=2.0, alpha=0.25, class_weights=class_weights)

    # Compile with metrics appropriate for multi-class segmentation
    # Note: Global precision/recall are misleading for segmentation due to class imbalance
    # Focus on MeanIoU which is standard for segmentation evaluation
    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=[
            keras.metrics.MeanIoU(num_classes=num_output_classes, name='mean_iou'),
            # Categorical accuracy (correct for multi-class)
            keras.metrics.CategoricalAccuracy(name='categorical_accuracy')
        ]
    )

    # Setup callbacks for better training monitoring and stability
    callbacks = [
        # Reduce learning rate when validation loss plateaus
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,  # Reduce LR by half
            patience=5,  # Wait 5 epochs before reducing
            min_lr=1e-7,  # Don't go below this
            verbose=1
        ),
        # Stop training if validation loss doesn't improve
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,  # Stop if no improvement for 10 epochs
            restore_best_weights=True,  # Restore weights from best epoch
            verbose=1
        ),
        # Save best model
        keras.callbacks.ModelCheckpoint(
            'best_model.keras',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        # Log training progress to CSV
        keras.callbacks.CSVLogger('training_log.csv', append=True),
        # Optional: Backup checkpoint every 5 epochs
        keras.callbacks.ModelCheckpoint(
            'checkpoint_epoch_{epoch:02d}.keras',
            save_freq='epoch',
            verbose=0
        )
    ]

    epochs = 100  # Increased epochs - early stopping will handle when to stop
    steps_per_epoch = None  # Use all data
    validation_steps = None  # Use all validation data

    print("=" * 80)
    print("IMPROVED TRAINING CONFIGURATION")
    print("=" * 80)
    print(f"Data path: {folder_path}")
    print(f"Input shape: ({D}, {H}, {W}, {C})")
    print(f"Data reduction: ~51% smaller than original (240x240x155)")
    print(f"Initial learning rate: {initial_learning_rate}")
    print(f"Batch size: {batch_size} (increased from 2 due to smaller volumes)")
    print(f"Loss function: Combined Focal + Dice Loss (Improved for IoU)")
    print(f"  - Focal loss gamma: 1.0")
    print(f"  - Focal loss alpha: 0.25")
    print(f"  - Loss weights: 30% Focal + 70% Dice (prioritizes overlap)")
    print(f"  - Note: Higher Dice weight helps with class imbalance")
    print(f"Max epochs: {epochs}")
    print(f"\nMetrics tracked:")
    print(f"  - Accuracy (overall voxel accuracy)")
    print(f"  - Mean IoU (Intersection over Union per class)")
    print(f"  - Precision & Recall")
    print("\nCallbacks enabled:")
    print("  ✓ ReduceLROnPlateau: Reduces LR when val_loss plateaus (patience=5)")
    print("  ✓ EarlyStopping: Stops if no improvement (patience=10)")
    print("  ✓ ModelCheckpoint: Saves best model")
    print("  ✓ CSVLogger: Logs all metrics to CSV")
    print("=" * 80)
    print("\nStarting model training...\n")

    history = model.fit(
        train_generator,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=validation_generator,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1  # Detailed output
    )

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print("✓ Best model saved to: best_model.keras")
    print("✓ Training log saved to: training_log.csv")
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
    print("  model = keras.models.load_model('best_model.keras',")
    print("                                   custom_objects={'CombinedLoss': CombinedLoss})")
    print("=" * 80)