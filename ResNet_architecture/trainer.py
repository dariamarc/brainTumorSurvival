import os
import sys
import tensorflow as tf
from tensorflow import keras
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from losses import DiceLoss
from prototype_projection import PrototypeProjector
from metrics import SegmentationMetrics, PrototypeMetrics, UtilizationMetrics


class PrototypeTrainer:
    """
    Three-phase trainer for PrototypeSegNet3D.

    Phase 1: Warm-up (ASPP + Prototypes Training)
    Phase 2: Joint Fine-tuning (Full Network Training)
    Phase 3: Prototype Projection & Refinement
    """

    def __init__(self, model, train_generator, val_generator,
                 checkpoint_dir='checkpoints', log_dir='logs'):
        self.model = model
        self.train_generator = train_generator
        self.val_generator = val_generator
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir

        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        # Initialize loss functions
        self._init_losses()

        # Initialize metrics
        self._init_metrics()

        # Training history
        self.history = {
            'phase1': [], 'phase2': [], 'phase3': []
        }

    def _init_losses(self):
        """Initialize all loss functions."""
        # Weighted Dice: 0 for background, 1 for tumor classes
        self.dice_loss = DiceLoss(
            class_weights=[0.0, 1.0, 1.0, 1.0],
            num_classes=self.model.num_classes
        )

    def _init_metrics(self):
        """Initialize metric calculators."""
        self.seg_metrics = SegmentationMetrics(num_classes=self.model.num_classes)
        self.proto_metrics = PrototypeMetrics(
            n_prototypes=self.model.n_prototypes,
            n_classes=self.model.num_classes
        )
        self.util_metrics = UtilizationMetrics(n_prototypes=self.model.n_prototypes)

    def _compute_metrics(self, masks, logits, similarities):
        """
        Compute all metrics for given predictions.

        Args:
            masks: Ground truth masks (B, D, H, W, num_classes)
            logits: Prediction logits (B, D, H, W, num_classes)
            similarities: Prototype similarity maps (B, D, H, W, n_prototypes)

        Returns:
            Dict with all computed metrics
        """
        seg = self.seg_metrics.compute_all(masks, logits)
        proto = self.proto_metrics.compute_all(masks, similarities)
        util = self.util_metrics.compute_all(similarities)
        return {**seg, **proto, **util}

    # ==================== PHASE SETUP ====================

    def _freeze_backbone(self):
        """Freeze backbone weights."""
        self.model.backbone.trainable = False
        print("Backbone: FROZEN")

    def _unfreeze_backbone(self):
        """Unfreeze backbone weights."""
        self.model.backbone.trainable = True
        print("Backbone: TRAINABLE")

    def _freeze_prototypes(self):
        """Freeze prototype layer weights."""
        self.model.prototype_layer.trainable = False
        print("Prototypes: FROZEN")

    def _unfreeze_prototypes(self):
        """Unfreeze prototype layer weights."""
        self.model.prototype_layer.trainable = True
        print("Prototypes: TRAINABLE")

    def _setup_phase1(self):
        """
        Phase 1: Warm-up
        - Frozen: Backbone
        - Trainable: ASPP, Prototypes, Classifier
        - LR: 1e-3
        """
        print("\n" + "=" * 50)
        print("PHASE 1: Warm-up Setup")
        print("=" * 50)

        self._freeze_backbone()
        self._unfreeze_prototypes()

        self.model.aspp.trainable = True
        self.model.classifier.trainable = True

        print("ASPP: TRAINABLE")
        print("Classifier: TRAINABLE")

        # Single optimizer for all trainable components
        self.optimizer = keras.optimizers.Adam(learning_rate=1e-3)
        print(f"Optimizer: Adam, LR=1e-3")

    def _setup_phase2(self):
        """
        Phase 2: Joint Fine-tuning
        - Frozen: None
        - Trainable: All
        - LR: 1e-4 (backbone), 5e-4 (others)
        """
        print("\n" + "=" * 50)
        print("PHASE 2: Joint Fine-tuning Setup")
        print("=" * 50)

        self._unfreeze_backbone()
        self._unfreeze_prototypes()

        self.model.aspp.trainable = True
        self.model.classifier.trainable = True

        print("All components: TRAINABLE")

        # Use two optimizers for differential learning rates
        self.optimizer_backbone = keras.optimizers.Adam(learning_rate=1e-4)
        self.optimizer_other = keras.optimizers.Adam(learning_rate=5e-4)
        print(f"Optimizer (backbone): Adam, LR=1e-4")
        print(f"Optimizer (ASPP/prototypes/classifier): Adam, LR=5e-4")

    def _setup_phase3(self):
        """
        Phase 3: Refinement (after projection)
        - Frozen: Prototypes
        - Trainable: Backbone, ASPP, Classifier
        - LR: 5e-5 (backbone), 1e-4 (others)
        """
        print("\n" + "=" * 50)
        print("PHASE 3: Refinement Setup")
        print("=" * 50)

        self._unfreeze_backbone()
        self._freeze_prototypes()

        self.model.aspp.trainable = True
        self.model.classifier.trainable = True

        print("ASPP: TRAINABLE")
        print("Classifier: TRAINABLE")

        # Two optimizers for differential learning rates
        self.optimizer_backbone = keras.optimizers.Adam(learning_rate=5e-5)
        self.optimizer_other = keras.optimizers.Adam(learning_rate=1e-4)
        print(f"Optimizer (backbone): Adam, LR=5e-5")
        print(f"Optimizer (ASPP/classifier): Adam, LR=1e-4")

    # ==================== LOSS COMPUTATION ====================

    def _compute_phase1_loss(self, y_true, y_pred, similarities):
        """
        Phase 1 loss: Weighted Dice only.
        Weights: [0, 1, 1, 1] - ignores background, focuses on tumor classes.
        """
        dice_loss = self.dice_loss(y_true, y_pred)

        loss_dict = {
            'total': dice_loss,
            'dice': dice_loss
        }

        return dice_loss, loss_dict

    def _compute_phase2_loss(self, y_true, y_pred, similarities, features):
        """
        Phase 2 loss: Weighted Dice only.
        Weights: [0, 1, 1, 1] - ignores background, focuses on tumor classes.
        """
        dice_loss = self.dice_loss(y_true, y_pred)

        loss_dict = {
            'total': dice_loss,
            'dice': dice_loss
        }

        return dice_loss, loss_dict

    def _compute_phase3_loss(self, y_true, y_pred, similarities):
        """
        Phase 3 loss: Weighted Dice only.
        Weights: [0, 1, 1, 1] - ignores background, focuses on tumor classes.
        """
        dice_loss = self.dice_loss(y_true, y_pred)

        loss_dict = {
            'total': dice_loss,
            'dice': dice_loss
        }

        return dice_loss, loss_dict

    def _downsample_masks(self, masks, target_shape):
        """Downsample masks to match feature resolution."""
        # Simple approach: use tf.image.resize with nearest neighbor
        batch_size = tf.shape(masks)[0]
        n_classes = masks.shape[-1]

        # Reshape for processing
        masks_reshaped = tf.reshape(masks, [-1, masks.shape[2], masks.shape[3], n_classes])

        # Resize H, W
        masks_hw = tf.image.resize(masks_reshaped, [target_shape[1], target_shape[2]], method='nearest')

        # Reshape back and handle D dimension
        masks_hw = tf.reshape(masks_hw, [batch_size, masks.shape[1], target_shape[1], target_shape[2], n_classes])

        # Resize D dimension (simple strided sampling)
        stride_d = masks.shape[1] // target_shape[0]
        indices = tf.range(0, masks.shape[1], stride_d)[:target_shape[0]]
        masks_downsampled = tf.gather(masks_hw, indices, axis=1)

        return masks_downsampled

    # ==================== TRAINING STEPS ====================

    @tf.function
    def _train_step_phase1(self, images, masks):
        """Training step for Phase 1."""
        with tf.GradientTape() as tape:
            logits, similarities = self.model(images, training=True)
            loss, loss_dict = self._compute_phase1_loss(masks, logits, similarities)

        # Get trainable variables (excludes frozen backbone)
        trainable_vars = self.model.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Clip gradients to prevent exploding gradients
        gradients, _ = tf.clip_by_global_norm(gradients, 1.0)

        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        return loss_dict

    @tf.function
    def _train_step_phase2(self, images, masks):
        """Training step for Phase 2 with differential learning rates."""
        with tf.GradientTape(persistent=True) as tape:
            logits, similarities, features = self.model.call_with_features(images, training=True)
            loss, loss_dict = self._compute_phase2_loss(masks, logits, similarities, features)

        # Separate variables by component
        backbone_vars = self.model.backbone.trainable_variables
        other_vars = (
            self.model.aspp.trainable_variables +
            self.model.prototype_layer.trainable_variables +
            self.model.classifier.trainable_variables
        )

        # Compute and apply gradients separately
        backbone_grads = tape.gradient(loss, backbone_vars)
        other_grads = tape.gradient(loss, other_vars)

        del tape

        # Clip gradients to prevent exploding gradients
        backbone_grads, _ = tf.clip_by_global_norm(backbone_grads, 1.0)
        other_grads, _ = tf.clip_by_global_norm(other_grads, 1.0)

        self.optimizer_backbone.apply_gradients(zip(backbone_grads, backbone_vars))
        self.optimizer_other.apply_gradients(zip(other_grads, other_vars))

        return loss_dict

    @tf.function
    def _train_step_phase3(self, images, masks):
        """Training step for Phase 3."""
        with tf.GradientTape(persistent=True) as tape:
            logits, similarities = self.model(images, training=True)
            loss, loss_dict = self._compute_phase3_loss(masks, logits, similarities)

        # Separate variables (prototypes are frozen)
        backbone_vars = self.model.backbone.trainable_variables
        other_vars = (
            self.model.aspp.trainable_variables +
            self.model.classifier.trainable_variables
        )

        backbone_grads = tape.gradient(loss, backbone_vars)
        other_grads = tape.gradient(loss, other_vars)

        del tape

        # Clip gradients to prevent exploding gradients
        backbone_grads, _ = tf.clip_by_global_norm(backbone_grads, 1.0)
        other_grads, _ = tf.clip_by_global_norm(other_grads, 1.0)

        self.optimizer_backbone.apply_gradients(zip(backbone_grads, backbone_vars))
        self.optimizer_other.apply_gradients(zip(other_grads, other_vars))

        return loss_dict

    def _validate(self, phase):
        """Run validation and return losses and metrics."""
        val_losses = []
        val_metrics = []

        for batch_idx in range(len(self.val_generator)):
            images, masks = self.val_generator[batch_idx]

            if phase == 1:
                logits, similarities = self.model(images, training=False)
                _, loss_dict = self._compute_phase1_loss(masks, logits, similarities)
            elif phase == 2:
                logits, similarities, features = self.model.call_with_features(images, training=False)
                _, loss_dict = self._compute_phase2_loss(masks, logits, similarities, features)
            else:
                logits, similarities = self.model(images, training=False)
                _, loss_dict = self._compute_phase3_loss(masks, logits, similarities)

            val_losses.append({k: float(v) for k, v in loss_dict.items()})

            # Compute metrics
            metrics = self._compute_metrics(masks, logits, similarities)
            val_metrics.append(metrics)

        # Average losses
        avg_losses = {}
        for key in val_losses[0].keys():
            avg_losses[key] = np.mean([l[key] for l in val_losses])

        # Average metrics (handle nested dict for proto_class_activations)
        avg_metrics = {}
        for key in val_metrics[0].keys():
            if key == 'proto_class_activations':
                # Average the activation matrices
                matrices = [np.array(m[key]) for m in val_metrics]
                avg_metrics[key] = np.mean(matrices, axis=0).tolist()
            else:
                avg_metrics[key] = np.mean([m[key] for m in val_metrics])

        return avg_losses, avg_metrics

    # ==================== PHASE TRAINING ====================

    def train_phase1(self, epochs=50, patience=10):
        """
        Phase 1: Warm-up training.

        Args:
            epochs: Number of epochs (default 50)
            patience: Early stopping patience (default 10)
        """
        self._setup_phase1()
        best_val_loss = float('inf')
        patience_counter = 0

        print(f"\nStarting Phase 1 training for {epochs} epochs...")

        for epoch in range(epochs):
            epoch_losses = []

            # Training loop
            pbar = tqdm(range(len(self.train_generator)), desc=f"Phase 1 - Epoch {epoch+1}/{epochs}")
            for batch_idx in pbar:
                images, masks = self.train_generator[batch_idx]
                loss_dict = self._train_step_phase1(images, masks)
                epoch_losses.append({k: float(v) for k, v in loss_dict.items()})

                pbar.set_postfix({'loss': f"{float(loss_dict['total']):.4f}"})

            # Compute epoch averages
            avg_train_loss = np.mean([l['total'] for l in epoch_losses])

            # Validation
            val_losses, val_metrics = self._validate(phase=1)
            avg_val_loss = val_losses['total']

            # Log
            log_entry = {
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'train_losses': {k: np.mean([l[k] for l in epoch_losses]) for k in epoch_losses[0]},
                'val_losses': val_losses,
                'val_metrics': val_metrics
            }
            self.history['phase1'].append(log_entry)

            # Print epoch summary with metrics
            print(f"Epoch {epoch+1}: train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}")
            print(f"  Dice: ET={val_metrics['dice_gd_enhancing']:.2f}, "
                  f"ED={val_metrics['dice_edema']:.2f}, "
                  f"NCR={val_metrics['dice_necrotic']:.2f}, "
                  f"Mean={val_metrics['dice_mean']:.2f}")
            print(f"  Purity: P0={val_metrics['purity_proto_0']:.2f}, "
                  f"P1={val_metrics['purity_proto_1']:.2f}, "
                  f"P2={val_metrics['purity_proto_2']:.2f}, "
                  f"Mean={val_metrics['purity_mean']:.2f}")

            # Early stopping check
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                self._save_checkpoint('phase1_best')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

            # Shuffle data
            self.train_generator.on_epoch_end()

        self._save_checkpoint('phase1_final')
        self._save_history()
        print("Phase 1 complete.")

    def train_phase2(self, epochs=150, patience=15):
        """
        Phase 2: Joint fine-tuning.

        Args:
            epochs: Number of epochs (default 150)
            patience: Early stopping patience (default 15)
        """
        self._setup_phase2()
        best_val_loss = float('inf')
        patience_counter = 0

        print(f"\nStarting Phase 2 training for {epochs} epochs...")

        for epoch in range(epochs):
            epoch_losses = []

            pbar = tqdm(range(len(self.train_generator)), desc=f"Phase 2 - Epoch {epoch+1}/{epochs}")
            for batch_idx in pbar:
                images, masks = self.train_generator[batch_idx]
                loss_dict = self._train_step_phase2(images, masks)
                epoch_losses.append({k: float(v) for k, v in loss_dict.items()})

                pbar.set_postfix({'loss': f"{float(loss_dict['total']):.4f}"})

            avg_train_loss = np.mean([l['total'] for l in epoch_losses])

            val_losses, val_metrics = self._validate(phase=2)
            avg_val_loss = val_losses['total']

            log_entry = {
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'train_losses': {k: np.mean([l[k] for l in epoch_losses]) for k in epoch_losses[0]},
                'val_losses': val_losses,
                'val_metrics': val_metrics
            }
            self.history['phase2'].append(log_entry)

            # Print epoch summary with metrics
            print(f"Epoch {epoch+1}: train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}")
            print(f"  Dice: ET={val_metrics['dice_gd_enhancing']:.2f}, "
                  f"ED={val_metrics['dice_edema']:.2f}, "
                  f"NCR={val_metrics['dice_necrotic']:.2f}, "
                  f"Mean={val_metrics['dice_mean']:.2f}")
            print(f"  Purity: P0={val_metrics['purity_proto_0']:.2f}, "
                  f"P1={val_metrics['purity_proto_1']:.2f}, "
                  f"P2={val_metrics['purity_proto_2']:.2f}, "
                  f"Mean={val_metrics['purity_mean']:.2f}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                self._save_checkpoint('phase2_best')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

            self.train_generator.on_epoch_end()

        self._save_checkpoint('phase2_final')
        self._save_history()
        print("Phase 2 complete.")

    def train_phase3(self, epochs=30, patience=10):
        """
        Phase 3: Projection and refinement.

        Args:
            epochs: Number of epochs (default 30)
            patience: Early stopping patience (default 10)
        """
        # Step 3a: Prototype projection
        print("\n" + "=" * 50)
        print("PHASE 3a: Prototype Projection")
        print("=" * 50)

        projector = PrototypeProjector(self.model)
        projection_info = projector.project_prototypes(self.train_generator)
        projector.apply_projection(projection_info)
        projector.save_projection(
            projection_info,
            os.path.join(self.checkpoint_dir, 'phase3_projection')
        )

        # Step 3b: Refinement training
        print("\n" + "=" * 50)
        print("PHASE 3b: Refinement Training")
        print("=" * 50)

        self._setup_phase3()
        best_val_loss = float('inf')
        patience_counter = 0

        print(f"\nStarting Phase 3 refinement for {epochs} epochs...")

        for epoch in range(epochs):
            epoch_losses = []

            pbar = tqdm(range(len(self.train_generator)), desc=f"Phase 3 - Epoch {epoch+1}/{epochs}")
            for batch_idx in pbar:
                images, masks = self.train_generator[batch_idx]
                loss_dict = self._train_step_phase3(images, masks)
                epoch_losses.append({k: float(v) for k, v in loss_dict.items()})

                pbar.set_postfix({'loss': f"{float(loss_dict['total']):.4f}"})

            avg_train_loss = np.mean([l['total'] for l in epoch_losses])

            val_losses, val_metrics = self._validate(phase=3)
            avg_val_loss = val_losses['total']

            log_entry = {
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'train_losses': {k: np.mean([l[k] for l in epoch_losses]) for k in epoch_losses[0]},
                'val_losses': val_losses,
                'val_metrics': val_metrics
            }
            self.history['phase3'].append(log_entry)

            # Print epoch summary with metrics
            print(f"Epoch {epoch+1}: train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}")
            print(f"  Dice: ET={val_metrics['dice_gd_enhancing']:.2f}, "
                  f"ED={val_metrics['dice_edema']:.2f}, "
                  f"NCR={val_metrics['dice_necrotic']:.2f}, "
                  f"Mean={val_metrics['dice_mean']:.2f}")
            print(f"  Purity: P0={val_metrics['purity_proto_0']:.2f}, "
                  f"P1={val_metrics['purity_proto_1']:.2f}, "
                  f"P2={val_metrics['purity_proto_2']:.2f}, "
                  f"Mean={val_metrics['purity_mean']:.2f}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                self._save_checkpoint('phase3_best')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

            self.train_generator.on_epoch_end()

        self._save_checkpoint('phase3_final')
        self._save_history()
        print("Phase 3 complete.")

    def run_full_training(self, phase1_epochs=50, phase2_epochs=150, phase3_epochs=30):
        """
        Run complete three-phase training.

        Args:
            phase1_epochs: Epochs for warm-up phase
            phase2_epochs: Epochs for joint training phase
            phase3_epochs: Epochs for refinement phase
        """
        print("\n" + "=" * 60)
        print("STARTING FULL THREE-PHASE TRAINING")
        print("=" * 60)

        start_time = datetime.now()

        # Phase 1
        self.train_phase1(epochs=phase1_epochs)

        # Phase 2
        self.train_phase2(epochs=phase2_epochs)

        # Phase 3
        self.train_phase3(epochs=phase3_epochs)

        elapsed = datetime.now() - start_time
        print("\n" + "=" * 60)
        print(f"TRAINING COMPLETE - Total time: {elapsed}")
        print("=" * 60)

    # ==================== CHECKPOINTING ====================

    def _save_checkpoint(self, name):
        """Save model checkpoint."""
        path = os.path.join(self.checkpoint_dir, f"{name}.weights.h5")
        self.model.save_weights(path)
        print(f"Checkpoint saved: {path}")

    def load_checkpoint(self, name):
        """Load model checkpoint."""
        path = os.path.join(self.checkpoint_dir, f"{name}.weights.h5")
        self.model.load_weights(path)
        print(f"Checkpoint loaded: {path}")

    def _convert_to_serializable(self, obj):
        """Recursively convert numpy types to Python types for JSON serialization."""
        if isinstance(obj, dict):
            return {k: self._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(v) for v in obj]
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    def _save_history(self):
        """Save training history."""
        path = os.path.join(self.log_dir, 'training_history.json')
        serializable_history = self._convert_to_serializable(self.history)
        with open(path, 'w') as f:
            json.dump(serializable_history, f, indent=2)
        print(f"History saved: {path}")

    def resume_from_phase(self, phase, checkpoint_name=None):
        """
        Resume training from a specific phase.

        Args:
            phase: Phase number to resume (1, 2, or 3)
            checkpoint_name: Optional checkpoint to load first
        """
        if checkpoint_name:
            self.load_checkpoint(checkpoint_name)

        if phase == 1:
            self.train_phase1()
            self.train_phase2()
            self.train_phase3()
        elif phase == 2:
            self.train_phase2()
            self.train_phase3()
        elif phase == 3:
            self.train_phase3()
        else:
            raise ValueError(f"Invalid phase: {phase}")

    def get_full_history(self):
        """
        Get combined training history from all phases for plotting.

        Returns:
            Dictionary with lists ready for matplotlib plotting:
            - epochs: [1, 2, 3, ...]
            - phase_boundaries: [phase1_end, phase2_end] epoch numbers
            - train_loss, val_loss: loss values per epoch
            - dice_gd_enhancing, dice_edema, dice_necrotic, dice_mean, dice_whole_tumor
            - purity_proto_0, purity_proto_1, purity_proto_2, purity_mean
            - max_act_proto_0/1/2, mean_act_proto_0/1/2, coverage_proto_0/1/2
        """
        combined = {
            'epochs': [],
            'phase_boundaries': [],
            'train_loss': [],
            'val_loss': [],
            # Segmentation metrics
            'dice_gd_enhancing': [],
            'dice_edema': [],
            'dice_necrotic': [],
            'dice_mean': [],
            'dice_whole_tumor': [],
            # Prototype purity metrics
            'purity_proto_0': [],
            'purity_proto_1': [],
            'purity_proto_2': [],
            'purity_mean': [],
            # Utilization metrics
            'max_act_proto_0': [],
            'max_act_proto_1': [],
            'max_act_proto_2': [],
            'mean_act_proto_0': [],
            'mean_act_proto_1': [],
            'mean_act_proto_2': [],
            'coverage_proto_0': [],
            'coverage_proto_1': [],
            'coverage_proto_2': [],
            # Loss components
            'loss_segmentation': [],
            'loss_purity': [],
        }

        epoch_counter = 0

        # Process each phase
        for phase_idx, phase_name in enumerate(['phase1', 'phase2', 'phase3']):
            phase_entries = self.history.get(phase_name, [])

            for entry in phase_entries:
                epoch_counter += 1
                combined['epochs'].append(epoch_counter)
                combined['train_loss'].append(entry.get('train_loss', 0))
                combined['val_loss'].append(entry.get('val_loss', 0))

                # Loss components
                if 'val_losses' in entry:
                    combined['loss_segmentation'].append(
                        entry['val_losses'].get('segmentation', 0)
                    )
                    combined['loss_purity'].append(
                        entry['val_losses'].get('purity', 0)
                    )

                # Metrics (from val_metrics)
                if 'val_metrics' in entry:
                    metrics = entry['val_metrics']
                    # Segmentation
                    combined['dice_gd_enhancing'].append(metrics.get('dice_gd_enhancing', 0))
                    combined['dice_edema'].append(metrics.get('dice_edema', 0))
                    combined['dice_necrotic'].append(metrics.get('dice_necrotic', 0))
                    combined['dice_mean'].append(metrics.get('dice_mean', 0))
                    combined['dice_whole_tumor'].append(metrics.get('dice_whole_tumor', 0))
                    # Purity
                    combined['purity_proto_0'].append(metrics.get('purity_proto_0', 0))
                    combined['purity_proto_1'].append(metrics.get('purity_proto_1', 0))
                    combined['purity_proto_2'].append(metrics.get('purity_proto_2', 0))
                    combined['purity_mean'].append(metrics.get('purity_mean', 0))
                    # Utilization
                    combined['max_act_proto_0'].append(metrics.get('max_act_proto_0', 0))
                    combined['max_act_proto_1'].append(metrics.get('max_act_proto_1', 0))
                    combined['max_act_proto_2'].append(metrics.get('max_act_proto_2', 0))
                    combined['mean_act_proto_0'].append(metrics.get('mean_act_proto_0', 0))
                    combined['mean_act_proto_1'].append(metrics.get('mean_act_proto_1', 0))
                    combined['mean_act_proto_2'].append(metrics.get('mean_act_proto_2', 0))
                    combined['coverage_proto_0'].append(metrics.get('coverage_proto_0', 0))
                    combined['coverage_proto_1'].append(metrics.get('coverage_proto_1', 0))
                    combined['coverage_proto_2'].append(metrics.get('coverage_proto_2', 0))

            # Record phase boundary
            if phase_entries and phase_idx < 2:  # Don't add boundary after last phase
                combined['phase_boundaries'].append(epoch_counter)

        return combined
