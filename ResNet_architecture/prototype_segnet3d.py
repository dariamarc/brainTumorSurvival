import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from backbone_resnet3d import ResNet3D
from aspp3d import ASPP3D
from prototype_layer import PrototypeLayer
from interpretable_classifier import InterpretableClassifier


class PrototypeSegNet3D(keras.Model):
    """
    Prototype-based 3D segmentation network for brain tumor segmentation.

    Architecture:
        Input (B, 160, 192, 128, 4)
            ↓
        ResNet3D backbone → (B, 20, 24, 16, 512)
            ↓
        ASPP3D → (B, 20, 24, 16, 256)
            ↓
        PrototypeLayer → (B, 20, 24, 16, 3) [similarity maps]
            ↓
        Trilinear upsample → (B, 160, 192, 128, 3)
            ↓
        1×1×1 Conv → (B, 160, 192, 128, 4) [class probabilities]

    Returns:
        segmentation_logits: (B, D, H, W, num_classes)
        similarity_maps: (B, D, H, W, n_prototypes)
    """

    def __init__(self,
                 input_shape=(160, 192, 128, 4),
                 num_classes=4,
                 n_prototypes=3,
                 backbone_channels=64,
                 aspp_out_channels=256,
                 dilation_rates=(2, 4, 8),
                 distance_type='l2',
                 activation_function='log',
                 **kwargs):
        super(PrototypeSegNet3D, self).__init__(**kwargs)

        self.input_shape_model = input_shape
        self.num_classes = num_classes
        self.n_prototypes = n_prototypes

        # 1. ResNet3D Backbone
        self.backbone = ResNet3D(
            in_channels=input_shape[-1],
            base_channels=backbone_channels,
            name='resnet3d_backbone'
        )
        # Output: (B, D/8, H/8, W/8, 512)

        # 2. ASPP for multi-scale context
        backbone_out_channels = backbone_channels * 8  # 512
        self.aspp = ASPP3D(
            in_channels=backbone_out_channels,
            out_channels=aspp_out_channels,
            dilation_rates=dilation_rates,
            name='aspp3d'
        )
        # Output: (B, D/8, H/8, W/8, 256)

        # 3. Prototype Layer
        self.prototype_layer = PrototypeLayer(
            n_prototypes=n_prototypes,
            prototype_dim=aspp_out_channels,
            distance_type=distance_type,
            activation_function=activation_function,
            name='prototype_layer'
        )
        # Output: (B, D/8, H/8, W/8, n_prototypes)

        # 4. Interpretable classifier (preserves prototype-class relationships)
        self.classifier = InterpretableClassifier(
            n_prototypes=n_prototypes,
            n_classes=num_classes,
            name='interpretable_classifier'
        )
        # Output: (B, D, H, W, num_classes)

    def call(self, inputs, training=False):
        # Store original spatial dimensions for upsampling
        input_shape = tf.shape(inputs)
        original_size = input_shape[1:4]  # (D, H, W)

        # 1. Extract features with backbone
        features = self.backbone(inputs, training=training)
        # (B, 20, 24, 16, 512)

        # 2. Multi-scale context with ASPP
        features = self.aspp(features, training=training)
        # (B, 20, 24, 16, 256)

        # 3. Compute prototype similarities
        similarity_maps = self.prototype_layer(features, training=training)
        # (B, 20, 24, 16, 3)

        # 4. Upsample similarity maps to original resolution
        similarity_maps_upsampled = self._trilinear_upsample(
            similarity_maps, original_size
        )
        # (B, 160, 192, 128, 3)

        # 5. Final classification
        segmentation_logits = self.classifier(similarity_maps_upsampled)
        # (B, 160, 192, 128, 4)

        return segmentation_logits, similarity_maps_upsampled

    def call_with_features(self, inputs, training=False):
        """
        Forward pass that also returns intermediate ASPP features.
        Used during training for computing clustering loss.

        Args:
            inputs: (B, D, H, W, C) input tensor
            training: whether in training mode

        Returns:
            segmentation_logits: (B, D, H, W, num_classes)
            similarity_maps: (B, D, H, W, n_prototypes) upsampled
            aspp_features: (B, D/8, H/8, W/8, 256) features at reduced resolution
        """
        input_shape = tf.shape(inputs)
        original_size = input_shape[1:4]

        # 1. Extract features with backbone
        backbone_features = self.backbone(inputs, training=training)

        # 2. Multi-scale context with ASPP
        aspp_features = self.aspp(backbone_features, training=training)

        # 3. Compute prototype similarities
        similarity_maps = self.prototype_layer(aspp_features, training=training)

        # 4. Upsample similarity maps to original resolution
        similarity_maps_upsampled = self._trilinear_upsample(
            similarity_maps, original_size
        )

        # 5. Final classification
        segmentation_logits = self.classifier(similarity_maps_upsampled)

        return segmentation_logits, similarity_maps_upsampled, aspp_features

    def _trilinear_upsample(self, x, target_size):
        """
        Upsample 3D tensor using trilinear interpolation.

        Args:
            x: (B, D, H, W, C) tensor
            target_size: (D', H', W') target spatial dimensions

        Returns:
            Upsampled tensor (B, D', H', W', C)
        """
        # TensorFlow doesn't have direct 3D resize, so we do it in two steps
        # First resize D dimension, then resize H, W

        current_shape = tf.shape(x)
        batch_size = current_shape[0]
        channels = current_shape[4]

        # Reshape to (B*D, H, W, C) for 2D resize of H, W
        x_reshaped = tf.reshape(x, [-1, current_shape[2], current_shape[3], channels])

        # Resize H, W dimensions
        x_resized_hw = tf.image.resize(
            x_reshaped,
            [target_size[1], target_size[2]],
            method='bilinear'
        )
        # (B*D, H', W', C)

        # Reshape back to (B, D, H', W', C)
        x_resized_hw = tf.reshape(
            x_resized_hw,
            [batch_size, current_shape[1], target_size[1], target_size[2], channels]
        )

        # Now resize D dimension
        # Transpose to (B, H', W', D, C), reshape to (B*H'*W', D, C)
        x_transposed = tf.transpose(x_resized_hw, [0, 2, 3, 1, 4])
        x_for_d_resize = tf.reshape(
            x_transposed,
            [-1, current_shape[1], channels]
        )

        # Add dummy dimension for 2D resize: (B*H'*W', D, 1, C)
        x_for_d_resize = tf.expand_dims(x_for_d_resize, axis=2)

        # Resize D dimension
        x_resized_d = tf.image.resize(
            x_for_d_resize,
            [target_size[0], 1],
            method='bilinear'
        )
        # (B*H'*W', D', 1, C)

        # Remove dummy dimension and reshape back
        x_resized_d = tf.squeeze(x_resized_d, axis=2)
        # (B*H'*W', D', C)

        x_resized_d = tf.reshape(
            x_resized_d,
            [batch_size, target_size[1], target_size[2], target_size[0], channels]
        )
        # (B, H', W', D', C)

        # Transpose back to (B, D', H', W', C)
        output = tf.transpose(x_resized_d, [0, 3, 1, 2, 4])

        return output

    def get_features(self, inputs, training=False):
        """Extract features before prototype layer (for initialization)."""
        features = self.backbone(inputs, training=training)
        features = self.aspp(features, training=training)
        return features

    def get_prototypes(self):
        """Return learned prototype vectors."""
        return self.prototype_layer.get_prototypes()

    def set_prototypes(self, prototypes):
        """Set prototype vectors (for data-driven initialization)."""
        self.prototype_layer.set_prototypes(prototypes)

    def get_prototype_contributions(self):
        """Get softmax-normalized weights showing prototype contributions per class."""
        return self.classifier.get_prototype_contributions()

    def get_classifier_weights(self):
        """Get raw classifier weights for inspection."""
        return self.classifier.get_weights_matrix()

    def get_config(self):
        config = super().get_config()
        config.update({
            'input_shape': self.input_shape_model,
            'num_classes': self.num_classes,
            'n_prototypes': self.n_prototypes
        })
        return config


def create_prototype_segnet3d(
        input_shape=(160, 192, 128, 4),
        num_classes=4,
        n_prototypes=3,
        backbone_channels=64,
        aspp_out_channels=256,
        dilation_rates=(2, 4, 8),
        distance_type='l2',
        activation_function='log'):
    """
    Factory function to create PrototypeSegNet3D.

    Args:
        input_shape: Input tensor shape (D, H, W, C)
        num_classes: Number of output classes (default 4: background + 3 tumor)
        n_prototypes: Number of prototypes (default 3, one per tumor class)
        backbone_channels: Base channels for ResNet (default 64)
        aspp_out_channels: ASPP output channels (default 256)
        dilation_rates: ASPP dilation rates (default (2, 4, 8))
        distance_type: 'l2' or 'cosine'
        activation_function: 'log', 'linear', or 'exp'

    Returns:
        PrototypeSegNet3D model
    """
    return PrototypeSegNet3D(
        input_shape=input_shape,
        num_classes=num_classes,
        n_prototypes=n_prototypes,
        backbone_channels=backbone_channels,
        aspp_out_channels=aspp_out_channels,
        dilation_rates=dilation_rates,
        distance_type=distance_type,
        activation_function=activation_function
    )
