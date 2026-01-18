import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class ResidualBlock3D(layers.Layer):
    """Basic 3D residual block with two conv layers and skip connection."""

    def __init__(self, filters, strides=1, downsample=False, **kwargs):
        super(ResidualBlock3D, self).__init__(**kwargs)
        self.filters = filters
        self.strides = strides
        self.downsample = downsample

        self.conv1 = layers.Conv3D(
            filters, kernel_size=3, strides=strides, padding='same',
            use_bias=False, kernel_initializer='he_normal'
        )
        self.bn1 = layers.BatchNormalization()

        self.conv2 = layers.Conv3D(
            filters, kernel_size=3, strides=1, padding='same',
            use_bias=False, kernel_initializer='he_normal'
        )
        self.bn2 = layers.BatchNormalization()

        if downsample:
            self.shortcut = keras.Sequential([
                layers.Conv3D(
                    filters, kernel_size=1, strides=strides, padding='same',
                    use_bias=False, kernel_initializer='he_normal'
                ),
                layers.BatchNormalization()
            ])
        else:
            self.shortcut = None

        self.relu = layers.ReLU()

    def call(self, x, training=False):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out, training=training)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out, training=training)

        if self.shortcut is not None:
            identity = self.shortcut(identity, training=training)

        out = out + identity
        out = self.relu(out)

        return out


class ResNet3D(keras.Model):
    """
    3D ResNet backbone for medical image segmentation.

    Architecture follows ResNet-18 pattern adapted for 3D:
    - Initial conv + pool: 160x192x128 -> 80x96x64
    - Stage 1 (64 ch): 80x96x64 -> 80x96x64
    - Stage 2 (128 ch): 80x96x64 -> 40x48x32
    - Stage 3 (256 ch): 40x48x32 -> 20x24x16
    - Stage 4 (512 ch): 20x24x16 -> 20x24x16

    Output: (B, 20, 24, 16, 512) - 1/8 spatial resolution
    """

    def __init__(self, in_channels=4, base_channels=64, **kwargs):
        super(ResNet3D, self).__init__(**kwargs)
        self.in_channels = in_channels
        self.base_channels = base_channels

        # Initial convolution block
        # Input: (B, 160, 192, 128, 4)
        self.conv1 = layers.Conv3D(
            base_channels, kernel_size=7, strides=2, padding='same',
            use_bias=False, kernel_initializer='he_normal', name='conv1'
        )
        self.bn1 = layers.BatchNormalization(name='bn1')
        self.relu = layers.ReLU(name='relu1')
        # Output: (B, 80, 96, 64, 64)

        # Residual stages
        # Stage 1: (B, 80, 96, 64, 64) -> (B, 80, 96, 64, 64)
        self.stage1 = self._make_stage(base_channels, num_blocks=2, stride=1, name='stage1')

        # Stage 2: (B, 80, 96, 64, 64) -> (B, 40, 48, 32, 128)
        self.stage2 = self._make_stage(base_channels * 2, num_blocks=2, stride=2, name='stage2')

        # Stage 3: (B, 40, 48, 32, 128) -> (B, 20, 24, 16, 256)
        self.stage3 = self._make_stage(base_channels * 4, num_blocks=2, stride=2, name='stage3')

        # Stage 4: (B, 20, 24, 16, 256) -> (B, 20, 24, 16, 512)
        self.stage4 = self._make_stage(base_channels * 8, num_blocks=2, stride=1, name='stage4')

    def _make_stage(self, filters, num_blocks, stride, name):
        """Create a stage with multiple residual blocks."""
        blocks = []

        # First block may downsample
        downsample = (stride != 1) or (filters != self.base_channels)
        blocks.append(ResidualBlock3D(
            filters, strides=stride, downsample=downsample, name=f'{name}_block1'
        ))

        # Remaining blocks
        for i in range(1, num_blocks):
            blocks.append(ResidualBlock3D(
                filters, strides=1, downsample=False, name=f'{name}_block{i+1}'
            ))

        return blocks

    def call(self, x, training=False):
        # Initial conv block
        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = self.relu(x)
        # (B, 80, 96, 64, 64)

        # Residual stages
        for block in self.stage1:
            x = block(x, training=training)
        # (B, 80, 96, 64, 64)

        for block in self.stage2:
            x = block(x, training=training)
        # (B, 40, 48, 32, 128)

        for block in self.stage3:
            x = block(x, training=training)
        # (B, 20, 24, 16, 256)

        for block in self.stage4:
            x = block(x, training=training)
        # (B, 20, 24, 16, 512)

        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            'in_channels': self.in_channels,
            'base_channels': self.base_channels
        })
        return config


def create_resnet3d_backbone(in_channels=4, base_channels=64):
    """
    Factory function to create ResNet3D backbone.

    Args:
        in_channels: Number of input channels (default 4 for MRI modalities)
        base_channels: Base number of channels (default 64)

    Returns:
        ResNet3D model

    Example:
        backbone = create_resnet3d_backbone(in_channels=4)
        # Input: (B, 160, 192, 128, 4)
        # Output: (B, 20, 24, 16, 512)
    """
    return ResNet3D(in_channels=in_channels, base_channels=base_channels)