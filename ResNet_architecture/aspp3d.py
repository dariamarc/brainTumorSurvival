import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class ASPP3D(layers.Layer):
    """
    3D Atrous Spatial Pyramid Pooling module for multi-scale context.

    Branches:
    - 1x1x1 convolution (no dilation)
    - 3x3x3 convolution with dilation rate 2
    - 3x3x3 convolution with dilation rate 4
    - 3x3x3 convolution with dilation rate 8
    - Global average pooling branch

    Input: (B, D, H, W, in_channels) - e.g., (B, 20, 24, 16, 512)
    Output: (B, D, H, W, out_channels) - e.g., (B, 20, 24, 16, 256)
    """

    def __init__(self, in_channels, out_channels, dilation_rates=(2, 4, 8), **kwargs):
        super(ASPP3D, self).__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dilation_rates = dilation_rates

        # Branch 1: 1x1x1 convolution
        self.conv1x1 = keras.Sequential([
            layers.Conv3D(out_channels, kernel_size=1, padding='same',
                          use_bias=False, kernel_initializer='he_normal'),
            layers.BatchNormalization(),
            layers.ReLU()
        ], name='aspp_conv1x1')

        # Branch 2: 3x3x3 dilated convolution (rate=2)
        self.conv_d1 = keras.Sequential([
            layers.Conv3D(out_channels, kernel_size=3, padding='same',
                          dilation_rate=dilation_rates[0],
                          use_bias=False, kernel_initializer='he_normal'),
            layers.BatchNormalization(),
            layers.ReLU()
        ], name=f'aspp_conv_d{dilation_rates[0]}')

        # Branch 3: 3x3x3 dilated convolution (rate=4)
        self.conv_d2 = keras.Sequential([
            layers.Conv3D(out_channels, kernel_size=3, padding='same',
                          dilation_rate=dilation_rates[1],
                          use_bias=False, kernel_initializer='he_normal'),
            layers.BatchNormalization(),
            layers.ReLU()
        ], name=f'aspp_conv_d{dilation_rates[1]}')

        # Branch 4: 3x3x3 dilated convolution (rate=8)
        self.conv_d3 = keras.Sequential([
            layers.Conv3D(out_channels, kernel_size=3, padding='same',
                          dilation_rate=dilation_rates[2],
                          use_bias=False, kernel_initializer='he_normal'),
            layers.BatchNormalization(),
            layers.ReLU()
        ], name=f'aspp_conv_d{dilation_rates[2]}')

        # Branch 5: Global average pooling
        # Note: Using LayerNormalization instead of BatchNorm because the input
        # is (B, 1, 1, 1, C) after global pooling - BatchNorm fails with single
        # spatial values (variance computation over 1 element causes NaN)
        self.global_pool_conv = keras.Sequential([
            layers.Conv3D(out_channels, kernel_size=1, padding='same',
                          use_bias=True, kernel_initializer='he_normal'),
            layers.LayerNormalization(),
            layers.ReLU()
        ], name='aspp_global_pool')

        # Fusion: concatenate all branches and reduce
        self.fusion = keras.Sequential([
            layers.Conv3D(out_channels, kernel_size=1, padding='same',
                          use_bias=False, kernel_initializer='he_normal'),
            layers.BatchNormalization(),
            layers.ReLU()
        ], name='aspp_fusion')

    def call(self, x, training=False):
        # Get spatial dimensions for upsampling global pooling branch
        input_shape = tf.shape(x)
        spatial_size = input_shape[1:4]  # (D, H, W)

        # Branch 1: 1x1x1 conv
        x1 = self.conv1x1(x, training=training)

        # Branch 2: dilated conv (rate=2)
        x2 = self.conv_d1(x, training=training)

        # Branch 3: dilated conv (rate=4)
        x3 = self.conv_d2(x, training=training)

        # Branch 4: dilated conv (rate=8)
        x4 = self.conv_d3(x, training=training)

        # Branch 5: global average pooling + upsample
        x5 = tf.reduce_mean(x, axis=[1, 2, 3], keepdims=True)  # (B, 1, 1, 1, C)
        x5 = self.global_pool_conv(x5, training=training)
        x5 = tf.image.resize(
            tf.reshape(x5, [-1, 1, tf.shape(x5)[1] * tf.shape(x5)[2], self.out_channels]),
            [spatial_size[0], spatial_size[1] * spatial_size[2]],
            method='bilinear'
        )
        x5 = tf.reshape(x5, [-1, spatial_size[0], spatial_size[1], spatial_size[2], self.out_channels])

        # Concatenate all branches
        concat = layers.concatenate([x1, x2, x3, x4, x5], axis=-1)

        # Fuse to output channels
        output = self.fusion(concat, training=training)

        return output

    def get_config(self):
        config = super().get_config()
        config.update({
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'dilation_rates': self.dilation_rates
        })
        return config


def create_aspp3d(in_channels=512, out_channels=256, dilation_rates=(2, 4, 8)):
    """
    Factory function to create ASPP3D module.

    Args:
        in_channels: Number of input channels (default 512 from ResNet backbone)
        out_channels: Number of output channels (default 256)
        dilation_rates: Tuple of dilation rates (default (2, 4, 8))

    Returns:
        ASPP3D layer
    """
    return ASPP3D(in_channels, out_channels, dilation_rates)
