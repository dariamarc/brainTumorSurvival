import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class ConvBlock(keras.layers.Layer):
    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = layers.Conv3D(filters, 3, padding='same', use_bias=False,
                                   kernel_initializer='he_normal')
        self.norm1 = layers.LayerNormalization()
        self.relu1 = layers.ReLU()
        self.conv2 = layers.Conv3D(filters, 3, padding='same', use_bias=False,
                                   kernel_initializer='he_normal')
        self.norm2 = layers.LayerNormalization()
        self.relu2 = layers.ReLU()

    def call(self, x):
        x = self.relu1(self.norm1(self.conv1(x)))
        x = self.relu2(self.norm2(self.conv2(x)))
        return x


class EncoderBlock(keras.layers.Layer):
    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.conv = ConvBlock(filters)
        self.pool = layers.MaxPool3D(pool_size=(1, 2, 2))

    def call(self, x):
        skip = self.conv(x)
        return skip, self.pool(skip)


class DecoderBlock(keras.layers.Layer):
    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.upsample = layers.Conv3DTranspose(filters, kernel_size=(1, 2, 2),
                                               strides=(1, 2, 2), padding='same',
                                               kernel_initializer='he_normal')
        self.concat = layers.Concatenate()
        self.conv   = ConvBlock(filters)

    def call(self, x, skip):
        x = self.upsample(x)
        x = self.concat([x, skip])
        return self.conv(x)


class UNet3D(keras.Model):
    """
    Plain 3D U-Net baseline (no prototypes).

    Input:  (B, D, H, W, 4)  — e.g. (1, 128, 192, 160, 4)
    Output: (B, D, H, W, n_classes) — logits

    Pools only H and W (1×2×2 kernels), preserving depth.
    Uses LayerNorm for stability at batch_size=1.

    Encoder channels: base → 2x → 4x
    Bottleneck:       8x
    Decoder channels: 4x → 2x → base
    """

    def __init__(self, n_classes=4, base_channels=32, **kwargs):
        super().__init__(**kwargs)
        c = base_channels
        self.enc1       = EncoderBlock(c)
        self.enc2       = EncoderBlock(c * 2)
        self.enc3       = EncoderBlock(c * 4)
        self.bottleneck = ConvBlock(c * 8)
        self.dropout    = layers.Dropout(0.2)
        self.dec3       = DecoderBlock(c * 4)
        self.dec2       = DecoderBlock(c * 2)
        self.dec1       = DecoderBlock(c)
        self.out_conv   = layers.Conv3D(n_classes, 1, kernel_initializer='glorot_uniform')

    def call(self, inputs, training=False):
        s1, x = self.enc1(inputs)
        s2, x = self.enc2(x)
        s3, x = self.enc3(x)

        x = self.bottleneck(x)
        x = self.dropout(x, training=training)  # only layer that uses training flag

        x = self.dec3(x, s3)
        x = self.dec2(x, s2)
        x = self.dec1(x, s1)

        return self.out_conv(x)
