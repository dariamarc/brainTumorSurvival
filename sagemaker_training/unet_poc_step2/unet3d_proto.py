import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ── Shared building blocks (identical to Step 1) ──────────────────────────────

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


# ── Step 1 model — used only to load pretrained weights ───────────────────────

class UNet3D(keras.Model):
    """Plain UNet3D (Step 1 architecture). Instantiated only to load weights."""

    def __init__(self, n_classes=4, base_channels=16, **kwargs):
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
        x = self.dropout(x, training=training)
        x = self.dec3(x, s3)
        x = self.dec2(x, s2)
        x = self.dec1(x, s1)
        return self.out_conv(x)


# ── Prototype bottleneck ───────────────────────────────────────────────────────

class UNet3DProto(keras.Model):
    """
    Step 2: UNet3D + prototype layer at bottleneck.

    Architecture is identical to Step 1 except the bottleneck is augmented:

        bottleneck_features           (B, D, H/8, W/8, 8c)
              ↓
        L2 distances to prototypes    (B, D, H/8, W/8, P)   P = protos_per_class * 3
              ↓
        log similarity                (B, D, H/8, W/8, P)
              ↓
        prototype_to_features (1x1)   (B, D, H/8, W/8, 8c)   kernel init = zeros
              ↓
        element-wise add with bottleneck
              ↓ decoder (unchanged)

    prototype_to_features is zero-initialized so the prototype contribution
    starts at zero, matching Step 1 behaviour at epoch 0.

    Background class gets no prototypes; prototypes are assigned to
    tumor classes only: class 1 (NCR), class 2 (ED), class 3 (ET).
    """

    # prototype class assignment: proto k belongs to tumor class (k // protos_per_class) + 1
    TUMOR_CLASSES = [1, 2, 3]  # NCR, ED, ET

    def __init__(self, n_classes=4, base_channels=16, protos_per_class=3, **kwargs):
        super().__init__(**kwargs)
        c                       = base_channels
        self.protos_per_class   = protos_per_class
        self.num_prototypes     = protos_per_class * len(self.TUMOR_CLASSES)  # 9
        self.proto_dim          = c * 8   # bottleneck channels (128 with base=16)

        # Encoder
        self.enc1 = EncoderBlock(c)
        self.enc2 = EncoderBlock(c * 2)
        self.enc3 = EncoderBlock(c * 4)

        # Bottleneck conv
        self.bottleneck = ConvBlock(c * 8)
        self.dropout    = layers.Dropout(0.2)

        # Prototype vectors: (P, proto_dim, 1, 1, 1)
        self.prototype_vectors = tf.Variable(
            tf.initializers.GlorotUniform()(shape=(self.num_prototypes, self.proto_dim, 1, 1, 1)),
            trainable=True,
            name='prototype_vectors'
        )

        # Maps similarity scores → bottleneck feature space.
        # Zero kernel: prototype contribution is 0 at init, grows during training.
        self.prototype_to_features = layers.Conv3D(
            self.proto_dim, kernel_size=1,
            kernel_initializer='zeros',
            bias_initializer='zeros',
            name='prototype_to_features'
        )

        # Decoder
        self.dec3     = DecoderBlock(c * 4)
        self.dec2     = DecoderBlock(c * 2)
        self.dec1     = DecoderBlock(c)
        self.out_conv = layers.Conv3D(n_classes, 1, kernel_initializer='glorot_uniform')

    # ── Prototype ops ──────────────────────────────────────────────────────────

    def _l2_distances(self, x):
        """
        x: (B, D, H, W, C)
        returns: (B, D, H, W, P)  — L2 distance from each voxel to each prototype
        """
        # proto_filters: (1, 1, 1, C, P) for conv3d
        proto_filters = tf.transpose(self.prototype_vectors, perm=[2, 3, 4, 1, 0])

        dot = tf.nn.conv3d(x, filters=proto_filters,
                           strides=[1, 1, 1, 1, 1], padding='SAME')   # (B, D, H, W, P)
        x2  = tf.reduce_sum(tf.square(x), axis=-1, keepdims=True)     # (B, D, H, W, 1)
        p2  = tf.reduce_sum(tf.square(self.prototype_vectors),
                            axis=[1, 2, 3, 4])                         # (P,)
        p2  = tf.reshape(p2, [1, 1, 1, 1, -1])                        # (1,1,1,1,P)

        dist = x2 - 2.0 * dot + p2
        dist = tf.maximum(dist, 1e-8)
        return tf.sqrt(dist)                                           # (B, D, H, W, P)

    def _similarities(self, distances):
        """log((d+1)/(d+eps)) — large when d is small (voxel close to prototype)."""
        return tf.math.log((distances + 1.0) / (distances + 1e-4))

    def _bottleneck_with_protos(self, x):
        """
        Runs the prototype layer and returns (augmented_features, similarities).
        Separated so we can call it from both call() and forward_with_similarities().
        """
        distances    = self._l2_distances(x)           # (B, D, H/8, W/8, P)
        similarities = self._similarities(distances)    # (B, D, H/8, W/8, P)
        proto_feat   = self.prototype_to_features(similarities)  # (B, D, H/8, W/8, 8c)
        return x + proto_feat, similarities

    # ── Forward pass ──────────────────────────────────────────────────────────

    def call(self, inputs, training=False):
        s1, x = self.enc1(inputs)
        s2, x = self.enc2(x)
        s3, x = self.enc3(x)

        x = self.bottleneck(x)
        x, _ = self._bottleneck_with_protos(x)
        x = self.dropout(x, training=training)

        x = self.dec3(x, s3)
        x = self.dec2(x, s2)
        x = self.dec1(x, s1)
        return self.out_conv(x)

    def forward_with_similarities(self, inputs):
        """Returns (logits, similarities) in one pass — used for prototype monitoring."""
        s1, x = self.enc1(inputs)
        s2, x = self.enc2(x)
        s3, x = self.enc3(x)

        x = self.bottleneck(x)
        x, similarities = self._bottleneck_with_protos(x)
        x = self.dropout(x, training=False)

        x = self.dec3(x, s3)
        x = self.dec2(x, s2)
        x = self.dec1(x, s1)
        return self.out_conv(x), similarities   # similarities: (B, D, H/8, W/8, P)
