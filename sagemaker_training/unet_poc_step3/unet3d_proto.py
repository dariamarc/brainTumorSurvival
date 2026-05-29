import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ── Shared building blocks ────────────────────────────────────────────────────

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


# ── Prototype model ───────────────────────────────────────────────────────────

class UNet3DProto(keras.Model):
    """
    Step 3: UNet3D + prototype bottleneck + prototype learning losses.

    Identical architecture to Step 2.  The only change is that forward_train()
    now returns (logits, distances) so the training loop can compute the three
    prototype losses:

        L_total = L_dice
                + clst_weight  * L_clustering    ← pull each proto toward its class
                - sep_weight   * L_separation     ← push each proto from other classes
                + div_weight   * L_diversity      ← spread same-class protos apart

    Background gets no prototypes.  Prototype k is assigned to tumor class:
        class(k) = k // protos_per_class + 1     (1=NCR, 2=ED, 3=ET)
    """

    TUMOR_CLASSES = [1, 2, 3]

    def __init__(self, n_classes=4, base_channels=16, protos_per_class=3, **kwargs):
        super().__init__(**kwargs)
        c                     = base_channels
        self.protos_per_class = protos_per_class
        self.num_prototypes   = protos_per_class * len(self.TUMOR_CLASSES)
        self.proto_dim        = c * 8

        self.enc1       = EncoderBlock(c)
        self.enc2       = EncoderBlock(c * 2)
        self.enc3       = EncoderBlock(c * 4)
        self.bottleneck = ConvBlock(c * 8)
        self.dropout    = layers.Dropout(0.2)

        self.prototype_vectors = tf.Variable(
            tf.initializers.GlorotUniform()(
                shape=(self.num_prototypes, self.proto_dim, 1, 1, 1)),
            trainable=True, name='prototype_vectors'
        )
        self.prototype_to_features = layers.Conv3D(
            self.proto_dim, kernel_size=1,
            kernel_initializer='zeros', bias_initializer='zeros',
            name='prototype_to_features'
        )

        self.dec3     = DecoderBlock(c * 4)
        self.dec2     = DecoderBlock(c * 2)
        self.dec1     = DecoderBlock(c)
        self.out_conv = layers.Conv3D(n_classes, 1, kernel_initializer='glorot_uniform')

    # ── Prototype ops ─────────────────────────────────────────────────────────

    def _l2_distances(self, x):
        """x: (B,D,H,W,C) → (B,D,H,W,P)"""
        proto_filters = tf.transpose(self.prototype_vectors, perm=[2, 3, 4, 1, 0])
        dot = tf.nn.conv3d(x, filters=proto_filters,
                           strides=[1, 1, 1, 1, 1], padding='SAME')
        x2  = tf.reduce_sum(tf.square(x), axis=-1, keepdims=True)
        p2  = tf.reshape(
            tf.reduce_sum(tf.square(self.prototype_vectors), axis=[1, 2, 3, 4]),
            [1, 1, 1, 1, -1])
        return tf.sqrt(tf.maximum(x2 - 2.0 * dot + p2, 1e-8))

    def _similarities(self, distances):
        return tf.math.log((distances + 1.0) / (distances + 1e-4))

    def _proto_bottleneck(self, x):
        """Returns (augmented_x, similarities, distances)."""
        distances    = self._l2_distances(x)
        similarities = self._similarities(distances)
        proto_feat   = self.prototype_to_features(similarities)
        return x + proto_feat, similarities, distances

    def _decode(self, x, s1, s2, s3, training):
        x = self.dropout(x, training=training)
        x = self.dec3(x, s3)
        x = self.dec2(x, s2)
        x = self.dec1(x, s1)
        return self.out_conv(x)

    # ── Forward passes ────────────────────────────────────────────────────────

    def call(self, inputs, training=False):
        s1, x = self.enc1(inputs)
        s2, x = self.enc2(x)
        s3, x = self.enc3(x)
        x = self.bottleneck(x)
        x, _, _ = self._proto_bottleneck(x)
        return self._decode(x, s1, s2, s3, training)

    def forward_train(self, inputs):
        """Returns (logits, distances) — distances needed for prototype losses."""
        s1, x = self.enc1(inputs)
        s2, x = self.enc2(x)
        s3, x = self.enc3(x)
        x = self.bottleneck(x)
        x, _, distances = self._proto_bottleneck(x)
        return self._decode(x, s1, s2, s3, training=True), distances

    def forward_with_similarities(self, inputs):
        """Returns (logits, similarities) — used for activation ratio monitoring."""
        s1, x = self.enc1(inputs)
        s2, x = self.enc2(x)
        s3, x = self.enc3(x)
        x = self.bottleneck(x)
        x, similarities, _ = self._proto_bottleneck(x)
        return self._decode(x, s1, s2, s3, training=False), similarities
