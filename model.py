import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


def features_imagenet1k_keras(model_name):
    """
    Placeholder for loading a 3D feature extractor.
    In a real scenario, this would load a 3D CNN backbone (e.g., a 3D ResNet, or adapt a 2D one).
    For this example, let's create a dummy 3D feature extractor that mimics downsampling.
    """
    if model_name == 'resnet50_ri':
        # This is a VERY simplified dummy encoder to show how intermediate outputs might be obtained.
        # In a real ResNet, you'd extract outputs after specific blocks.

        input_tensor = keras.Input(shape=(None, None, None, 4))  # (D, H, W, C_in)

        # Encoder Block 1 (e.g., initial conv and pool)
        conv1 = layers.Conv3D(32, 3, activation='relu', padding='same')(input_tensor)
        pool1 = layers.MaxPool3D(pool_size=(2, 2, 2))(conv1)  # Downsamples by 2

        # Encoder Block 2
        conv2 = layers.Conv3D(64, 3, activation='relu', padding='same')(pool1)
        pool2 = layers.MaxPool3D(pool_size=(2, 2, 2))(conv2)  # Downsamples by 2 (total 4x)

        # Encoder Block 3 (Bottleneck)
        conv3 = layers.Conv3D(128, 3, activation='relu', padding='same')(pool2)
        # No pool here, this is the bottleneck output

        # Return a model that outputs features from different stages
        # In a real U-Net, you'd carefully select where to get these skip connections.
        return keras.Model(inputs=input_tensor, outputs=[conv1, conv2, conv3])
    else:
        raise ValueError(f"Unknown feature extractor: {model_name}")


def init_resnet3d_features_keras(model, in_channels):
    print(f"Initializing dummy ResNet 3D features for {in_channels} channels.")


class MProtoNet3D_Segmentation_Keras(keras.Model):
    def __init__(self, in_size=(160, 240, 240, 4), num_classes=3, features='resnet50_ri',
                 prototype_shape=(21, 128, 1, 1, 1), init_weights=True, f_dist='l2',
                 prototype_activation_function='log', **kwargs):
        super(MProtoNet3D_Segmentation_Keras, self).__init__(**kwargs)
        self.input_shape_keras = in_size
        self.num_classes = num_classes
        self.prototype_shape_tuple = prototype_shape
        self.num_prototypes = prototype_shape[0]
        self.epsilon = 1e-4
        self.f_dist = f_dist
        self.prototype_activation_function = prototype_activation_function

        # getting the prototype shape and the prototype number per class
        assert self.num_prototypes % self.num_classes == 0, \
            "Number of prototypes must be a multiple of the number of classes"
        self.prototype_class_identity = tf.constant(
            np.eye(self.num_classes).repeat(self.num_prototypes // self.num_classes, axis=0), dtype=tf.float32)
        self.num_prototypes_per_class = self.num_prototypes // self.num_classes

        while len(self.prototype_shape_tuple) < 5:
            self.prototype_shape_tuple += (1,)

        self.prototype_shape = tf.constant(self.prototype_shape_tuple, dtype=tf.int32)

        # --- ENCODER (Contracting Path) ---
        self.features_extractor_model = features_imagenet1k_keras(features)

        dummy_input = tf.zeros((1,) + self.input_shape_keras, dtype=tf.float32)
        dummy_encoder_outputs = self.features_extractor_model(dummy_input)

        # --- BOTTLENECK/ADD-ONS ---
        self.add_ons = keras.Sequential([
            layers.Conv3D(self.prototype_shape_tuple[1], 1, use_bias=True, activation='relu',
                          kernel_initializer='he_normal', bias_initializer='zeros'),
            layers.BatchNormalization(),
            layers.Conv3D(self.prototype_shape_tuple[1], 1, use_bias=True,
                          kernel_initializer='he_normal', bias_initializer='zeros'),
            layers.BatchNormalization()
        ], name="add_ons_bottleneck")

        # Prototypes as learnable variables
        self.prototype_vectors = tf.Variable(tf.random.uniform(self.prototype_shape_tuple), name='prototype_vectors')
        self.ones = tf.constant(np.ones(self.prototype_shape_tuple[1:]), dtype=tf.float32)

        # --- DECODER (Expanding Path) ---
        # Calculate channel numbers from encoder outputs
        conv1_channels = dummy_encoder_outputs[0].shape[-1]  # 32
        conv2_channels = dummy_encoder_outputs[1].shape[-1]  # 64
        conv3_channels = dummy_encoder_outputs[2].shape[-1]  # 128

        # Decoder blocks - designed to reach exact target dimensions
        self.upsample_block1 = self._build_decoder_block(conv3_channels)  # 128 channels
        self.upsample_block2 = self._build_decoder_block(conv2_channels)  # 64 channels
        self.upsample_block3 = self._build_decoder_block(conv1_channels)  # 32 channels
        self.upsample_block4 = self._build_decoder_block(16)  # 16 channels

        # --- FINAL SEGMENTATION HEAD ---
        self.segmentation_head = layers.Conv3D(self.num_classes, kernel_size=1, activation=None,
                                               name='segmentation_output',
                                               kernel_initializer='glorot_uniform',
                                               bias_initializer='zeros')

    def _build_decoder_block(self, output_channels):
        return keras.Sequential([
            layers.Conv3DTranspose(output_channels, kernel_size=(1, 2, 2), strides=(1, 2, 2), padding='same',
                                   kernel_initializer='he_normal', bias_initializer='zeros'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv3D(output_channels, kernel_size=3, padding='same',
                          kernel_initializer='he_normal', bias_initializer='zeros'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv3D(output_channels, kernel_size=3, padding='same',
                          kernel_initializer='he_normal', bias_initializer='zeros'),
            layers.BatchNormalization(),
            layers.ReLU()
        ])

    def l2_convolution_3D(self, x):
        proto_filters = tf.transpose(self.prototype_vectors, perm=[2, 3, 4, 1, 0])
        dot_product = tf.nn.conv3d(x, filters=proto_filters, strides=(1, 1, 1, 1, 1), padding='SAME')
        x2 = tf.reduce_sum(tf.square(x), axis=-1, keepdims=True)
        p2 = tf.reduce_sum(tf.square(self.prototype_vectors), axis=[1, 2, 3, 4], keepdims=True)
        p2 = tf.transpose(p2, perm=[1, 2, 3, 4, 0])
        distances = x2 - 2 * dot_product + p2
        distances = tf.maximum(distances, self.epsilon)
        distances = tf.sqrt(distances)
        return distances

    def cosine_convolution_3D(self, x):
        proto_filters = tf.transpose(self.prototype_vectors, perm=[2, 3, 4, 1, 0])
        dot_product = tf.nn.conv3d(x, filters=proto_filters, strides=(1, 1, 1, 1, 1), padding='SAME')
        x_norm = tf.norm(x, axis=-1, keepdims=True)
        p_norm = tf.norm(self.prototype_vectors, axis=[1, 2, 3, 4], keepdims=True)
        p_norm = tf.transpose(p_norm, perm=[1, 2, 3, 4, 0])
        similarities = dot_product / (x_norm * p_norm + self.epsilon)
        return similarities

    def distance_2_similarity(self, distances):
        if self.prototype_activation_function == 'log':
            return tf.math.log((distances + 1.) / (distances + self.epsilon))
        elif self.prototype_activation_function == 'linear':
            return -distances
        elif self.prototype_activation_function == 'exp':
            return tf.exp(-distances)
        else:
            raise NotImplementedError

    def call(self, inputs, training=False):
        # Get encoder outputs: conv1, conv2, conv3, bottleneck
        encoder_outputs = self.features_extractor_model(inputs, training=training)

        f_level1 = encoder_outputs[0]  # (B, 155, 240, 240, 32)
        f_level2 = encoder_outputs[1]  # (B, 155, 120, 120, 64)
        f_level3 = encoder_outputs[2]  # (B, 155, 60, 60, 128)
        f_bottleneck = encoder_outputs[3]  # (B, 155, 30, 30, 128)

        # Process bottleneck features
        f_processed = self.add_ons(f_bottleneck, training=training)  # (B, 155, 30, 30, 128)

        # Prototype learning (optional - computed but not used in decoder)
        if self.f_dist == 'l2':
            prototype_voxel_distances = self.l2_convolution_3D(f_processed)
            prototype_voxel_similarities = self.distance_2_similarity(prototype_voxel_distances)
        elif self.f_dist == 'cosine':
            prototype_voxel_similarities = self.cosine_convolution_3D(f_processed)
        else:
            raise NotImplementedError

        # Decoder path - progressive upsampling
        up = f_processed  # Start: (B, 155, 30, 30, 128)

        # Upsample 1: 30x30 -> 60x60
        up = self.upsample_block1(up, training=training)  # (B, 155, 60, 60, 128)
        up = layers.concatenate([up, f_level3], axis=-1)  # (B, 155, 60, 60, 256)

        # Upsample 2: 60x60 -> 120x120
        up = self.upsample_block2(up, training=training)  # (B, 155, 120, 120, 64)
        up = layers.concatenate([up, f_level2], axis=-1)  # (B, 155, 120, 120, 128)

        # Upsample 3: 120x120 -> 240x240
        up = self.upsample_block3(up, training=training)  # (B, 155, 240, 240, 32)
        up = layers.concatenate([up, f_level1], axis=-1)  # (B, 155, 240, 240, 64)

        # Final upsample to match exactly
        final_features = self.upsample_block4(up, training=training)  # (B, 155, 240, 240, 16)

        # Segmentation head
        segmentation_logits = self.segmentation_head(final_features, training=training)  # (B, 155, 240, 240, 3)

        return segmentation_logits
