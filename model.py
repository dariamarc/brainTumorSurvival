import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


def features_imagenet1k_keras(model_name):
    """
    Placeholder for loading a 3D feature extractor.
    Creates a proper 3D encoder that matches the expected dimensions.
    """
    if model_name == 'resnet50_ri':
        input_tensor = keras.Input(shape=(None, None, None, 4))  # (D, H, W, C_in)

        # Encoder Block 1 - keep full resolution for skip connection
        conv1 = layers.Conv3D(32, 3, activation='relu', padding='same')(input_tensor)
        # (160, 240, 240, 32)

        # Pool to reduce spatial dimensions for conv2
        pool1 = layers.MaxPool3D(pool_size=(1, 2, 2))(conv1)  # (160, 120, 120, 32)

        # Encoder Block 2
        conv2 = layers.Conv3D(64, 3, activation='relu', padding='same')(pool1)
        # (160, 120, 120, 64)

        # Pool to reduce spatial dimensions for conv3
        pool2 = layers.MaxPool3D(pool_size=(1, 2, 2))(conv2)  # (160, 60, 60, 64)

        # Encoder Block 3
        conv3 = layers.Conv3D(128, 3, activation='relu', padding='same')(pool2)
        # (160, 60, 60, 128)

        # Pool to reduce spatial dimensions for bottleneck
        pool3 = layers.MaxPool3D(pool_size=(1, 2, 2))(conv3)  # (160, 30, 30, 128)

        # Encoder Block 4 (Bottleneck)
        conv4 = layers.Conv3D(128, 3, activation='relu', padding='same')(pool3)
        # (160, 30, 30, 128) - this is the bottleneck

        # Return 4 outputs for U-Net skip connections
        return keras.Model(inputs=input_tensor, outputs=[conv1, conv2, conv3, conv4])
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

        print("Encoder output shapes:")
        for i, output in enumerate(dummy_encoder_outputs):
            print(f"  Level {i + 1}: {output.shape}")

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

        # Prototype-to-features layer (maps prototype similarities to feature space)
        self.prototype_to_features = layers.Conv3D(
            128, kernel_size=1,
            activation='relu',
            padding='same',
            name='prototype_to_features',
            kernel_initializer='he_normal',
            bias_initializer='zeros'
        )

        # Decoder blocks with correct upsampling
        self.upsample_block1 = self._build_decoder_block(conv3_channels)  # 128 channels
        self.upsample_block2 = self._build_decoder_block(conv2_channels)  # 64 channels
        self.upsample_block3 = self._build_decoder_block(conv1_channels)  # 32 channels
        self.upsample_block4 = self._build_decoder_block(16)  # 16 channels

        # --- FINAL PROCESSING AND SEGMENTATION HEAD ---
        self.final_conv = layers.Conv3D(16, kernel_size=3, padding='same', activation='relu',
                                        name='final_processing',
                                        kernel_initializer='he_normal', bias_initializer='zeros')

        self.segmentation_head = layers.Conv3D(self.num_classes, kernel_size=1, activation=None,
                                               name='segmentation_output',
                                               kernel_initializer='glorot_uniform',
                                               bias_initializer='zeros')

    def _build_decoder_block(self, output_channels):
        """Build a decoder block that upsamples by 2x in spatial dimensions only"""
        return keras.Sequential([
            # Upsample by 2x in H and W dimensions only, keep D dimension same
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

        f_level1 = encoder_outputs[0]  # (B, 160, 240, 240, 32)
        f_level2 = encoder_outputs[1]  # (B, 160, 120, 120, 64)
        f_level3 = encoder_outputs[2]  # (B, 160, 60, 60, 128)
        f_bottleneck = encoder_outputs[3]  # (B, 160, 30, 30, 128)

        # Process bottleneck features
        f_processed = self.add_ons(f_bottleneck, training=training)  # (B, 160, 30, 30, 128)

        # Prototype learning - NOW INTEGRATED INTO DECODER!
        if self.f_dist == 'l2':
            prototype_voxel_distances = self.l2_convolution_3D(f_processed)
            prototype_voxel_similarities = self.distance_2_similarity(prototype_voxel_distances)
        elif self.f_dist == 'cosine':
            prototype_voxel_similarities = self.cosine_convolution_3D(f_processed)
        else:
            raise NotImplementedError

        # Map prototype similarities to feature space
        # Shape: (B, D, H, W, num_prototypes) -> (B, D, H, W, 128)
        prototype_features = self.prototype_to_features(prototype_voxel_similarities, training=training)

        # Decoder path - progressive upsampling using prototype features
        up = prototype_features  # Start: (B, 160, 30, 30, 128) - NOW USES PROTOTYPES!

        # Upsample 1: 30x30 -> 60x60
        up = self.upsample_block1(up, training=training)  # (B, 160, 60, 60, 128)
        up = layers.concatenate([up, f_level3], axis=-1)  # (B, 160, 60, 60, 256)

        # Upsample 2: 60x60 -> 120x120
        up = self.upsample_block2(up, training=training)  # (B, 160, 120, 120, 64)
        up = layers.concatenate([up, f_level2], axis=-1)  # (B, 160, 120, 120, 128)

        # Upsample 3: 120x120 -> 240x240
        up = self.upsample_block3(up, training=training)  # (B, 160, 240, 240, 32)
        up = layers.concatenate([up, f_level1], axis=-1)  # (B, 160, 240, 240, 64)

        # Final processing - no upsampling needed, just process features
        final_features = self.final_conv(up, training=training)  # (B, 160, 240, 240, 16)

        # Segmentation head
        segmentation_logits = self.segmentation_head(final_features, training=training)  # (B, 160, 240, 240, 3)

        return segmentation_logits