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
    # Dummy function for ResNet 3D feature initialization
    print(f"Initializing dummy ResNet 3D features for {in_channels} channels.")


# --- MProtoNet3D_Segmentation_Keras Class ---
class MProtoNet3D_Segmentation_Keras(keras.Model):
    def __init__(self, in_size=(155, 240, 240, 4), num_classes=3, features='resnet50_ri',
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

        assert self.num_prototypes % self.num_classes == 0, \
            "Number of prototypes must be a multiple of the number of classes"
        self.prototype_class_identity = tf.constant(
            np.eye(self.num_classes).repeat(self.num_prototypes // self.num_classes, axis=0), dtype=tf.float32)
        self.num_prototypes_per_class = self.num_prototypes // self.num_classes

        while len(self.prototype_shape_tuple) < 5:
            self.prototype_shape_tuple += (1,)

        # FIX: Keep prototype_shape_tuple as regular Python tuple for indexing
        # Only create TF constant when needed for TF operations
        self.prototype_shape = tf.constant(self.prototype_shape_tuple, dtype=tf.int32)

        # --- ENCODER (Contracting Path) ---
        self.features_extractor_model = features_imagenet1k_keras(features)

        dummy_input = tf.zeros((1,) + self.input_shape_keras, dtype=tf.float32)
        # Ensure dummy_input is float32
        dummy_encoder_outputs = self.features_extractor_model(dummy_input)

        # --- BOTTLENECK/ADD-ONS (where prototypes might interact) ---
        # FIX: Use prototype_shape_tuple[1] (Python int) instead of self.prototype_shape[1] (TF tensor)
        self.add_ons = keras.Sequential([
            layers.Conv3D(self.prototype_shape_tuple[1], 1, use_bias=True, activation='relu',
                          kernel_initializer='he_normal', bias_initializer='zeros'),
            layers.BatchNormalization(),
            layers.Conv3D(self.prototype_shape_tuple[1], 1, use_bias=True,
                          kernel_initializer='he_normal', bias_initializer='zeros'),  # No activation here
            layers.BatchNormalization()
        ], name="add_ons_bottleneck")

        # Prototypes as learnable variables
        self.prototype_vectors = tf.Variable(tf.random.uniform(self.prototype_shape_tuple), name='prototype_vectors')
        self.ones = tf.constant(np.ones(self.prototype_shape_tuple[1:]), dtype=tf.float32)

        # --- DECODER (Expanding Path) ---
        decoder_ch_1 = dummy_encoder_outputs[1].shape[-1]
        decoder_ch_2 = dummy_encoder_outputs[0].shape[-1]
        decoder_ch_3 = 16

        self.upsample_block1 = self._build_decoder_block(decoder_ch_1)
        self.upsample_block2 = self._build_decoder_block(decoder_ch_2)
        self.upsample_block3 = self._build_decoder_block(decoder_ch_3)

        # --- FINAL SEGMENTATION HEAD ---
        self.segmentation_head = layers.Conv3D(self.num_classes, kernel_size=1, activation=None,
                                               name='segmentation_output',
                                               kernel_initializer='glorot_uniform',
                                               bias_initializer='zeros')  # Common for final linear layer

    def _build_decoder_block(self, output_channels):
        return keras.Sequential([
            layers.Conv3DTranspose(output_channels, kernel_size=2, strides=2, padding='same',
                                   kernel_initializer='he_normal', bias_initializer='zeros'),  # Added initializer
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv3D(output_channels, kernel_size=3, padding='same',
                          kernel_initializer='he_normal', bias_initializer='zeros'),  # Added initializer
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv3D(output_channels, kernel_size=3, padding='same',
                          kernel_initializer='he_normal', bias_initializer='zeros'),  # Added initializer
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

    def _match_spatial_dimensions(self, tensor_to_resize, target_tensor):
        """
        Resize tensor_to_resize to match the spatial dimensions of target_tensor.
        Uses cropping or padding as needed.
        """
        target_shape = tf.shape(target_tensor)
        current_shape = tf.shape(tensor_to_resize)

        # Extract spatial dimensions (D, H, W) - indices 1, 2, 3
        target_d, target_h, target_w = target_shape[1], target_shape[2], target_shape[3]
        current_d, current_h, current_w = current_shape[1], current_shape[2], current_shape[3]

        # Calculate differences
        d_diff = target_d - current_d
        h_diff = target_h - current_h
        w_diff = target_w - current_w

        # Use tf.cond for conditional operations in graph mode
        def pad_depth():
            pad_before = d_diff // 2
            pad_after = d_diff - pad_before
            return tf.pad(tensor_to_resize,
                          [[0, 0], [pad_before, pad_after], [0, 0], [0, 0], [0, 0]])

        def crop_depth():
            crop_before = (-d_diff) // 2
            crop_after = crop_before + target_d
            return tensor_to_resize[:, crop_before:crop_after, :, :, :]

        def no_change_depth():
            return tensor_to_resize

        # Handle depth dimension
        result = tf.cond(d_diff > 0, pad_depth,
                         lambda: tf.cond(d_diff < 0, crop_depth, no_change_depth))

        # For height and width, let's use a simpler approach with tf.image.resize_with_crop_or_pad
        # But since that's for 2D, we'll implement a simpler version using slicing and padding

        # Update current shape after depth adjustment
        current_shape = tf.shape(result)
        current_h, current_w = current_shape[2], current_shape[3]
        h_diff = target_h - current_h
        w_diff = target_w - current_w

        def pad_height():
            pad_before = h_diff // 2
            pad_after = h_diff - pad_before
            return tf.pad(result,
                          [[0, 0], [0, 0], [pad_before, pad_after], [0, 0], [0, 0]])

        def crop_height():
            crop_before = (-h_diff) // 2
            crop_after = crop_before + target_h
            return result[:, :, crop_before:crop_after, :, :]

        def no_change_height():
            return result

        # Handle height dimension
        result = tf.cond(h_diff > 0, pad_height,
                         lambda: tf.cond(h_diff < 0, crop_height, no_change_height))

        # Update current shape after height adjustment
        current_shape = tf.shape(result)
        current_w = current_shape[3]
        w_diff = target_w - current_w

        def pad_width():
            pad_before = w_diff // 2
            pad_after = w_diff - pad_before
            return tf.pad(result,
                          [[0, 0], [0, 0], [0, 0], [pad_before, pad_after], [0, 0]])

        def crop_width():
            crop_before = (-w_diff) // 2
            crop_after = crop_before + target_w
            return result[:, :, :, crop_before:crop_after, :]

        def no_change_width():
            return result

        # Handle width dimension
        result = tf.cond(w_diff > 0, pad_width,
                         lambda: tf.cond(w_diff < 0, crop_width, no_change_width))

        return result

    def call(self, inputs, training=False):
        encoder_outputs = self.features_extractor_model(inputs, training=training)

        f_level1 = encoder_outputs[0]
        f_level2 = encoder_outputs[1]
        f_bottleneck = encoder_outputs[2]

        f_processed = self.add_ons(f_bottleneck, training=training)

        if self.f_dist == 'l2':
            prototype_voxel_distances = self.l2_convolution_3D(f_processed)
            prototype_voxel_similarities = self.distance_2_similarity(prototype_voxel_distances)
        elif self.f_dist == 'cosine':
            prototype_voxel_similarities = self.cosine_convolution_3D(f_processed)
        else:
            raise NotImplementedError

        up = f_processed

        up = self.upsample_block1(up, training=training)
        # Match spatial dimensions before concatenation
        f_level2_matched = self._match_spatial_dimensions(f_level2, up)
        up = layers.concatenate([up, f_level2_matched], axis=-1)

        up = self.upsample_block2(up, training=training)
        # Match spatial dimensions before concatenation
        f_level1_matched = self._match_spatial_dimensions(f_level1, up)
        up = layers.concatenate([up, f_level1_matched], axis=-1)

        final_features = self.upsample_block3(up, training=training)

        segmentation_logits = self.segmentation_head(final_features, training=training)

        return segmentation_logits
