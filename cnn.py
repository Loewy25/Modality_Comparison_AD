from tensorflow.keras.layers import Conv3D, Input, LeakyReLU, Add, GlobalAveragePooling3D, Dense, Dropout, SpatialDropout3D, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
import tensorflow_addons as tfa
import nibabel as nib

import numpy as np
from nilearn.image import resample_img, new_img_like, reorder_img
from scipy.ndimage import zoom
import tensorflow as tfwh
from tensorflow_addons.layers import InstanceNormalization


def resample_to_spacing(data, original_spacing, new_spacing, interpolation='linear'):
    # Assuming the last dimension is the channel and should not be resampled
    zoom_factors = [o / n for o, n in zip(original_spacing, new_spacing)] + [1]
    return zoom(data, zoom_factors, order=1 if interpolation == 'linear' else 0)


def calculate_origin_offset(new_spacing, original_spacing):
    return [(o - n) / 2 for o, n in zip(original_spacing, new_spacing)]


def pad_image_to_shape(image, target_shape=(128, 128, 128)):
    # Check if the image has a 4th dimension (like a channel)
    has_channel = image.ndim == 4

    # Adjust target shape if the image has a channel dimension
    target_shape_adjusted = target_shape + (image.shape[3],) if has_channel else target_shape

    # Calculate the padding required in each dimension
    padding = [(0, max(target_shape_adjusted[dim] - image.shape[dim], 0)) for dim in range(image.ndim)]

    # Apply zero-padding to the data
    new_data = np.pad(image.get_fdata(), padding, mode='constant', constant_values=0)
    
    # Adjust the affine to account for the new shape
    new_affine = np.copy(image.affine)
    
    # Create and return a new NIfTI-like image with the padded data
    return new_img_like(image, new_data, affine=new_affine)



def resize(image, new_shape=(128, 128, 128), interpolation="linear"):
    # Reorder the image and resample it with the desired interpolation
    image = reorder_img(image, resample=interpolation)
    
    # Calculate the zoom levels needed for the new shape
    zoom_level = np.divide(new_shape, image.shape[:3])
    
    # Calculate the new spacing for the image
    new_spacing = np.divide(image.header.get_zooms()[:3], zoom_level)
    
    # Resample the image data to the new spacing
    new_data = resample_to_spacing(image.get_fdata(), image.header.get_zooms()[:3], new_spacing, 
                                   interpolation=interpolation)
    # Copy and adjust the affine transformation matrix for the new spacing
    new_affine = np.copy(image.affine)
    np.fill_diagonal(new_affine, new_spacing.tolist() + [1])
    new_affine[:3, 3] += calculate_origin_offset(new_spacing, image.header.get_zooms()[:3])
    
    # Create and return a new NIfTI-like image
    return new_img_like(image, new_data, affine=new_affine)


# Define the convolution block with hyperparameter options

def convolution_block(x, filters, kernel_size=(3, 3, 3), strides=(1, 1, 1),
                      regularization_rate=1e-5, normalization_type='instance'):
    x = Conv3D(filters, kernel_size, strides=strides, padding='same',
               kernel_regularizer=l2(regularization_rate))(x)
    if normalization_type == 'instance':
        x = InstanceNormalization()(x)
    elif normalization_type == 'batch':
        x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    return x

# Define the context module with dropout
def context_module(x, filters, dropout_rate=0.3, normalization_type='instance'):
    x = convolution_block(x, filters, normalization_type=normalization_type)
    x = SpatialDropout3D(dropout_rate)(x)
    x = convolution_block(x, filters, normalization_type=normalization_type)
    return x

# Define the CNN model with hyperparameters
class CNNHyperModel(HyperModel):
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def build(self, hp):
        filters = hp.Int('filters', min_value=4, max_value=16, step=4)
        dropout_rate = hp.Float('dropout_rate', min_value=0.3, max_value=0.5, step=0.1)
        regularization_rate = 1e-5
        normalization_type = hp.Choice('normalization_type', ['instance', 'batch'])
        learning_rate = hp.Float('learning_rate', min_value=1e-6, max_value=1e-4, sampling='LOG')
        augmentation_level = hp.Int('augmentation_level', min_value=1, max_value=5, step=1)
        batch_size = hp.Choice('batch_size', values=[5, 10]) 

        inputs = Input(shape=self.input_shape)
        x = convolution_block(inputs, filters=filters, regularization_rate=regularization_rate, normalization_type=normalization_type)
        conv1_out = x

        # Context 1
        x = context_module(x, filters=filters, dropout_rate=dropout_rate, normalization_type=normalization_type)
        x = Add()([x, conv1_out])
        x = convolution_block(x, filters=filters * 2, strides=(2, 2, 2), regularization_rate=regularization_rate, normalization_type=normalization_type)
        conv2_out = x

        # Context 2
        x = context_module(x, filters=filters * 2, dropout_rate=dropout_rate, normalization_type=normalization_type)
        x = Add()([x, conv2_out])
        x = convolution_block(x, filters=filters * 4, strides=(2, 2, 2), regularization_rate=regularization_rate, normalization_type=normalization_type)
        conv3_out = x

        # Context 3
        x = context_module(x, filters=filters * 4, dropout_rate=dropout_rate, normalization_type=normalization_type)
        x = Add()([x, conv3_out])
        x = convolution_block(x, filters=filters * 8, strides=(2, 2, 2), regularization_rate=regularization_rate, normalization_type=normalization_type)
        conv4_out = x

        # Context 4
        x = context_module(x, filters=filters * 8, dropout_rate=dropout_rate, normalization_type=normalization_type)
        x = Add()([x, conv4_out])
        x = convolution_block(x, filters=filters * 16, strides=(2, 2, 2), regularization_rate=regularization_rate, normalization_type=normalization_type)
        
        # Context 5
        x = context_module(x, filters=filters * 16, dropout_rate=dropout_rate, normalization_type=normalization_type)

        # Global Average Pooling and Dropout layer before Dense layer
        x = GlobalAveragePooling3D()(x)
        x = Dropout(dropout_rate)(x)

        # Dense layer with softmax activation
        outputs = Dense(2, activation='softmax')(x)

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=learning_rate),
                      loss='categorical_crossentropy',
                      metrics=['accuracy', AUC(name='auc')])
        return model

