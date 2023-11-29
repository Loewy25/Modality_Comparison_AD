from tensorflow.keras.layers import Conv3D, Input, LeakyReLU, Add, GlobalAveragePooling3D, Dense, Dropout, SpatialDropout3D
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
import tensorflow_addons as tfa
import numpy as np
from nilearn.image import resample_img, new_img_like


def pad_image_to_shape(image, target_shape=(128, 128, 128)):
    # Calculate the padding required in each dimension
    padding = [(0, max(target_shape[dim] - image.shape[dim], 0)) for dim in range(3)]
    
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



def convolution_block(x, filters, kernel_size=(3,3,3), strides=(1,1,1)):
    x = Conv3D(filters, kernel_size, strides=strides, padding='same', kernel_regularizer=l2(1e-5))(x)
    x = tfa.layers.InstanceNormalization()(x)
    x = LeakyReLU()(x)
    return x

def context_module(x, filters):
    # First convolution block
    x = convolution_block(x, filters)
    # Dropout layer
    x = SpatialDropout3D(0.3)(x) 
    # Second convolution block
    x = convolution_block(x, filters)
    return x

def create_cnn_model():
    input_img = Input(shape=(128, 128, 128, 1))
    x = convolution_block(input_img, 16, strides=(1,1,1))
    conv1_out = x

    # Context 1
    x = context_module(x, 16)
    x = Add()([x, conv1_out])
    x = convolution_block(x, 32, strides=(2,2,2))
    conv2_out = x

    # Context 2
    x = context_module(x, 32)
    x = Add()([x, conv2_out])
    x = convolution_block(x, 64, strides=(2,2,2))
    conv3_out = x

    # Context 3
    x = context_module(x, 64)
    x = Add()([x, conv3_out])
    x = convolution_block(x, 128, strides=(2,2,2))
    conv4_out = x

    # Context 4
    x = context_module(x, 128)
    x = Add()([x, conv4_out])
    x = convolution_block(x, 256, strides=(2,2,2))
    
    # Context 5
    x = context_module(x, 256)

    # Global Average Pooling
    x = GlobalAveragePooling3D()(x)

    # Dropout layer as described in the paper
    x = SpatialDropout3D(0.3)(x)   # The paper mentioned a dropout layer after GAP

    # Dense layer with 7 output nodes as described in the paper
    output = Dense(2, activation='softmax')(x) 

    model = Model(inputs=input_img, outputs=output)
    model.summary()

    return model
