import os
import numpy as np
import nibabel as nib
import tensorflow as tf
from tensorflow.keras.layers import (Conv3D, LeakyReLU, Add, Concatenate, 
                                     Conv3DTranspose, GlobalAveragePooling3D, 
                                     Dense, AveragePooling3D)
from tensorflow.keras.models import Model
from tensorflow_addons.layers import InstanceNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from scipy.stats import zscore
from scipy.ndimage import zoom

# Import data loading functions
from data_loading import generate_data_path_less, generate, binarylabel

from tensorflow.keras.layers import Concatenate

class DenseUNetGenerator:
    """Class for the DenseU-Net based Generator with 13 Dense Blocks, feature map concatenation, 7 Transition Layers, and 7 Upsampling Layers."""

    def __init__(self, input_shape=(128, 128, 128, 1)):
        self.input_shape = input_shape
        self.model = self.build_generator()

    def convolution_block(self, x, filters, kernel_size=(3, 3, 3), strides=(1, 1, 1)):
        x = Conv3D(filters, kernel_size, strides=strides, padding='same')(x)
        x = InstanceNormalization()(x)
        x = LeakyReLU(0.2)(x)
        return x

    def dense_block(self, x, filters, num_layers=2):
        concatenated_features = [x]  # Start with the input feature map
        for _ in range(num_layers):
            x = self.convolution_block(x, filters)
            concatenated_features.append(x)  # Append each convolution output to the list
            x = Concatenate()(concatenated_features)  # Concatenate all feature maps so far
        return x

    def transition_layer(self, x, filters):
        x = Conv3D(filters, kernel_size=(1, 1, 1), padding='same')(x)
        x = InstanceNormalization()(x)
        x = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(x)
        return x

    def upsampling_block(self, x, skip, filters):
        x = Conv3DTranspose(filters, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding='same')(x)
        x = Concatenate()([x, skip])  # Skip connection concatenation
        x = InstanceNormalization()(x)
        x = LeakyReLU(0.2)(x)
        return x

    def build_generator(self):
        input_img = tf.keras.Input(shape=self.input_shape)
        skips = []
        
        # Downsampling Path with 7 Dense Blocks and 7 Transition Layers
        x = self.convolution_block(input_img, 64)
        for filters in [64, 128, 256, 512, 512, 512, 512]:  # 7 dense blocks with 7 transitions
            x = self.dense_block(x, filters)
            skips.append(x)
            x = self.transition_layer(x, filters)

        # Upsampling Path with 7 Upsampling Layers and 6 Dense Blocks
        for filters in reversed([512, 512, 512, 512, 256, 128, 64]):
            x = self.upsampling_block(x, skips.pop(), filters)
            x = self.dense_block(x, filters)  # Dense block after each upsampling
        
        # Final convolution to output PET image with Tanh activation
        output = Conv3D(1, kernel_size=(1, 1, 1), activation='tanh')(x)
        return Model(inputs=input_img, outputs=output)

import tensorflow as tf
from tensorflow.keras.layers import (Conv3D, Add, BatchNormalization, ReLU, 
                                     MaxPooling3D, GlobalAveragePooling3D)
from tensorflow.keras.models import Model

class ResNetEncoder:
    """Class for the ResNet-34 Encoder Network following the original architecture."""

    def __init__(self, input_shape=(128, 128, 128, 1)):
        self.input_shape = input_shape
        self.model = self.build_encoder()

    def residual_block(self, x, filters, strides=(1, 1, 1), downsample=False):
        shortcut = x
        if downsample:
            # Adjust the shortcut path when downsampling
            shortcut = Conv3D(filters, kernel_size=(1, 1, 1), strides=strides, padding='same')(x)
            shortcut = BatchNormalization()(shortcut)

        # First convolutional layer
        x = Conv3D(filters, kernel_size=(3, 3, 3), strides=strides, padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        # Second convolutional layer
        x = Conv3D(filters, kernel_size=(3, 3, 3), padding='same')(x)
        x = BatchNormalization()(x)

        # Add the shortcut connection
        x = Add()([shortcut, x])
        x = ReLU()(x)
        return x

    def build_encoder(self):
        input_img = tf.keras.Input(shape=self.input_shape)
        
        # Initial Convolution and Pooling Layer
        x = Conv3D(64, kernel_size=(7, 7, 7), strides=(2, 2, 2), padding='same')(input_img)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = MaxPooling3D(pool_size=(3, 3, 3), strides=(2, 2, 2), padding='same')(x)

        # Residual Blocks for ResNet-34
        # Stage 1: 3 Blocks, 64 Filters
        for _ in range(3):
            x = self.residual_block(x, 64)

        # Stage 2: 4 Blocks, 128 Filters, Downsample on first block
        x = self.residual_block(x, 128, strides=(2, 2, 2), downsample=True)
        for _ in range(3):
            x = self.residual_block(x, 128)

        # Stage 3: 6 Blocks, 256 Filters, Downsample on first block
        x = self.residual_block(x, 256, strides=(2, 2, 2), downsample=True)
        for _ in range(5):
            x = self.residual_block(x, 256)

        # Stage 4: 3 Blocks, 512 Filters, Downsample on first block
        x = self.residual_block(x, 512, strides=(2, 2, 2), downsample=True)
        for _ in range(2):
            x = self.residual_block(x, 512)

        # Global Average Pooling at the end
        x = GlobalAveragePooling3D()(x)
        return Model(inputs=input_img, outputs=x)

# Example of initializing and summarizing the model
input_shape = (128, 128, 128, 1)
encoder = ResNetEncoder(input_shape)
encoder.model.summary()


class Discriminator:
    """Class for the Discriminator network."""

    def __init__(self, input_shape=(128, 128, 128, 1)):
        self.input_shape = input_shape
        self.model = self.build_discriminator()

    def convolution_block(self, x, filters, kernel_size=(3, 3, 3), strides=(2, 2, 2)):
        x = Conv3D(filters, kernel_size, strides=strides, padding='same')(x)
        x = InstanceNormalization()(x)
        x = LeakyReLU(0.2)(x)
        return x

    def build_discriminator(self):
        input_img = tf.keras.Input(shape=self.input_shape)
        x = self.convolution_block(input_img, 32)
        x = self.convolution_block(x, 64)
        x = self.convolution_block(x, 128)
        x = self.convolution_block(x, 256)
        x = GlobalAveragePooling3D()(x)
        output = Dense(1, activation='sigmoid')(x)
        return Model(inputs=input_img, outputs=output)

import tensorflow as tf
from tensorflow.keras.layers import Conv3D, Conv3DTranspose, Add, ReLU, BatchNormalization, GlobalAveragePooling3D, Dense, MaxPooling3D, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError, MeanAbsoluteError
from tensorflow.keras.applications import VGG16

# Helper function to compute perceptual loss
def perceptual_loss(real, generated):
    vgg = VGG16(include_top=False, weights='imagenet', input_shape=real.shape[1:])
    vgg.trainable = False
    real_features = vgg(real)
    generated_features = vgg(generated)
    return tf.reduce_mean(tf.abs(real_features - generated_features))

# BMGAN Class with Integrated Loss Functions
class BMGAN:
    def __init__(self, generator, discriminator, encoder, input_shape=(128, 128, 128, 1), lambda1=10.0, lambda2=0.5):
        self.generator = generator
        self.discriminator = discriminator
        self.encoder = encoder
        self.lambda1 = lambda1  # Weight for L1 Loss
        self.lambda2 = lambda2  # Weight for Perceptual Loss
        self.build_bmgan(input_shape)

    def build_bmgan(self, input_shape):
        # GAN setup
        real_input = tf.keras.Input(shape=input_shape)
        fake_output = self.generator(real_input)
        validity = self.discriminator(fake_output)

        self.bmgan_model = Model(real_input, validity)

        # Compile discriminator with LSGAN loss
        self.discriminator.compile(optimizer=Adam(0.0002, beta_1=0.5), loss='mse')

        # Compile BMGAN model with combined losses
        self.bmgan_model.compile(optimizer=Adam(0.0002, beta_1=0.5), loss=self.combined_loss)

    def combined_loss(self, y_true, y_pred):
        # Least-Square GAN Loss for Generator
        lsgan_loss = MeanSquaredError()(y_true, y_pred)

        # L1 Pixel-wise Loss
        l1_loss = MeanAbsoluteError()(self.real_pet, self.generated_pet)

        # Perceptual Loss
        p_loss = perceptual_loss(self.real_pet, self.generated_pet)

        return lsgan_loss + self.lambda1 * l1_loss + self.lambda2 * p_loss

    def train(self, mri_images, pet_images, epochs, batch_size):
        real_labels = tf.ones((batch_size, 1))
        fake_labels = tf.zeros((batch_size, 1))

        for epoch in range(epochs):
            # Select random batch for training
            idx = tf.random.uniform([batch_size], minval=0, maxval=mri_images.shape[0], dtype=tf.int32)
            real_mri, real_pet = tf.gather(mri_images, idx), tf.gather(pet_images, idx)

            # Generate synthetic PET images
            generated_pet = self.generator(real_mri)

            # Train Discriminator
            d_loss_real = self.discriminator.train_on_batch(real_pet, real_labels)
            d_loss_fake = self.discriminator.train_on_batch(generated_pet, fake_labels)
            d_loss = 0.5 * (d_loss_real + d_loss_fake)

            # Train Generator (BMGAN)
            g_loss = self.bmgan_model.train_on_batch(real_mri, real_labels)

            # Print losses
            print(f"{epoch+1}/{epochs}, D loss: {d_loss:.4f}, G loss: {g_loss:.4f}")




# Utility functions for data loading and resizing
def load_mri_pet_data(task):
    images_pet, images_mri, labels = generate_data_path_less()
    pet_data = generate(images_pet, labels, task)
    mri_data = generate(images_mri, labels, task)

    mri_resized = []
    pet_resized = []

    for mri_path, pet_path in zip(mri_data, pet_data):
        mri_img = nib.load(mri_path).get_fdata()
        pet_img = nib.load(pet_path).get_fdata()
        mri_img = zscore(mri_img, axis=None)
        pet_img = zscore(pet_img, axis=None)
        mri_resized.append(resize_image(mri_img, (128, 128, 128)))
        pet_resized.append(resize_image(pet_img, (128, 128, 128)))

    return np.expand_dims(np.array(mri_resized), -1), np.expand_dims(np.array(pet_resized), -1)

def resize_image(image, target_shape):
    zoom_factors = [target_shape[i] / image.shape[i] for i in range(3)]
    return zoom(image, zoom_factors, order=1)

# Main function
if __name__ == '__main__':
    task = 'cd'
    mri_data, pet_data = load_mri_pet_data(task)
    mri_train, mri_gen, pet_train, _ = train_test_split(mri_data, pet_data, test_size=0.33, random_state=42)

    bmgan = BMGAN(input_shape=(128, 128, 128, 1))
    bmgan.compile()
    bmgan.train(mri_train, pet_train, epochs=100, batch_size=4)

    generated_pet_images = bmgan.predict(mri_gen)
    print("Generated PET images shape:", generated_pet_images.shape)

