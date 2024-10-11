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

class DenseUNetGenerator:
    """Class for the DenseU-Net based Generator."""

    def __init__(self, input_shape=(128, 128, 128, 1)):
        self.input_shape = input_shape
        self.model = self.build_generator()

    def convolution_block(self, x, filters, kernel_size=(3, 3, 3), strides=(1, 1, 1)):
        x = Conv3D(filters, kernel_size, strides=strides, padding='same')(x)
        x = InstanceNormalization()(x)
        x = LeakyReLU(0.2)(x)
        return x

    def dense_block(self, x, filters):
        x1 = self.convolution_block(x, filters)
        x2 = self.convolution_block(x1, filters)
        return Add()([x, x2])

    def transition_layer(self, x, filters):
        x = Conv3D(filters, kernel_size=(1, 1, 1), padding='same')(x)
        x = InstanceNormalization()(x)
        x = AveragePooling3D(pool_size=(2, 2, 2))(x)
        return x

    def upsampling_block(self, x, skip, filters):
        x = Conv3DTranspose(filters, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding='same')(x)
        x = Concatenate()([x, skip])
        x = InstanceNormalization()(x)
        x = LeakyReLU(0.2)(x)
        return x

    def build_generator(self):
        input_img = tf.keras.Input(shape=self.input_shape)
        skips = []
        x = self.convolution_block(input_img, 64)
        for filters in [64, 128, 256, 512]:
            x = self.dense_block(x, filters)
            skips.append(x)
            if filters < 512:
                x = self.transition_layer(x, filters * 2)
        for filters in reversed([256, 128, 64]):
            x = self.upsampling_block(x, skips.pop(), filters)
        x = self.dense_block(x, 64)
        output = Conv3D(1, kernel_size=(1, 1, 1), activation='tanh')(x)
        return Model(inputs=input_img, outputs=output)

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

class BMGAN:
    """Bidirectional Mapping Generative Adversarial Network (BMGAN)."""

    def __init__(self, input_shape=(128, 128, 128, 1)):
        self.generator = DenseUNetGenerator(input_shape).model
        self.discriminator = Discriminator(input_shape).model
        self.bmgan = self.build_bmgan()

    def build_bmgan(self):
        self.discriminator.trainable = False
        gan_input = tf.keras.Input(shape=(128, 128, 128, 1))
        generated_pet = self.generator(gan_input)
        gan_output = self.discriminator(generated_pet)
        return Model(inputs=gan_input, outputs=gan_output)

    def compile(self):
        self.generator_optimizer = Adam(0.0002, beta_1=0.5)
        self.discriminator_optimizer = Adam(0.0002, beta_1=0.5)
        self.bmgan.compile(optimizer=self.generator_optimizer, loss='binary_crossentropy')
        self.discriminator.compile(optimizer=self.discriminator_optimizer, loss='binary_crossentropy')

    def train(self, mri_train, pet_train, epochs=100, batch_size=4):
        real_labels = np.ones((batch_size, 1))
        fake_labels = np.zeros((batch_size, 1))

        for epoch in range(epochs):
            idx = np.random.randint(0, mri_train.shape[0], batch_size)
            real_mri = mri_train[idx]
            real_pet = pet_train[idx]

            generated_pet = self.generator.predict(real_mri)

            d_loss_real = self.discriminator.train_on_batch(real_pet, real_labels)
            d_loss_fake = self.discriminator.train_on_batch(generated_pet, fake_labels)
            d_loss = 0.5 * (d_loss_real + d_loss_fake)

            g_loss = self.bmgan.train_on_batch(real_mri, real_labels)

            print(f"Epoch {epoch + 1}/{epochs} - D Loss: {d_loss:.4f}, G Loss: {g_loss:.4f}")

    def predict(self, mri_data):
        return self.generator.predict(mri_data)

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

