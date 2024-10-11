import os
import numpy as np
import nibabel as nib
import tensorflow as tf
from tensorflow.keras.layers import (Conv3D, Input, LeakyReLU, Add, 
                                     Concatenate, Conv3DTranspose, 
                                     AveragePooling3D, GlobalAveragePooling3D, 
                                     Dense, SpatialDropout3D)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow_addons.layers import InstanceNormalization
from sklearn.model_selection import train_test_split
from scipy.stats import zscore
from scipy.ndimage import zoom

# Import your own data loading functions
from data_loading import generate_data_path_less, generate, binarylabel

# Function to ensure a directory exists
def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Function for a single convolution block with Instance Normalization
def convolution_block(x, filters, kernel_size=(3, 3, 3), strides=(1, 1, 1)):
    x = Conv3D(filters, kernel_size, strides=strides, padding='same')(x)
    x = InstanceNormalization()(x)
    x = LeakyReLU(0.2)(x)
    return x

# Dense block with two directly connected convolution layers
def dense_block(x, filters):
    x1 = convolution_block(x, filters)
    x2 = convolution_block(x1, filters)
    return Add()([x, x2])

# Transition layer to control feature map growth
def transition_layer(x, filters):
    x = Conv3D(filters, kernel_size=(1, 1, 1), padding='same')(x)
    x = InstanceNormalization()(x)
    x = AveragePooling3D(pool_size=(2, 2, 2))(x)
    return x

# Upsampling block using transpose convolution and skip connections
def upsampling_block(x, skip, filters):
    x = Conv3DTranspose(filters, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding='same')(x)
    x = Concatenate()([x, skip])
    x = InstanceNormalization()(x)
    x = LeakyReLU(0.2)(x)
    return x

# Generator network based on DenseU-Net
def create_generator(input_shape=(128, 128, 128, 1)):
    input_img = Input(shape=input_shape)
    skips = []
    x = convolution_block(input_img, 64)
    for filters in [64, 128, 256, 512]:
        x = dense_block(x, filters)
        skips.append(x)
        if filters < 512:
            x = transition_layer(x, filters * 2)
    for filters in reversed([256, 128, 64]):
        x = upsampling_block(x, skips.pop(), filters)
    x = dense_block(x, 64)
    output = Conv3D(1, kernel_size=(1, 1, 1), activation='tanh')(x)
    return Model(inputs=input_img, outputs=output)

# Discriminator network
def create_discriminator(input_shape=(128, 128, 128, 1)):
    input_img = Input(shape=input_shape)
    x = convolution_block(input_img, 32, strides=(2, 2, 2))
    x = convolution_block(x, 64, strides=(2, 2, 2))
    x = convolution_block(x, 128, strides=(2, 2, 2))
    x = convolution_block(x, 256, strides=(2, 2, 2))
    x = GlobalAveragePooling3D()(x)
    output = Dense(1, activation='sigmoid')(x)
    return Model(inputs=input_img, outputs=output)

# Resize image to the target shape
def resize_image(image, target_shape):
    zoom_factors = [target_shape[i] / image.shape[i] for i in range(3)]
    return zoom(image, zoom_factors, order=1)

# Load and preprocess MRI and PET data
def load_mri_pet_data(task):
    images_pet, images_mri, labels = generate_data_path_less()
    pet_data, _ = generate(images_pet, labels, task), labels
    mri_data, _ = generate(images_mri, labels, task), labels

    mri_resized = []
    pet_resized = []

    for mri_path, pet_path in zip(mri_data, pet_data):
        mri_img = nib.load(mri_path).get_fdata()
        pet_img = nib.load(pet_path).get_fdata()

        # Normalize images
        mri_img = zscore(mri_img, axis=None)
        pet_img = zscore(pet_img, axis=None)

        # Resize images to (128, 128, 128)
        mri_resized.append(resize_image(mri_img, (128, 128, 128)))
        pet_resized.append(resize_image(pet_img, (128, 128, 128)))

    return np.expand_dims(np.array(mri_resized), -1), np.expand_dims(np.array(pet_resized), -1)

# Train the BMGAN
def train_bmgan(mri_train, pet_train, epochs=100, batch_size=4):
    generator = create_generator()
    discriminator = create_discriminator()
    
    gan_input = Input(shape=(128, 128, 128, 1))
    generated_pet = generator(gan_input)
    discriminator.trainable = False
    gan_output = discriminator(generated_pet)
    gan = Model(inputs=gan_input, outputs=gan_output)
    gan.compile(optimizer=Adam(0.0002), loss='binary_crossentropy')
    discriminator.compile(optimizer=Adam(0.0002), loss='binary_crossentropy')
    
    real_labels = np.ones((batch_size, 1))
    fake_labels = np.zeros((batch_size, 1))
    
    for epoch in range(epochs):
        idx = np.random.randint(0, mri_train.shape[0], batch_size)
        real_mri = mri_train[idx]
        real_pet = pet_train[idx]
        
        generated_pet = generator.predict(real_mri)
        
        d_loss_real = discriminator.train_on_batch(real_pet, real_labels)
        d_loss_fake = discriminator.train_on_batch(generated_pet, fake_labels)
        d_loss = 0.5 * (d_loss_real + d_loss_fake)
        
        g_loss = gan.train_on_batch(real_mri, real_labels)
        
        print(f"Epoch {epoch + 1}/{epochs} - D Loss: {d_loss:.4f}, G Loss: {g_loss:.4f}")
    
    return generator

if __name__ == '__main__':
    task = 'cd'  # Task identifier
    mri_data, pet_data = load_mri_pet_data(task)
    
    mri_train, mri_gen, pet_train, _ = train_test_split(mri_data, pet_data, test_size=0.33, random_state=42)
    
    best_generator = train_bmgan(mri_train, pet_train)
    
    generated_pet_images = best_generator.predict(mri_gen)
    print("Generated PET images shape:", generated_pet_images.shape)
