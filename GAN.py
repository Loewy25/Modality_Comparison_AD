import os
import numpy as np
import nibabel as nib
import tensorflow as tf
from tensorflow.keras.layers import (Conv3D, ReLU, Add, Concatenate,
                                     Conv3DTranspose, GlobalAveragePooling3D, LeakyReLU,
                                     Dense, AveragePooling3D, MaxPooling3D, BatchNormalization)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from scipy.stats import zscore
from scipy.ndimage import zoom
from tensorflow.keras.applications import VGG16
from tensorflow.keras import mixed_precision

# Import data loading functions (Ensure these are correctly implemented)
from data_loading import generate_data_path_less, generate, binarylabel

# ------------------------------------------------------------
# Custom InstanceNormalization Layer
# ------------------------------------------------------------
class InstanceNormalization(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-5, **kwargs):
        super(InstanceNormalization, self).__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        # Create scale and offset parameters
        self.scale = self.add_weight(
            shape=(input_shape[-1],),
            initializer='ones',
            trainable=True,
            name='scale'
        )
        self.offset = self.add_weight(
            shape=(input_shape[-1],),
            initializer='zeros',
            trainable=True,
            name='offset'
        )
        super(InstanceNormalization, self).build(input_shape)

    def call(self, inputs):
        # Compute mean and variance across spatial dimensions (excluding batch and channels)
        mean, variance = tf.nn.moments(inputs, axes=[1,2,3], keepdims=True)
        # Normalize
        inv = tf.math.rsqrt(variance + self.epsilon)
        normalized = (inputs - mean) * inv
        return self.scale * normalized + self.offset

    def get_config(self):
        config = super(InstanceNormalization, self).get_config()
        config.update({"epsilon": self.epsilon})
        return config

# ------------------------------------------------------------
# DenseUNetGenerator Class
# ------------------------------------------------------------
class DenseUNetGenerator:
    """Class for the DenseU-Net based Generator with Dense Blocks and Transition Layers."""

    def __init__(self, input_shape=(128, 128, 128, 1)):
        self.input_shape = input_shape
        self.model = self.build_generator()

    def convolution_block(self, x, filters, kernel_size=(3, 3, 3), strides=(1, 1, 1)):
        x = Conv3D(filters, kernel_size, strides=strides, padding='same')(x)
        x = InstanceNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)
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
        x = MaxPooling3D(pool_size=(2, 2, 2))(x)
        return x

    def upsampling_block(self, x, skip, filters):
        x = Conv3DTranspose(filters, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding='same')(x)
        x = Concatenate()([x, skip])  # Skip connection concatenation
        x = InstanceNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)
        return x

    def build_generator(self):
        input_img = tf.keras.Input(shape=self.input_shape)
        skips = []
        filters_list = [64, 128, 256, 512, 512, 512]  # Adjusted to match desired depth

        # Downsampling Path with Dense Blocks and Transition Layers
        x = self.convolution_block(input_img, 64)
        for filters in filters_list:
            x = self.dense_block(x, filters)
            skips.append(x)
            x = self.transition_layer(x, filters)

        # Bottleneck Dense Block
        x = self.dense_block(x, 512)

        # Upsampling Path with Upsampling Blocks and Dense Blocks
        for filters in reversed(filters_list):
            x = self.upsampling_block(x, skips.pop(), filters)
            x = self.dense_block(x, filters)  # Dense block after each upsampling

        # Final convolution to output PET image with Tanh activation
        output = Conv3D(1, kernel_size=(1, 1, 1), activation='tanh')(x)
        return Model(inputs=input_img, outputs=output)

# ------------------------------------------------------------
# ResNetEncoder Class with KL-Divergence Constraint
# ------------------------------------------------------------
class ResNetEncoder:
    """Class for the ResNet-34 Encoder Network with KL-Divergence constraint."""

    def __init__(self, input_shape=(128, 128, 128, 1), latent_dim=512):
        self.input_shape = input_shape
        self.latent_dim = latent_dim
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

        x = GlobalAveragePooling3D()(x)

        # Output mean and log variance for KL-divergence
        z_mean = Dense(self.latent_dim)(x)
        z_log_var = Dense(self.latent_dim)(x)

        return Model(inputs=input_img, outputs=[z_mean, z_log_var])

# ------------------------------------------------------------
# Discriminator Class with Patch-Level Discrimination
# ------------------------------------------------------------
class Discriminator:
    """Class for the Discriminator network with patch-level discrimination."""

    def __init__(self, input_shape=(128, 128, 128, 1)):
        self.input_shape = input_shape
        self.model = self.build_discriminator()

    def convolution_block(self, x, filters, kernel_size=(3, 3, 3)):
        x = Conv3D(filters, kernel_size, padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = MaxPooling3D(pool_size=(2, 2, 2))(x)
        return x

    def build_discriminator(self):
        input_img = tf.keras.Input(shape=self.input_shape)
        x = self.convolution_block(input_img, 32)
        x = self.convolution_block(x, 64)
        x = self.convolution_block(x, 128)
        x = self.convolution_block(x, 256)

        # Patch-level output
        x = Conv3D(1, kernel_size=(3, 3, 3), padding='same', activation='sigmoid')(x)
        return Model(inputs=input_img, outputs=x)

# ------------------------------------------------------------
# BMGAN Class with Integrated Loss Functions
# ------------------------------------------------------------
class BMGAN:
    def __init__(self, generator, discriminator, encoder, input_shape=(128, 128, 128, 1), lambda1=10.0, lambda2=0.5):
        self.generator = generator
        self.discriminator = discriminator
        self.encoder = encoder
        self.lambda1 = lambda1  # Weight for L1 Loss
        self.lambda2 = lambda2  # Weight for Perceptual Loss
        self.input_shape = input_shape
        self.vgg_model = self.get_vgg_model()  # Load pretrained VGG model for perceptual loss
        self.build_bmgan()

    def get_vgg_model(self):
        # Load the VGG16 model with ImageNet weights (pretrained)
        vgg = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

        # Extract features from an intermediate layer (before the second max-pooling operation)
        output = vgg.get_layer('block2_pool').output  # Adjust layer as needed

        # Create the feature extraction model
        model = Model(inputs=vgg.input, outputs=output)
        model.trainable = False  # Freeze the VGG model

        return model

    def perceptual_loss(self, y_true, y_pred):
        """
        Compute the perceptual loss using VGG features by processing all slices in a batched manner.
        Args:
            y_true (Tensor): Ground truth images, shape (batch_size, depth, height, width, channels)
            y_pred (Tensor): Generated images, shape (batch_size, depth, height, width, channels)
        Returns:
            Tensor: Scalar perceptual loss value
        """
        # Get the shape parameters
        batch_size = tf.shape(y_true)[0]
        depth = tf.shape(y_true)[1]
        height = tf.shape(y_true)[2]
        width = tf.shape(y_true)[3]
        channels = tf.shape(y_true)[4]

        # Reshape to combine batch and depth dimensions
        # New shape: (batch_size * depth, height, width, channels)
        y_true_reshaped = tf.reshape(y_true, [batch_size * depth, height, width, channels])
        y_pred_reshaped = tf.reshape(y_pred, [batch_size * depth, height, width, channels])

        # Resize to VGG16 input size (224x224)
        y_true_resized = tf.image.resize(y_true_reshaped, [224, 224])
        y_pred_resized = tf.image.resize(y_pred_reshaped, [224, 224])

        # Convert grayscale to RGB by repeating channels
        y_true_rgb = tf.image.grayscale_to_rgb(y_true_resized)
        y_pred_rgb = tf.image.grayscale_to_rgb(y_pred_resized)

        # Pass through VGG model to extract features
        y_true_features = self.vgg_model(y_true_rgb)
        y_pred_features = self.vgg_model(y_pred_rgb)

        # Compute the mean absolute error between the features
        perceptual_loss = tf.reduce_mean(tf.abs(y_true_features - y_pred_features))

        return perceptual_loss

    def l1_perceptual_loss(self, y_true, y_pred):
        # L1 Loss
        l1_loss = tf.reduce_mean(tf.abs(y_true - y_pred))
        # Perceptual Loss using VGG16
        perceptual_loss = self.perceptual_loss(y_true, y_pred)
        return l1_loss + self.lambda2 * perceptual_loss

    def kl_divergence_loss(self, y_true, y_pred):
        # y_pred is the kl_loss computed during model building
        return y_pred

    def lsgan_loss(self, y_true, y_pred):
        return tf.reduce_mean(tf.square(y_pred - y_true))

    def build_bmgan(self):
        # Compile the discriminator
        self.discriminator.compile(
            optimizer=Adam(0.0002, beta_1=0.5),
            loss='mse',
            metrics=['accuracy']
        )

        # Input images
        real_mri = tf.keras.Input(shape=self.input_shape)
        fake_pet = self.generator(real_mri)  # Generator is on default device

        # Encoder outputs
        real_pet = tf.keras.Input(shape=self.input_shape)
        z_mean_real_pet, z_log_var_real_pet = self.encoder(real_pet)
        z_mean_fake_pet, z_log_var_fake_pet = self.encoder(fake_pet)

        # Discriminator output for generated images (fake PET)
        validity_fake = self.discriminator(fake_pet)

        # Compute KL-divergence loss
        kl_loss_real = -0.5 * tf.reduce_mean(
            1 + z_log_var_real_pet - tf.square(z_mean_real_pet) - tf.exp(z_log_var_real_pet)
        )
        kl_loss_fake = -0.5 * tf.reduce_mean(
            1 + z_log_var_fake_pet - tf.square(z_mean_fake_pet) - tf.exp(z_log_var_fake_pet)
        )

        # Define the combined model
        self.combined = Model(
            inputs=[real_mri, real_pet],
            outputs=[validity_fake, fake_pet, kl_loss_real, kl_loss_fake]
        )

        # Compile the combined model
        self.combined.compile(
            optimizer=Adam(0.0002, beta_1=0.5),
            loss=[
                self.lsgan_loss,           # For validity_fake
                self.l1_perceptual_loss,  # For fake_pet
                self.kl_divergence_loss,  # For kl_loss_real
                self.kl_divergence_loss   # For kl_loss_fake
            ],
            loss_weights=[1, self.lambda1, self.lambda2, self.lambda2]
        )

    def train(self, mri_images, pet_images, epochs, batch_size):
        real_labels = np.ones((batch_size,) + (8, 8, 8, 1))  # Adjusted for patch-level output
        fake_labels = np.zeros((batch_size,) + (8, 8, 8, 1))

        for epoch in range(epochs):
            # Select random batch for training
            idx = np.random.randint(0, mri_images.shape[0], batch_size)
            real_mri = mri_images[idx]
            real_pet = pet_images[idx]

            # Convert data to tensors
            real_mri_tensor = tf.convert_to_tensor(real_mri, dtype=tf.float32)
            real_pet_tensor = tf.convert_to_tensor(real_pet, dtype=tf.float32)

            # Generate synthetic PET images
            generated_pet = self.generator.predict_on_batch(real_mri_tensor)

            # Train Discriminator
            d_loss_real = self.discriminator.train_on_batch(real_pet, real_labels)
            d_loss_fake = self.discriminator.train_on_batch(generated_pet, fake_labels)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # Train Generator (Combined Model)
            g_loss = self.combined.train_on_batch(
                [real_mri, real_pet],
                [real_labels, real_pet, np.zeros((batch_size,)), np.zeros((batch_size,))]
            )

            # Print losses
            print(f"Epoch {epoch+1}/{epochs}, D loss: {d_loss[0]:.4f}, "
                  f"D acc.: {d_loss[1]*100:.2f}%, G loss: {g_loss[0]:.4f}")

# ------------------------------------------------------------
# Data Loading and Preprocessing Functions
# ------------------------------------------------------------
def load_mri_pet_data(task):
    """
    Load and preprocess MRI and PET data.
    Args:
        task (str): Task identifier.
    Returns:
        Tuple of numpy arrays: (mri_data, pet_data)
    """
    images_pet, images_mri, labels = generate_data_path_less()
    pet_data, label = generate(images_pet, labels, task)
    mri_data, label = generate(images_mri, labels, task)

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
    """
    Resize a 3D image to the target shape using zoom.
    Args:
        image (numpy.ndarray): 3D image.
        target_shape (tuple): Desired shape.
    Returns:
        numpy.ndarray: Resized image.
    """
    zoom_factors = [target_shape[i] / image.shape[i] for i in range(3)]
    return zoom(image, zoom_factors, order=1)

# ------------------------------------------------------------
# Utility Function to Save Images
# ------------------------------------------------------------
def save_images(image, file_path):
    """
    Save a 3D image as a NIfTI file.
    Args:
        image (numpy.ndarray): 3D image.
        file_path (str): Destination file path.
    """
    nib.save(nib.Nifti1Image(image, np.eye(4)), file_path)

# ------------------------------------------------------------
# Main Function
# ------------------------------------------------------------
if __name__ == '__main__':

    # List available GPUs
    physical_gpus = tf.config.list_physical_devices('GPU')
    print("Available GPUs:", physical_gpus)

    if physical_gpus:
        try:
            # Enable memory growth for each GPU
            for gpu in physical_gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("Enabled memory growth for all GPUs.")
        except RuntimeError as e:
            print(e)

    # Define task and experiment info
    task = 'cd'
    info = 'experiment1'  # New parameter for the subfolder

    # Load MRI and PET data
    print("Loading MRI and PET data...")
    mri_data, pet_data = load_mri_pet_data(task)
    print(f"Loaded {mri_data.shape[0]} MRI and PET image pairs.")

    # Split data into training (2/3) and test (1/3)
    print("Splitting data into training and testing sets...")
    mri_train, mri_gen, pet_train, pet_gen = train_test_split(
        mri_data, pet_data, test_size=0.33, random_state=42
    )
    print(f"Training set: {mri_train.shape[0]} samples")
    print(f"Testing set: {mri_gen.shape[0]} samples")

    input_shape = (128, 128, 128, 1)

    # Initialize generator
    print("Initializing Generator")
    generator = DenseUNetGenerator(input_shape).model
    generator.summary()

    # Initialize discriminator
    print("Initializing Discriminator")
    discriminator = Discriminator(input_shape).model
    discriminator.summary()

    # Initialize encoder
    print("Initializing Encoder")
    encoder = ResNetEncoder(input_shape).model
    encoder.summary()

    # Initialize BMGAN model
    print("Building BMGAN model...")
    bmgan = BMGAN(generator, discriminator, encoder, input_shape)

    # Train the model
    print("Starting training...")
    bmgan.train(mri_train, pet_train, epochs=5, batch_size=1)

    # Create directories to store the results
    output_dir_mri = f'gan/{task}/{info}/mri'
    output_dir_pet = f'gan/{task}/{info}/pet'

    os.makedirs(output_dir_mri, exist_ok=True)
    os.makedirs(output_dir_pet, exist_ok=True)
    print(f"Created directories: {output_dir_mri}, {output_dir_pet}")

    # Predict PET images for the test MRI data
    print("Generating PET images for the test set...")
    generated_pet_images = generator.predict(mri_gen)

    # Save the test MRI data and the generated PET images in their respective folders
    print("Saving generated PET images and corresponding MRI scans...")
    for i in range(len(mri_gen)):
        mri_file_path = os.path.join(output_dir_mri, f'mri_{i}.nii.gz')
        pet_file_path = os.path.join(output_dir_pet, f'generated_pet_{i}.nii.gz')

        # Save MRI and generated PET images
        save_images(mri_gen[i, :, :, :, 0], mri_file_path)  # Save MRI
        save_images(generated_pet_images[i, :, :, :, 0], pet_file_path)  # Save generated PET

    # Print confirmation
    print(f"Saved {len(mri_gen)} MRI and corresponding generated PET images in 'gan/{task}/{info}'")


