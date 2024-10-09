#!/usr/bin/env python

import argparse
import os
import numpy as np
import nibabel as nib
import tensorflow as tf
from tensorflow.keras.layers import (
    Conv3D, Input, LeakyReLU, Add,
    GlobalAveragePooling3D, Dense, Dropout,
    SpatialDropout3D, BatchNormalization
)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical, Sequence
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.metrics import AUC
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from nilearn import plotting
from scipy.stats import zscore
from scipy.ndimage import zoom, rotate
import matplotlib.pyplot as plt

# Import Keras Tuner
import keras_tuner as kt

class Utils:
    """Utility functions for directory management and image resizing."""

    @staticmethod
    def ensure_directory_exists(directory):
        if not os.path.exists(directory):
            os.makedirs(directory)

    @staticmethod
    def resize_image(image, target_shape):
        if len(image.shape) == 4:
            # Assuming the last dimension is the channel dimension
            spatial_dims = image.shape[:3]
            zoom_factors = [target_shape[i] / spatial_dims[i] for i in range(3)] + [1]
        elif len(image.shape) == 3:
            zoom_factors = [target_shape[i] / image.shape[i] for i in range(3)]
        else:
            raise ValueError(f"Unexpected image shape: {image.shape}. Expected 3D or 4D image.")

        # Apply zoom to the image
        resized_image = zoom(image, zoom_factors, order=1)
        return resized_image


class CNNModel:
    """Class to create and manage the 3D CNN model."""

    @staticmethod
    def convolution_block(x, filters, kernel_size=(3, 3, 3), strides=(1, 1, 1), l2_reg=1e-5):
        x = Conv3D(filters, kernel_size, strides=strides, padding='same',
                   kernel_regularizer=l2(l2_reg))(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        return x

    @staticmethod
    def context_module(x, filters, dropout_rate=0.05, l2_reg=1e-5):
        x = CNNModel.convolution_block(x, filters, l2_reg=l2_reg)
        x = SpatialDropout3D(dropout_rate)(x)
        x = CNNModel.convolution_block(x, filters, l2_reg=l2_reg)
        return x

    @staticmethod
    def create_model(hp, input_shape=(128, 128, 128, 1), num_classes=2):
        """
        Build and compile the model with hyperparameters from Keras Tuner.

        Hyperparameter Search Space:
        - dropout_rate: Float between 0.0 and 0.5 with step size 0.1
        - l2_reg: Log-uniform distribution between 1e-6 and 1e-4
        - learning_rate: Log-uniform distribution between 1e-5 and 1e-3
        """
        input_img = Input(shape=input_shape)

        # Define hyperparameters and search space
        dropout_rate = hp.Float('dropout_rate', min_value=0.0, max_value=0.5, step=0.1)
        l2_reg = hp.Float('l2_reg', min_value=1e-6, max_value=1e-4, sampling='log')
        learning_rate = hp.Float('learning_rate', min_value=1e-5, max_value=1e-3, sampling='log')

        # Conv1 block (16 filters)
        x = CNNModel.convolution_block(input_img, 16, l2_reg=l2_reg)
        conv1_out = x
        x = CNNModel.context_module(x, 16, dropout_rate=dropout_rate, l2_reg=l2_reg)
        x = Add()([x, conv1_out])

        # Conv2 block (32 filters, stride 2)
        x = CNNModel.convolution_block(x, 32, strides=(2, 2, 2), l2_reg=l2_reg)
        conv2_out = x
        x = CNNModel.context_module(x, 32, dropout_rate=dropout_rate, l2_reg=l2_reg)
        x = Add()([x, conv2_out])

        # Conv3 block (64 filters, stride 2)
        x = CNNModel.convolution_block(x, 64, strides=(2, 2, 2), l2_reg=l2_reg)
        conv3_out = x
        x = CNNModel.context_module(x, 64, dropout_rate=dropout_rate, l2_reg=l2_reg)
        x = Add()([x, conv3_out])

        # Conv4 block (128 filters, stride 2)
        x = CNNModel.convolution_block(x, 128, strides=(2, 2, 2), l2_reg=l2_reg)
        conv4_out = x
        x = CNNModel.context_module(x, 128, dropout_rate=dropout_rate, l2_reg=l2_reg)
        x = Add()([x, conv4_out])

        # Conv5 block (256 filters, stride 2)
        x = CNNModel.convolution_block(x, 256, strides=(2, 2, 2), l2_reg=l2_reg)
        conv5_out = x
        x = CNNModel.context_module(x, 256, dropout_rate=dropout_rate, l2_reg=l2_reg)
        x = Add()([x, conv5_out])

        # Global Average Pooling
        x = GlobalAveragePooling3D()(x)

        # Dropout for regularization
        x = Dropout(dropout_rate)(x)

        # Dense layer with softmax for classification
        output = Dense(num_classes, activation='softmax')(x)

        model = Model(inputs=input_img, outputs=output)
        model.summary()

        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy', AUC(name='auc')]
        )

        return model


class DataLoader:
    """Class to handle data loading and preprocessing."""

    @staticmethod
    def loading_mask_3d(task, modality):
        """
        Load and preprocess 3D medical images based on the specified task and modality.

        Returns:
        - file_paths (list): List of file paths for the specified modality and task.
        - labels (list): Corresponding binary labels.
        - original_imgs (list): List of original NIfTI images (for Grad-CAM reference).
        """
        # Placeholder for data loading logic
        # Replace this with your actual data loading code
        num_samples = 50
        file_paths = [f"image_{i}_{modality}.nii.gz" for i in range(num_samples)]
        labels = np.random.randint(0, 2, size=num_samples)
        original_imgs = file_paths.copy()  # Assuming original images are the same

        return file_paths, labels, original_imgs

    @staticmethod
    def augment_data(image, flip_prob=0.1, rotate_prob=0.1):
        """
        Apply random augmentation to a single image.

        Returns:
        - augmented_image (np.ndarray): Augmented image array.
        """
        img_aug = image.copy()
        # Randomly flip along each axis with specified probability
        if np.random.rand() < flip_prob:
            img_aug = np.flip(img_aug, axis=0)  # Flip along x-axis
        if np.random.rand() < flip_prob:
            img_aug = np.flip(img_aug, axis=1)  # Flip along y-axis
        if np.random.rand() < flip_prob:
            img_aug = np.flip(img_aug, axis=2)  # Flip along z-axis

        # Random rotation with specified probability
        if np.random.rand() < rotate_prob:
            angle = np.random.uniform(-10, 10)  # Rotate between -10 and 10 degrees
            img_aug = rotate(img_aug, angle, axes=(1, 2), reshape=False, order=1)

        return img_aug


class DataGenerator(Sequence):
    """Keras Sequence Data Generator for loading and preprocessing 3D medical images."""

    def __init__(self, file_paths, labels, batch_size=8,
                 target_shape=(128, 128, 128), num_classes=2,
                 augment=False, flip_prob=0.1, rotate_prob=0.1):
        """
        Initialization.
        """
        self.file_paths = file_paths
        self.labels = labels
        self.batch_size = batch_size
        self.target_shape = target_shape
        self.num_classes = num_classes
        self.augment = augment
        self.flip_prob = flip_prob
        self.rotate_prob = rotate_prob
        self.indices = np.arange(len(self.file_paths))

    def __len__(self):
        """Denotes the number of batches per epoch."""
        return int(np.ceil(len(self.file_paths) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data."""
        # Generate indices of the batch
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]

        # Initialize arrays
        batch_x = np.empty((len(batch_indices), *self.target_shape, 1), dtype=np.float32)
        batch_y = np.empty((len(batch_indices), self.num_classes), dtype=np.float32)

        for i, idx in enumerate(batch_indices):
            # Load NIfTI image
            # Replace this with your actual image loading code
            img_data = np.random.rand(*self.target_shape)
            img_data = zscore(img_data, axis=None)

            # Resize the image (if needed)
            img_resized = Utils.resize_image(img_data, self.target_shape)

            # Apply augmentation if enabled
            if self.augment:
                img_resized = DataLoader.augment_data(img_resized,
                                                      flip_prob=self.flip_prob,
                                                      rotate_prob=self.rotate_prob)

            # Expand dimensions and assign to batch_x
            batch_x[i, ..., 0] = img_resized

            # Assign label
            batch_y[i] = to_categorical(self.labels[idx], num_classes=self.num_classes)

        return batch_x, batch_y

    def on_epoch_end(self):
        """Updates indexes after each epoch."""
        np.random.shuffle(self.indices)


class GradCAM:
    """Class to compute and save Grad-CAM heatmaps."""

    @staticmethod
    def make_gradcam_heatmap(model, img, last_conv_layer_name, pred_index=None):
        grad_model = tf.keras.models.Model(
            [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img)
            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            loss = predictions[:, pred_index]

        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2, 3))

        conv_outputs = conv_outputs[0]
        conv_outputs = conv_outputs * pooled_grads
        heatmap = tf.reduce_sum(conv_outputs, axis=-1)

        # Apply ReLU and normalize
        heatmap = np.maximum(heatmap, 0)
        if np.max(heatmap) != 0:
            heatmap /= np.max(heatmap)
        else:
            heatmap = np.zeros_like(heatmap)

        return heatmap

    @staticmethod
    def save_gradcam(heatmap, original_img,
                     task, modality, layer_name, class_idx, info, save_dir='./grad-cam'):
        save_dir = os.path.join(save_dir, info, task, modality)
        Utils.ensure_directory_exists(save_dir)

        # Calculate the zoom factors based on the actual dimensions
        zoom_factors = [original_dim / heatmap_dim for original_dim, heatmap_dim in zip(original_img.shape[:3], heatmap.shape)]

        # Upsample the heatmap to the original image size
        upsampled_heatmap = zoom(heatmap, zoom=zoom_factors, order=1)

        # Create a NIfTI image with a dummy affine (since original_img is dummy data)
        heatmap_img = nib.Nifti1Image(upsampled_heatmap, affine=np.eye(4))

        # Normalize the upsampled heatmap
        data = heatmap_img.get_fdata()
        min_val = np.min(data)
        max_val = np.max(data)
        if max_val - min_val != 0:
            normalized_data = (data - min_val) / (max_val - min_val)
        else:
            normalized_data = np.zeros_like(data)

        heatmap_img = nib.Nifti1Image(normalized_data, heatmap_img.affine)

        # Save the 3D NIfTI file
        nifti_file_name = f"gradcam_{task}_{modality}_class{class_idx}_{layer_name}.nii.gz"
        nifti_save_path = os.path.join(save_dir, nifti_file_name)
        nib.save(heatmap_img, nifti_save_path)
        print(f'3D Grad-CAM heatmap saved at {nifti_save_path}')

        # Plot the glass brain
        output_glass_brain_path = os.path.join(save_dir, f'glass_brain_{task}_{modality}_{layer_name}_class{class_idx}.png')
        plotting.plot_glass_brain(heatmap_img, colorbar=True, plot_abs=True,
                                  cmap='jet', output_file=output_glass_brain_path)
        print(f'Glass brain plot saved at {output_glass_brain_path}')

    @staticmethod
    def apply_gradcam_all_layers_average(model, imgs, original_imgs,
                                         task, modality, info):
        # Identify convolutional layers
        conv_layers = []
        cumulative_scales = []
        cumulative_scale = 1
        for layer in model.layers:
            if isinstance(layer, Conv3D):
                conv_layers.append(layer.name)
                # Update cumulative scaling factor if stride is not 1
                if layer.strides[0] != 1:
                    cumulative_scale *= layer.strides[0]
                cumulative_scales.append(cumulative_scale)

        print("Cumulative scaling factors for each convolutional layer:")
        for idx, (layer_name, scale_value) in enumerate(zip(conv_layers, cumulative_scales)):
            print(f"Layer {layer_name}: cumulative_scale = {scale_value}")

        for idx, conv_layer_name in enumerate(conv_layers):
            cumulative_scale = cumulative_scales[idx]
            for class_idx in range(2):  # Loop through both class indices (class 0 and class 1)
                accumulated_heatmap = None
                for i, img in enumerate(imgs):
                    heatmap = GradCAM.make_gradcam_heatmap(model, img, conv_layer_name,
                                                           pred_index=class_idx)
                    print(f"Heatmap shape for layer {conv_layer_name} and class {class_idx}: {heatmap.shape}")
                    if accumulated_heatmap is None:
                        accumulated_heatmap = heatmap
                    else:
                        accumulated_heatmap += heatmap

                avg_heatmap = accumulated_heatmap / len(imgs)
                # Use the first original image as reference
                GradCAM.save_gradcam(avg_heatmap, original_imgs[0],
                                     task, modality, conv_layer_name, class_idx, info)


class CNNTrainable:
    """Class to encapsulate the model training logic for Keras Tuner."""

    def __init__(self, task, modality, info):
        self.task = task
        self.modality = modality
        self.info = info

        # Load data
        self.file_paths, self.labels, self.original_file_paths = DataLoader.loading_mask_3d(task, modality)
        self.X_file_paths = np.array(self.file_paths)
        self.Y = np.array(self.labels)

        # Split data into training and validation
        self.train_idx, self.val_idx = train_test_split(
            np.arange(len(self.X_file_paths)), test_size=0.2, stratify=self.Y, random_state=42
        )
        self.train_file_paths = self.X_file_paths[self.train_idx]
        self.val_file_paths = self.X_file_paths[self.val_idx]
        self.train_labels = self.Y[self.train_idx]
        self.val_labels = self.Y[self.val_idx]

    def build_model(self, hp):
        """Builds the model using hyperparameters from Keras Tuner."""
        model = CNNModel.create_model(hp)
        return model

    def train(self):
        """Performs hyperparameter tuning and trains the best model."""
        # Create data generators
        batch_size = 5
        self.train_generator = DataGenerator(
            file_paths=self.train_file_paths,
            labels=self.train_labels,
            batch_size=batch_size,
            augment=True,
            flip_prob=0.2,
            rotate_prob=0.2
        )
        self.val_generator = DataGenerator(
            file_paths=self.val_file_paths,
            labels=self.val_labels,
            batch_size=batch_size,
            augment=False
        )

        # Define the tuner with max_trials
        tuner = kt.Hyperband(
            self.build_model,
            objective='val_auc',
            max_epochs=50,
            factor=3,
            hyperband_iterations=2,  # Controls the number of times to repeat the Hyperband algorithm
            directory='hyperband_dir',
            project_name='hyperband_project'
        )

        # Early stopping and Reduce LR callbacks
        early_stopping = EarlyStopping(
            monitor='val_auc',
            patience=10,
            mode='max',
            verbose=1,
            restore_best_weights=True
        )

        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            mode='min',
            verbose=1
        )

        # Run the hyperparameter search
        tuner.search(
            self.train_generator,
            validation_data=self.val_generator,
            epochs=50,
            callbacks=[early_stopping, reduce_lr]
        )

        # Retrieve the number of trials conducted
        trials = tuner.oracle.get_best_trials(num_trials=100)
        print(f"Number of trials conducted: {len(trials)}")

        # Get the best model
        self.best_model = tuner.get_best_models(num_models=1)[0]

        # Train the best model on the full training data
        self.history = self.best_model.fit(
            self.train_generator,
            validation_data=self.val_generator,
            epochs=50,
            callbacks=[early_stopping, reduce_lr]
        )

        # Plot training vs validation loss
        self.plot_training_validation_loss()

    def plot_training_validation_loss(self):
        """Plots and saves the training vs validation loss graph."""
        save_dir = os.path.join('./grad-cam', self.info, self.task, self.modality)
        Utils.ensure_directory_exists(save_dir)

        plt.figure(figsize=(10, 6))
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.title('Training Loss vs Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.ylim(0, 1)
        plt.legend(loc='upper right')
        loss_plot_path = os.path.join(save_dir, 'loss_vs_val_loss.png')
        plt.savefig(loss_plot_path)
        plt.close()
        print(f'Loss vs Validation Loss plot saved at {loss_plot_path}')

    def get_best_model(self):
        """Returns the best trained model."""
        return self.best_model


class Trainer:
    """Class to handle model training and evaluation."""

    @staticmethod
    def tune_model(task, modality, info):
        cnn_trainable = CNNTrainable(task, modality, info)
        cnn_trainable.train()
        return cnn_trainable.get_best_model()


def main():
    task = 'cd'  # Update as per your task
    modality = 'MRI'  # 'MRI' or 'PET'
    info = 'test'  # Additional info for saving results

    # Train the model with hyperparameter tuning and get the best trained model
    best_model = Trainer.tune_model(task, modality, info)

    # Apply Grad-CAM using the trained model
    # For Grad-CAM, select a subset of images to avoid excessive computation
    file_paths, labels, original_file_paths = DataLoader.loading_mask_3d(task, modality)
    sample_indices = np.random.choice(len(file_paths), size=10, replace=False)
    sampled_file_paths = [file_paths[i] for i in sample_indices]
    sampled_original_imgs = [np.random.rand(128, 128, 128) for _ in sample_indices]  # Dummy images

    # Create a list of expanded images for Grad-CAM
    imgs = []
    for file_path in sampled_file_paths:
        # Replace this with your actual image loading code
        img_data = np.random.rand(128, 128, 128)
        img_data = zscore(img_data, axis=None)
        resized_data = Utils.resize_image(img_data, target_shape=(128, 128, 128))
        resized_data = np.expand_dims(resized_data, axis=-1)  # Add channel dimension
        resized_data = np.expand_dims(resized_data, axis=0)  # Add batch dimension
        imgs.append(resized_data)

    # Apply Grad-CAM to all sampled images
    GradCAM.apply_gradcam_all_layers_average(best_model, imgs, sampled_original_imgs, task, modality, info)


if __name__ == "__main__":
    main()

