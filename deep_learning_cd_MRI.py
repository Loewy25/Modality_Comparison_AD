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
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, CSVLogger
from tensorflow.keras.metrics import AUC
from sklearn.model_selection import StratifiedKFold
from nilearn import plotting
from scipy.stats import zscore
from scipy.ndimage import zoom, rotate
import matplotlib.pyplot as plt

# Import Ray Tune
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.integration.keras import TuneReportCallback
from ray.tune.suggest.basic_variant import BasicVariantGenerator

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
        Build and compile the model with hyperparameters from the configuration.
        """
        input_img = Input(shape=input_shape)

        # Retrieve hyperparameters
        dropout_rate = hp['dropout_rate']
        l2_reg = hp['l2_reg']
        learning_rate = hp['learning_rate']

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
        # Uncomment the following line if you want to see the model summary
        # model.summary()

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
                     task, modality, layer_name, class_idx, info, fold_idx, save_dir='./grad-cam'):
        save_dir = os.path.join(save_dir, info, f"fold_{fold_idx}", task, modality)
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
                                         task, modality, info, fold_idx):
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
                                     task, modality, conv_layer_name, class_idx, info, fold_idx)


def get_hyperparameter_search_space():
    config = {
        "dropout_rate": tune.uniform(0.0, 0.5),
        "l2_reg": tune.loguniform(1e-6, 1e-4),
        "learning_rate": tune.loguniform(1e-5, 1e-3),
        "flip_prob": tune.uniform(0.0, 0.5),
        "rotate_prob": tune.uniform(0.0, 0.5),
        "batch_size": tune.choice([4, 8, 12]),
        "epochs": 150  # You can adjust this or make it a hyperparameter as well
    }
    return config


def main():
    task = 'cd'  # Update as per your task
    modality = 'MRI'  # 'MRI' or 'PET'
    info = 'test'  # Additional info for saving results

    # Initialize Ray
    ray.init()

    # Load data
    file_paths, labels, original_file_paths = DataLoader.loading_mask_3d(task, modality)
    X_file_paths = np.array(file_paths)
    Y = np.array(labels)

    # Set up stratified k-fold cross-validation
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    for fold_idx, (train_indices, val_indices) in enumerate(skf.split(X_file_paths, Y)):
        print(f"\nStarting Fold {fold_idx + 1}/3")
        train_file_paths = X_file_paths[train_indices]
        val_file_paths = X_file_paths[val_indices]
        train_labels = Y[train_indices]
        val_labels = Y[val_indices]

        # Create an instance of CNNTrainable for this fold
        cnn_trainable = CNNTrainable(
            task, modality, info,
            train_file_paths, train_labels,
            val_file_paths, val_labels
        )

        # Get hyperparameter search space
        config = get_hyperparameter_search_space()

        # Define the scheduler
        scheduler = ASHAScheduler(
            metric="val_auc",
            mode="max",
            max_t=150,
            grace_period=30,
            reduction_factor=2
        )

        # Define the search algorithm (Random Search)
        search_alg = BasicVariantGenerator()

        # Execute the hyperparameter search
        analysis = tune.run(
            tune.with_parameters(cnn_trainable.train),
            resources_per_trial={"cpu": 1, "gpu": 1},  # Adjust based on your resources
            config=config,
            metric="val_auc",
            mode="max",
            num_samples=20,  # Adjust based on your computational budget
            scheduler=scheduler,
            search_alg=search_alg,
            name=f"ray_tune_experiment_fold_{fold_idx + 1}",
            local_dir="./ray_results"  # Directory to save results
        )

        # Get the best trial
        best_trial = analysis.get_best_trial("val_auc", "max", "last")
        print(f"Best trial config for Fold {fold_idx + 1}: {best_trial.config}")
        print(f"Best trial final validation AUC for Fold {fold_idx + 1}: {best_trial.last_result['val_auc']}")

        # Retrain the model with the best hyperparameters on the training data
        final_model, history = cnn_trainable.retrain_best_model(best_trial.config)

        # Plot training and validation loss curves
        plt.figure()
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history.get('val_loss', []), label='Validation Loss')
        plt.title(f'Training and Validation Loss - Fold {fold_idx + 1}')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        loss_curve_path = f'loss_curve_fold_{fold_idx + 1}.png'
        plt.savefig(loss_curve_path)
        plt.close()
        print(f'Loss curve saved at {loss_curve_path}')

        # Apply Grad-CAM using the trained model
        # For Grad-CAM, select a subset of images to avoid excessive computation
        sample_indices = np.random.choice(len(val_file_paths), size=10, replace=False)
        sampled_file_paths = [val_file_paths[i] for i in sample_indices]
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

        # Apply Grad-CAM to all sampled images using final_model
        GradCAM.apply_gradcam_all_layers_average(
            final_model, imgs, sampled_original_imgs, task, modality, info, fold_idx + 1
        )

    # Shutdown Ray after completion
    ray.shutdown()


class CNNTrainable:
    """Class to encapsulate the model training logic for Ray Tune."""

    def __init__(self, task, modality, info,
                 train_file_paths, train_labels,
                 val_file_paths, val_labels):
        self.task = task
        self.modality = modality
        self.info = info
        self.train_file_paths = train_file_paths
        self.train_labels = train_labels
        self.val_file_paths = val_file_paths
        self.val_labels = val_labels

    def train(self, config):
        """Training function compatible with Ray Tune."""
        # Build the model using the sampled hyperparameters
        hp = config
        model = CNNModel.create_model(hp)

        # Build data generators
        batch_size = hp['batch_size']
        train_generator = DataGenerator(
            file_paths=self.train_file_paths,
            labels=self.train_labels,
            batch_size=batch_size,
            augment=True,
            flip_prob=hp['flip_prob'],
            rotate_prob=hp['rotate_prob']
        )
        val_generator = DataGenerator(
            file_paths=self.val_file_paths,
            labels=self.val_labels,
            batch_size=batch_size,
            augment=False
        )

        # Define callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=50,
                mode='min',
                verbose=1,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                mode='min',
                verbose=1
            ),
            TuneReportCallback(
                {
                    "val_loss": "val_loss",
                    "val_accuracy": "val_accuracy",
                    "val_auc": "val_auc"
                },
                on="epoch_end"
            )
        ]

        # Train the model
        model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=hp.get('epochs', 150),
            callbacks=callbacks,
            verbose=0
        )

        # Save the model checkpoint
        with tune.checkpoint_dir(step=0) as checkpoint_dir:
            model.save_weights(os.path.join(checkpoint_dir, "checkpoint"))

def retrain_best_model(self, best_hp):
        """Retrain the model with the best hyperparameters on the training data."""
        # Build the model with best hyperparameters
        model = CNNModel.create_model(best_hp)
    
        # Get batch size and other hyperparameters
        batch_size = best_hp['batch_size']
    
        # Build data generators
        train_generator = DataGenerator(
            file_paths=self.train_file_paths,
            labels=self.train_labels,
            batch_size=batch_size,
            augment=True,
            flip_prob=best_hp['flip_prob'],
            rotate_prob=best_hp['rotate_prob']
        )
        val_generator = DataGenerator(
            file_paths=self.val_file_paths,
            labels=self.val_labels,
            batch_size=batch_size,
            augment=False
        )
    
        # Define callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=50,
                mode='min',
                verbose=1,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                mode='min',
                verbose=1
            ),
            CSVLogger(f'training_log_fold_{self.fold_idx}.csv')
        ]
    
        # Retrain the model
        history = model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=best_hp.get('epochs', 150),
            callbacks=callbacks,
            verbose=1
        )
    
        # Evaluate the model on the validation data
        val_loss, val_accuracy, val_auc = model.evaluate(val_generator, verbose=0)
        print(f"\nRetrained model performance on validation data for Fold {self.fold_idx}:")
        print(f"  Validation Loss: {val_loss}")
        print(f"  Validation Accuracy: {val_accuracy}")
        print(f"  Validation AUC: {val_auc}")
    
        # Optionally, save the final model
        model.save(f'final_model_fold_{self.fold_idx}.h5')
    
        return model, history


if __name__ == "__main__":
    main()

