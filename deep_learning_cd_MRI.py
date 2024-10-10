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
from scipy.stats import zscore
from scipy.ndimage import zoom, rotate
import matplotlib.pyplot as plt

# Import Ray Tune
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.train.tensorflow.keras import ReportCheckpointCallback

from ray import air
from ray.tune.search.basic_variant import BasicVariantGenerator
from data_loading import generate_data_path_less, generate, binarylabel


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
        # Assuming generate_data_path_less() and generate() are defined elsewhere
        images_pet, images_mri, labels = generate_data_path_less()
        if modality == 'PET':
            file_paths, binary_labels = generate(images_pet, labels, task)
        elif modality == 'MRI':
            file_paths, binary_labels = generate(images_mri, labels, task)
        else:
            raise ValueError(f"Unsupported modality: {modality}")
        binary_labels = binarylabel(binary_labels,task)
        
        return file_paths, binary_labels


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
            # Load NIfTI image using the actual file path
            nifti_img = nib.load(self.file_paths[idx])
            img_data = nifti_img.get_fdata()

            # Normalize image data
            img_data = zscore(img_data, axis=None)

            # Resize the image
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


class CNNTrainable:
    """Class to encapsulate the model training logic for Ray Tune."""

    def __init__(self, task, modality, info,
                 train_file_paths, train_labels,
                 val_file_paths, val_labels, fold_idx):
        self.task = task
        self.modality = modality
        self.info = info
        self.train_file_paths = train_file_paths
        self.train_labels = train_labels
        self.val_file_paths = val_file_paths
        self.val_labels = val_labels
        self.fold_idx = fold_idx  # Store fold index

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
            ReportCheckpointCallback()
        ]

        # Train the model
        model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=hp.get('epochs', 30),
            callbacks=callbacks,
            verbose=0
        )

        # The ReportCheckpointCallback handles reporting metrics and checkpoints

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
            epochs=best_hp.get('epochs', 30),
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
        model_save_path = f'final_model_fold_{self.fold_idx}.h5'
        model.save(model_save_path)
        print(f'Final model saved at {model_save_path}')
    
        return model, history


def main():
    task = 'cd'  # Update as per your task
    modality = 'MRI'  # 'MRI' or 'PET'
    info = 'test'  # Additional info for saving results

    # Initialize Ray
    ray.init(ignore_reinit_error=True)

    # Load data
    file_paths, labels = DataLoader.loading_mask_3d(task, modality)
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
            val_file_paths, val_labels,
            fold_idx + 1  # Pass fold index
        )

        # Get hyperparameter search space
        config = get_hyperparameter_search_space()

        # Define the scheduler
        scheduler = ASHAScheduler(
            max_t=30,
            grace_period=10,
            reduction_factor=2
        )

        # Define the search algorithm (Basic Variant Generator)
        search_alg = BasicVariantGenerator()

        # Set storage_path to your desired absolute path with 'file://' scheme
        storage_path = 'file:///home/l.peiwang/Modality_Comparison_AD/ray_result'

        # Ensure the storage directory exists
        os.makedirs('/home/l.peiwang/Modality_Comparison_AD/ray_result', exist_ok=True)
        
        from ray.tune import TuneConfig
        
        # Wrap the training function to specify resources
        train_fn = tune.with_resources(
            tune.with_parameters(cnn_trainable.train),
            resources={"cpu": 1, "gpu": 1}  # Adjust based on your needs
        )
        
        # Initialize the tuner with the wrapped training function
        tuner = tune.Tuner(
            train_fn,
            param_space=config,
            tune_config=TuneConfig(
                metric="val_auc",
                mode="max",
                num_samples=10,
                scheduler=scheduler,
                search_alg=search_alg,
                max_concurrent_trials=2,  # Limit concurrency to available GPUs
            ),
            run_config=air.RunConfig(
                name=f"ray_tune_experiment_fold_{fold_idx + 1}",
                storage_path=storage_path,
            ),
        )

        results = tuner.fit()

        # Get the best result
        best_result = results.get_best_result(metric="val_auc", mode="max")
        best_config = best_result.config
        best_val_auc = best_result.metrics["val_auc"]

        print(f"Best trial config for Fold {fold_idx + 1}: {best_config}")
        print(f"Best trial final validation AUC for Fold {fold_idx + 1}: {best_val_auc}")

        # Retrain the model with the best hyperparameters on the training data
        final_model, history = cnn_trainable.retrain_best_model(best_config)

        # Plot training and validation loss curves
        plt.figure()
        plt.plot(history.history['loss'], label='Training Loss')
        if 'val_loss' in history.history:
            plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title(f'Training and Validation Loss - Fold {fold_idx + 1}')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        loss_curve_path = f'loss_curve_fold_{fold_idx + 1}.png'
        plt.savefig(loss_curve_path)
        plt.close()
        print(f'Loss curve saved at {loss_curve_path}')

    # Shutdown Ray after completion
    ray.shutdown()


if __name__ == "__main__":
    main()


