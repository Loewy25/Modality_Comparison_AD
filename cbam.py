import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from nilearn import plotting
from scipy.stats import zscore
from scipy.ndimage import zoom, rotate

# Import your own data loading functions
from data_loading import generate_data_path_less, generate, binarylabel

# Import TensorFlow and Keras
import tensorflow as tf
from tensorflow.keras.layers import (Conv3D, Input, LeakyReLU, Add,
                                     GlobalAveragePooling3D, Dense, Dropout,
                                     SpatialDropout3D, BatchNormalization, Multiply, Reshape, Concatenate)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical, Sequence
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import StratifiedKFold

# Import Ray and Ray Tune
import ray
from ray import tune
from ray.tune.schedulers import HyperBandScheduler
from ray.train.tensorflow.keras import ReportCheckpointCallback

# Initialize Ray
ray.init(ignore_reinit_error=True)

# -------------------------------
# Utility Classes
# -------------------------------

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

class CBAM3D(tf.keras.layers.Layer):
    """Convolutional Block Attention Module for 3D CNNs."""

    def __init__(self, channels, reduction_ratio=16, kernel_size=7, **kwargs):
        super(CBAM3D, self).__init__(**kwargs)
        self.channels = channels
        self.reduction_ratio = reduction_ratio
        self.kernel_size = kernel_size

        # Channel Attention components
        self.global_avg_pool = GlobalAveragePooling3D()
        self.global_max_pool = GlobalMaxPooling3D()
        self.shared_dense1 = Dense(self.channels // self.reduction_ratio,
                                   activation='relu',
                                   kernel_initializer='he_normal',
                                   use_bias=True,
                                   bias_initializer='zeros')
        self.shared_dense2 = Dense(self.channels,
                                   activation='sigmoid',
                                   kernel_initializer='he_normal',
                                   use_bias=True,
                                   bias_initializer='zeros')

        # Spatial Attention components
        self.conv_spatial = Conv3D(filters=1,
                                   kernel_size=(self.kernel_size, self.kernel_size, self.kernel_size),
                                   strides=(1, 1, 1),
                                   padding='same',
                                   activation='sigmoid',
                                   kernel_initializer='he_normal',
                                   use_bias=False)

    def call(self, inputs):
        # ----- Channel Attention Module -----
        # Feature descriptor on the channel axis
        avg_pool = self.global_avg_pool(inputs)  # Shape: (batch, channels)
        avg_pool = self.shared_dense1(avg_pool)
        avg_pool = self.shared_dense2(avg_pool)  # Shape: (batch, channels)

        max_pool = self.global_max_pool(inputs)  # Shape: (batch, channels)
        max_pool = self.shared_dense1(max_pool)
        max_pool = self.shared_dense2(max_pool)  # Shape: (batch, channels)

        # Combine and apply sigmoid activation
        channel_attention = Add()([avg_pool, max_pool])  # Shape: (batch, channels)
        channel_attention = Reshape((1, 1, 1, self.channels))(channel_attention)  # Shape: (batch, 1, 1, 1, channels)
        channel_refined = Multiply()([inputs, channel_attention])  # Shape: (batch, D, H, W, channels)

        # ----- Spatial Attention Module -----
        # Feature descriptor on the spatial axis
        avg_spatial = tf.reduce_mean(channel_refined, axis=-1, keepdims=True)  # Shape: (batch, D, H, W, 1)
        max_spatial = tf.reduce_max(channel_refined, axis=-1, keepdims=True)   # Shape: (batch, D, H, W, 1)
        concat = Concatenate(axis=-1)([avg_spatial, max_spatial])              # Shape: (batch, D, H, W, 2)

        # Apply convolution to get spatial attention map
        spatial_attention = self.conv_spatial(concat)                         # Shape: (batch, D, H, W, 1)
        spatial_refined = Multiply()([channel_refined, spatial_attention])    # Shape: (batch, D, H, W, channels)

        return spatial_refined

    def get_config(self):
        config = super(CBAM3D, self).get_config()
        config.update({
            'channels': self.channels,
            'reduction_ratio': self.reduction_ratio,
            'kernel_size': self.kernel_size
        })
        return config

class CNNModel:
    """Class to create and manage the 3D CNN model with CBAM attention mechanisms."""

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
    def create_model(input_shape=(128, 128, 128, 1), num_classes=2, dropout_rate=0.05, l2_reg=1e-5, use_cbam=True):
        input_img = Input(shape=input_shape)

        # Conv1 block (16 filters)
        x = CNNModel.convolution_block(input_img, 16, l2_reg=l2_reg)
        conv1_out = x
        if use_cbam:
            x = CBAM3D(channels=16)(x)
        x = CNNModel.context_module(x, 16, dropout_rate=dropout_rate, l2_reg=l2_reg)
        x = Add()([x, conv1_out])

        # Conv2 block (32 filters, stride 2)
        x = CNNModel.convolution_block(x, 32, strides=(2, 2, 2), l2_reg=l2_reg)
        conv2_out = x
        if use_cbam:
            x = CBAM3D(channels=32)(x)
        x = CNNModel.context_module(x, 32, dropout_rate=dropout_rate, l2_reg=l2_reg)
        x = Add()([x, conv2_out])

        # Conv3 block (64 filters, stride 2)
        x = CNNModel.convolution_block(x, 64, strides=(2, 2, 2), l2_reg=l2_reg)
        conv3_out = x
        if use_cbam:
            x = CBAM3D(channels=64)(x)
        x = CNNModel.context_module(x, 64, dropout_rate=dropout_rate, l2_reg=l2_reg)
        x = Add()([x, conv3_out])

        # Conv4 block (128 filters, stride 2)
        x = CNNModel.convolution_block(x, 128, strides=(2, 2, 2), l2_reg=l2_reg)
        conv4_out = x
        if use_cbam:
            x = CBAM3D(channels=128)(x)
        x = CNNModel.context_module(x, 128, dropout_rate=dropout_rate, l2_reg=l2_reg)
        x = Add()([x, conv4_out])

        # Conv5 block (256 filters, stride 2)
        x = CNNModel.convolution_block(x, 256, strides=(2, 2, 2), l2_reg=l2_reg)
        conv5_out = x
        if use_cbam:
            x = CBAM3D(channels=256)(x)
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

        return model

class DataLoader:
    """Class to handle data loading and preprocessing."""

    @staticmethod
    def loading_mask_3d(task, modality):
        """
        Load and preprocess 3D medical images based on the specified task and modality.

        Parameters:
        - task (str): The specific task identifier.
        - modality (str): The imaging modality ('MRI' or 'PET').

        Returns:
        - file_paths (list): List of file paths for the specified modality and task.
        - labels (list or np.ndarray): Corresponding binary labels.
        """
        images_pet, images_mri, labels = generate_data_path_less()

        if modality == 'PET':
            file_paths, train_label = generate(images_pet, labels, task)
        elif modality == 'MRI':
            file_paths, train_label = generate(images_mri, labels, task)
        else:
            raise ValueError(f"Unsupported modality: {modality}")

        # Binary labeling based on the task
        train_label = binarylabel(train_label, task)

        return file_paths, train_label

    @staticmethod
    def augment_data(image, flip_prob=0.1, rotate_prob=0.1):
        """
        Apply random augmentation to a single image.

        Parameters:
        - image (np.ndarray): 3D image array.
        - flip_prob (float): Probability of flipping along each axis.
        - rotate_prob (float): Probability of applying rotation.

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
            img_aug = rotate(img_aug, angle, axes=(0, 1), reshape=False, order=1)  # Rotate in x-y plane

        return img_aug

# -------------------------------
# DataGenerator Class
# -------------------------------

class DataGenerator(Sequence):
    """Keras Sequence Data Generator for loading and preprocessing 3D medical images."""

    def __init__(self, file_paths, labels, batch_size=8, 
                 target_shape=(128, 128, 128), num_classes=2, 
                 augment=False, flip_prob=0.1, rotate_prob=0.1):
        """
        Initialization.

        Parameters:
        - file_paths (list): List of file paths to NIfTI images.
        - labels (list or np.ndarray): Corresponding labels.
        - batch_size (int): Size of the batches.
        - target_shape (tuple): Desired shape of the images.
        - num_classes (int): Number of classes for classification.
        - augment (bool): Whether to apply data augmentation.
        - flip_prob (float): Probability of flipping along each axis.
        - rotate_prob (float): Probability of applying rotation.
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
            nifti_img = nib.load(self.file_paths[idx])
            img_data = nifti_img.get_fdata()
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
            batch_y[i] = self.labels[idx]

        return batch_x, batch_y

    def on_epoch_end(self):
        """Updates indexes after each epoch."""
        np.random.shuffle(self.indices)

# -------------------------------
# Trainer Class
# -------------------------------

class Trainer:
    """Class to handle model training and hyperparameter tuning."""

    @staticmethod
    def train_model(config, train_file_paths, train_labels, val_file_paths, val_labels):
        """
        Train the model with the given configuration.

        Parameters:
        - config (dict): Hyperparameter configuration.
        - train_file_paths (list): List of training file paths.
        - train_labels (list or np.ndarray): Training labels.
        - val_file_paths (list): List of validation file paths.
        - val_labels (list or np.ndarray): Validation labels.
        """
        # Unpack hyperparameters from the config dictionary
        learning_rate = config["learning_rate"]
        batch_size = config["batch_size"]
        dropout_rate = config["dropout_rate"]
        l2_reg = config["l2_reg"]
        flip_prob = config["flip_prob"]
        rotate_prob = config["rotate_prob"]

        # Create training and validation data generators
        train_generator = DataGenerator(
            file_paths=train_file_paths,
            labels=train_labels,
            batch_size=batch_size,
            augment=True,
            flip_prob=flip_prob,
            rotate_prob=rotate_prob
        )

        val_generator = DataGenerator(
            file_paths=val_file_paths,
            labels=val_labels,
            batch_size=batch_size,
            augment=False
        )

        # Create model with hyperparameters
        model = CNNModel.create_model(
            input_shape=(128, 128, 128, 1),
            num_classes=2,
            dropout_rate=dropout_rate,
            l2_reg=l2_reg,
            use_cbam=True  # CBAM is integral and always used
        )
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
        )

        # Define callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=50,
            mode='min',
            verbose=0,
            restore_best_weights=True
        )

        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            mode='min',
            verbose=0
        )

        # Use ReportCheckpointCallback for Ray Tune
        report_checkpoint_callback = ReportCheckpointCallback()

        # Train the model
        model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=150,  # Set a high epoch count; early stopping will handle termination
            callbacks=[
                early_stopping,
                reduce_lr,
                report_checkpoint_callback
            ],
            verbose=0
        )

        # Report metrics to Ray Tune
        val_loss, val_accuracy, val_auc = model.evaluate(val_generator, verbose=0)
        tune.report(val_loss=val_loss, val_accuracy=val_accuracy, val_auc=val_auc)

        # After training, clear the session and collect garbage to free memory
        tf.keras.backend.clear_session()
        import gc
        gc.collect()

    @staticmethod
    def tune_model(X_file_paths, Y_labels, task, modality, info):
        """
        Perform hyperparameter tuning using Ray Tune.

        Parameters:
        - X_file_paths (list): List of all file paths.
        - Y_labels (list or np.ndarray): Corresponding labels.
        - task (str): Task identifier.
        - modality (str): Imaging modality ('MRI' or 'PET').
        - info (str): Additional information for saving results.

        Returns:
        - best_model (tf.keras.Model): The best trained model.
        """
        # Define search space excluding CBAM as it's integral
        config = {
            "learning_rate": tune.loguniform(1e-5, 1e-3),
            "batch_size": tune.choice([4, 8, 16]),
            "dropout_rate": tune.uniform(0.0, 0.5),
            "l2_reg": tune.loguniform(1e-5, 1e-3),
            "flip_prob": tune.uniform(0.0, 0.3),
            "rotate_prob": tune.uniform(0.0, 0.3),
        }

        # Scheduler for early stopping bad trials
        scheduler = HyperBandScheduler(
            time_attr="training_iteration",
            max_t=150,  # Maximum number of epochs
            reduction_factor=3
        )

        # Split data using StratifiedKFold
        stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        splits = list(stratified_kfold.split(X_file_paths, Y_labels.argmax(axis=1)))

        def objective(config):
            """
            Objective function for each Ray Tune trial.

            Parameters:
            - config (dict): Hyperparameter configuration.
            """
            # Use the first fold for tuning
            train_idx, val_idx = splits[0]
            train_file_paths = [X_file_paths[i] for i in train_idx]
            val_file_paths = [X_file_paths[i] for i in val_idx]
            train_labels = Y_labels[train_idx]
            val_labels = Y_labels[val_idx]

            Trainer.train_model(
                config=config,
                train_file_paths=train_file_paths,
                train_labels=train_labels,
                val_file_paths=val_file_paths,
                val_labels=val_labels
            )

        # Execute tuning
        analysis = tune.run(
            objective,
            resources_per_trial={"cpu": 1, "gpu": 1},  # Adjust based on your hardware
            config=config,
            metric="val_auc",  # Use AUC for selecting the best model
            mode="max",
            num_samples=20,  # Number of hyperparameter configurations to try
            scheduler=scheduler,
            name="hyperparameter_tuning",
            max_concurrent_trials=4  # Adjust based on available GPUs
        )

        # Get the best trial
        best_trial = analysis.get_best_trial("val_auc", "max", "last")
        print("Best trial config: {}".format(best_trial.config))
        print("Best trial final validation AUC: {:.4f}".format(best_trial.last_result["val_auc"]))

        # Retrain the best model on the entire training data using the best hyperparameters
        best_config = best_trial.config

        # Use the first fold for final training
        train_idx, val_idx = splits[0]
        X_train = [X_file_paths[i] for i in train_idx]
        X_val = [X_file_paths[i] for i in val_idx]
        Y_train = Y_labels[train_idx]
        Y_val = Y_labels[val_idx]

        # Create data generators with best hyperparameters
        train_generator = DataGenerator(
            file_paths=X_train,
            labels=Y_train,
            batch_size=best_config["batch_size"],
            augment=True,
            flip_prob=best_config["flip_prob"],
            rotate_prob=best_config["rotate_prob"]
        )

        val_generator = DataGenerator(
            file_paths=X_val,
            labels=Y_val,
            batch_size=best_config["batch_size"],
            augment=False
        )

        # Create the best model
        best_model = CNNModel.create_model(
            input_shape=(128, 128, 128, 1),
            num_classes=2,
            dropout_rate=best_config["dropout_rate"],
            l2_reg=best_config["l2_reg"],
            use_cbam=True  # CBAM is integral
        )
        best_model.compile(
            optimizer=Adam(learning_rate=best_config["learning_rate"]),
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
        )

        # Define callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=50,
            mode='min',
            verbose=1,
            restore_best_weights=True
        )

        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            mode='min',
            verbose=1
        )

        # Retrain on the full data
        history = best_model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=400,  # Set a high epoch count; early stopping will handle termination
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )

        # Evaluate on validation set
        y_val_pred = best_model.predict(val_generator)
        final_auc = roc_auc_score(Y_val[:, 1], y_val_pred[:, 1])
        print(f'Final AUC on validation data: {final_auc:.4f}')

        # Plot training history
        Trainer.plot_history(history, info, task, modality)

        return best_model

    @staticmethod
    def plot_history(history, info, task, modality):
        """
        Plot the training and validation loss and AUC.

        Parameters:
        - history (History): Keras History object.
        - info (str): Additional information for saving plots.
        - task (str): Task identifier.
        - modality (str): Imaging modality ('MRI' or 'PET').
        """
        Utils.ensure_directory_exists('training_plots')
        plot_dir = os.path.join('training_plots', info, task, modality)
        Utils.ensure_directory_exists(plot_dir)

        # Plot Loss
        plt.figure(figsize=(10, 5))
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(loc='upper right')
        loss_plot_path = os.path.join(plot_dir, 'loss_plot.png')
        plt.savefig(loss_plot_path)
        plt.close()
        print(f'Loss plot saved at {loss_plot_path}')

        # Plot AUC
        plt.figure(figsize=(10, 5))
        plt.plot(history.history['auc'], label='Train AUC')
        plt.plot(history.history['val_auc'], label='Validation AUC')
        plt.title('Model AUC')
        plt.ylabel('AUC')
        plt.xlabel('Epoch')
        plt.legend(loc='lower right')
        auc_plot_path = os.path.join(plot_dir, 'auc_plot.png')
        plt.savefig(auc_plot_path)
        plt.close()
        print(f'AUC plot saved at {auc_plot_path}')

# -------------------------------
# Main Execution Flow
# -------------------------------

def main():
    task = 'cd'  # Update as per your task
    modality = 'MRI'  # 'MRI' or 'PET'
    info = 'test2'  # Additional info for saving results

    # Load your data
    file_paths, labels = DataLoader.loading_mask_3d(task, modality)
    X_file_paths = np.array(file_paths)  # Convert to NumPy array for indexing
    Y = to_categorical(labels, num_classes=2)

    # Convert labels to numpy array if not already
    Y = np.array(Y)

    # Train the model with hyperparameter tuning and get the best trained model
    best_model = Trainer.tune_model(X_file_paths, Y, task, modality, info)

    # Shutdown Ray
    ray.shutdown()

if __name__ == '__main__':
    main()

