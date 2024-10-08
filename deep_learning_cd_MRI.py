import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import (Conv3D, Input, LeakyReLU, Add,
                                     GlobalAveragePooling3D, Dense, Dropout,
                                     SpatialDropout3D, BatchNormalization)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.metrics import AUC
from tensorflow.keras.utils import to_categorical, Sequence
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from nilearn import plotting
from scipy.stats import zscore
from scipy.ndimage import zoom, rotate

# Import your own data loading functions
from data_loading import generate_data_path_less, generate, binarylabel

# Import Ray and Ray Tune
import ray
from ray import tune
from ray.tune.schedulers import HyperBandScheduler
from ray.train.tensorflow.keras import ReportCheckpointCallback

# Initialize Ray
ray.init(ignore_reinit_error=True)

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
    def create_model(input_shape=(128, 128, 128, 1), num_classes=2, dropout_rate=0.05, l2_reg=1e-5):
        input_img = Input(shape=input_shape)

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
        - labels (list): Corresponding binary labels.
        - original_imgs (list): List of original NIfTI images (for Grad-CAM reference).
        """
        images_pet, images_mri, labels = generate_data_path_less()
        original_imgs = []  # Initialize the list to store original images

        if modality == 'PET':
            data_train, train_label = generate(images_pet, labels, task)
        elif modality == 'MRI':
            data_train, train_label = generate(images_mri, labels, task)
        else:
            raise ValueError(f"Unsupported modality: {modality}")

        # Binary labeling based on the task
        train_label = binarylabel(train_label, task)

        return data_train, train_label, images_pet if modality == 'PET' else images_mri

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
            img_aug = rotate(img_aug, angle, axes=(1, 2), reshape=False, order=1)

        return img_aug

class DataGenerator(Sequence):
    """Keras Sequence Data Generator for loading and preprocessing 3D medical images."""

    def __init__(self, file_paths, labels, original_file_paths, batch_size=8, 
                 target_shape=(128, 128, 128), num_classes=2, 
                 augment=False, flip_prob=0.1, rotate_prob=0.1):
        """
        Initialization.

        Parameters:
        - file_paths (list): List of file paths to NIfTI images.
        - labels (list or np.ndarray): Corresponding labels.
        - original_file_paths (list): List of original NIfTI file paths for Grad-CAM.
        - batch_size (int): Size of the batches.
        - target_shape (tuple): Desired shape of the images.
        - num_classes (int): Number of classes for classification.
        - augment (bool): Whether to apply data augmentation.
        - flip_prob (float): Probability of flipping along each axis.
        - rotate_prob (float): Probability of applying rotation.
        """
        self.file_paths = file_paths
        self.labels = labels
        self.original_file_paths = original_file_paths
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

        # Create a NIfTI image with the same affine as the original image
        heatmap_img = nib.Nifti1Image(upsampled_heatmap, original_img.affine)

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

class Trainer:
    """Class to handle model training and evaluation."""

    @staticmethod
    def train_model(config, train_file_paths, train_labels, val_file_paths, val_labels, 
                   original_train_file_paths, task, modality, info):
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
            original_file_paths=original_train_file_paths,
            batch_size=batch_size,
            augment=True,
            flip_prob=flip_prob,
            rotate_prob=rotate_prob
        )

        val_generator = DataGenerator(
            file_paths=val_file_paths,
            labels=val_labels,
            original_file_paths=original_train_file_paths,
            batch_size=batch_size,
            augment=False
        )

        # Create model with hyperparameters
        model = CNNModel.create_model(
            input_shape=(128, 128, 128, 1),
            num_classes=2,
            dropout_rate=dropout_rate,
            l2_reg=l2_reg
        )
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy', AUC(name='auc')]
        )

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

        # Use ReportCheckpointCallback
        report_checkpoint_callback = ReportCheckpointCallback()

        # Train the model
        model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=150,
            callbacks=[
                early_stopping,
                reduce_lr,
                report_checkpoint_callback
            ],
            verbose=0
        )

        # Evaluate on validation data
        val_loss, val_accuracy, val_auc = model.evaluate(val_generator, verbose=0)
        tune.report(val_loss=val_loss, val_accuracy=val_accuracy, val_auc=val_auc)

        # After training, clear the session and collect garbage to free memory
        tf.keras.backend.clear_session()
        import gc
        gc.collect()

    @staticmethod
    def tune_model(X_file_paths, Y_labels, original_file_paths, task, modality, info):
        # Define search space including augmentation probabilities
        config = {
            "learning_rate": tune.loguniform(1e-5, 1e-3),
            "batch_size": tune.choice([4, 8]),
            "dropout_rate": tune.uniform(0.0, 0.5),
            "l2_reg": tune.loguniform(1e-6, 1e-4),
            "flip_prob": tune.uniform(0.0, 0.5),
            "rotate_prob": tune.uniform(0.0, 0.5),
        }

        # Scheduler for early stopping bad trials
        scheduler = HyperBandScheduler(
            time_attr="training_iteration",
            max_t=150,  # Maximum number of epochs
        )

        # Define the objective function for each trial
        def objective(config):
            # Split data using StratifiedKFold
            stratified_kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=2)
            for train_idx, val_idx in stratified_kfold.split(X_file_paths, Y_labels.argmax(axis=1)):
                train_file_paths = [X_file_paths[i] for i in train_idx]
                val_file_paths = [X_file_paths[i] for i in val_idx]
                train_labels = Y_labels[train_idx]
                val_labels = Y_labels[val_idx]

                Trainer.train_model(
                    config=config,
                    train_file_paths=train_file_paths,
                    train_labels=train_labels,
                    val_file_paths=val_file_paths,
                    val_labels=val_labels,
                    original_train_file_paths=[original_file_paths[i] for i in train_idx],
                    task=task,
                    modality=modality,
                    info=info
                )

        # Execute tuning
        analysis = tune.run(
            objective,
            resources_per_trial={"cpu": 1, "gpu": 1},  # Adjust based on availability
            config=config,
            metric="val_auc",  # Use AUC for selecting the best model
            mode="max",
            num_samples=20,
            scheduler=scheduler,
            name="hyperparameter_tuning",
            max_concurrent_trials=4  # Utilize up to 4 GPUs if available
        )

        # Get the best trial
        best_trial = analysis.get_best_trial("val_auc", "max", "last")
        print("Best trial config: {}".format(best_trial.config))
        print("Best trial final validation AUC: {}".format(
            best_trial.last_result["val_auc"]))

        # Retrain the best model on the full training data
        best_config = best_trial.config

        # Split data using StratifiedKFold and select the first fold for final training
        stratified_kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=2)
        train_idx, val_idx = next(stratified_kfold.split(X_file_paths, Y_labels.argmax(axis=1)))
        X_train = [X_file_paths[i] for i in train_idx]
        X_val = [X_file_paths[i] for i in val_idx]
        Y_train = Y_labels[train_idx]
        Y_val = Y_labels[val_idx]
        original_train = [original_file_paths[i] for i in train_idx]

        # Create data generators with best hyperparameters
        train_generator = DataGenerator(
            file_paths=X_train,
            labels=Y_train,
            original_file_paths=original_train,
            batch_size=best_config["batch_size"],
            augment=True,
            flip_prob=best_config["flip_prob"],
            rotate_prob=best_config["rotate_prob"]
        )

        val_generator = DataGenerator(
            file_paths=X_val,
            labels=Y_val,
            original_file_paths=original_train,  # Use training original images for Grad-CAM
            batch_size=best_config["batch_size"],
            augment=False
        )

        # Create the best model
        best_model = CNNModel.create_model(
            input_shape=(128, 128, 128, 1),
            num_classes=2,
            dropout_rate=best_config["dropout_rate"],
            l2_reg=best_config["l2_reg"]
        )
        best_model.compile(
            optimizer=Adam(learning_rate=best_config["learning_rate"]),
            loss='categorical_crossentropy',
            metrics=['accuracy', AUC(name='auc')]
        )

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
            epochs=800,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )

        # Evaluate on validation set
        y_val_pred = best_model.predict(val_generator)
        final_auc = roc_auc_score(Y_val[:, 1], y_val_pred[:, 1])
        print(f'Final AUC on validation data: {final_auc:.4f}')

        return best_model

def main():
    task = 'cd'  # Update as per your task
    modality = 'MRI'  # 'MRI' or 'PET'
    info = 'test'  # Additional info for saving results

    # Load your data
    file_paths, labels, original_file_paths = DataLoader.loading_mask_3d(task, modality)
    X_file_paths = file_paths  # List of file paths
    X_file_paths = np.array(X_file_paths)
    Y = to_categorical(labels, num_classes=2)

    # Convert labels to numpy array if not already
    Y = np.array(Y)

    # Train the model with hyperparameter tuning and get the best trained model
    best_model = Trainer.tune_model(X_file_paths, Y, original_file_paths, task, modality, info)

    # Apply Grad-CAM using the trained model
    # For Grad-CAM, select a subset of images to avoid excessive computation
    sample_indices = np.random.choice(len(X_file_paths), size=10, replace=False)
    sampled_file_paths = [X_file_paths[i] for i in sample_indices]
    sampled_original_imgs = [original_file_paths[i] for i in sample_indices]

    # Create a list of expanded images for Grad-CAM
    imgs = []
    for file_path in sampled_file_paths:
        nifti_img = nib.load(file_path)
        img_data = nifti_img.get_fdata()
        img_data = zscore(img_data, axis=None)
        resized_data = Utils.resize_image(img_data, target_shape=(128, 128, 128))
        resized_data = np.expand_dims(resized_data, axis=-1)  # Add channel dimension
        resized_data = np.expand_dims(resized_data, axis=0)  # Add batch dimension
        imgs.append(resized_data)

    # Apply Grad-CAM to all sampled images
    GradCAM.apply_gradcam_all_layers_average(best_model, imgs, sampled_original_imgs, task, modality, info)

    # Shutdown Ray
    ray.shutdown()

if __name__ == '__main__':
    main()

