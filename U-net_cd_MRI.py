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
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from nilearn import plotting
from scipy.stats import zscore
from scipy.ndimage import zoom, rotate
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from ray import tune
from ray.tune.schedulers import HyperBandScheduler
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.metrics import AUC
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
        images_pet, images_mri, labels = generate_data_path_less()
        original_imgs = []  # Initialize the list to store original images

        if modality == 'PET':
            data_train, train_label = generate(images_pet, labels, task)
        elif modality == 'MRI':
            data_train, train_label = generate(images_mri, labels, task)
        else:
            raise ValueError(f"Unsupported modality: {modality}")

        train_data = []
        target_shape = (128, 128, 128)

        for i in range(len(data_train)):
            nifti_img = nib.load(data_train[i])
            original_imgs.append(nifti_img)  # Store the original NIfTI image

            reshaped_data = nifti_img.get_fdata()
            reshaped_data = zscore(reshaped_data, axis=None)

            # Resize the image to (128, 128, 128)
            resized_data = Utils.resize_image(reshaped_data, target_shape)

            train_data.append(resized_data)

        train_label = binarylabel(train_label, task)
        train_data = np.array(train_data)

        return train_data, train_label, original_imgs

    @staticmethod
    def augment_data(X, flip_prob=0.1, rotate_prob=0.1):
        augmented_X = []
        for img in X:
            img_aug = img.copy()
            # Randomly flip along each axis with specified probability
            if np.random.rand() < flip_prob:
                img_aug = np.flip(img_aug, axis=1)  # Flip along x-axis
            if np.random.rand() < flip_prob:
                img_aug = np.flip(img_aug, axis=2)  # Flip along y-axis

            # Random rotation with specified probability
            if np.random.rand() < rotate_prob:
                angle = np.random.uniform(-10, 10)  # Rotate between -10 and 10 degrees
                img_aug = rotate(img_aug, angle, axes=(1, 2), reshape=False, order=1)

            augmented_X.append(img_aug)
        return np.array(augmented_X)


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
    def tune_model_nested_cv(X, Y, task, modality, info):
      # Define the cross-validation strategy
        n_splits = 3
        stratified_kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=2)
        
        # Initialize variables to store results
        fold_results = []
        
        # Iterate over each fold
        for fold, (train_idx, val_idx) in enumerate(stratified_kfold.split(X, Y.argmax(axis=1)), 1):
            print(f"\nStarting fold {fold}/{n_splits}")
            X_train, X_val = X[train_idx], X[val_idx]
            Y_train, Y_val = Y[train_idx], Y[val_idx]
            
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
                max_t=4,  # Maximum number of epochs
              
            )
            
            # Execute tuning for the current fold
            analysis = tune.run(
                tune.with_parameters(Trainer.train_model, X_train=X_train, Y_train=Y_train, X_val=X_val, Y_val=Y_val),
                resources_per_trial={"cpu": 1, "gpu": 1},  # Use 1 GPU per trial
                config=config,
                metric="val_auc",  # Use AUC for selecting the best model
                mode="max",
                num_samples=8,
                scheduler=scheduler,
                name=f"hyperparameter_tuning_fold_{fold}",
                max_concurrent_trials=4  # Utilize up to 4 GPUs
            )
            
            # Get the best trial for the current fold
            best_trial = analysis.get_best_trial("val_auc", "max", "last")
            print(f"Best trial config for fold {fold}: {best_trial.config}")
            print(f"Best trial final validation AUC for fold {fold}: {best_trial.last_result['val_auc']}")
            
            # Train the best model on the current fold's training data
            best_config = best_trial.config
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
            
            # Apply data augmentation with best probabilities
            X_train_augmented = DataLoader.augment_data(
                X_train,
                flip_prob=best_config["flip_prob"],
                rotate_prob=best_config["rotate_prob"]
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
            
            # Retrain on the current fold's training data
            history = best_model.fit(
                X_train_augmented, Y_train,
                validation_data=(X_val, Y_val),
                epochs=5,
                batch_size=best_config["batch_size"],
                callbacks=[early_stopping, reduce_lr],
                verbose=1
            )
            
            # Evaluate on validation set
            y_val_pred = best_model.predict(X_val)
            final_auc = roc_auc_score(Y_val[:, 1], y_val_pred[:, 1])
            print(f'Final AUC on validation data for fold {fold}: {final_auc:.4f}')
            
            # Store the result
            fold_results.append(final_auc)
        
        # Compute the average AUC across all folds
        average_auc = np.mean(fold_results)
        print(f'\nAverage AUC across all {n_splits} folds: {average_auc:.4f}')
        
        return average_auc

    @staticmethod
    def train_model(config, X_train, Y_train, X_val, Y_val):
        # Unpack hyperparameters from the config dictionary
        learning_rate = config["learning_rate"]
        batch_size = config["batch_size"]
        dropout_rate = config["dropout_rate"]
        l2_reg = config["l2_reg"]
        flip_prob = config["flip_prob"]
        rotate_prob = config["rotate_prob"]

        # Apply data augmentation with specified probabilities
        X_train_augmented = DataLoader.augment_data(X_train, flip_prob=flip_prob, rotate_prob=rotate_prob)

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
            X_train_augmented, Y_train,
            validation_data=(X_val, Y_val),
            epochs=800,
            batch_size=batch_size,
            callbacks=[
                early_stopping,
                reduce_lr,
                report_checkpoint_callback
            ],
            verbose=0
        )

        # Report metrics to Ray Tune
        val_loss, val_accuracy, val_auc = model.evaluate(X_val, Y_val, verbose=0)
        tune.report(val_loss=val_loss, val_accuracy=val_accuracy, val_auc=val_auc)

        # After training, clear the session and collect garbage to free memory
        tf.keras.backend.clear_session()
        import gc
        gc.collect()


def main():
    task = 'cd'  # Update as per your task
    modality = 'MRI'  # 'MRI' or 'PET'
    info = 'test'  # Additional info for saving results

    # Load your data
    train_data, train_label, original_imgs = DataLoader.loading_mask_3d(task, modality)
    X = np.array(train_data)
    X = np.expand_dims(X, axis=-1)  # Add channel dimension if not already present
    Y = to_categorical(train_label, num_classes=2)

    # Train the model with hyperparameter tuning and get the best trained model
    best_model = Trainer.tune_model_nested_cv(X, Y, task, modality, info)

    # Shutdown Ray
    ray.shutdown()


if __name__ == '__main__':
    main()

