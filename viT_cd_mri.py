import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import (Input, Dense, Conv3D, Dropout, LayerNormalization,
                                     Add, Reshape, MultiHeadAttention)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.metrics import AUC
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from nilearn import plotting
from scipy.stats import zscore
from scipy.ndimage import zoom, rotate
import keras_tuner as kt
import random

# Import your own data loading functions
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
        """Applies random flipping and rotation to the data based on given probabilities."""
        augmented_X = []
        for img in X:
            img_aug = img.copy()
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
                axes = np.random.choice([(0, 1), (0, 2), (1, 2)])
                img_aug = rotate(img_aug, angle, axes=axes, reshape=False, order=1)

            augmented_X.append(img_aug)
        return np.array(augmented_X)

class ViTModel:
    """Class to create and manage the ViT model for 3D images."""

    @staticmethod
    def transformer_block(x, num_heads, mlp_dim, dropout_rate):
        # Layer normalization
        x_norm1 = LayerNormalization(epsilon=1e-6)(x)
        # Multi-Head Self-Attention
        attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=x.shape[-1] // num_heads)(x_norm1, x_norm1)
        # Dropout
        attention_output = Dropout(dropout_rate)(attention_output)
        # Residual connection
        x1 = Add()([x, attention_output])

        # Layer normalization
        x_norm2 = LayerNormalization(epsilon=1e-6)(x1)
        # MLP
        mlp_output = Dense(mlp_dim, activation='gelu')(x_norm2)
        mlp_output = Dropout(dropout_rate)(mlp_output)
        mlp_output = Dense(x.shape[-1])(mlp_output)
        mlp_output = Dropout(dropout_rate)(mlp_output)
        # Residual connection
        x2 = Add()([x1, mlp_output])

        return x2

    @staticmethod
    def create_model(input_shape=(128, 128, 128, 1), num_classes=2, patch_size=(16, 16, 16),
                     num_layers=8, d_model=128, num_heads=8, mlp_dim=512, dropout_rate=0.1):
        inputs = Input(shape=input_shape)

        # Compute number of patches
        num_patches = (input_shape[0] // patch_size[0]) * \
                      (input_shape[1] // patch_size[1]) * \
                      (input_shape[2] // patch_size[2])

        # Patch embedding
        x = Conv3D(filters=d_model, kernel_size=patch_size, strides=patch_size, padding='valid')(inputs)
        x = Reshape((num_patches, d_model))(x)

        # Class token
        class_token = tf.Variable(tf.zeros((1, 1, d_model)), trainable=True, name='class_token')
        batch_size = tf.shape(x)[0]
        class_tokens = tf.broadcast_to(class_token, [batch_size, 1, d_model])
        x = tf.concat([class_tokens, x], axis=1)

        # Positional embeddings
        num_positions = num_patches + 1  # +1 for the class token
        positional_embedding = tf.Variable(
            initial_value=tf.random.normal(shape=(1, num_positions, d_model)),
            trainable=True,
            name='positional_embedding'
        )
        x = x + positional_embedding

        # Transformer blocks
        for _ in range(num_layers):
            x = ViTModel.transformer_block(x, num_heads, mlp_dim, dropout_rate)

        # Final layer normalization
        x = LayerNormalization(epsilon=1e-6)(x)
        # Classification head
        class_token_output = x[:, 0]  # Extract the class token
        x = Dense(mlp_dim, activation='gelu')(class_token_output)
        x = Dropout(dropout_rate)(x)
        outputs = Dense(num_classes, activation='softmax')(x)

        model = Model(inputs=inputs, outputs=outputs)
        model.summary()

        return model

class Trainer:
    """Class to handle model training and evaluation."""

    @staticmethod
    def build_model(hp):
        """Builds and compiles the model using hyperparameters from Keras Tuner."""
        # Sample hyperparameters
        learning_rate = hp.Float('learning_rate', 1e-5, 1e-3, sampling='log')
        dropout_rate = hp.Float('dropout_rate', 0.1, 0.5, step=0.1)
        num_layers = hp.Int('num_layers', min_value=4, max_value=8, step=2)
        d_model = hp.Choice('d_model', [64, 128, 256])
        num_heads = hp.Choice('num_heads', [4, 8])
        mlp_dim = hp.Choice('mlp_dim', [128, 256, 512])

        # Ensure d_model is divisible by num_heads
        if d_model % num_heads != 0:
            d_model = num_heads * (d_model // num_heads)
        
        # Build the model
        model = ViTModel.create_model(
            input_shape=(128, 128, 128, 1),
            num_classes=2,
            patch_size=(16, 16, 16),  # Adjust patch size if needed
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            dropout_rate=dropout_rate
        )
        
        # Compile the model
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy', AUC(name='auc')]
        )
        return model

    @staticmethod
    def tune_model_nested_cv(X, Y, task, modality, info, n_splits=3, max_trials=8, executions_per_trial=1):
        """Performs hyperparameter tuning using nested cross-validation."""
        # Define the cross-validation strategy
        stratified_kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=2)

        # Initialize variables to store results
        fold_results = []
        best_hyperparameters = []

        # Iterate over each fold
        for fold, (train_idx, val_idx) in enumerate(stratified_kfold.split(X, Y.argmax(axis=1)), 1):
            print(f"\nStarting fold {fold}/{n_splits}")
            X_train, X_val = X[train_idx], X[val_idx]
            Y_train, Y_val = Y[train_idx], Y[val_idx]
            X_train_augmented = DataLoader.augment_data(
                X_train, 
                flip_prob=0.5, 
                rotate_prob=0.5
            )
            tuner_dir = os.path.join('keras_tuner_dir', task, modality, info, f"fold_{fold}")
            os.makedirs(tuner_dir, exist_ok=True)
            # Define the search space within the build_model function
            tuner = kt.RandomSearch(
                hypermodel=Trainer.build_model,
                objective=kt.Objective("val_auc", direction="max"),
                max_trials=max_trials,
                executions_per_trial=executions_per_trial,
                directory=tuner_dir,
                project_name=f"hyperparam_tuning_fold_{fold}",
                seed=42
            )

            # Define callbacks
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=20,
                    mode='min',
                    verbose=1,
                    restore_best_weights=True
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    mode='min',
                    verbose=1
                )
            ]

            # Perform hyperparameter search
            tuner.search(
                X_train_augmented, Y_train,
                validation_data=(X_val, Y_val),
                epochs=100,  # Set a high number; early stopping will handle it
                batch_size=1,  # Adjust batch size based on memory constraints
                callbacks=callbacks,
                verbose=1,
            )

            # Retrieve the best hyperparameters
            best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
            best_hyperparameters.append(best_hps)
            print(f"Best hyperparameters for fold {fold}:")
            print(f"Learning Rate: {best_hps.get('learning_rate')}")
            print(f"Dropout Rate: {best_hps.get('dropout_rate')}")
            print(f"Number of Layers: {best_hps.get('num_layers')}")
            print(f"d_model (Embedding Dimension): {best_hps.get('d_model')}")
            print(f"Number of Heads: {best_hps.get('num_heads')}")
            print(f"MLP Dimension: {best_hps.get('mlp_dim')}")

            # Build a new model with the best hyperparameters
            final_model = Trainer.build_model(best_hps)

            # Define callbacks for final training
            final_callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=20,
                    mode='min',
                    verbose=1,
                    restore_best_weights=True
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    mode='min',
                    verbose=1
                )
            ]

            # Train the final model on augmented data
            history = final_model.fit(
                X_train_augmented, Y_train,
                validation_data=(X_val, Y_val),
                epochs=100,  # Adjust as needed
                batch_size=1,  # Adjust batch size based on memory constraints
                callbacks=final_callbacks,
                verbose=1
            )

            # Evaluate the final model on validation data
            val_loss, val_accuracy, val_auc = final_model.evaluate(X_val, Y_val, verbose=0)
            print(f"Final Validation AUC for fold {fold}: {val_auc}")

            # Store the result
            fold_results.append(val_auc)

            # Optionally, save the best model
            # final_model.save(f"best_model_fold_{fold}.h5")

        # Compute the average AUC across all folds
        average_auc = np.mean(fold_results)
        print(f'\nAverage AUC across all {n_splits} folds: {average_auc:.4f}')

        return average_auc

def main():
    # Clear the entire keras_tuner_dir to start fresh
    task = 'cd'  # Update as per your task
    modality = 'MRI'  # 'MRI' or 'PET'
    info = 'vit_model'  # Additional info for saving results

    # Load your data
    train_data, train_label, original_imgs = DataLoader.loading_mask_3d(task, modality)
    X = np.array(train_data)
    X = np.expand_dims(X, axis=-1)  # Add channel dimension if not already present
    Y = to_categorical(train_label, num_classes=2)

    # Train the model with hyperparameter tuning and get the average AUC
    average_auc = Trainer.tune_model_nested_cv(X, Y, task, modality, info)
    print(f"Average AUC across all folds: {average_auc:.4f}")

if __name__ == '__main__':
    # Set seeds for reproducibility
    seed = 42
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    main()
