import os
import numpy as np
import nibabel as nib
import tensorflow as tf
from tensorflow.keras.layers import (Input, Dense, LayerNormalization, Dropout, Add, Reshape)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.metrics import AUC
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import StratifiedKFold
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
        zoom_factors = [target_shape[i] / image.shape[i] for i in range(3)]
        resized_image = zoom(image, zoom_factors, order=1)
        return resized_image

class DataLoader:
    """Class to handle data loading and preprocessing."""

    @staticmethod
    def loading_mask_3d(task, modality):
        images_pet, images_mri, labels = generate_data_path_less()
        original_imgs = []

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
            original_imgs.append(nifti_img)

            reshaped_data = nifti_img.get_fdata()
            reshaped_data = zscore(reshaped_data, axis=None)

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
            if np.random.rand() < flip_prob:
                img_aug = np.flip(img_aug, axis=0)
            if np.random.rand() < flip_prob:
                img_aug = np.flip(img_aug, axis=1)
            if np.random.rand() < flip_prob:
                img_aug = np.flip(img_aug, axis=2)

            if np.random.rand() < rotate_prob:
                angle = np.random.uniform(-10, 10)
                axes = np.random.choice([(0, 1), (0, 2), (1, 2)])
                img_aug = rotate(img_aug, angle, axes=axes, reshape=False, order=1)

            augmented_X.append(img_aug)
        return np.array(augmented_X)

class ViTModelBuilder:
    """Class to build the Vision Transformer model."""

    def __init__(self, input_shape=(128, 128, 128), num_classes=2, patch_size=(16, 16, 16),
                 d_model=128, num_heads=8, d_ff=256, num_layers=8, dropout_rate=0.1):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.d_model = d_model  # Embedding dimension
        self.num_heads = num_heads
        self.d_ff = d_ff  # Hidden layer size in MLP
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate

    # Custom Layer for Class Token
    class ClassToken(tf.keras.layers.Layer):
        def build(self, input_shape):
            self.cls_token = self.add_weight(
                shape=(1, 1, input_shape[-1]),
                initializer='random_normal',
                trainable=True,
                name='cls_token'
            )
            super().build(input_shape)

        def call(self, inputs):
            batch_size = tf.shape(inputs)[0]
            cls_tokens = tf.broadcast_to(self.cls_token, [batch_size, 1, inputs.shape[-1]])
            return tf.concat([cls_tokens, inputs], axis=1)

    # Custom Layer for Positional Embeddings
    class AddPositionEmbs(tf.keras.layers.Layer):
        def build(self, input_shape):
            self.pos_emb = self.add_weight(
                shape=(1, input_shape[1], input_shape[2]),
                initializer='random_normal',
                trainable=True,
                name='pos_embedding'
            )
            super().build(input_shape)

        def call(self, inputs):
            return inputs + self.pos_emb

    # Transformer block adapted for ViT
    def transformer_block(self, inputs):
        # Layer normalization
        x_norm1 = LayerNormalization(epsilon=1e-6)(inputs)
        # Multi-Head Self-Attention
        attention_output = tf.keras.layers.MultiHeadAttention(
            num_heads=self.num_heads, key_dim=self.d_model // self.num_heads
        )(x_norm1, x_norm1)
        # Dropout
        attention_output = Dropout(self.dropout_rate)(attention_output)
        # Residual connection
        x1 = Add()([inputs, attention_output])

        # Layer normalization
        x_norm2 = LayerNormalization(epsilon=1e-6)(x1)
        # MLP with two layers
        mlp_output = Dense(self.d_ff, activation='gelu')(x_norm2)
        mlp_output = Dropout(self.dropout_rate)(mlp_output)
        mlp_output = Dense(self.d_model)(mlp_output)
        # Dropout
        mlp_output = Dropout(self.dropout_rate)(mlp_output)
        # Residual connection
        x2 = Add()([x1, mlp_output])

        return x2

    # Patch embedding layer
    def patch_embedding_layer(self):
        inputs = Input(shape=self.input_shape) 
        # Compute number of patches
        num_patches = (self.input_shape[0] // self.patch_size[0]) * \
                      (self.input_shape[1] // self.patch_size[1]) * \
                      (self.input_shape[2] // self.patch_size[2])

        # Extract patches and flatten them
        patches = tf.image.extract_patches(
            images=inputs,
            sizes=[1, self.patch_size[0], self.patch_size[1], self.patch_size[2], 1],
            strides=[1, self.patch_size[0], self.patch_size[1], self.patch_size[2], 1],
            rates=[1, 1, 1, 1, 1],
            padding='VALID'
        )
        patches_shape = tf.shape(patches)
        patch_volume = patches_shape[-1]
        x = tf.reshape(patches, [patches_shape[0], num_patches, patch_volume])

        # Linear projection of flattened patches to embedding dimension
        x = Dense(self.d_model)(x)

        return Model(inputs=inputs, outputs=x, name='patch_embedding')

    # Create the ViT model
    def create_model(self):
        inputs = Input(shape=self.input_shape + (1,))

        # Patch Embedding
        x = self.patch_embedding_layer()(inputs)

        # Add class token and positional embeddings
        x = self.ClassToken()(x)
        x = self.AddPositionEmbs()(x)

        # Transformer layers
        for _ in range(self.num_layers):
            x = self.transformer_block(x)

        # Layer normalization
        x = LayerNormalization(epsilon=1e-6)(x)

        # Use the class token output for classification
        class_token_output = x[:, 0]

        # Classification head (single linear layer)
        output = Dense(self.num_classes, activation='softmax')(class_token_output)

        model = Model(inputs=inputs, outputs=output)
        model.summary()

        return model

class Trainer:
    """Class to handle model training and evaluation."""

    @staticmethod
    def build_model(hp):
        """Builds and compiles the model using hyperparameters from Keras Tuner."""
        # Sample hyperparameters
        dropout_rate = hp.Float('dropout_rate', 0.0, 0.5, step=0.1)
        num_layers = hp.Int('num_layers', min_value=4, max_value=8, step=2)
        d_model = hp.Choice('d_model', [64, 128, 256])
        num_heads = hp.Choice('num_heads', [4, 8])
        d_ff = hp.Choice('d_ff', [128, 256, 512])
        patch_size = hp.Choice('patch_size', [(8, 8, 8), (16, 16, 16)])

        # Ensure d_model is divisible by num_heads
        if d_model % num_heads != 0:
            d_model = num_heads * (d_model // num_heads)

        # Build the model
        vit_builder = ViTModelBuilder(
            input_shape=(128, 128, 128),
            num_classes=2,
            patch_size=patch_size,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            num_layers=num_layers,
            dropout_rate=dropout_rate
        )
        model = vit_builder.create_model()

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
        stratified_kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=2)

        fold_results = []
        best_hyperparameters = []

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
                )
            ]

            # Perform hyperparameter search
            tuner.search(
                X_train_augmented, Y_train,
                validation_data=(X_val, Y_val),
                epochs=80,
                batch_size=5,
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
            print(f"d_ff (Feed-Forward Dimension): {best_hps.get('d_ff')}")

            # Build a new model with the best hyperparameters
            final_model = Trainer.build_model(best_hps)

            # Define callbacks for final training
            final_callbacks = [
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
                )
            ]

            # Train the final model on augmented data
            history = final_model.fit(
                X_train_augmented, Y_train,
                validation_data=(X_val, Y_val),
                epochs=250,
                batch_size=5,
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
    task = 'cd'  # Update as per your task
    modality = 'MRI'  # 'MRI' or 'PET'
    info = 'vit_model_final'  # Additional info for saving results

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

