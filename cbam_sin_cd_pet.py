import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import (Conv3D, Input, LeakyReLU, Add,
                                     GlobalAveragePooling3D, Dense, Dropout,
                                     SpatialDropout3D, BatchNormalization,
                                     GlobalMaxPooling3D, Reshape, multiply, Concatenate)
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

# Import your own data loading functions
from data_loading import generate_data_path_less, generate, binarylabel

# Utility function to ensure a directory exists
def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

class DataLoader:
    def __init__(self, task, modality):
        self.task = task
        self.modality = modality
        self.original_imgs = []  # List to store original images
        self.target_shape = (128, 128, 128)

    # Function to resize an image to the target shape using interpolation
    def resize_image(self, image, target_shape):
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

    # Function to load and preprocess data
    def load_data(self):
        images_pet, images_mri, labels = generate_data_path_less()
        if self.modality == 'PET':
            data_paths, data_labels = generate(images_pet, labels, self.task)
        elif self.modality == 'MRI':
            data_paths, data_labels = generate(images_mri, labels, self.task)
        else:
            raise ValueError(f"Unsupported modality: {self.modality}")

        data = []
        for i in range(len(data_paths)):
            nifti_img = nib.load(data_paths[i])
            self.original_imgs.append(nifti_img)  # Store the original NIfTI image

            reshaped_data = nifti_img.get_fdata()
            reshaped_data = zscore(reshaped_data, axis=None)

            # Resize the image to the target shape
            resized_data = self.resize_image(reshaped_data, self.target_shape)
            data.append(resized_data)

        data_labels = binarylabel(data_labels, self.task)
        data = np.array(data)

        return data, data_labels

class ModelBuilder:
    def __init__(self, input_shape=(128, 128, 128, 1), num_classes=2):
        self.input_shape = input_shape
        self.num_classes = num_classes

    # CBAM Block
    @staticmethod
    def cbam_block(x, reduction_ratio=8):
        # Channel Attention Module
        channel_avg_pool = GlobalAveragePooling3D()(x)
        channel_max_pool = GlobalMaxPooling3D()(x)

        shared_mlp = Dense(x.shape[-1] // reduction_ratio, activation='relu')
        
        channel_avg = shared_mlp(channel_avg_pool)
        channel_max = shared_mlp(channel_max_pool)

        channel_avg = Dense(x.shape[-1], activation='sigmoid')(channel_avg)
        channel_max = Dense(x.shape[-1], activation='sigmoid')(channel_max)

        channel_attention = Add()([channel_avg, channel_max])
        channel_attention = Reshape((1, 1, 1, x.shape[-1]))(channel_attention)
        x = multiply([x, channel_attention])

        # Spatial Attention Module
        avg_pool = tf.reduce_mean(x, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(x, axis=-1, keepdims=True)

        spatial_concat = Concatenate(axis=-1)([avg_pool, max_pool])
        spatial_attention = Conv3D(1, kernel_size=(7, 7, 7), strides=(1, 1, 1),
                                   padding='same', activation='sigmoid')(spatial_concat)
        x = multiply([x, spatial_attention])

        return x

    # Function for a single convolution block with Batch Normalization
    def convolution_block(self, x, filters, kernel_size=(3, 3, 3), strides=(1, 1, 1), 
                          l2_reg=1e-5, use_cbam=False):
        x = Conv3D(filters, kernel_size, strides=strides, padding='same',
                   kernel_regularizer=l2(l2_reg))(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        if use_cbam:
            x = self.cbam_block(x)  # Apply CBAM if specified
        return x

    # Context module: two convolution blocks with optional dropout
    def context_module(self, x, filters, dropout_rate=0.2, l2_reg=1e-5, use_cbam=False):
        x = self.convolution_block(x, filters, l2_reg=l2_reg, use_cbam=use_cbam)
        x = SpatialDropout3D(dropout_rate)(x)
        x = self.convolution_block(x, filters, l2_reg=l2_reg, use_cbam=use_cbam)
        return x

    # Create the model with CBAM
    def create_model(self, dropout_rate=0.1, l2_reg=8.5475e-05, reduction_ratio=4):
        input_img = Input(shape=self.input_shape)

        # Conv1 block (16 filters) with CBAM
        x = self.convolution_block(input_img, 16, l2_reg=l2_reg, use_cbam=True)
        conv1_out = x
        x = self.context_module(x, 16, dropout_rate=dropout_rate, l2_reg=l2_reg, use_cbam=True)
        x = Add()([x, conv1_out])

        # Conv2 block (32 filters, stride 2) with CBAM
        x = self.convolution_block(x, 32, strides=(2, 2, 2), l2_reg=l2_reg, use_cbam=True)
        conv2_out = x
        x = self.context_module(x, 32, dropout_rate=dropout_rate, l2_reg=l2_reg, use_cbam=True)
        x = Add()([x, conv2_out])

        # Conv3 block (64 filters, stride 2) without CBAM
        x = self.convolution_block(x, 64, strides=(2, 2, 2), l2_reg=l2_reg, use_cbam=False)
        conv3_out = x
        x = self.context_module(x, 64, dropout_rate=dropout_rate, l2_reg=l2_reg, use_cbam=False)
        x = Add()([x, conv3_out])

        # Conv4 block (128 filters, stride 2) with CBAM
        x = self.convolution_block(x, 128, strides=(2, 2, 2), l2_reg=l2_reg, use_cbam=True)
        conv4_out = x
        x = self.context_module(x, 128, dropout_rate=dropout_rate, l2_reg=l2_reg, use_cbam=True)
        x = Add()([x, conv4_out])

        # Conv5 block (256 filters, stride 2) without CBAM
        x = self.convolution_block(x, 256, strides=(2, 2, 2), l2_reg=l2_reg, use_cbam=False)
        conv5_out = x
        x = self.context_module(x, 256, dropout_rate=dropout_rate, l2_reg=l2_reg, use_cbam=False)
        x = Add()([x, conv5_out])

        # Global Average Pooling
        x = GlobalAveragePooling3D()(x)

        # Dropout for regularization
        x = Dropout(dropout_rate)(x)

        # Dense layer with softmax for classification
        output = Dense(self.num_classes, activation='softmax')(x)

        model = Model(inputs=input_img, outputs=output)
        model.summary()

        return model

class Trainer:
    def __init__(self, model_builder, data_loader, task, modality, info):
        self.model_builder = model_builder
        self.data_loader = data_loader
        self.task = task
        self.modality = modality
        self.info = info
        self.save_dir = os.path.join('./grad-cam', self.info, self.task, self.modality)
        ensure_directory_exists(self.save_dir)
        self.histories = []
        self.best_auc = 0
        self.best_model = None

    # Function to perform data augmentation by flipping and rotating images
    def augment_data(self, X, flip_prob=0.5, rotate_prob=0.5):
        """Applies random flipping and rotation to the data based on given probabilities."""
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

    # Function to plot training and validation loss and save the figure
    def plot_training_validation_loss(self):
        # Initialize lists to collect losses
        train_losses = []
        val_losses = []

        for history in self.histories:
            train_losses.extend(history.history['loss'])
            val_losses.extend(history.history['val_loss'])

        # Create the plot
        plt.figure(figsize=(10, 6))

        # Plot the training and validation loss
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')

        # Add title and labels
        plt.title('Training Loss vs Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')

        # Set y-axis limits to focus on 0-1 range
        plt.ylim(0, 1)

        # Add a legend
        plt.legend(loc='upper right')

        # Save the plot
        loss_plot_path = os.path.join(self.save_dir, 'loss_vs_val_loss.png')
        plt.savefig(loss_plot_path)
        plt.close()  # Close the figure to avoid displaying it in notebooks
        print(f'Loss vs Validation Loss plot saved at {loss_plot_path}')

    # Function to train the model and return the best trained model
    def train_model(self, X, Y):
        stratified_kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=2)
        all_y_val = []
        all_y_val_pred = []
        all_auc_scores = []

        for fold_num, (train_idx, val_idx) in enumerate(stratified_kfold.split(X, Y.argmax(axis=1))):
            print(f"Starting fold {fold_num + 1}")
            X_train, X_val = X[train_idx], X[val_idx]
            Y_train, Y_val = Y[train_idx], Y[val_idx]

            # Apply data augmentation to X_train
            X_train_augmented = self.augment_data(X_train)

            model = self.model_builder.create_model()
            model.compile(optimizer=Adam(learning_rate=0.00067187),
                          loss='categorical_crossentropy',
                          metrics=['accuracy', AUC(name='auc')])

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

            history = model.fit(X_train_augmented, Y_train,
                                batch_size=5,
                                epochs=800,  # Adjust epochs as needed
                                validation_data=(X_val, Y_val),
                                callbacks=[early_stopping, reduce_lr])
            self.histories.append(history)

            # Get predictions after training
            y_val_pred = model.predict(X_val)
            y_val_pred_auc = roc_auc_score(Y_val[:, 1], y_val_pred[:, 1])

            # Track the best validation AUC during training for comparison
            best_val_auc = max(history.history['val_auc'])
            print(f'Best val_auc during training for fold {fold_num + 1}: {best_val_auc:.4f}')

            # Calculate and print final AUC after restoring the best weights
            final_auc = roc_auc_score(Y_val[:, 1], y_val_pred[:, 1])
            print(f'Final AUC for fold {fold_num + 1}: {final_auc:.4f}')

            if final_auc > self.best_auc:
                self.best_auc = final_auc
                self.best_model = model  # Save the best model

            all_y_val.extend(Y_val[:, 1])
            all_y_val_pred.extend(y_val_pred[:, 1])
            all_auc_scores.append(final_auc)

        # Plot and save loss vs validation loss graph
        self.plot_training_validation_loss()

        # Compute average AUC across all folds
        average_auc = sum(all_auc_scores) / len(all_auc_scores)
        print(f'Average AUC across all folds: {average_auc:.4f}')

        return self.best_model

class GradCAMVisualizer:
    def __init__(self, model, data_loader, task, modality, info):
        self.model = model
        self.data_loader = data_loader
        self.task = task
        self.modality = modality
        self.info = info
        self.save_dir = os.path.join('./grad-cam', self.info, self.task, self.modality)
        ensure_directory_exists(self.save_dir)

    # Function to compute Grad-CAM for a given layer and class index
    def make_gradcam_heatmap(self, img, last_conv_layer_name, pred_index=None):
        grad_model = tf.keras.models.Model(
            [self.model.inputs], [self.model.get_layer(last_conv_layer_name).output, self.model.output]
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

    # Function to save Grad-CAM heatmap and plot glass brain
    def save_gradcam(self, heatmap, original_img, layer_name, class_idx):
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
        nifti_file_name = f"gradcam_{self.task}_{self.modality}_class{class_idx}_{layer_name}.nii.gz"
        nifti_save_path = os.path.join(self.save_dir, nifti_file_name)
        nib.save(heatmap_img, nifti_save_path)
        print(f'3D Grad-CAM heatmap saved at {nifti_save_path}')

        # Plot the glass brain
        output_glass_brain_path = os.path.join(self.save_dir, f'glass_brain_{self.task}_{self.modality}_{layer_name}_class{class_idx}.png')
        plotting.plot_glass_brain(heatmap_img, colorbar=True, plot_abs=True,
                                  cmap='jet', output_file=output_glass_brain_path)
        print(f'Glass brain plot saved at {output_glass_brain_path}')

    # Function to apply Grad-CAM for all layers across all dataset images and save averaged heatmaps
    def apply_gradcam_all_layers_average(self, imgs):
        # Identify convolutional layers
        conv_layers = []
        cumulative_scales = []
        cumulative_scale = 1
        for layer in self.model.layers:
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
                    heatmap = self.make_gradcam_heatmap(img, conv_layer_name,
                                                        pred_index=class_idx)
                    print(f"Heatmap shape for layer {conv_layer_name} and class {class_idx}: {heatmap.shape}")
                    if accumulated_heatmap is None:
                        accumulated_heatmap = heatmap
                    else:
                        accumulated_heatmap += heatmap

                avg_heatmap = accumulated_heatmap / len(imgs)
                # Use the first original image as reference
                self.save_gradcam(avg_heatmap, self.data_loader.original_imgs[0],
                                  conv_layer_name, class_idx)

class MedicalImageClassifier:
    def __init__(self, task, modality, info):
        self.task = task
        self.modality = modality
        self.info = info
        self.data_loader = DataLoader(task, modality)
        self.model_builder = ModelBuilder()
        self.trainer = Trainer(self.model_builder, self.data_loader, task, modality, info)
        self.gradcam_visualizer = None  # Will be initialized after training

    # Main execution function
    def run(self):
        # Load your data
        train_data, train_label = self.data_loader.load_data()
        X = np.array(train_data)
        X = np.expand_dims(X, axis=-1)  # Add channel dimension if not already present
        Y = to_categorical(train_label, num_classes=2)

        # Train the model and get the best trained model
        best_model = self.trainer.train_model(X, Y)

        # Initialize GradCAMVisualizer with the best model
        self.gradcam_visualizer = GradCAMVisualizer(best_model, self.data_loader, self.task, self.modality, self.info)

        # Apply Grad-CAM using the trained model
        imgs = [np.expand_dims(X[i], axis=0) for i in range(X.shape[0])]

        # Apply Grad-CAM to all images
        self.gradcam_visualizer.apply_gradcam_all_layers_average(imgs)

# Main execution
if __name__ == '__main__':
    task = 'cd'  # Update as per your task
    modality = 'PET'  # 'MRI' or 'PET'
    info = 'cbam_single'  # Additional info for saving results

    classifier = MedicalImageClassifier(task, modality, info)
    classifier.run()

