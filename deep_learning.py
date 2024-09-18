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
from nilearn.input_data import NiftiMasker
from nilearn import plotting
from scipy.stats import zscore
from scipy.ndimage import zoom

# Import your own data loading functions
from data_loading import generate_data_path, generate, binarylabel

# Function to ensure a directory exists
def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Function for a single convolution block with Batch Normalization
def convolution_block(x, filters, kernel_size=(3, 3, 3), strides=(1, 1, 1)):
    x = Conv3D(filters, kernel_size, strides=strides, padding='same',
               kernel_regularizer=l2(1e-3))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    return x

# Context module: two convolution blocks with optional dropout
def context_module(x, filters):
    x = convolution_block(x, filters)
    x = SpatialDropout3D(0.4)(x)
    x = convolution_block(x, filters)
    return x

# Full 3D CNN classification architecture with context blocks
def create_3d_cnn(input_shape=(128, 128, 128, 1), num_classes=2):
    input_img = Input(shape=input_shape)

    # Conv1 block (16 filters)
    x = convolution_block(input_img, 16)
    conv1_out = x
    x = context_module(x, 16)
    x = Add()([x, conv1_out])

    # Conv2 block (32 filters, stride 2)
    x = convolution_block(x, 32, strides=(2, 2, 2))
    conv2_out = x
    x = context_module(x, 32)
    x = Add()([x, conv2_out])

    # Conv3 block (64 filters, stride 2)
    x = convolution_block(x, 64, strides=(2, 2, 2))
    conv3_out = x
    x = context_module(x, 64)
    x = Add()([x, conv3_out])

    # Conv4 block (128 filters, stride 2)
    x = convolution_block(x, 128, strides=(2, 2, 2))
    conv4_out = x
    x = context_module(x, 128)
    x = Add()([x, conv4_out])

    # Conv5 block (256 filters, stride 2)
    x = convolution_block(x, 256, strides=(2, 2, 2))
    conv5_out = x
    x = context_module(x, 256)
    x = Add()([x, conv5_out])

    # Global Average Pooling
    x = GlobalAveragePooling3D()(x)

    # Dropout for regularization
    x = Dropout(0.4)(x)

    # Dense layer with softmax for classification
    output = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=input_img, outputs=output)
    model.summary()

    return model

# Function to resize an image to the target shape using interpolation
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

# Function to compute Grad-CAM for a given layer and class index
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

# Function to plot training and validation loss and save the figure
def plot_training_validation_loss(histories, save_dir):
    # Initialize lists to collect losses
    train_losses = []
    val_losses = []

    for history in histories:
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
    loss_plot_path = os.path.join(save_dir, 'loss_vs_val_loss.png')
    plt.savefig(loss_plot_path)
    plt.close()  # Close the figure to avoid displaying it in notebooks
    print(f'Loss vs Validation Loss plot saved at {loss_plot_path}')


# Function to save Grad-CAM heatmap and plot glass brain
def save_gradcam(heatmap, original_img,
                 task, modality, layer_name, class_idx, info, save_dir='./grad-cam'):
    save_dir = os.path.join(save_dir, info, task, modality)
    ensure_directory_exists(save_dir)

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

# Function to apply Grad-CAM for all layers across all dataset images and save averaged heatmaps
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
                heatmap = make_gradcam_heatmap(model, img, conv_layer_name,
                                               pred_index=class_idx)
                print(f"Heatmap shape for layer {conv_layer_name} and class {class_idx}: {heatmap.shape}")
                if accumulated_heatmap is None:
                    accumulated_heatmap = heatmap
                else:
                    accumulated_heatmap += heatmap

            avg_heatmap = accumulated_heatmap / len(imgs)
            # Use the first original image as reference
            save_gradcam(avg_heatmap, original_imgs[0],
                         task, modality, conv_layer_name, class_idx, info)

# Function to perform data augmentation by flipping images along axes
def augment_data(X):
    augmented_X = []
    for img in X:
        img_aug = img.copy()
        # Randomly flip along each axis with 50% probability
        if np.random.rand() < 0.5:
            img_aug = np.flip(img_aug, axis=1)  # Flip along x-axis
        if np.random.rand() < 0.5:
            img_aug = np.flip(img_aug, axis=2)  # Flip along y-axis
        if np.random.rand() < 0.5:
            img_aug = np.flip(img_aug, axis=3)  # Flip along z-axis
        augmented_X.append(img_aug)
    return np.array(augmented_X)

# Function to train the model and return the best trained model
def train_model(X, Y, task, modality, info):
    stratified_kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=2)
    all_y_val = []
    all_y_val_pred = []
    all_auc_scores = []
    histories = []

    # Directory to save loss vs val-loss plot
    save_dir = os.path.join('./grad-cam', info, task, modality)
    ensure_directory_exists(save_dir)

    best_auc = 0
    best_model = None

    for fold_num, (train_idx, val_idx) in enumerate(stratified_kfold.split(X, Y.argmax(axis=1))):
        print(f"Starting fold {fold_num + 1}")
        X_train, X_val = X[train_idx], X[val_idx]
        Y_train, Y_val = Y[train_idx], Y[val_idx]

        # Apply data augmentation to X_train
        X_train_augmented = augment_data(X_train)

        model = create_3d_cnn(input_shape=(128, 128, 128, 1), num_classes=2)
        model.compile(optimizer=Adam(learning_rate=5e-4),
                      loss='categorical_crossentropy',
                      metrics=['accuracy', AUC(name='auc')])

        early_stopping = EarlyStopping(
            monitor='val_auc',
            patience=50,
            mode='max',
            verbose=1,
            restore_best_weights=True
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_auc',
            factor=0.5,
            patience=10,
            mode='max',
            verbose=1
        )

        history = model.fit(X_train_augmented, Y_train,
                            batch_size=5,
                            epochs=800,  # Adjust epochs as needed
                            validation_data=(X_val, Y_val),
                            callbacks=[early_stopping, reduce_lr])
        histories.append(history)

        # Get predictions after training
        y_val_pred = model.predict(X_val)
        y_val_pred_auc = roc_auc_score(Y_val[:, 1], y_val_pred[:, 1])

        # Track the best validation AUC during training for comparison
        best_val_auc = max(history.history['val_auc'])
        print(f'Best val_auc during training for fold {fold_num + 1}: {best_val_auc:.4f}')

        # Calculate and print final AUC after restoring the best weights
        final_auc = roc_auc_score(Y_val[:, 1], y_val_pred[:, 1])
        print(f'Final AUC for fold {fold_num + 1}: {final_auc:.4f}')

        if final_auc > best_auc:
            best_auc = final_auc
            best_model = model  # Save the best model

        all_y_val.extend(Y_val[:, 1])
        all_y_val_pred.extend(y_val_pred[:, 1])
        all_auc_scores.append(final_auc)

    # Plot and save loss vs validation loss graph
    plot_training_validation_loss(histories, save_dir)

    # Compute average AUC across all folds
    average_auc = sum(all_auc_scores) / len(all_auc_scores)
    print(f'Average AUC across all folds: {average_auc:.4f}')

    return best_model

# Function to load data
def loading_mask_3d(task, modality):
    images_pet, images_mri, labels = generate_data_path()
    original_imgs = []  # Initialize the list to store original images

    if modality == 'PET':
        data_train, train_label = generate(images_pet, labels, task)
    elif modality == 'MRI':
        data_train, train_label = generate(images_mri, labels, task)
    else:
        raise ValueError(f"Unsupported modality: {modality}")

    # Update the mask path to your actual mask image path
    mask_path = '/home/l.peiwang/MR-PET-Classfication/mask_gm_p4_new4.nii'  # Replace with your actual mask path
    masker = NiftiMasker(mask_img=mask_path)
    mask_affine = nib.load(mask_path).affine
    print(f"Mask affine:\n{mask_affine}")

    train_data = []
    target_shape = (128, 128, 128)

    from nibabel.orientations import aff2axcodes

    for i in range(len(data_train)):
        nifti_img = nib.load(data_train[i])
        affine = nifti_img.affine
        original_imgs.append(nifti_img)  # Store the original NIfTI image
      
        image_orientation = aff2axcodes(nifti_img.affine)
        mask_orientation = aff2axcodes(mask_affine)
        print(f"Original image orientation: {image_orientation}")
        print(f"Mask image orientation: {mask_orientation}")

        if image_orientation != mask_orientation:
            nifti_img = nib.as_closest_canonical(nifti_img)
            affine = nifti_img.affine  # Update affine after reorientation

        # Apply the mask
        masked_data = masker.fit_transform(nifti_img)
        reshaped_data = masker.inverse_transform(masked_data).get_fdata()
        reshaped_data = zscore(reshaped_data, axis=None)

        # Resize the image to (128, 128, 128)
        resized_data = resize_image(reshaped_data, target_shape)

        train_data.append(resized_data)

    train_label = binarylabel(train_label, task)
    train_data = np.array(train_data)

    return train_data, train_label, masker, original_imgs

# Main execution
if __name__ == '__main__':
    task = 'pc'  # Update as per your task
    modality = 'MRI'  # 'MRI' or 'PET'
    info = '5_context_from_16_0.4_dropout_1e3_with_augmentation'  # Additional info for saving results

    # Load your data
    train_data, train_label, masker, original_imgs = loading_mask_3d(task, modality)
    X = np.array(train_data)
    X = np.expand_dims(X, axis=-1)  # Add channel dimension if not already present
    Y = to_categorical(train_label, num_classes=2)

    # Train the model and get the best trained model
    best_model = train_model(X, Y, task, modality, info)

    # Apply Grad-CAM using the trained model
    imgs = [np.expand_dims(X[i], axis=0) for i in range(X.shape[0])]

    # Apply Grad-CAM to all images
    apply_gradcam_all_layers_average(best_model, imgs, original_imgs, task, modality, info)


