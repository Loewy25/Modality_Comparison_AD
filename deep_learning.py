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
from nilearn.image import resample_to_img
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
    x = SpatialDropout3D(0.57)(x)
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
    x = Dropout(0.57)(x)

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

# Function to adjust affine for resizing
def adjust_affine_for_resizing(original_affine, original_shape, new_shape):
    scales = [original_shape[i] / new_shape[i] for i in range(3)]
    new_affine = original_affine.copy()
    new_affine[:3, :3] = original_affine[:3, :3] @ np.diag(scales)
    return new_affine

# Function to adjust affine for the cumulative scaling of the layer
def adjust_affine_for_layer(original_affine, cumulative_scale):
    scaling_factors = [cumulative_scale] * 3
    new_affine = original_affine.copy()
    new_affine[:3, :3] = original_affine[:3, :3] @ np.diag(scaling_factors)
    return new_affine


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

    # Add a legend
    plt.legend(loc='upper right')

    # Save the plot
    loss_plot_path = os.path.join(save_dir, 'loss_vs_val_loss.png')
    plt.savefig(loss_plot_path)
    plt.close()  # Close the figure to avoid displaying it in notebooks
    print(f'Loss vs Validation Loss plot saved at {loss_plot_path}')

# Function to save Grad-CAM heatmap and plot glass brain using the stored affine
def save_gradcam(heatmap, adjusted_affine, cumulative_scale, original_img,
                 task, modality, layer_name, class_idx, info, save_dir='./grad-cam'):
    save_dir = os.path.join(save_dir, info, task, modality)
    ensure_directory_exists(save_dir)

    # Adjust the affine for the downscaling due to convolutional strides
    print(f"Adjusted affine before layer scaling for layer {layer_name}:\n{adjusted_affine}")
    layer_affine = adjust_affine_for_layer(adjusted_affine, cumulative_scale)
    print(f"Layer affine after applying cumulative scale {cumulative_scale}:\n{layer_affine}")
    heatmap_img = nib.Nifti1Image(heatmap, layer_affine)

    # Resample the heatmap to the space of the original image
    print("Resampling heatmap to original image space...")
    resampled_heatmap_img = resample_to_img(heatmap_img, original_img, interpolation='continuous')

    # Normalize the resampled heatmap
    data = resampled_heatmap_img.get_fdata()
    min_val = np.min(data)
    max_val = np.max(data)
    if max_val - min_val != 0:
        normalized_data = (data - min_val) / (max_val - min_val)
    else:
        normalized_data = np.zeros_like(data)

    resampled_heatmap_img = nib.Nifti1Image(normalized_data, resampled_heatmap_img.affine)

    # Save the 3D NIfTI file
    nifti_file_name = f"gradcam_{task}_{modality}_class{class_idx}_{layer_name}.nii.gz"
    nifti_save_path = os.path.join(save_dir, nifti_file_name)
    nib.save(resampled_heatmap_img, nifti_save_path)
    print(f'3D Grad-CAM heatmap saved at {nifti_save_path}')

    # Print affines and shapes for debugging
    print(f"Resampled heatmap affine:\n{resampled_heatmap_img.affine}")
    print(f"Resampled heatmap shape: {resampled_heatmap_img.shape}")
    print(f"Original image affine:\n{original_img.affine}")
    print(f"Original image shape: {original_img.shape}")

    # Plot the glass brain
    output_glass_brain_path = os.path.join(save_dir, f'glass_brain_{task}_{modality}_{layer_name}_class{class_idx}.png')
    plotting.plot_glass_brain(resampled_heatmap_img, colorbar=True, plot_abs=True,
                              cmap='jet', output_file=output_glass_brain_path)
    print(f'Glass brain plot saved at {output_glass_brain_path}')

# Function to apply Grad-CAM for all layers across all dataset images and save averaged heatmaps
def apply_gradcam_all_layers_average(model, imgs, adjusted_affines, original_imgs,
                                     task, modality, info):
    # Identify convolutional layers
    conv_layers = [layer.name for layer in model.layers if isinstance(layer, Conv3D)]
    # Calculate cumulative scaling factors based on strides
    cumulative_scales = []
    scale = 1
    for layer in model.layers:
        if isinstance(layer, Conv3D):
            strides = layer.strides
            scale *= strides[0]  # Assuming strides are the same in all dimensions
            cumulative_scales.append(scale)

    # Ensure that cumulative_scales and conv_layers have the same length
    cumulative_scales = cumulative_scales[:len(conv_layers)]

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
            # Use the first adjusted affine and original image as reference
            save_gradcam(avg_heatmap, adjusted_affines[0], cumulative_scale,
                         original_imgs[0], task, modality, conv_layer_name, class_idx, info)

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

        model = create_3d_cnn(input_shape=(128, 128, 128, 1), num_classes=2)
        model.compile(optimizer=Adam(learning_rate=5e-4),
                      loss='categorical_crossentropy',
                      metrics=['accuracy', AUC(name='auc')])

        early_stopping = EarlyStopping(monitor='val_loss', patience=50, verbose=1, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1)

        history = model.fit(X_train, Y_train,
                            batch_size=5,
                            epochs=800,  # Reduced epochs for testing
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
    adjusted_affines = []
    original_imgs = []  # Initialize the list to store original images

    if modality == 'PET':
        data_train, train_label = generate(images_pet, labels, task)
    elif modality == 'MRI':
        data_train, train_label = generate(images_mri, labels, task)
    else:
        raise ValueError(f"Unsupported modality: {modality}")

    # Update the mask path to your actual mask image path
    mask_path = '/home/l.peiwang/MR-PET-Classfication/mask_gm_p4_new4.nii'
    masker = NiftiMasker(mask_img=mask_path)
    mask_affine = nib.load(mask_path).affine
    print(f"Mask affine:\n{mask_affine}")

    train_data = []
    target_shape = (128, 128, 128)

    for i in range(len(data_train)):
        print(f"\nProcessing image {i + 1}/{len(data_train)}")
        nifti_img = nib.load(data_train[i])
        affine = nifti_img.affine
        print(f"Original affine for image {i + 1}:\n{affine}")
        original_imgs.append(nifti_img)  # Store the original NIfTI image

        # Check orientation
        from nibabel.orientations import aff2axcodes
        image_orientation = aff2axcodes(nifti_img.affine)
        mask_orientation = aff2axcodes(mask_affine)
        print(f"Original image orientation: {image_orientation}")
        print(f"Mask image orientation: {mask_orientation}")

        if image_orientation != mask_orientation:
            print(f"Reorienting image {i + 1} to match mask orientation...")
            nifti_img = nib.as_closest_canonical(nifti_img)
            affine = nifti_img.affine  # Update affine after reorientation
            print(f"New affine for image {i + 1} after reorientation:\n{affine}")
            print(f"New image orientation: {aff2axcodes(affine)}")

        # Apply the mask
        masked_data = masker.fit_transform(nifti_img)
        reshaped_data = masker.inverse_transform(masked_data).get_fdata()
        reshaped_data = zscore(reshaped_data, axis=None)

        # Resize the image to (128, 128, 128)
        resized_data = resize_image(reshaped_data, target_shape)

        train_data.append(resized_data)
        new_affine = adjust_affine_for_resizing(affine, reshaped_data.shape, target_shape)
        print(f"Adjusted affine after resizing for image {i + 1}:\n{new_affine}")
        adjusted_affines.append(new_affine)

    train_label = binarylabel(train_label, task)
    train_data = np.array(train_data)

    return train_data, train_label, masker, adjusted_affines, original_imgs

# Main execution
if __name__ == '__main__':
    task = 'cd'  # Update as per your task
    modality = 'MRI'  # 'MRI' or 'PET'
    info = '5_context_from_16_0.5_dropout_1e3'  # Additional info for saving results

    # Load your data
    train_data, train_label, masker, adjusted_affines, original_imgs = loading_mask_3d(task, modality)
    X = np.array(train_data)
    X = np.expand_dims(X, axis=-1)  # Add channel dimension if not already present
    Y = to_categorical(train_label, num_classes=2)

    # Train the model and get the best trained model
    best_model = train_model(X, Y, task, modality, info)

    # Apply Grad-CAM using the trained model
    imgs = [np.expand_dims(X[i], axis=0) for i in range(X.shape[0])]

    # Test with a single image to simplify debugging (optional)
    # Uncomment the following lines to test with one image
    # test_img = imgs[0]
    # test_adjusted_affine = adjusted_affines[0]
    # test_original_img = original_imgs[0]
    # apply_gradcam_all_layers_average(best_model, [test_img], [test_adjusted_affine], [test_original_img],
    #                                  task, modality, info)

    # Apply Grad-CAM to all images
    apply_gradcam_all_layers_average(best_model, imgs, adjusted_affines, original_imgs, task, modality, info)

