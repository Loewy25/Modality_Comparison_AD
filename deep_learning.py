from tensorflow.keras.layers import Conv3D, Input, LeakyReLU, Add, GlobalAveragePooling3D, Dense, Dropout, SpatialDropout3D, BatchNormalization  # Add BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
import tensorflow_addons as tfa
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.metrics import AUC
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import backend as K
from nilearn.image import reorder_img, new_img_like
from tensorflow.keras.losses import CategoricalCrossentropy
import nibabel as nib
from nilearn.input_data import NiftiMasker
from nilearn import plotting
from scipy.stats import zscore
import os

# Additional Imports (as per your request)
from data_loading import loading_mask, generate_data_path, generate, binarylabel
from main import nested_crossvalidation_multi_kernel, nested_crossvalidation

# Function for a single convolution block with Batch Normalization
def convolution_block(x, filters, kernel_size=(3, 3, 3), strides=(1, 1, 1)):
    x = Conv3D(filters, kernel_size, strides=strides, padding='same', kernel_regularizer=l2(1e-3))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    return x

def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Context module: two convolution blocks with optional dropout
def context_module(x, filters):
    x = convolution_block(x, filters)
    x = SpatialDropout3D(0.5)(x)
    x = convolution_block(x, filters)
    return x

# Full 3D CNN classification architecture with 4 context blocks
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

    # Conv4 block (128 filters, stride 2)
    x = convolution_block(x, 256, strides=(2, 2, 2))
    conv5_out = x
    x = context_module(x, 256)
    x = Add()([x, conv5_out])
    
    # Global Average Pooling
    x = GlobalAveragePooling3D()(x)
    
    # Dropout for regularization
    x = Dropout(0.5)(x)
    
    # Dense layer with softmax for classification
    output = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=input_img, outputs=output)
    model.summary()
    
    return model

# Modified pad_image_to_shape function to return padding values
def pad_image_to_shape(image, target_shape=(128, 128, 128)):
    if image.shape[-1] == 1:
        spatial_shape = image.shape[:-1]
    else:
        spatial_shape = image.shape

    padding = [(max((target_shape[i] - spatial_shape[i]) // 2, 0),
                max((target_shape[i] - spatial_shape[i]) - (target_shape[i] - spatial_shape[i]) // 2, 0))
               for i in range(3)]

    padded_image = np.pad(image, [(p[0], p[1]) for p in padding] + [(0, 0)], mode='constant', constant_values=0)
    
    return padded_image, padding  # Return padding information

# Function to load the data, mask it, and return padding information
def loading_mask_3d(task, modality):
    images_pet, images_mri, labels = generate_data_path()
    
    if modality == 'PET':
        data_train, train_label = generate(images_pet, labels, task)
    elif modality == 'MRI':
        data_train, train_label = generate(images_mri, labels, task)
    
    masker = NiftiMasker(mask_img='/home/l.peiwang/MR-PET-Classfication/mask_gm_p4_new4.nii')
    
    train_data = []
    paddings = []
    affines = []  # List to store the affine matrices
    target_shape = (128, 128, 128)
    
    for i in range(len(data_train)):
        nifti_img = nib.load(data_train[i])
        affine = nifti_img.affine  # Store the affine matrix
        masked_data = masker.fit_transform(nifti_img)
        reshaped_data = masker.inverse_transform(masked_data).get_fdata()
        reshaped_data = zscore(reshaped_data, axis=None)
        
        padded_data, padding = pad_image_to_shape(reshaped_data, target_shape=target_shape)
        
        train_data.append(padded_data)
        paddings.append(padding)
        affines.append(affine)  # Store the affine matrix for later use
    
    train_label = binarylabel(train_label, task)
    train_data = np.array(train_data)
    
    return train_data, train_label, masker, paddings, affines  

# Function to remove padding based on stored padding values
def remove_padding(heatmap, padding):
    slices = tuple(slice(p[0], -p[1] if p[1] > 0 else None) for p in padding)
    heatmap_unpadded = heatmap[slices]
    return heatmap_unpadded

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
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2, 3))  # Adjust this for 3D

    conv_outputs = conv_outputs[0]
    conv_outputs = tf.reduce_mean(conv_outputs * pooled_grads, axis=-1)

    heatmap = tf.maximum(conv_outputs, 0)
    heatmap = heatmap / tf.math.reduce_max(heatmap)
    
    return heatmap.numpy()

# Function to save Grad-CAM 3D heatmap and plot glass brain
# Function to save Grad-CAM 3D heatmap and plot glass brain using the stored affine
def save_gradcam(heatmap, img, padding, affine, task, modality, layer_name, class_idx, info, save_dir='./grad-cam'):
    save_dir = os.path.join(save_dir, info, task, modality)
    ensure_directory_exists(save_dir)
    
    # Remove padding from the heatmap
    heatmap_unpadded = remove_padding(heatmap, padding)
    
    # Create a NIfTI image using the stored affine matrix
    nifti_img = nib.Nifti1Image(heatmap_unpadded, affine)  # Use the stored affine matrix
    
    # Save the 3D NIfTI file
    nifti_file_name = f"gradcam_{task}_{modality}_class{class_idx}_{layer_name}.nii.gz"
    nifti_save_path = os.path.join(save_dir, nifti_file_name)
    nib.save(nifti_img, nifti_save_path)
    print(f'3D Grad-CAM heatmap saved at {nifti_save_path}')
    
    # Now, plot the glass brain
    output_glass_brain_path = os.path.join(save_dir, f'glass_brain_{task}_{modality}_{layer_name}_class{class_idx}.png')
    plotting.plot_glass_brain(nifti_img, colorbar=True, plot_abs=True, cmap='jet', output_file=output_glass_brain_path)
    print(f'Glass brain plot saved at {output_glass_brain_path}')


# Function to apply Grad-CAM for all layers across all dataset images and save averaged heatmaps
def apply_gradcam_all_layers_average(model, imgs, task, modality, paddings, info):
    conv_layers = [layer.name for layer in model.layers if 'conv' in layer.name]
    
    for conv_layer_name in conv_layers:
        for class_idx in range(2):  # Loop through both class indices (class 0 and class 1)
            accumulated_heatmap = None
            for i, img in enumerate(imgs):
                heatmap = make_gradcam_heatmap(model, img, conv_layer_name, pred_index=class_idx)
                if accumulated_heatmap is None:
                    accumulated_heatmap = heatmap
                else:
                    accumulated_heatmap += heatmap
            
            avg_heatmap = accumulated_heatmap / len(imgs)
            save_gradcam(avg_heatmap, imgs[0], paddings[0], masker, task, modality, conv_layer_name, class_idx,info)

def train_model(X, Y):
    stratified_kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=2)
    all_y_val = []
    all_y_val_pred = []
    all_auc_scores = []

    for fold_num, (train_idx, val_idx) in enumerate(stratified_kfold.split(X, Y.argmax(axis=1))):
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
                            epochs=800,
                            validation_data=(X_val, Y_val),
                            callbacks=[early_stopping, reduce_lr])

        y_val_pred = model.predict(X_val)
        all_y_val.extend(Y_val[:, 1])
        all_y_val_pred.extend(y_val_pred[:, 1])

        auc_score = roc_auc_score(Y_val[:, 1], y_val_pred[:, 1])
        all_auc_scores.append(auc_score)
        print(f'AUC for fold {fold_num + 1}: {auc_score:.4f}')

    average_auc = sum(all_auc_scores) / len(all_auc_scores)
    print(f'Average AUC across all folds: {average_auc:.4f}')

# Example usage:
task = 'dm'
modality = 'PET'
info='5_context_from_16_0.5_dropout_1e3'

# Load your data
train_data, train_label, masker, paddings = loading_mask_3d(task, modality)
X = np.array(train_data)
Y = to_categorical(train_label, num_classes=2)

# Create and compile the model
model = create_3d_cnn(input_shape=(128, 128, 128, 1), num_classes=2)

# Train the model (if needed)
train_model(X, Y)

# Apply Grad-CAM to all images and compute the average
imgs = [np.expand_dims(X[i], axis=0) for i in range(X.shape[0])]
apply_gradcam_all_layers_average(model, imgs, task, modality, paddings, info)
