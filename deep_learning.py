from tensorflow.keras.layers import Conv3D, Input, LeakyReLU, Add, GlobalAveragePooling3D, Dense, Dropout, SpatialDropout3D
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
from tensorflow.keras import backend as K
from nilearn.image import reorder_img, new_img_like
from tensorflow.keras.losses import CategoricalCrossentropy
import nibabel as nib
from nilearn.input_data import NiftiMasker

# Additional Imports (as per your request)
from data_loading import loading_mask, generate_data_path, generate, binarylabel
from main import nested_crossvalidation_multi_kernel, nested_crossvalidation


# Function for a single convolution block with Batch Normalization
def convolution_block(x, filters, kernel_size=(3, 3, 3), strides=(1, 1, 1)):
    x = Conv3D(filters, kernel_size, strides=strides, padding='same', kernel_regularizer=l2(1e-3))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    return x

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
    
    # Global Average Pooling
    x = GlobalAveragePooling3D()(x)
    
    # Dropout for regularization
    x = Dropout(0.5)(x)
    
    # Dense layer with softmax for classification
    output = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=input_img, outputs=output)
    model.summary()
    
    return model


# Data augmentation (mirroring along each axis with a 50% probability)
def augment_data(image):
    augmented_image = image.copy()
    for axis in range(3):
        if np.random.rand() > 0.5:
            augmented_image = np.flip(augmented_image, axis=axis)
    return augmented_image

# Modified pad_image_to_shape function to return padding values
def pad_image_to_shape(image, target_shape=(128, 128, 128)):
    """
    Pads or crops an image to the target shape (128, 128, 128).
    Returns the padded image and the amount of padding applied.
    """
    if image.shape[-1] == 1:
        spatial_shape = image.shape[:-1]
    else:
        spatial_shape = image.shape  # Handle the case if no channel dimension is present

    print("Current spatial shape:", spatial_shape)

    padding = [(max((target_shape[i] - spatial_shape[i]) // 2, 0),
                max((target_shape[i] - spatial_shape[i]) - (target_shape[i] - spatial_shape[i]) // 2, 0))
               for i in range(3)]

    padded_image = np.pad(image, [(p[0], p[1]) for p in padding] + [(0, 0)], mode='constant', constant_values=0)
    
    return padded_image, padding  # Return padding information


# Modified function to handle loading with padding and return padding values
def loading_mask_3d(task, modality):
    images_pet, images_mri, labels = generate_data_path()
    
    if modality == 'PET':
        data_train, train_label = generate(images_pet, labels, task)
    elif modality == 'MRI':
        data_train, train_label = generate(images_mri, labels, task)
    
    masker = NiftiMasker(mask_img='/home/l.peiwang/MR-PET-Classfication/mask_gm_p4_new4.nii')
    
    train_data = []
    paddings = []  # Store padding info
    target_shape = (128, 128, 128)
    
    for i in range(len(data_train)):
        nifti_img = nib.load(data_train[i])
        masked_data = masker.fit_transform(nifti_img)
        reshaped_data = masker.inverse_transform(masked_data).get_fdata()
        reshaped_data = zscore(reshaped_data, axis=None)
        
        padded_data, padding = pad_image_to_shape(reshaped_data, target_shape=target_shape)
        
        train_data.append(padded_data)
        paddings.append(padding)
    
    train_label = binarylabel(train_label, task)
    train_data = np.array(train_data)
    
    return train_data, train_label, masker, paddings


# New function to remove padding based on stored padding values
def remove_padding(heatmap, padding):
    slices = tuple(slice(p[0], -p[1] if p[1] > 0 else None) for p in padding)
    heatmap_unpadded = heatmap[slices]
    return heatmap_unpadded


# Modified save_gradcam to handle padding removal
def save_gradcam(heatmap, img, padding, masker, task, modality, layer_name, class_idx, save_dir='./grad-cam'):
    ensure_directory_exists(save_dir)
    
    # Remove padding from heatmap
    heatmap_unpadded = remove_padding(heatmap, padding)
    
    # Rescale the heatmap back to the original space using the masker
    heatmap_rescaled = masker.inverse_transform(heatmap_unpadded)
    
    nifti_file_name = f"gradcam_{task}_{modality}_class{class_idx}_{layer_name}.nii.gz"
    nifti_save_path = os.path.join(save_dir, nifti_file_name)
    
    nifti_img = new_img_like(masker.mask_img_, heatmap_rescaled.get_fdata())
    nib.save(nifti_img, nifti_save_path)
    
    output_slice_path = os.path.join(save_dir, f'stat_map_{task}_{modality}_{layer_name}_class{class_idx}.png')
    
    plotting.plot_stat_map(
        nifti_img,
        display_mode='x',
        cut_coords=range(0, 51, 5),
        title=f'Grad-CAM Slices for {layer_name}, Class {class_idx}',
        cmap='jet',
        output_file=output_slice_path,
        threshold=0.2,
        vmax=1
    )
    
    print(f'Grad-CAM stat map saved at {output_slice_path}')


# Function to apply Grad-CAM for all layers and save heatmaps
def apply_gradcam_all_layers(model, img, task, modality, paddings):
    conv_layers = [layer.name for layer in model.layers if 'conv' in layer.name]
    
    for class_idx in range(2):
        for conv_layer_name in conv_layers:
            heatmap = make_gradcam_heatmap(model, img, conv_layer_name, pred_index=class_idx)
            save_gradcam(heatmap, img, paddings[0], masker, task, modality, conv_layer_name, class_idx)


# Define your task and modality
task = 'cd'
modality = 'PET'

# Load your data
train_data, train_label, masker, paddings = loading_mask_3d(task, modality)
X = np.array(train_data)
Y = to_categorical(train_label, num_classes=2)

# Create and compile the model
model = create_3d_cnn(input_shape=(128, 128, 128, 1), num_classes=2)

# Train the model
train_model(X, Y)

# Select an image from your dataset (e.g., the first one)
img = np.expand_dims(X[0], axis=0)

# Apply Grad-CAM
apply_gradcam_all_layers(model, img, task, modality, paddings)
