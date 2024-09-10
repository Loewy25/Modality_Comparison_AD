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
# Additional Imports (as per your request
from data_loading import loading_mask, generate_data_path, generate, binarylabel
from main import nested_crossvalidation_multi_kernel, nested_crossvalidation
import nibabel as nib
from tensorflow.keras.layers import Conv3D, Input, LeakyReLU, Add, GlobalAveragePooling3D, Dense, Dropout, SpatialDropout3D
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
import tensorflow_addons as tfa
from tensorflow.keras.optimizers import Adam
from nilearn.input_data import NiftiMasker
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.metrics import AUC
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

def convolution_block(x, filters, kernel_size=(3, 3, 3), strides=(1, 1, 1)):
    x = Conv3D(filters, kernel_size, strides=strides, padding='same', kernel_regularizer=l2(1e-4))(x)
    x = tfa.layers.InstanceNormalization()(x)
    x = LeakyReLU()(x)
    return x

# Context module: two convolution blocks with optional dropout
def context_module(x, filters):
    x = convolution_block(x, filters)
    x = SpatialDropout3D(0.3)(x)
    x = convolution_block(x, filters)
    return x

# Full 3D CNN classification architecture based on the encoder path with reduced number of filters
def create_3d_cnn(input_shape=(128, 128, 128, 1), num_classes=2):
    # Input layer
    input_img = Input(shape=input_shape)
    
    # Conv1 block (8 filters)
    x = convolution_block(input_img, 8)
    conv1_out = x
    
    # Context 1 (8 filters)
    x = context_module(x, 8)
    x = Add()([x, conv1_out])
    
    # Conv2 block (16 filters, stride 2)
    x = convolution_block(x, 16, strides=(2, 2, 2))
    conv2_out = x
    
    # Context 2 (16 filters)
    x = context_module(x, 16)
    x = Add()([x, conv2_out])
    
    # Conv3 block (32 filters, stride 2)
    x = convolution_block(x, 32, strides=(2, 2, 2))
    conv3_out = x
    
    # Context 3 (32 filters)
    x = context_module(x, 32)
    x = Add()([x, conv3_out])
    
    # Conv4 block (64 filters, stride 2)
    x = convolution_block(x, 64, strides=(2, 2, 2))
    conv4_out = x
    
    # Context 4 (64 filters)
    x = context_module(x, 64)
    x = Add()([x, conv4_out])
    
    # Global Average Pooling
    x = GlobalAveragePooling3D()(x)
    
    # Dropout for regularization
    x = Dropout(0.4)(x)
    
    # Dense layer with softmax for classification
    output = Dense(num_classes, activation='softmax')(x)
    
    # Create model
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

# Training loop with added print statements to confirm shapes
def train_model(X, Y):
    stratified_kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    all_y_val = []
    all_y_val_pred = []
    all_auc_scores = []

    # Print shape of X after loading

    for fold_num, (train_idx, val_idx) in enumerate(stratified_kfold.split(X, Y.argmax(axis=1))):
        # Print the shape of the input data X to confirm what it looks like
        
        X_train, X_val = X[train_idx], X[val_idx]
        Y_train, Y_val = Y[train_idx], Y[val_idx]


        # Debug inside the augment_data function: check X[i] shape before augmentation
        def augment_data_debug(image):
            print(f"Shape of image before augmentation: {image.shape}")
            augmented_image = image.copy()
            for axis in range(3):  # Assuming a 3D image
                if np.random.rand() > 0.5:
                    augmented_image = np.flip(augmented_image, axis=axis)
            print(f"Shape of image after augmentation: {augmented_image.shape}")
            return augmented_image

        # Augment the training data
        X_train_augmented = np.array([augment_data_debug(X[i]) for i in train_idx])
        # Create and compile the model
        model = create_3d_cnn(input_shape=(128, 128, 128, 1), num_classes=2)
        model.compile(optimizer=Adam(learning_rate=5e-4),
                      loss='CategoricalCrossentropy',  
                      metrics=['accuracy', AUC(name='auc')])


        # Callbacks for early stopping and learning rate reduction
        early_stopping = EarlyStopping(monitor='val_loss', patience=50, verbose=1, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1)

        # Train the model
        history = model.fit(X_train_augmented, Y_train,
                            batch_size=5,
                            epochs=200,
                            validation_data=(X_val, Y_val),
                            callbacks=[early_stopping, reduce_lr])

        # Make predictions on the validation set
        y_val_pred = model.predict(X_val)
        all_y_val.extend(Y_val[:, 1])
        all_y_val_pred.extend(y_val_pred[:, 1])

        # Calculate AUC for the current fold
        auc_score = roc_auc_score(Y_val[:, 1], y_val_pred[:, 1])
        all_auc_scores.append(auc_score)
        print(f"AUC for fold {fold_num + 1}: {auc_score:.4f}")

    # Calculate and print the average AUC across all folds
    average_auc = sum(all_auc_scores) / len(all_auc_scores)
    print(f"Average AUC across all folds: {average_auc:.4f}")



from nilearn.image import resample_img
import numpy as np

def pad_image_to_shape(image, target_shape=(128, 128, 128)):
    """
    Pads or crops an image to the target shape of (128, 128, 128).
    Assumes that the input may already include a channel dimension.
    """
    # Assume the last dimension is the channel if its size is 1
    if image.shape[-1] == 1:
        spatial_shape = image.shape[:-1]
    else:
        spatial_shape = image.shape  # Handle the case if no channel dimension is present

    print("Current spatial shape:", spatial_shape)

    # Calculate the padding needed for each spatial dimension (height, width, depth)
    padding = [(max((target_shape[i] - spatial_shape[i]) // 2, 0),
                max((target_shape[i] - spatial_shape[i]) - (target_shape[i] - spatial_shape[i]) // 2, 0))
               for i in range(3)]

    # Apply padding to the spatial dimensions
    padded_image = np.pad(image, [(p[0], p[1]) for p in padding] + [(0,0)], mode='constant', constant_values=0)

    # If the image's spatial dimensions are larger than the target shape, crop it
    start = [(spatial_shape[i] - target_shape[i]) // 2 if spatial_shape[i] > target_shape[i] else 0 for i in range(3)]
    end = [start[i] + target_shape[i] if spatial_shape[i] > target_shape[i] else target_shape[i] for i in range(3)]
    slices = tuple(slice(start[dim], end[dim]) for dim in range(3)) + (slice(None),)  # Preserve the channel dimension
    cropped_image = padded_image[slices]

    return cropped_image





from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore
import nibabel as nib
import numpy as np
from nilearn.input_data import NiftiMasker

def loading_mask_3d(task, modality):
    """
    Load the data, apply NiftiMasker, apply Z-score normalization, and pad the images to the target shape (128, 128, 128).
    """
    # Loading and generating data
    images_pet, images_mri, labels = generate_data_path()
    
    if modality == 'PET':
        data_train, train_label = generate(images_pet, labels, task)
    elif modality == 'MRI':
        data_train, train_label = generate(images_mri, labels, task)
    
    # Instantiate NiftiMasker
    masker = NiftiMasker(mask_img='/home/l.peiwang/MR-PET-Classfication/mask_gm_p4_new4.nii')
    
    train_data = []
    target_shape = (128, 128, 128)  # Define your desired shape (128, 128, 128)
    
    for i in range(len(data_train)):
        # Load the NIfTI image using nibabel (data_train[i] is a file path)
        nifti_img = nib.load(data_train[i])
        
        # Apply the mask, which flattens the image into 1D
        masked_data = masker.fit_transform(nifti_img)
        
        # Reshape the flattened data back into the original 3D shape
        reshaped_data = masker.inverse_transform(masked_data).get_fdata()
        
        # Z-score normalization in 3D
        reshaped_data = zscore(reshaped_data, axis=None)  # Normalize the whole 3D volume
        
        # Resize or pad the image to the target shape using the updated function
        padded_data = pad_image_to_shape(reshaped_data, target_shape=target_shape)
        
        # Add reshaped 3D data to the list
        train_data.append(padded_data)
    
    # Convert the train_label to binary if necessary
    train_label = binarylabel(train_label, task)
    
    # Convert list to numpy array for consistency
    train_data = np.array(train_data)
    
    return train_data, train_label, masker


import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical



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

# Function to overlay the heatmap on the original image and save it
def save_gradcam(heatmap, img, task, modality, layer_name, class_idx, save_dir='./grad-cam'):
    # Ensure the save directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Generate the filename with task, modality, and class index
    file_name = f"gradcam_{task}_{modality}_class{class_idx}_{layer_name}.png"
    save_path = os.path.join(save_dir, file_name)
    
    # Plot and save the heatmap overlaid on the original image
    plt.figure(figsize=(10, 10))
    plt.title(f"Grad-CAM for {layer_name}, Class {class_idx}")
    
    # Normalize and display the original image as grayscale
    img_display = img[0, ..., 0]  # Assuming grayscale input, adjust for RGB if needed
    img_display = (img_display - img_display.min()) / (img_display.max() - img_display.min())  # Normalize to [0,1]
    
    plt.imshow(img_display, cmap='gray')  # Show original image in grayscale
    plt.imshow(heatmap, cmap='jet', alpha=0.5)  # Overlay Grad-CAM heatmap
    plt.axis('off')
    
    # Save the plot as an image
    plt.savefig(save_path)
    plt.close()  # Close the plot to avoid displaying it

# Function to apply Grad-CAM for all convolutional layers and save heatmaps for both classes
def apply_gradcam_all_layers(model, img, task, modality):
    # Get all convolutional layers in the model
    conv_layers = [layer.name for layer in model.layers if 'conv' in layer.name]
    
    # Loop through both class indices (class 0 and class 1)
    for class_idx in range(2):
        for conv_layer_name in conv_layers:
            # Generate the Grad-CAM heatmap for each conv layer and each class
            heatmap = make_gradcam_heatmap(model, img, conv_layer_name, pred_index=class_idx)
            
            # Save the heatmap
            save_gradcam(heatmap, img, task, modality, conv_layer_name, class_idx)

# Example usage:
task = 'cd'
modality = 'MRI'
train_data, train_label, masker = loading_mask_3d(task, modality)  # Assume function is available
X = np.array(train_data)
Y = to_categorical(train_label, num_classes=2)

# Train the model
train_model(X, Y)  # Assume the model is already trained

# Pass a single input image to Grad-CAM (assuming train_data[0] is the image)
img = np.expand_dims(X[0], axis=0)  # Add batch dimension

# Apply Grad-CAM and save heatmaps for both classes and each convolutional layer
apply_gradcam_all_layers(model, img, task, modality)

