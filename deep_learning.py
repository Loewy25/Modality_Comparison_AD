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
    
    # Conv1 block (8 filters instead of 16)
    x = convolution_block(input_img, 8)
    conv1_out = x
    
    # Context 1 (8 filters)
    x = context_module(x, 8)
    x = Add()([x, conv1_out])
    
    # Conv2 block (16 filters instead of 32, stride 2)
    x = convolution_block(x, 16, strides=(2, 2, 2))
    conv2_out = x
    
    # Context 2 (16 filters)
    x = context_module(x, 16)
    x = Add()([x, conv2_out])
    
    # Conv3 block (32 filters instead of 64, stride 2)
    x = convolution_block(x, 32, strides=(2, 2, 2))
    conv3_out = x
    
    # Context 3 (32 filters)
    x = context_module(x, 32)
    x = Add()([x, conv3_out])
    
    # Conv4 block (64 filters instead of 128, stride 2)
    x = convolution_block(x, 64, strides=(2, 2, 2))
    conv4_out = x
    
    # Context 4 (64 filters)
    x = context_module(x, 64)
    x = Add()([x, conv4_out])
    
    # Conv5 block (128 filters instead of 256, stride 2)
    x = convolution_block(x, 128, strides=(2, 2, 2))
    conv5_out = x
    
    # Context 5 (128 filters)
    x = context_module(x, 128)
    x = Add()([x, conv5_out])
    
    # Global Average Pooling
    x = GlobalAveragePooling3D()(x)
    
    # Dropout
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
def train_model(X, Y, class_weights):
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
                            callbacks=[early_stopping, reduce_lr],
                            class_weight=class_weights)

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


# Calculate class weights manually
def calculate_class_weights(labels):
    classes, class_counts = np.unique(labels, return_counts=True)
    total_samples = len(labels)
    class_weights = {int(c): total_samples / (len(classes) * count) for c, count in zip(classes, class_counts)}
    return class_weights


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





def loading_mask_3d(task, modality):
    """
    Load the data, apply NiftiMasker, and pad the images to the target shape (128, 128, 128).
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
        
        # Resize or pad the image to the target shape using the updated function
        padded_data = pad_image_to_shape(reshaped_data, target_shape=target_shape)
        
        # Add reshaped 3D data to the list
        train_data.append(padded_data)
    
    # Convert the train_label to binary if necessary
    train_label = binarylabel(train_label, task)
    
    # Convert list to numpy array for consistency
    train_data = np.array(train_data)
    
    return train_data, train_label, masker

task = 'pc'
modality = 'MRI'
# Example usage
train_data, train_label, masker = loading_mask_3d(task, modality)  # Assume function is available
X = np.array(train_data)
Y = to_categorical(train_label, num_classes=2)

# Calculate class weights manually
class_weights = calculate_class_weights(train_label)
import tensorflow as tf
print("Built with CUDA:", tf.test.is_built_with_cuda())
print("Available GPU Devices:", tf.config.list_physical_devices('GPU'))


# Train the model
train_model(X, Y, class_weights)

