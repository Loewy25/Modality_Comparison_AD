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

# Additional Imports (as per your request
from data_loading import loading_mask, generate_data_path, generate, binarylabel
from main import nested_crossvalidation_multi_kernel, nested_crossvalidation

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

# Convolution block with L2 regularization, InstanceNorm, and Leaky ReLU
def convolution_block(x, filters, kernel_size=(3, 3, 3), strides=(1, 1, 1)):
    x = Conv3D(filters, kernel_size, strides=strides, padding='same', kernel_regularizer=l2(1e-5))(x)
    x = tfa.layers.InstanceNormalization()(x)
    x = LeakyReLU()(x)
    return x

# Context module: two convolution blocks with optional dropout
def context_module(x, filters):
    x = convolution_block(x, filters)
    x = SpatialDropout3D(0.4)(x)
    x = convolution_block(x, filters)
    return x

# Full 3D CNN classification architecture based on the encoder path
def create_3d_cnn(input_shape=(128, 128, 128, 1), num_classes=7):
    # Input layer
    input_img = Input(shape=input_shape)
    
    # Conv1 block (16 filters)
    x = convolution_block(input_img, 16)
    conv1_out = x
    
    # Context 1 (16 filters)
    x = context_module(x, 16)
    x = Add()([x, conv1_out])
    
    # Conv2 block (32 filters, stride 2)
    x = convolution_block(x, 32, strides=(2, 2, 2))
    conv2_out = x
    
    # Context 2 (32 filters)
    x = context_module(x, 32)
    x = Add()([x, conv2_out])
    
    # Conv3 block (64 filters, stride 2)
    x = convolution_block(x, 64, strides=(2, 2, 2))
    conv3_out = x
    
    # Context 3 (64 filters)
    x = context_module(x, 64)
    x = Add()([x, conv3_out])
    
    # Conv4 block (128 filters, stride 2)
    x = convolution_block(x, 128, strides=(2, 2, 2))
    conv4_out = x
    
    # Context 4 (128 filters)
    x = context_module(x, 128)
    x = Add()([x, conv4_out])
    
    # Conv5 block (256 filters, stride 2)
    x = convolution_block(x, 256, strides=(2, 2, 2))
    conv5_out = x
    
    # Context 5 (256 filters)
    x = context_module(x, 256)
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
    print(f"Shape of X after loading: {X.shape}")

    for fold_num, (train_idx, val_idx) in enumerate(stratified_kfold.split(X, Y.argmax(axis=1))):
        # Print the shape of the input data X to confirm what it looks like
        print(f"Shape of X (full dataset): {X.shape}")
        
        X_train, X_val = X[train_idx], X[val_idx]
        Y_train, Y_val = Y[train_idx], Y[val_idx]

        # Print the shape of X_train and X_val
        print(f"Shape of X_train: {X_train.shape}")
        print(f"Shape of X_val: {X_val.shape}")

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

        # Print the shape before passing it into the model
        print(f"Shape of X_train_augmented: {X_train_augmented.shape}")

        # Create and compile the model
        model = create_3d_cnn(input_shape=(128, 128, 128, 1), num_classes=7)
        model.compile(optimizer=Adam(learning_rate=5e-4),
                      loss='categorical_crossentropy',
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
def pad_image_to_shape(image, target_shape=(128, 128, 128)):
    """Pads or crops an image to the target shape."""
    current_shape = image.shape
    padding = [(0, max(target_shape[i] - current_shape[i], 0)) for i in range(3)]
    padded_image = np.pad(image, padding, mode='constant', constant_values=0)
    
    # If the image is larger than the target shape, crop it
    slices = [slice(0, min(current_shape[i], target_shape[i])) for i in range(3)]
    return padded_image[slices[0], slices[1], slices[2]]

def loading_mask_3d(task, modality):
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
        
        # Resize or pad the image to the target shape
        padded_data = pad_image_to_shape(reshaped_data, target_shape=target_shape)
        
        # Add reshaped 3D data to the list
        train_data.append(padded_data)
    
    # Convert the train_label to binary if necessary
    train_label = binarylabel(train_label, task)
    
    # Convert list to numpy array for consistency
    train_data = np.array(train_data)
    
    # Add channel dimension for the CNN (128, 128, 128, 1)
    train_data = train_data[..., np.newaxis]
    
    return train_data, train_label, masker

task = 'cd'
modality = 'PET'
# Example usage
train_data, train_label, masker = loading_mask_3d(task, modality)  # Assume function is available
X = np.array(train_data)
Y = to_categorical(train_label, num_classes=2)

# Calculate class weights manually
class_weights = calculate_class_weights(train_label)

# Train the model
train_model(X, Y, class_weights)
