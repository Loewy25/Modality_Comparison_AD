from tensorflow.keras.layers import Conv3D, Input, LeakyReLU, Add, GlobalAveragePooling3D, Dense, Dropout, SpatialDropout3D, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
import tensorflow_addons as tfa
import nibabel as nib

import numpy as np
from nilearn.image import resample_img, new_img_like, reorder_img
from scipy.ndimage import zoom
import tensorflow as tfwh
from tensorflow_addons.layers import InstanceNormalization
from kerastuner import Hyperband, HyperModel, RandomSearch
from kerastuner.engine import tuner as tuner_module
from kerastuner.engine import hyperparameters as hp_module

from data_loading import generate, generate_data_path, binarylabel
from sklearn.model_selection import StratifiedShuffleSplit
from nilearn.input_data import NiftiMasker
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.metrics import AUC
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K 
import scipy.ndimage
from keras_tuner import Objective
from kerastuner import Hyperband, HyperModel, RandomSearch
from kerastuner.engine import tuner as tuner_module
from kerastuner.engine import hyperparameters as hp_module

def augment_data(image, augmentation_level=1):
    augmented_image = image.copy()
    
    if augmentation_level == 0:
        return image 
        
    # Level 1 (Very Low): Basic flipping
    if augmentation_level >= 1:
        for axis in range(3):
            if np.random.rand() > 0.5:
                augmented_image = np.flip(augmented_image, axis=axis)

    # Level 2 (Low): Flipping and slight rotation
    if augmentation_level >= 2:
        if np.random.rand() > 0.5:
            angle = np.random.uniform(-5, 5)
            augmented_image = scipy.ndimage.rotate(augmented_image, angle, axes=(0, 1), reshape=False)

    # Level 3 (Medium): Includes brightness adjustment
    if augmentation_level >= 3:
        if np.random.rand() > 0.5:
            brightness_factor = np.random.uniform(0.9, 1.1)
            augmented_image *= brightness_factor

    # Level 4 (High): Adds contrast adjustment
    if augmentation_level >= 4:
        if np.random.rand() > 0.5:
            contrast_factor = np.random.uniform(0.9, 1.1)
            mean = np.mean(augmented_image)
            augmented_image = (augmented_image - mean) * contrast_factor + mean

    # Level 5 (Very High): Incorporates random noise
    if augmentation_level == 5:
        if np.random.rand() > 0.5:
            noise = np.random.normal(0, 0.01, augmented_image.shape)
            augmented_image += noise

    return augmented_image


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, images, labels, batch_size, augmentation_level):
        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        self.augmentation_level = augmentation_level

    def __len__(self):
        return np.ceil(len(self.images) / self.batch_size).astype(np.int)

    def __getitem__(self, idx):
        batch_x = self.images[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]

        return np.array([
            augment_data(file, self.augmentation_level) for file in batch_x]), np.array(batch_y)

    def on_epoch_end(self):
        # This method is called at the end of each epoch and could be used to shuffle the data
        pass

class MyTuner(Hyperband):
    def run_trial(self, trial, *args, **kwargs):
        # Retrieve hyperparameters
        augmentation_level = trial.hyperparameters.Int('augmentation_level', 1, 5, step=1)
        batch_size = trial.hyperparameters.get('batch_size')

        # Set up k-fold stratified cross-validation
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        val_scores = []

        for train_indices, val_indices in skf.split(X_train_full, Y_train_full.argmax(axis=1)):
            X_train, X_val = X_train_full[train_indices], X_train_full[val_indices]
            Y_train, Y_val = Y_train_full[train_indices], Y_train_full[val_indices]

            # Data augmentation for the current fold
            train_data_gen = DataGenerator(X_train, Y_train, batch_size, augmentation_level)
            val_data_gen = DataGenerator(X_val, Y_val, batch_size, 0)

            # Train the model
            model = self.hypermodel.build(trial.hyperparameters)
            model.fit(train_data_gen, epochs=kwargs['epochs'], validation_data=val_data_gen)

            val_scores.append(model.evaluate(val_data_gen))

        avg_val_score = np.mean(val_scores)
        self.oracle.update_trial(trial.trial_id, {'val_auc': avg_val_score})
        self.save_model(trial.trial_id, model)




def loading_mask(task,modality):
    #Loading and generating data
    images_pet,images_mri,labels=generate_data_path()
    if modality == 'PET':
        data_train,train_label=generate(images_pet,labels,task)
    if modality == 'MRI':
        data_train,train_label=generate(images_mri,labels,task)
    masker = NiftiMasker(mask_img='/home/l.peiwang/MR-PET-Classfication/mask_gm_p4_new4.nii')
    train_data=[]
    for i in range(len(data_train)):
        # Apply masker and inverse transform
        masked_data = masker.fit_transform(data_train[i])
        masked_image = masker.inverse_transform(masked_data)

        # Resize the image
        resized_image = resize(masked_image)

        # Convert the resized NIfTI image to a numpy array
        resized_data = resized_image.get_fdata()
        train_data.append(resized_data)
      
    train_label=binarylabel(train_label,task)
    
    return train_data,train_label,masker

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))



task = 'cd'
modality = 'PET'
train_data, train_label, masker = loading_mask(task, modality)
X = np.array(train_data)
Y = to_categorical(train_label, num_classes=2)

# Split the dataset into an 80% training set and a 20% internal testing set
X_train_full, X_internal_test, Y_train_full, Y_internal_test = train_test_split(
    X, Y, test_size=0.2, random_state=42, stratify=Y.argmax(axis=1))

# Initialize the hypermodel with the input shape
hypermodel = CNNHyperModel(input_shape=(128, 128, 128, 1))

# Callbacks for the hyperparameter tuning phase
early_stopping_tuner = EarlyStopping(monitor='val_loss', patience=20)
reduce_lr_tuner = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10)

# Callbacks for the final model training phase
early_stopping_final = EarlyStopping(monitor='val_loss', patience=50)
reduce_lr_final = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10)




tuner = MyTuner(
    hypermodel,
    objective=Objective('val_auc', direction='max'),  # Specify the direction for the AUC metric
    max_epochs=200,
    factor=2,
    directory='./result',
    project_name='cnn_hyperparam_tuning'
)



tuner.search(
    X_train_full, Y_train_full,
    epochs=400,
    callbacks=[early_stopping_tuner, reduce_lr_tuner]
)


# Retrieve the best hyperparameters
best_hps = tuner.get_best_hyperparameters()[0]
print(best_hps)

# Retrieve the best augmentation level from the hyperparameters
best_augmentation_level = best_hps.get('augmentation_level')

# Apply data augmentation to the training data using the best augmentation level
X_train_full_augmented = np.array([augment_data(x, best_augmentation_level) for x in X_train_full])

# Build the model with the best hyperparameters
model = tuner.hypermodel.build(best_hps)

# Retrain the model on the full training data
best_batch_size = best_hps.get('batch_size')
model.fit(
    X_train_full_augmented, Y_train_full,
    batch_size=best_batch_size,
    epochs=400,
    callbacks=[early_stopping_final, reduce_lr_final]
)

# Evaluate on the internal testing set
y_internal_test_pred = model.predict(X_internal_test)
final_auc = roc_auc_score(Y_internal_test[:, 1], y_internal_test_pred[:, 1])
print(f"Final AUC on internal testing set: {final_auc:.4f}")



def resample_to_spacing(data, original_spacing, new_spacing, interpolation='linear'):
    # Assuming the last dimension is the channel and should not be resampled
    zoom_factors = [o / n for o, n in zip(original_spacing, new_spacing)] + [1]
    return zoom(data, zoom_factors, order=1 if interpolation == 'linear' else 0)


def calculate_origin_offset(new_spacing, original_spacing):
    return [(o - n) / 2 for o, n in zip(original_spacing, new_spacing)]


def pad_image_to_shape(image, target_shape=(128, 128, 128)):
    # Check if the image has a 4th dimension (like a channel)
    has_channel = image.ndim == 4

    # Adjust target shape if the image has a channel dimension
    target_shape_adjusted = target_shape + (image.shape[3],) if has_channel else target_shape

    # Calculate the padding required in each dimension
    padding = [(0, max(target_shape_adjusted[dim] - image.shape[dim], 0)) for dim in range(image.ndim)]

    # Apply zero-padding to the data
    new_data = np.pad(image.get_fdata(), padding, mode='constant', constant_values=0)
    
    # Adjust the affine to account for the new shape
    new_affine = np.copy(image.affine)
    
    # Create and return a new NIfTI-like image with the padded data
    return new_img_like(image, new_data, affine=new_affine)



def resize(image, new_shape=(128, 128, 128), interpolation="linear"):
    # Reorder the image and resample it with the desired interpolation
    image = reorder_img(image, resample=interpolation)
    
    # Calculate the zoom levels needed for the new shape
    zoom_level = np.divide(new_shape, image.shape[:3])
    
    # Calculate the new spacing for the image
    new_spacing = np.divide(image.header.get_zooms()[:3], zoom_level)
    
    # Resample the image data to the new spacing
    new_data = resample_to_spacing(image.get_fdata(), image.header.get_zooms()[:3], new_spacing, 
                                   interpolation=interpolation)
    # Copy and adjust the affine transformation matrix for the new spacing
    new_affine = np.copy(image.affine)
    np.fill_diagonal(new_affine, new_spacing.tolist() + [1])
    new_affine[:3, 3] += calculate_origin_offset(new_spacing, image.header.get_zooms()[:3])
    
    # Create and return a new NIfTI-like image
    return new_img_like(image, new_data, affine=new_affine)


# Define the convolution block with hyperparameter options

def convolution_block(x, filters, kernel_size=(3, 3, 3), strides=(1, 1, 1),
                      regularization_rate=1e-5, normalization_type='instance'):
    x = Conv3D(filters, kernel_size, strides=strides, padding='same',
               kernel_regularizer=l2(regularization_rate))(x)
    if normalization_type == 'instance':
        x = InstanceNormalization()(x)
    elif normalization_type == 'batch':
        x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    return x

# Define the context module with dropout
def context_module(x, filters, dropout_rate=0.3, normalization_type='instance'):
    x = convolution_block(x, filters, normalization_type=normalization_type)
    x = SpatialDropout3D(dropout_rate)(x)
    x = convolution_block(x, filters, normalization_type=normalization_type)
    return x

# Define the CNN model with hyperparameters
class CNNHyperModel(HyperModel):
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def build(self, hp):
        filters = hp.Int('filters', min_value=4, max_value=16, step=4)
        dropout_rate = hp.Float('dropout_rate', min_value=0.3, max_value=0.5, step=0.1)
        regularization_rate = 1e-5
        normalization_type = hp.Choice('normalization_type', ['instance', 'batch'])
        learning_rate = hp.Float('learning_rate', min_value=1e-6, max_value=1e-4, sampling='LOG')
        augmentation_level = hp.Int('augmentation_level', min_value=1, max_value=5, step=1)
        batch_size = hp.Choice('batch_size', values=[5, 10]) 

        inputs = Input(shape=self.input_shape)
        x = convolution_block(inputs, filters=filters, regularization_rate=regularization_rate, normalization_type=normalization_type)
        conv1_out = x

        # Context 1
        x = context_module(x, filters=filters, dropout_rate=dropout_rate, normalization_type=normalization_type)
        x = Add()([x, conv1_out])
        x = convolution_block(x, filters=filters * 2, strides=(2, 2, 2), regularization_rate=regularization_rate, normalization_type=normalization_type)
        conv2_out = x

        # Context 2
        x = context_module(x, filters=filters * 2, dropout_rate=dropout_rate, normalization_type=normalization_type)
        x = Add()([x, conv2_out])
        x = convolution_block(x, filters=filters * 4, strides=(2, 2, 2), regularization_rate=regularization_rate, normalization_type=normalization_type)
        conv3_out = x

        # Context 3
        x = context_module(x, filters=filters * 4, dropout_rate=dropout_rate, normalization_type=normalization_type)
        x = Add()([x, conv3_out])
        x = convolution_block(x, filters=filters * 8, strides=(2, 2, 2), regularization_rate=regularization_rate, normalization_type=normalization_type)
        conv4_out = x

        # Context 4
        x = context_module(x, filters=filters * 8, dropout_rate=dropout_rate, normalization_type=normalization_type)
        x = Add()([x, conv4_out])
        x = convolution_block(x, filters=filters * 16, strides=(2, 2, 2), regularization_rate=regularization_rate, normalization_type=normalization_type)
        
        # Context 5
        x = context_module(x, filters=filters * 16, dropout_rate=dropout_rate, normalization_type=normalization_type)

        # Global Average Pooling and Dropout layer before Dense layer
        x = GlobalAveragePooling3D()(x)
        x = Dropout(dropout_rate)(x)

        # Dense layer with softmax activation
        outputs = Dense(2, activation='softmax')(x)

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=learning_rate),
                      loss='categorical_crossentropy',
                      metrics=['accuracy', AUC(name='auc')])
        return model

