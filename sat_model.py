from cnn import pad_image_to_shape, resize, convolution_block, context_module, CNNHyperModel

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
            val_data_gen = DataGenerator(X_val, Y_val, batch_size, augmentation_level)

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

