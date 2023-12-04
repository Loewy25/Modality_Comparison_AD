from cnn import pad_image_to_shape, resize, convolution_block, context_module, create_cnn_model
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
        resized_image = pad_image_to_shape(masked_image)

        # Convert the resized NIfTI image to a numpy array
        resized_data = resized_image.get_fdata()
        train_data.append(resized_data)
      
    train_label=binarylabel(train_label,task)
    
    return train_data,train_label,masker

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))



# Loading the dataset
task = 'cd'
modality = 'PET'
train_data, train_label, masker = loading_mask(task, modality)
X = np.array(train_data)
Y = to_categorical(train_label, num_classes=2)

# Apply StratifiedKFold on the entire dataset
stratified_kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
all_y_test = []
all_y_test_pred = []
all_auc_scores = []

for fold_num, (train_val_idx, test_idx) in enumerate(stratified_kfold.split(X, Y.argmax(axis=1))):
    X_train_val, Y_train_val = X[train_val_idx], Y[train_val_idx]
    X_test, Y_test = X[test_idx], Y[test_idx]

    stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_idx, val_idx in stratified_split.split(X_train_val, Y_train_val.argmax(axis=1)):
        X_train, Y_train = X_train_val[train_idx], Y_train_val[train_idx]
        X_val, Y_val = X_train_val[val_idx], Y_train_val[val_idx]

    X_train_augmented = np.array([augment_data(X_train[i], augmentation_level=5) for i in range(len(X_train))])

    with tf.distribute.MirroredStrategy().scope():
        model = create_cnn_model()
        model.compile(optimizer=Adam(5e-4), loss='categorical_crossentropy', metrics=['accuracy', AUC(name='auc')])

        early_stopping = EarlyStopping(monitor='val_loss', patience=50, verbose=1, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1)

        history = model.fit(X_train_augmented, Y_train, batch_size=5, epochs=200, validation_data=(X_val, Y_val), callbacks=[early_stopping, reduce_lr])

    y_test_pred = model.predict(X_test)
    all_y_test.extend(Y_test[:, 1])
    all_y_test_pred.extend(y_test_pred[:, 1])

    auc_score = roc_auc_score(Y_test[:, 1], y_test_pred[:, 1])
    all_auc_scores.append(auc_score)
    print(f"AUC for fold {fold_num + 1}: {auc_score:.4f}")

    plt.figure()
    plt.plot(history.history['auc'], label='Train AUC')
    plt.plot(history.history['val_auc'], label='Validation AUC', linestyle='--')
    plt.title(f'Fold {fold_num + 1} - AUC')
    plt.xlabel('Epochs')
    plt.ylabel('AUC')
    plt.legend()
    plt.show()

    K.clear_session()

average_auc = sum(all_auc_scores) / len(all_auc_scores)
print(f"Average AUC across all test sets: {average_auc:.4f}")
