from cnn import pad_image_to_shape, resize, convolution_block, context_module, create_cnn_model
from data_loading import generate, generate_data_path, binarylabel

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



def augment_data(image):
    augmented_image = image.copy()
    for axis in range(3):
        if np.random.rand() > 0.5:
            augmented_image = np.flip(augmented_image, axis=axis)
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

# Apply StratifiedKFold on the entire dataset
stratified_kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
all_y_val = []
all_y_val_pred = []
all_auc_scores = []

for fold_num, (train, val) in enumerate(stratified_kfold.split(X, Y.argmax(axis=1))):
    X_train_augmented = np.array([augment_data(X[i]) for i in train])
    Y_train = Y[train]

    with tf.distribute.MirroredStrategy().scope():
        model = create_cnn_model()
        model.compile(optimizer=Adam(5e-4), loss='categorical_crossentropy', metrics=['accuracy', AUC(name='auc')])

        early_stopping = EarlyStopping(monitor='val_loss', patience=50, verbose=1, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1)

        history = model.fit(X_train_augmented, Y_train, batch_size=1, epochs=200, validation_data=(X[val], Y[val]), callbacks=[early_stopping, reduce_lr])

    y_val_pred = model.predict(X[val])
    all_y_val.extend(Y[val][:, 1])
    all_y_val_pred.extend(y_val_pred[:, 1])

    # AUC for the current fold
    auc_score = roc_auc_score(Y[val][:, 1], y_val_pred[:, 1])
    all_auc_scores.append(auc_score)
    print(f"AUC for fold {fold_num + 1}: {auc_score:.4f}")

    # Plotting AUC history
    plt.figure()
    plt.plot(history.history['auc'], label='Train AUC')
    plt.plot(history.history['val_auc'], label='Validation AUC', linestyle='--')
    plt.title(f'Fold {fold_num + 1} - AUC')
    plt.xlabel('Epochs')
    plt.ylabel('AUC')
    plt.legend()
    plt.show()

    clear_session()

# Calculate and print the average AUC across all folds
average_auc = sum(all_auc_scores) / len(all_auc_scores)
print(f"Average AUC across all folds: {average_auc:.4f}")
