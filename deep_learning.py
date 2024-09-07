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
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
from data_loading import loading_mask
from main import nested_crossvalidation_multi_kernel, nested_crossvalidation

# Data augmentation function: Mirroring with probability of 0.5
def augment_data(image):
    augmented_image = image.copy()
    for axis in range(3):
        if np.random.rand() > 0.5:
            augmented_image = np.flip(augmented_image, axis=axis)
    return augmented_image

# Convolution block with L2 regularization
def convolution_block(x, filters, kernel_size=(3,3,3), strides=(1,1,1)):
    x = Conv3D(filters, kernel_size, strides=strides, padding='same', kernel_regularizer=l2(1e-5))(x)
    x = tfa.layers.InstanceNormalization()(x)
    x = LeakyReLU()(x)
    return x

# Context module
def context_module(x, filters):
    x = convolution_block(x, filters)
    x = SpatialDropout3D(0.4)(x)
    x = convolution_block(x, filters)
    return x

# CNN Model definition
def create_cnn_model():
    input_img = Input(shape=(128, 128, 128, 1))
    x = convolution_block(input_img, 16, strides=(1,1,1))
    conv1_out = x

    # Context 1
    x = context_module(x, 16)
    x = Add()([x, conv1_out])
    x = convolution_block(x, 32, strides=(2,2,2))
    conv2_out = x

    # Context 2
    x = context_module(x, 32)
    x = Add()([x, conv2_out])
    x = convolution_block(x, 64, strides=(2,2,2))
    conv3_out = x

    # Context 3
    x = context_module(x, 64)
    x = Add()([x, conv3_out])
    x = convolution_block(x, 128, strides=(2,2,2))
    conv4_out = x

    # Context 4
    x = context_module(x, 128)
    x = Add()([x, conv4_out])
    x = convolution_block(x, 256, strides=(2,2,2))
    
    # Context 5
    x = context_module(x, 256)

    # Global Average Pooling
    x = GlobalAveragePooling3D()(x)

    # Dropout layer after GAP
    x = Dropout(0.4)(x)

    # Output layer
    output = Dense(2, activation='softmax')(x)

    model = Model(inputs=input_img, outputs=output)
    model.summary()

    return model

# Training loop using StratifiedKFold
def train_model(X, Y):
    stratified_kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    all_y_val = []
    all_y_val_pred = []
    all_auc_scores = []

    for fold_num, (train, val) in enumerate(stratified_kfold.split(X, Y.argmax(axis=1))):
        X_train_augmented = np.array([augment_data(X[i]) for i in train])
        Y_train = Y[train]

        # Create and compile the model
        with tf.distribute.MirroredStrategy().scope():
            model = create_cnn_model()
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
                                validation_data=(X[val], Y[val]),
                                callbacks=[early_stopping, reduce_lr])

        # Make predictions on the validation set
        y_val_pred = model.predict(X[val])
        all_y_val.extend(Y[val][:, 1])
        all_y_val_pred.extend(y_val_pred[:, 1])

        # Calculate AUC for the current fold
        auc_score = roc_auc_score(Y[val][:, 1], y_val_pred[:, 1])
        all_auc_scores.append(auc_score)
        print(f"AUC for fold {fold_num + 1}: {auc_score:.4f}")

        # Plot the AUC history for the current fold
        plt.figure()
        plt.plot(history.history['auc'], label='Train AUC')
        plt.plot(history.history['val_auc'], label='Validation AUC', linestyle='--')
        plt.title(f'Fold {fold_num + 1} - AUC')
        plt.xlabel('Epochs')
        plt.ylabel('AUC')
        plt.legend()
        plt.show()

        K.clear_session()

    # Calculate and print the average AUC across all folds
    average_auc = sum(all_auc_scores) / len(all_auc_scores)
    print(f"Average AUC across all folds: {average_auc:.4f}")

# Example data loading
task = 'cd'
modality = 'PET'
train_data, train_label, masker = loading_mask(task, modality)
X = np.array(train_data)
Y = to_categorical(train_label, num_classes=2)

# Train the model without class weights
train_model(X, Y)

