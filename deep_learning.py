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

# Additional Imports (as per your request)
from data_loading import loading_mask
from main import nested_crossvalidation_multi_kernel, nested_crossvalidation

from tensorflow.keras.layers import Conv3D, Input, LeakyReLU, Add, GlobalAveragePooling3D, Dense, Dropout, SpatialDropout3D
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
import tensorflow_addons as tfa
from tensorflow.keras.optimizers import Adam
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

# Training loop
def train_model(X, Y, class_weights):
    stratified_kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    all_y_val = []
    all_y_val_pred = []
    all_auc_scores = []

    for fold_num, (train_idx, val_idx) in enumerate(stratified_kfold.split(X, Y.argmax(axis=1))):
        X_train, X_val = X[train_idx], X[val_idx]
        Y_train, Y_val = Y[train_idx], Y[val_idx]

        # Augment the training data
        X_train_augmented = np.array([augment_data(X[i]) for i in train_idx])

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
task = 'cd'
modality = 'PET'
# Example usage
train_data, train_label, masker = loading_mask(task, modality)  # Assume function is available
X = np.array(train_data)
Y = to_categorical(train_label, num_classes=7)

# Calculate class weights manually
class_weights = calculate_class_weights(train_label)

# Train the model
train_model(X, Y, class_weights)
