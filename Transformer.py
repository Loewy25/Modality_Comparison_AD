import os
import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, Add, Flatten, MultiHeadAttention, Reshape, Conv3D, Embedding

from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
import nibabel as nib
from nilearn.maskers import NiftiMasker  # Use the new import path as per the warning
from scipy.stats import zscore
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from data_loading import generate_data_path, generate, binarylabel

# Function to ensure a directory exists
def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Transformer block
def transformer_block(inputs, num_heads, d_model, d_ff, dropout_rate=0.1):
    # Multi-Head Self-Attention
    attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(inputs, inputs)
    attention_output = Dropout(dropout_rate)(attention_output)
    out1 = LayerNormalization(epsilon=1e-6)(Add()([inputs, attention_output]))  # Residual connection + Layer norm

    # Feed-Forward Network
    ffn_output = Dense(d_ff, activation='relu')(out1)
    ffn_output = Dense(d_model)(ffn_output)
    ffn_output = Dropout(dropout_rate)(ffn_output)
    out2 = LayerNormalization(epsilon=1e-6)(Add()([out1, ffn_output]))  # Residual connection + Layer norm
    
    return out2

# Patch embedding layer for 3D volumes using Conv3D
def patch_embedding_layer(input_shape, patch_size, d_model):
    inputs = Input(shape=input_shape + (1,))  # Adding a channel dimension for 3D images
    
    # Use Conv3D to extract patches and project to the embedding dimension
    x = Conv3D(
        filters=d_model,
        kernel_size=patch_size,
        strides=patch_size,
        padding='valid'
    )(inputs)
    
    # Flatten the patches into a sequence
    seq_len = (input_shape[0] // patch_size[0]) * (input_shape[1] // patch_size[1]) * (input_shape[2] // patch_size[2])
    x = Reshape((seq_len, d_model))(x)
    
    return Model(inputs=inputs, outputs=x)

# Full Transformer-based classification model
def create_transformer_model(input_shape=(91, 109, 91), patch_size=(7, 7, 7), d_model=128, num_heads=8, d_ff=256, num_layers=4, num_classes=2, dropout_rate=0.2):
    inputs = Input(shape=input_shape + (1,))  # Adding a channel dimension
    
    # Patch Embedding
    patch_embedding_model = patch_embedding_layer(input_shape, patch_size, d_model)
    x = patch_embedding_model(inputs)
    
    # Add positional encoding
    num_patches = x.shape[1]
    positions = tf.range(start=0, limit=num_patches, delta=1)
    pos_encoding = Embedding(input_dim=num_patches, output_dim=d_model)(positions)
    x = x + pos_encoding

    # Transformer layers
    for _ in range(num_layers):
        x = transformer_block(x, num_heads, d_model, d_ff, dropout_rate)

    # Global Average Pooling
    x = Flatten()(x)

    # Dense layers for classification
    x = Dropout(dropout_rate)(x)
    output = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=output)
    model.summary()

    return model

# Function to plot training and validation loss and save the figure in the specified save_dir
def plot_training_validation_loss(histories, save_dir):
    # Initialize lists to collect losses
    train_losses = []
    val_losses = []

    for history in histories:
        train_losses.extend(history.history['loss'])
        val_losses.extend(history.history['val_loss'])

    # Create the plot
    plt.figure(figsize=(10, 6))

    # Plot the training and validation loss
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')

    # Add title and labels
    plt.title('Training Loss vs Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    # Set y-axis limits to focus on 0-1 range
    plt.ylim(0, 1)

    # Add a legend
    plt.legend(loc='upper right')

    # Ensure the directory exists and save the plot
    ensure_directory_exists(save_dir)
    loss_plot_path = os.path.join(save_dir, 'loss_vs_val_loss.png')
    plt.savefig(loss_plot_path)
    plt.close()  # Close the figure to avoid displaying it in notebooks
    print(f'Loss vs Validation Loss plot saved at {loss_plot_path}')

# Function to train the model using Stratified K-Fold Cross Validation
def train_model(X, Y, task, modality, info):
    stratified_kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=2)
    all_auc_scores = []
    histories = []

    save_dir = os.path.join('./transformer', info, task, modality)
    ensure_directory_exists(save_dir)

    best_auc = 0
    best_model = None

    for fold_num, (train_idx, val_idx) in enumerate(stratified_kfold.split(X, Y.argmax(axis=1))):
        print(f"Starting fold {fold_num + 1}")
        X_train, X_val = X[train_idx], X[val_idx]
        Y_train, Y_val = Y[train_idx], Y[val_idx]

        model = create_transformer_model(input_shape=X.shape[1:4], num_classes=2)
        model.compile(optimizer=Adam(learning_rate=5e-4),
                      loss='categorical_crossentropy',
                      metrics=['accuracy', AUC(name='auc')])

        early_stopping = EarlyStopping(
            monitor='val_auc',
            patience=100,
            mode='max',
            verbose=1,
            restore_best_weights=True
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            mode='min',
            verbose=1
        )

        history = model.fit(X_train, Y_train,
                            batch_size=5,
                            epochs=800,
                            validation_data=(X_val, Y_val),
                            callbacks=[early_stopping, reduce_lr])
        histories.append(history)

        # Get predictions after training
        y_val_pred = model.predict(X_val)
        y_val_pred_auc = roc_auc_score(Y_val[:, 1], y_val_pred[:, 1])

        final_auc = roc_auc_score(Y_val[:, 1], y_val_pred[:, 1])
        print(f'Final AUC for fold {fold_num + 1}: {final_auc:.4f}')

        if final_auc > best_auc:
            best_auc = final_auc
            best_model = model

        all_auc_scores.append(final_auc)

    # Plot and save loss vs validation loss graph
    plot_training_validation_loss(histories, save_dir)

    average_auc = sum(all_auc_scores) / len(all_auc_scores)
    print(f'Average AUC across all folds: {average_auc:.4f}')

    return best_model

# Function to load data
def loading_mask_3d(task, modality):
    images_pet, images_mri, labels = generate_data_path()
    original_imgs = []  # Initialize the list to store original images

    if modality == 'PET':
        data_train, train_label = generate(images_pet, labels, task)
    elif modality == 'MRI':
        data_train, train_label = generate(images_mri, labels, task)
    else:
        raise ValueError(f"Unsupported modality: {modality}")

    mask_path = '/home/l.peiwang/MR-PET-Classfication/mask_gm_p4_new4.nii'
    masker = NiftiMasker(mask_img=mask_path, target_affine=nib.load(mask_path).affine)  # Avoid resampling warnings

    train_data = []

    for i in range(len(data_train)):
        nifti_img = nib.load(data_train[i])
        original_imgs.append(nifti_img)

        masked_data = masker.fit_transform(nifti_img)
        reshaped_data = masker.inverse_transform(masked_data).get_fdata()
        reshaped_data = zscore(reshaped_data, axis=None)

        train_data.append(reshaped_data)

    train_label = binarylabel(train_label, task)
    train_data = np.array(train_data)

    return train_data, train_label, masker, original_imgs

# Main execution
if __name__ == '__main__':
    task = 'cd'  # Update as per your task
    modality = 'MRI'  # 'MRI' or 'PET'
    info = 'transformer_model_head_8_drop_0.2'

    train_data, train_label, masker, original_imgs = loading_mask_3d(task, modality)
    X = np.array(train_data)
    X = np.expand_dims(X, axis=-1)  # Add channel dimension
    Y = to_categorical(train_label, num_classes=2)

    best_model = train_model(X, Y, task, modality, info)

