from nilearn.input_data import NiftiMasker

from data_loading import loading_mask, generate_data_path,generate,binarylabel
from utils import normalize_features, threshold_p_values
from plot_utils import plot_glass_brain, plot_stat_map
from main import hyperparameter_tuning_visual_cov_V3


import numpy as np

def loading_mask(task, modality):
    # Loading and generating data
    images_pet, images_mri, labels = generate_data_path()

    if modality == 'PET':
        data_train, train_label = generate(images_pet, labels, task)
    elif modality == 'MRI':
        data_train, train_label = generate(images_mri, labels, task)

    masker = NiftiMasker(mask_img='/home/l.peiwang/MR-PET-Classfication/mask_gm_p4_new4.nii')

    # Filtering logic for 'mci' labels
    filtered_data_train = []
    filtered_train_label = []
    mci_count = 0
    for i in range(len(train_label)):
        if train_label[i] == 'mci':
            if mci_count < 151:  # Accept only first 151 'mci' instances
                filtered_data_train.append(data_train[i])
                filtered_train_label.append(train_label[i])
                mci_count += 1
        else:  # Accept all 'cn' instances
            filtered_data_train.append(data_train[i])
            filtered_train_label.append(train_label[i])

    # Applying masker to the data
    train_data = []
    for data in filtered_data_train:
        a = masker.fit_transform(data)
        train_data.append(a)

    # Process labels
    train_label = binarylabel(filtered_train_label, task)
    train_data = np.array(train_data).reshape(np.array(train_label).shape[0], 122597)

    return train_data, train_label, masker

task="cm"
method="PET"
threshold=0.01

image1,label1,masker=loading_mask(task,method)


control_indices = [i for i, label in enumerate(label1) if label == 0]
image = normalize_features(image1, control_indices)
average_single_weights,average_corrected_weights,average_permuted_single_weights,average_permuted_corrected_weights=hyperparameter_tuning_visual_cov_V3(image,label1,[30],5,3,1000)
average_single_weights=1-average_single_weights
average_permuted_corrected_weights=1-average_permuted_corrected_weights
small_value = 1e-10
average_permuted_corrected_weights[average_permuted_corrected_weights == 0] = small_value
average_single_weights[average_single_weights == 0] = small_value
average_single_weights_5 = threshold_p_values(average_single_weights,threshold=threshold)
average_corrected_weights_5 = threshold_p_values(average_corrected_weights,threshold=threshold)
average_permuted_single_weights_5 = threshold_p_values(average_permuted_single_weights,threshold=threshold)
average_permuted_corrected_weights_5 = threshold_p_values(average_permuted_corrected_weights,threshold=threshold)
average_single_weights_5 = masker.inverse_transform(average_single_weights_5)
average_corrected_weights_5 = masker.inverse_transform(average_corrected_weights_5)
average_permuted_single_weights_5 = masker.inverse_transform(average_permuted_single_weights_5)
average_permuted_corrected_weights_5 = masker.inverse_transform(average_permuted_corrected_weights_5)
plot_glass_brain(average_single_weights_5, 'average_single_weights_masking_2', task, method,vmax=threshold)
plot_stat_map(average_single_weights_5, 0, 'average_single_weights_masking_2', task, method,vmax=threshold)
plot_glass_brain(average_corrected_weights_5, 'average_corrected_weights_masking_2', task, method,vmax=threshold)
plot_stat_map(average_corrected_weights_5, 0, 'average_corrected_weights_masking_2', task, method,vmax=threshold)
plot_glass_brain(average_permuted_single_weights_5, 'average_permuted_single_weights_masking_2', task, method,vmax=threshold)
plot_stat_map(average_permuted_single_weights_5, 0, 'average_permuted_single_weights_masking_2', task, method,vmax=threshold)
plot_glass_brain(average_permuted_corrected_weights_5, 'average_permuted_corrected_weights_masking_2', task, method,vmax=threshold)
plot_stat_map(average_permuted_corrected_weights_5, 0, 'average_permuted_corrected_weights_masking_2', task, method,vmax=threshold)
