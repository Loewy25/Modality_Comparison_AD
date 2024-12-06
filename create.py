import os
import h5py
import numpy as np
import nibabel as nib
from scipy.stats import zscore
from skimage.transform import resize

import dicom2nifti
import glob
import math
import nibabel as nib
import nilearn as nil
import numpy as np
import os
import pandas as pd
import pickle
import random
import scipy.ndimage as ndi
import statsmodels.stats.contingency_tables as ct
import time
from collections import Counter
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from nilearn import image, plotting
from nilearn.input_data import NiftiMasker
from nilearn.masking import (apply_mask, compute_brain_mask,
                             compute_multi_brain_mask, intersect_masks, unmask)
from nilearn.plotting import plot_roi, plot_stat_map, show
from numpy import mean, std
from numpy.linalg import inv
from scipy.stats import chi2_contingency, norm
from sklearn import metrics, svm
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_curve, precision_recall_fscore_support,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import (GridSearchCV, KFold, StratifiedKFold,
                                     cross_val_predict, cross_val_score,
                                     train_test_split)
from sklearn.preprocessing import Binarizer, label_binarize
from sklearn.svm import LinearSVC, SVC




def generate(images,labels,task):
    imagesData=[]
    labelsData=[]
    cn=0
    pcn=0
    dementia=0
    mci=0
    for i in range(len(images)):
      if labels[i]=='CN':
        cn+=1
      if labels[i]=='MCI':
        mci+=1
      if labels[i]=='Dementia':
        dementia+=1
      if labels[i]=='PCN':
        pcn+=1
    print("Number of CN subjects:")
    print(cn)
    print("Number of PCN subjects:")
    print(pcn)
    print("Number of MCI subjects:")
    print(mci)
    print("Number of Dementia subjects:")
    print(dementia)
    if task == "cd":
        for i in range(len(images)):
            if labels[i] == "CN":
                imagesData.append(images[i])
                labelsData.append(labels[i])
            if labels[i] == "Dementia":
                imagesData.append(images[i])
                labelsData.append(labels[i])
    if task == "cm":
        for i in range(len(images)):
            if labels[i] == "CN":
                imagesData.append(images[i])
                labelsData.append(labels[i])
            if labels[i] == "MCI":
                imagesData.append(images[i])
                labelsData.append(labels[i])
    if task == "dm":
        for i in range(len(images)):
            if labels[i] == "Dementia":
                imagesData.append(images[i])
                labelsData.append(labels[i])
            if labels[i] == "MCI":
                imagesData.append(images[i])
                labelsData.append(labels[i])
    if task == "pc":
        for i in range(len(images)):
            if labels[i] == "PCN":
                imagesData.append(images[i])
                labelsData.append(labels[i])
            if labels[i] == "CN":
                imagesData.append(images[i])
                labelsData.append(labels[i])   
    if task == 'cdm':
        for i in range(len(images)):
            if labels[i] == "CN":
                imagesData.append(images[i])
                labelsData.append(labels[i])
            if labels[i] == "Dementia" or labels[i] == 'MCI':
                imagesData.append(images[i])
                labelsData.append(labels[i])
    print("lenth of dataset: ")
    print(len(labelsData))
      
        
    return imagesData,labelsData


def generate_data_path():
    files=['/scratch/jjlee/Singularity/ADNI/bids/derivatives/table_preclinical_cross-sectional.csv','/scratch/jjlee/Singularity/ADNI/bids/derivatives/table_cn_cross-sectional.csv','/scratch/jjlee/Singularity/ADNI/bids/derivatives/table_cdr_0p5_apos_cross-sectional.csv','/scratch/jjlee/Singularity/ADNI/bids/derivatives/table_cdr_gt_0p5_apos_cross-sectional.csv']
    class_labels=['PCN','CN','MCI','Dementia']
    pet_paths = []
    mri_paths = []
    class_labels_out = []

    for file, class_label in zip(files, class_labels):
        df = pd.read_csv(file)

        for _, row in df.iterrows():
            # Extract sub-xxxx and ses-xxxx from original paths
            sub_ses_info = "/".join(row['FdgFilename'].split("/")[8:10])

            # Generate new directory
            new_directory = os.path.join('/scratch/l.peiwang/derivatives_new', sub_ses_info, 'pet')

            # Get all files that match the pattern but then exclude ones that contain 'icv'
            pet_files = [f for f in glob.glob(new_directory + '/*FDG*') if 'icv' not in f]
            mri_files = glob.glob(new_directory + '/*detJ*icv*')
            if pet_files and mri_files:  # If both lists are not empty
                pet_paths.append(pet_files[0])  # Append the first PET file found
                mri_paths.append(mri_files[0])  # Append the first MRI file found
                class_labels_out.append(class_label)  # Associate class label with the path

    return pet_paths, mri_paths, class_labels_out


def generate_data_path_less():
    files=['/scratch/jjlee/Singularity/ADNI/bids/derivatives/table_preclinical_cross-sectional.csv','/scratch/jjlee/Singularity/ADNI/bids/derivatives/table_cn_cross-sectional.csv','/scratch/jjlee/Singularity/ADNI/bids/derivatives/table_cdr_0p5_apos_cross-sectional.csv','/scratch/jjlee/Singularity/ADNI/bids/derivatives/table_cdr_gt_0p5_apos_cross-sectional.csv']
    class_labels=['PCN','CN','MCI','Dementia']
    pet_paths = []
    mri_paths = []
    class_labels_out = []

    for file, class_label in zip(files, class_labels):
        df = pd.read_csv(file)

        for _, row in df.iterrows():
            # Extract sub-xxxx and ses-xxxx from original paths
            sub_ses_info = "/".join(row['FdgFilename'].split("/")[8:10])

            # Generate new directory
            new_directory = os.path.join('/scratch/l.peiwang/derivatives_less', sub_ses_info, 'anat')

            # Get all files that match the pattern but then exclude ones that contain 'icv'
            pet_files = [f for f in glob.glob(new_directory + '/*FDG*') if 'icv' not in f]
            mri_files = glob.glob(new_directory + '/*brain*')
            if pet_files and mri_files:  # If both lists are not empty
                pet_paths.append(pet_files[0])  # Append the first PET file found
                mri_paths.append(mri_files[0])  # Append the first MRI file found
                class_labels_out.append(class_label)  # Associate class label with the path

    return pet_paths, mri_paths, class_labels_out


def binarylabel(train_label,mode):
    if mode=="cd" or mode=="cdm":
        for i in range(len(train_label)):
            if train_label[i]=="CN":
                train_label[i]=0
            else:
                train_label[i]=1
    if mode=="cm":
        for i in range(len(train_label)):
            if train_label[i]=="CN":
                train_label[i]=0
            else:
                train_label[i]=1
    if mode=="dm":
        for i in range(len(train_label)):
            if train_label[i]=="Dementia":
                train_label[i]=1
            else:
                train_label[i]=0
    if mode=="pc":
        for i in range(len(train_label)):
            if train_label[i]=="CN":
                train_label[i]=0
            else:
                train_label[i]=1
    if mode=="pm":
        for i in range(len(train_label)):
            if train_label[i]=="EMCI":
                train_label[i]=1
            else:
                train_label[i]=0
    return train_label



def loading_mask(task,modality):
    #Loading and generating data
    images_pet,images_mri,labels=generate_data_path_less()
    if modality == 'PET':
        data_train,train_label=generate(images_pet,labels,task)
    if modality == 'MRI':
        data_train,train_label=generate(images_mri,labels,task)
    masker = NiftiMasker(mask_img='/home/l.peiwang/MR-PET-Classfication/mask_gm_p4_new4.nii')
    train_data=[]
    for i in range(len(data_train)):
        a=masker.fit_transform(data_train[i])
        train_data.append(a)

    train_label=binarylabel(train_label,task)
    train_data=np.array(train_data).reshape(np.array(train_label).shape[0],122597)
    
    return train_data,train_label,masker



def load_mri_pet_data(task):
    """
    Load and preprocess MRI and PET data.
    Args:
        task (str): Task identifier.
    Returns:
        Tuple of numpy arrays: (mri_data, pet_data, labels)
    """
    images_pet, images_mri, labels = generate_data_path_less()
    pet_data, label = generate(images_pet, labels, task)
    mri_data, label = generate(images_mri, labels, task)

    mri_resized = []
    pet_resized = []

    for mri_path, pet_path in zip(mri_data, pet_data):
        mri_img = nib.load(mri_path).get_fdata()
        pet_img = nib.load(pet_path).get_fdata()
        mri_img = zscore(mri_img, axis=None)
        pet_img = zscore(pet_img, axis=None)
        mri_resized.append(resize_image(mri_img, (96, 112, 96)))
        pet_resized.append(resize_image(pet_img, (96, 112, 96)))

    return np.expand_dims(np.array(mri_resized), 1), np.expand_dims(np.array(pet_resized), 1), label



def create_hdf5(task="cd"):
    """
    Create three separate HDF5 files (train, validation, and test) from MRI and PET data 
    with random tabular data and store them in the specified scratch directory.
    """
    # Load MRI, PET, and labels
    mri_data, pet_data, labels = load_mri_pet_data(task)

    # Generate random tabular data (6 features)
    num_samples = len(labels)
    tabular_data = np.random.rand(num_samples, 6)  # Random values for the tabular data
    tabular_data[:, 0] = np.random.randint(50, 90, size=num_samples)  # Random age
    tabular_data[:, 1] = np.random.randint(0, 2, size=num_samples)    # Random gender (0 or 1)

    # Generate random attributes
    patient_ids = np.random.randint(1000, 9999, size=num_samples)  # Random patient IDs
    visit_codes = np.random.choice(["BL", "M12", "M24"], size=num_samples)  # Random visit codes

    # Meta-information for tabular data
    columns = ["age", "gender", "education", "MMSE", "ADAS-Cog-13", "ApoE4"]
    mean_tabular = np.mean(tabular_data, axis=0)
    std_tabular = np.std(tabular_data, axis=0)

    # Define output directory in scratch
    output_dir = os.path.join("/scratch/l.peiwang", "datasets")  # Replace with your scratch directory if needed
    os.makedirs(output_dir, exist_ok=True)

    # Split the dataset into train, val, test (70%, 15%, 15%)
    train_end = int(num_samples * 0.7)
    val_end = int(num_samples * 0.85)

    train_indices = np.arange(0, train_end)
    val_indices = np.arange(train_end, val_end)
    test_indices = np.arange(val_end, num_samples)

    def write_hdf5(file_path, indices):
        with h5py.File(file_path, "w") as f:
            for i, idx in enumerate(indices):
                image_id = f"ID_{i:03d}"
                group = f.create_group(image_id)

                # MRI/PET data
                mri_group = group.create_group("MRI")
                mri_group.create_dataset("T1", data=mri_data[idx], compression="gzip")

                pet_group = group.create_group("PET")
                pet_group.create_dataset("FDG", data=pet_data[idx], compression="gzip")

                # Tabular data
                group.create_dataset("tabular", data=tabular_data[idx])

                # Attributes
                group.attrs["DX"] = labels[idx]
                group.attrs["RID"] = patient_ids[idx]
                group.attrs["VISCODE"] = visit_codes[idx]

            # Add stats for the tabular data
            stats_group = f.create_group("stats")
            tabular_stats = stats_group.create_group("tabular")
            tabular_stats.create_dataset("columns", data=np.array(columns, dtype='S'))
            tabular_stats.create_dataset("mean", data=mean_tabular)
            tabular_stats.create_dataset("stddev", data=std_tabular)

        print(f"HDF5 file created and saved to {file_path} with {len(indices)} subjects.")

    # Write train, val, test files using fixed filenames in the scratch directory
    train_file_path = os.path.join(output_dir, "train.h5")
    val_file_path = os.path.join(output_dir, "val.h5")
    test_file_path = os.path.join(output_dir, "test.h5")

    write_hdf5(train_file_path, train_indices)
    write_hdf5(val_file_path, val_indices)
    write_hdf5(test_file_path, test_indices)


# Run the modified function
create_hdf5(task="cd")
