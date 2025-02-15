import os
import numpy as np
import nibabel as nib
from nilearn.maskers import NiftiMasker
from scipy.ndimage import zoom
from scipy.stats import zscore
import hashlib
from data_loading import generate_data_path_less, generate, binarylabel

from main import nested_crossvalidation_multi_kernel, nested_crossvalidation

def resize_image(image, target_shape):
    """
    Resize a 3D image to the target shape using zoom.
    Args:
        image (numpy.ndarray): 3D image.
        target_shape (tuple): Desired shape.
    Returns:
        numpy.ndarray: Resized image.
    """
    zoom_factors = [target_shape[i] / image.shape[i] for i in range(3)]
    return zoom(image, zoom_factors, order=1)

def save_image(image, file_path):
    """
    Save a 3D image as a NIfTI file with affine np.eye(4).
    Args:
        image (numpy.ndarray): 3D image.
        file_path (str): Destination file path.
    """
    nib.save(nib.Nifti1Image(image, np.eye(4)), file_path)

def preprocess_and_save_original_mri(task):
    """
    Preprocess original MRI data and save as NIfTI files with affine np.eye(4).
    Args:
        task (str): Task identifier.
    Returns:
        List of file paths to the saved preprocessed MRI images.
    """
    images_pet, images_mri, labels = generate_data_path_less()
    mri_data_paths, labels = generate(images_mri, labels, task)

    preprocessed_mri_paths = []
    for mri_path in mri_data_paths:
        mri_img = nib.load(mri_path).get_fdata()
        mri_img = zscore(mri_img, axis=None)
        mri_img_resized = resize_image(mri_img, (128, 128, 128))
        # Save the preprocessed image
        filename = os.path.basename(mri_path)
        preprocessed_path = os.path.join('/tmp/preprocessed_mri', filename)
        os.makedirs(os.path.dirname(preprocessed_path), exist_ok=True)
        save_image(mri_img_resized, preprocessed_path)
        preprocessed_mri_paths.append(preprocessed_path)

    return preprocessed_mri_paths, labels

def match_mri_by_files(preprocessed_mri_paths, preprocessed_gan_mri_paths, labels):
    """
    Match preprocessed original MRI images with GAN MRI images by comparing the NIfTI files directly.
    Args:
        preprocessed_mri_paths (list): Paths to preprocessed original MRI images.
        preprocessed_gan_mri_paths (list): Paths to GAN MRI images.
        labels (list): Labels corresponding to the original MRI images.
    Returns:
        Tuple of matched GAN MRI paths and their labels.
    """
    matched_gan_mri_paths = []
    matched_labels = []

    # Create a set of hashes for the preprocessed original MRI images
    mri_hashes = {}
    for path, label in zip(preprocessed_mri_paths, labels):
        img_data = nib.load(path).get_fdata()
        img_hash = hashlib.md5(img_data.tobytes()).hexdigest()
        mri_hashes[img_hash] = label

    # Compare GAN MRI images to original MRI images
    for gan_path in preprocessed_gan_mri_paths:
        gan_img_data = nib.load(gan_path).get_fdata()
        gan_img_hash = hashlib.md5(gan_img_data.tobytes()).hexdigest()
        if gan_img_hash in mri_hashes:
            matched_gan_mri_paths.append(gan_path)
            matched_labels.append(mri_hashes[gan_img_hash])
        else:
            print(f"No match found for {gan_path}")

    return matched_gan_mri_paths, matched_labels


import os
import glob
import pandas as pd
from itertools import combinations

def generate_data():
    base_csv_dir = "/ceph/chpc/shared/aristeidis_sotiras_group/l.peiwang_scratch/derivatives_new"
    tasks = ['cd','cm','md','pc']
    
    # Mapping from pair of tasks to final class label
    intersection_map = {
        frozenset(['cd','cm']): 'CN',
        frozenset(['cd','md']): 'Dementia',
        frozenset(['cm','md']): 'MCI',
        frozenset(['pc','cm']): 'CN',
        frozenset(['pc','cd']): 'CN'
    }
    
    # Dictionary to hold subject info
    subjects = {}
    
    def extract_sub_ses(filepath):
        parts = filepath.split('/')
        sub_ses = []
        for p in parts:
            if p.startswith('sub-') or p.startswith('ses-'):
                sub_ses.append(p)
        if len(sub_ses) == 2:
            return "/".join(sub_ses)
        return None

    # Read CSVs and store info
    for task in tasks:
        mri_csv = os.path.join(base_csv_dir, f"{task}_MRI_data.csv")
        pet_csv = os.path.join(base_csv_dir, f"{task}_PET_data.csv")
        
        mri_df = pd.read_csv(mri_csv, header=None)
        pet_df = pd.read_csv(pet_csv, header=None)
        
        for mri_line, pet_line in zip(mri_df[0], pet_df[0]):
            mri_path = mri_line.split(',')[0].strip()
            pet_path = pet_line.split(',')[0].strip()

            sub_ses_info = extract_sub_ses(mri_path)
            if sub_ses_info is None:
                sub_ses_info = extract_sub_ses(pet_path)
            if sub_ses_info is None:
                continue

            if sub_ses_info not in subjects:
                subjects[sub_ses_info] = {'tasks': set()}
            subjects[sub_ses_info]['tasks'].add(task)
            # Just store one MRI and PET path as reference
            # (This doesn't have to be perfect, since we will use new_directory to find actual files)
            if 'mri_path' not in subjects[sub_ses_info]:
                subjects[sub_ses_info]['mri_path'] = mri_path
            if 'pet_path' not in subjects[sub_ses_info]:
                subjects[sub_ses_info]['pet_path'] = pet_path

    pet_paths = []
    mri_paths = []
    class_labels_out = []

    # Determine class for each subject
    for sub_ses_info, data in subjects.items():
        tasks_present = data['tasks']
        if len(tasks_present) < 2:
            # Not enough info to deduce class
            continue
        
        assigned_label = None
        for combo in combinations(tasks_present, 2):
            pair = frozenset(combo)
            if pair in intersection_map:
                assigned_label = intersection_map[pair]
                break
        
        if assigned_label is None:
            continue

        # Same logic as old function
        # Use derivatives_less as mentioned before
        # If you have changed directory structure, adjust here
        new_directory = os.path.join('/scratch/l.peiwang/derivatives_less', sub_ses_info, 'anat')
        
        pet_files = [f for f in glob.glob(new_directory + '/*FDG*') if 'icv' not in f]
        mri_files = glob.glob(new_directory + '/*brain*')
        if pet_files and mri_files:
            pet_paths.append(pet_files[0])
            mri_paths.append(mri_files[0])
            class_labels_out.append(assigned_label)

    return pet_paths, mri_paths, class_labels_out




def main():
    task = 'cd'       # Example task identifier
    info = 'gan/cd/new_Paramid_40_30_110_zeta_0.7'  # Example subfolder identifier

    # Step 1: Preprocess original MRI data and save as NIfTI files
    print("Preprocessing original MRI data and saving as NIfTI files...")
    preprocessed_mri_paths, labels = preprocess_and_save_original_mri(task)

    # Step 2: Load GAN MRI, which were saved after preprocessing with affine np.eye(4)
    print("Loading GAN MRI images...")
    gan_mri_dir = f'/ceph/chpc/home/l.peiwang/Modality_Comparison_AD/gan/{task}/{info}/mri'
    preprocessed_gan_mri_files = sorted([f for f in os.listdir(gan_mri_dir) if f.endswith('.nii.gz')])
    preprocessed_gan_mri_paths = [os.path.join(gan_mri_dir, f) for f in preprocessed_gan_mri_files]

    # Step 3: Match GAN MRI images with preprocessed original MRI images
    print("Matching GAN MRI images with preprocessed original MRI images to assign labels...")
    matched_gan_mri_paths, matched_labels = match_mri_by_files(preprocessed_mri_paths, preprocessed_gan_mri_paths, labels)

    if not matched_gan_mri_paths:
        print("Error: No matches found between GAN MRI images and preprocessed original MRI images.")
        return

    # Step 4: Load GAN MRI, Real PET, and Synthesized PET images
    print("Loading GAN MRI, Real PET, and Synthesized PET images...")
    real_pet_dir = f'/ceph/chpc/home/l.peiwang/Modality_Comparison_AD/gan/{task}/{info}/real_pet'
    generated_pet_dir = f'/ceph/chpc/home/l.peiwang/Modality_Comparison_AD/gan/{task}/{info}/pet'

    real_pet_files = sorted([f for f in os.listdir(real_pet_dir) if f.endswith('.nii.gz')])
    generated_pet_files = sorted([f for f in os.listdir(generated_pet_dir) if f.endswith('.nii.gz')])

    real_pet_paths = [os.path.join(real_pet_dir, f) for f in real_pet_files]
    generated_pet_paths = [os.path.join(generated_pet_dir, f) for f in generated_pet_files]

    # Ensure paths are aligned
    real_pet_paths = real_pet_paths[:len(matched_gan_mri_paths)]
    generated_pet_paths = generated_pet_paths[:len(matched_gan_mri_paths)]

    # Step 5: Load images and resize back to original shape
    print("Loading images and resizing back to original shape...")
    gan_mri_data = []
    real_pet_data = []
    generated_pet_data = []

    for gan_path, real_pet_path, gen_pet_path in zip(matched_gan_mri_paths, real_pet_paths, generated_pet_paths):
        # Load GAN MRI
        gan_img = nib.load(gan_path).get_fdata()
        gan_img_resized = resize_image(gan_img, (91, 109, 91))
        gan_mri_data.append(gan_img_resized)

        # Load Real PET
        real_pet_img = nib.load(real_pet_path).get_fdata()
        real_pet_img_resized = resize_image(real_pet_img, (91, 109, 91))
        real_pet_data.append(real_pet_img_resized)

        # Load Generated PET
        gen_pet_img = nib.load(gen_pet_path).get_fdata()
        gen_pet_img_resized = resize_image(gen_pet_img, (91, 109, 91))
        generated_pet_data.append(gen_pet_img_resized)

    # Step 6: Apply mask to the data
    print("Applying mask to the data...")
    mask_img = nib.load('/ceph/chpc/home/l.peiwang/MR-PET-Classfication/mask_gm_p4_new4.nii')
    masker = NiftiMasker(mask_img=mask_img)
    masker.fit()  # Since mask_img is already provided, we can fit the masker without data

    # Create NIfTI images with the correct affine
    affine = mask_img.affine

    gan_mri_images = [nib.Nifti1Image(data, affine=affine) for data in gan_mri_data]
    real_pet_images = [nib.Nifti1Image(data, affine=affine) for data in real_pet_data]
    generated_pet_images = [nib.Nifti1Image(data, affine=affine) for data in generated_pet_data]

    # Transform data using the masker
    print("Transforming data using the masker...")
    processed_gan_mri = masker.transform(gan_mri_images)
    processed_real_pet = masker.transform(real_pet_images)
    processed_generated_pet = masker.transform(generated_pet_images)

    # Convert labels to binary labels
    binary_labels = binarylabel(matched_labels, task)

    # Step 7: Perform classification
    print("Performing classification on GAN MRI data...")
    performance_mri, _, _, _ = nested_crossvalidation(
        processed_gan_mri, binary_labels, 'GAN_MRI', task
    )

    print("Performing classification on Real PET data...")
    performance_real_pet, _, _, _ = nested_crossvalidation(
        processed_real_pet, binary_labels, 'Real_PET', task
    )

    print("Performing classification on Generated PET data...")
    performance_generated_pet, _, _, _ = nested_crossvalidation(
        processed_generated_pet, binary_labels, 'Generated_PET', task
    )


if __name__ == "__main__":
    main()
