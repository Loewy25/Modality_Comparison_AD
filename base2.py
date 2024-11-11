import os
import numpy as np
import nibabel as nib
from nilearn.maskers import NiftiMasker
import time
import hashlib
from data_loading import generate_data_path_less, generate, binarylabel

# Assuming these functions are defined elsewhere
from main import nested_crossvalidation_multi_kernel, nested_crossvalidation

def compute_image_hash(image_path):
    """
    Compute a hash for the image data array at the given path.
    
    Args:
        image_path (str): Path to the image file.
        
    Returns:
        str: Hash value of the image data.
    """
    img = nib.load(image_path)
    data_array = img.get_fdata()
    data_bytes = data_array.tobytes()
    return hashlib.md5(data_bytes).hexdigest()

def loading_mask(task, modality):
    # Loading and generating data
    images_pet, images_mri, labels = generate_data_path_less()
    if modality == 'PET':
        data_train, train_label = generate(images_pet, labels, task)
    elif modality == 'MRI':
        data_train, train_label = generate(images_mri, labels, task)
    else:
        raise ValueError("Invalid modality specified")
    
    masker = NiftiMasker(mask_img='/home/l.peiwang/MR-PET-Classfication/mask_gm_p4_new4.nii')
    
    # Apply binary labels
    train_label = binarylabel(train_label, task)
    
    return data_train, train_label, masker

def load_mri_data(mri_paths, masker):
    """
    Load and preprocess MRI images.
    
    Args:
        mri_paths (list): List of MRI image file paths.
        masker (NiftiMasker): Masker used for preprocessing.
    
    Returns:
        numpy.ndarray: Flattened MRI data.
    """
    print("Fitting masker and transforming MRI data...")
    mri_data = masker.fit_transform(mri_paths)
    return mri_data

def match_mri_hashes_and_labels(original_mri_paths, original_labels):
    """
    Create a mapping from MRI data hashes to their labels.
    
    Args:
        original_mri_paths (list): List of file paths to original MRI images.
        original_labels (numpy.ndarray): Corresponding labels.
    
    Returns:
        dict: Mapping from image data hashes to labels.
    """
    label_mapping = {}
    print("Creating label mapping for original MRI images using data hashes...")
    for path, label in zip(original_mri_paths, original_labels):
        image_hash = compute_image_hash(path)
        label_mapping[image_hash] = label
    return label_mapping

def assign_labels_to_gan_mri_by_hash(gan_mri_paths, original_mri_hash_mapping):
    """
    Assign labels to GAN-saved MRI images by matching data hashes.
    
    Args:
        gan_mri_paths (list): Paths to GAN-saved MRI images.
        original_mri_hash_mapping (dict): Mapping from image data hashes to labels.
    
    Returns:
        list: Assigned labels for GAN-saved MRI images.
    """
    assigned_labels = []
    for gan_path in gan_mri_paths:
        image_hash = compute_image_hash(gan_path)
        if image_hash in original_mri_hash_mapping:
            label = original_mri_hash_mapping[image_hash]
        else:
            label = -1  # Assign -1 if no match found
            print(f"No match found for {gan_path}")
        assigned_labels.append(label)
    return assigned_labels

def assign_labels_to_pets(real_pet_paths, generated_pet_paths, gan_labels):
    """
    Assign labels to real PET and generated PET images based on GAN-saved MRI labels.
    
    Args:
        real_pet_paths (list): Paths to real PET images.
        generated_pet_paths (list): Paths to generated PET images.
        gan_labels (list): Assigned labels for GAN-saved MRI images.
    
    Returns:
        Tuple: (real_pet_labels, generated_pet_labels)
    """
    # Assuming the order of real_pet_paths and generated_pet_paths matches gan_mri_paths
    real_pet_labels = gan_labels.copy()
    generated_pet_labels = gan_labels.copy()
    
    return real_pet_labels, generated_pet_labels

def load_pet_data(pet_paths, masker):
    """
    Load and preprocess PET images.
    
    Args:
        pet_paths (list): List of PET image file paths.
        masker (NiftiMasker): Masker used for preprocessing.
    
    Returns:
        numpy.ndarray: Flattened PET data.
    """
    print("Transforming PET data using fitted masker...")
    pet_data = masker.transform(pet_paths)
    return pet_data

def main():
    task = 'cd'       # Example task identifier
    info = 'trying2'  # Example subfolder identifier
    
    # Step 1: Load original MRI data with labels and file paths
    print("Loading original MRI data with labels and file paths...")
    original_mri_paths, original_labels, masker = loading_mask(task=task, modality='MRI')
    
    # Step 2: Load and preprocess original MRI data
    processed_original_mri = load_mri_data(original_mri_paths, masker)
    
    # Debug: Verify original MRI paths
    print(f"Number of original MRI paths: {len(original_mri_paths)}")
    if len(original_mri_paths) > 0:
        print(f"First 5 original MRI paths: {original_mri_paths[:5]}")
    
    # Step 3: Load GAN-saved MRI, real PET, and generated PET image paths
    print("Loading GAN-saved MRI, real PET, and generated PET image paths...")
    gan_mri_paths, real_pet_paths, generated_pet_paths = load_gan_saved_data(task=task, info=info)
    
    # Debug: Verify GAN-saved MRI paths
    print(f"Number of GAN-saved MRI paths: {len(gan_mri_paths)}")
    if len(gan_mri_paths) > 0:
        print(f"First 5 GAN-saved MRI paths: {gan_mri_paths[:5]}")
    
    # Step 4: Match GAN-saved MRI images with original MRI images to assign labels
    print("Matching GAN-saved MRI images with original MRI images to assign labels...")
    original_mri_hash_mapping = match_mri_hashes_and_labels(original_mri_paths, original_labels)
    gan_labels = assign_labels_to_gan_mri_by_hash(gan_mri_paths, original_mri_hash_mapping)
    
    # Check for unmatched images
    unmatched = [i for i, label in enumerate(gan_labels) if label == -1]
    if unmatched:
        print(f"Number of unmatched GAN-saved MRI images: {len(unmatched)}")
        # Optionally, remove unmatched entries
        gan_mri_paths = [path for i, path in enumerate(gan_mri_paths) if gan_labels[i] != -1]
        real_pet_paths = [path for i, path in enumerate(real_pet_paths) if gan_labels[i] != -1]
        generated_pet_paths = [path for i, path in enumerate(generated_pet_paths) if gan_labels[i] != -1]
        gan_labels = [label for label in gan_labels if label != -1]
    
    # Step 5: Assign labels to real PET and generated PET images
    print("Assigning labels to real PET and generated PET images...")
    real_pet_labels, generated_pet_labels = assign_labels_to_pets(real_pet_paths, generated_pet_paths, gan_labels)
    
    # Step 6: Load and preprocess real PET and generated PET images
    print("Loading and preprocessing real PET images...")
    processed_real_pet = load_pet_data(real_pet_paths, masker)
    
    print("Loading and preprocessing generated PET images...")
    processed_generated_pet = load_pet_data(generated_pet_paths, masker)
    
    # Now, you have:
    # - processed_real_pet: numpy array of real PET data
    # - processed_generated_pet: numpy array of generated PET data
    # - processed_original_mri: numpy array of original MRI data
    # - real_pet_labels: list of labels for real PET data
    # - generated_pet_labels: list of labels for generated PET data
    
    # Step 7: Perform classification comparisons using nested_crossvalidation
    print("Performing classification on real PET data...")
    start_time = time.time()
    performance_real_pet, all_y_test_real_pet, all_y_prob_real_pet, all_predictions_real_pet = nested_crossvalidation(
        processed_real_pet, real_pet_labels, 'Real_PET_z', task
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Real PET classification took {elapsed_time:.2f} seconds.")
    
    print("\nPerforming classification on generated PET data...")
    start_time = time.time()
    performance_generated_pet, all_y_test_generated_pet, all_y_prob_generated_pet, all_predictions_generated_pet = nested_crossvalidation(
        processed_generated_pet, generated_pet_labels, 'Generated_PET_z', task
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Generated PET classification took {elapsed_time:.2f} seconds.")
  
    print("\nPerforming classification on original MRI data...")
    start_time = time.time()
    performance_mri, all_y_test_mri, all_y_prob_mri, all_predictions_mri  = nested_crossvalidation(
        processed_original_mri, original_labels, 'MRI', task
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Original MRI classification took {elapsed_time:.2f} seconds.")
  
    # Optional: Print performance metrics
    print("\nClassification Performance on Real PET Data:")
    print(performance_real_pet)
    
    print("\nClassification Performance on Generated PET Data:")
    print(performance_generated_pet)
    
    print("\nClassification Performance on Original MRI Data:")
    print(performance_mri)
    
    # Optional: Save the processed data and labels for future use
    np.save('processed_real_pet.npy', processed_real_pet)
    np.save('processed_generated_pet.npy', processed_generated_pet)
    np.save('processed_original_mri.npy', processed_original_mri)
    np.save('labels_real_pet.npy', np.array(real_pet_labels))
    np.save('labels_generated_pet.npy', np.array(generated_pet_labels))
    np.save('labels_original_mri.npy', np.array(original_labels))
    
    print("\nAll steps completed successfully.")

if __name__ == "__main__":
    main()
