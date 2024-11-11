import os
import numpy as np
import nibabel as nib
from nilearn.maskers import NiftiMasker
import time
from data_loading import loading_mask  # Ensure this imports the updated loading_mask
from main import nested_crossvalidation_multi_kernel, nested_crossvalidation

def load_gan_saved_data(task, info):
    """
    Load file paths for saved GAN MRI, real PET, and generated PET images.
    
    Args:
        task (str): Task identifier (e.g., 'cd').
        info (str): Subfolder identifier used during GAN training (e.g., 'trying2').
    
    Returns:
        Tuple: (gan_mri_paths, real_pet_paths, generated_pet_paths)
    """
    gan_mri_dir = f'gan/{task}/{info}/mri'
    real_pet_dir = f'gan/{task}/{info}/real_pet'
    generated_pet_dir = f'gan/{task}/{info}/pet'
    
    gan_mri_files = sorted([f for f in os.listdir(gan_mri_dir) if f.endswith('.nii.gz')])
    real_pet_files = sorted([f for f in os.listdir(real_pet_dir) if f.endswith('.nii.gz')])
    generated_pet_files = sorted([f for f in os.listdir(generated_pet_dir) if f.endswith('.nii.gz')])
    
    gan_mri_paths = [os.path.join(gan_mri_dir, f) for f in gan_mri_files]
    real_pet_paths = [os.path.join(real_pet_dir, f) for f in real_pet_files]
    generated_pet_paths = [os.path.join(generated_pet_dir, f) for f in generated_pet_files]
    
    return gan_mri_paths, real_pet_paths, generated_pet_paths

def match_mri_labels(original_mri_paths, original_labels):
    """
    Create a mapping from original MRI filenames to their labels.
    
    Args:
        original_mri_paths (list): List of file paths to original MRI images.
        original_labels (numpy.ndarray): Corresponding labels.
    
    Returns:
        dict: Mapping from original MRI filenames to labels.
    """
    label_mapping = {}
    print("Creating label mapping for original MRI images...")
    for path, label in zip(original_mri_paths, original_labels):
        filename = os.path.basename(path)
        label_mapping[filename] = label
    return label_mapping

def assign_labels_to_gan_mri(gan_mri_paths, original_mri_mapping):
    """
    Assign labels to GAN-saved MRI images by matching filenames.
    
    Args:
        gan_mri_paths (list): Paths to GAN-saved MRI images.
        original_mri_mapping (dict): Mapping from original MRI filenames to labels.
    
    Returns:
        list: Assigned labels for GAN-saved MRI images.
    """
    assigned_labels = []
    for gan_path in gan_mri_paths:
        gan_filename = os.path.basename(gan_path)
        if gan_filename in original_mri_mapping:
            label = original_mri_mapping[gan_filename]
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
    pet_data = []
    for pet_path in pet_paths:
        img = nib.load(pet_path)
        masked_img = masker.transform(img)  # transform expects a Nifti1Image
        pet_data.append(masked_img.flatten())
    
    return np.array(pet_data)  # Shape: [N, Features]

def main():
    task = 'cd'       # Example task identifier
    info = 'trying2'  # Example subfolder identifier
    
    # Step 1: Load original MRI data with labels and file paths
    print("Loading original MRI data with labels and file paths...")
    original_mri_paths, original_labels, masker, processed_original_mri = loading_mask(task=task, modality='MRI')
    
    # Debug: Verify original MRI paths
    print(f"Number of original MRI paths: {len(original_mri_paths)}")
    if len(original_mri_paths) > 0:
        print(f"First 5 original MRI paths: {original_mri_paths[:5]}")
    
    # Step 2: Load GAN-saved MRI, real PET, and generated PET image paths
    print("Loading GAN-saved MRI, real PET, and generated PET image paths...")
    gan_mri_paths, real_pet_paths, generated_pet_paths = load_gan_saved_data(task=task, info=info)
    
    # Debug: Verify GAN-saved MRI paths
    print(f"Number of GAN-saved MRI paths: {len(gan_mri_paths)}")
    if len(gan_mri_paths) > 0:
        print(f"First 5 GAN-saved MRI paths: {gan_mri_paths[:5]}")
    
    # Step 3: Match GAN-saved MRI images with original MRI images to assign labels
    print("Matching GAN-saved MRI images with original MRI images to assign labels...")
    original_mri_mapping = match_mri_labels(original_mri_paths, original_labels)
    gan_labels = assign_labels_to_gan_mri(gan_mri_paths, original_mri_mapping)
    
    # Check for unmatched images
    unmatched = [i for i, label in enumerate(gan_labels) if label == -1]
    if unmatched:
        print(f"Number of unmatched GAN-saved MRI images: {len(unmatched)}")
        # Optionally, remove unmatched entries
        gan_mri_paths = [path for i, path in enumerate(gan_mri_paths) if gan_labels[i] != -1]
        real_pet_paths = [path for i, path in enumerate(real_pet_paths) if gan_labels[i] != -1]
        generated_pet_paths = [path for i, path in enumerate(generated_pet_paths) if gan_labels[i] != -1]
        gan_labels = [label for label in gan_labels if label != -1]
    
    # Step 4: Assign labels to real PET and generated PET images
    print("Assigning labels to real PET and generated PET images...")
    real_pet_labels, generated_pet_labels = assign_labels_to_pets(real_pet_paths, generated_pet_paths, gan_labels)
    
    # Step 5: Load and preprocess real PET and generated PET images
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
    
    # Step 6: Perform classification comparisons using nested_crossvalidation
    print("Performing classification on real PET data...")
    start_time = time.time()
    performance_real_pet, all_y_test_real_pet, all_y_prob_real_pet, all_predictions_real_pet = nested_crossvalidation(
        processed_real_pet, real_pet_labels, 'Real_PET_z', 'cd'
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Real PET classification took {elapsed_time:.2f} seconds.")
    
    print("\nPerforming classification on generated PET data...")
    start_time = time.time()
    performance_generated_pet, all_y_test_generated_pet, all_y_prob_generated_pet, all_predictions_generated_pet = nested_crossvalidation(
        processed_generated_pet, generated_pet_labels, 'Generated_PET_z', 'cd'
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Generated PET classification took {elapsed_time:.2f} seconds.")
  
    print("\nPerforming classification on original MRI data...")
    start_time = time.time()
    performance_mri, all_y_test_mri, all_y_prob_mri, all_predictions_mri  = nested_crossvalidation(
        processed_original_mri, original_labels, 'mri', 'cd'
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
