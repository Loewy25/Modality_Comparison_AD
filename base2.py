import os
import numpy as np
import nibabel as nib
from nilearn.maskers import NiftiMasker
import time
import hashlib
from data_loading import generate_data_path_less, generate, binarylabel

from main import nested_crossvalidation_multi_kernel, nested_crossvalidation

def compute_image_hash(image_path):
    """
    Compute a hash for the image data array at the given path.
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
    
    # Load a mask image that has the same affine as your images
    mask_img = nib.load('/home/l.peiwang/MR-PET-Classfication/mask_gm_p4_new4.nii')

    # Optionally, resample the mask to match the images to avoid resampling during transformation
    sample_img = nib.load(data_train[0])
    if not np.allclose(mask_img.affine, sample_img.affine):
        print("Resampling mask to match the images' affine.")
        from nilearn.image import resample_to_img
        mask_img = resample_to_img(mask_img, sample_img, interpolation='nearest')

    masker = NiftiMasker(mask_img=mask_img)

    # Apply binary labels
    train_label = binarylabel(train_label, task)
    
    return data_train, train_label, masker

def load_mri_data(mri_paths, masker):
    """
    Load and preprocess MRI images.
    """
    print("Fitting masker and transforming MRI data...")
    mri_data = masker.fit_transform(mri_paths)
    return mri_data

def load_gan_saved_data(task, info):
    """
    Load file paths for saved GAN MRI, real PET, and generated PET images.
    """
    base_dir = '/home/l.peiwang/Modality_Comparison_AD/gan'  # Updated base directory
    gan_mri_dir = os.path.join(base_dir, f'{task}/{info}/mri')
    real_pet_dir = os.path.join(base_dir, f'{task}/{info}/real_pet')
    generated_pet_dir = os.path.join(base_dir, f'{task}/{info}/pet')

    gan_mri_files = sorted([f for f in os.listdir(gan_mri_dir) if f.endswith('.nii.gz')])
    real_pet_files = sorted([f for f in os.listdir(real_pet_dir) if f.endswith('.nii.gz')])
    generated_pet_files = sorted([f for f in os.listdir(generated_pet_dir) if f.endswith('.nii.gz')])

    gan_mri_paths = [os.path.join(gan_mri_dir, f) for f in gan_mri_files]
    real_pet_paths = [os.path.join(real_pet_dir, f) for f in real_pet_files]
    generated_pet_paths = [os.path.join(generated_pet_dir, f) for f in generated_pet_files]

    return gan_mri_paths, real_pet_paths, generated_pet_paths

def match_mri_hashes_and_labels(original_mri_paths, original_labels):
    """
    Create a mapping from MRI data hashes to their labels.
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
    """
    real_pet_labels = gan_labels.copy()
    generated_pet_labels = gan_labels.copy()
    return real_pet_labels, generated_pet_labels

def load_pet_data(pet_paths, masker):
    """
    Load and preprocess PET images.
    """
    if not pet_paths:
        raise ValueError("Error: pet_paths is empty. No PET images to process.")
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
    print(f"Number of GAN MRI paths: {len(gan_mri_paths)}")
    if len(gan_mri_paths) > 0:
        print(f"First 5 GAN MRI paths: {gan_mri_paths[:5]}")

    # Step 4: Match GAN-saved MRI images with original MRI images to assign labels
    print("Matching GAN-saved MRI images with original MRI images to assign labels...")
    original_mri_hash_mapping = match_mri_hashes_and_labels(original_mri_paths, original_labels)
    gan_labels = assign_labels_to_gan_mri_by_hash(gan_mri_paths, original_mri_hash_mapping)

    # Check for unmatched images
    unmatched_indices = [i for i, label in enumerate(gan_labels) if label == -1]
    if unmatched_indices:
        print(f"Number of unmatched GAN MRI images: {len(unmatched_indices)}")
        # Remove unmatched entries
        indices_to_keep = [i for i, label in enumerate(gan_labels) if label != -1]
        gan_mri_paths = [gan_mri_paths[i] for i in indices_to_keep]
        real_pet_paths = [real_pet_paths[i] for i in indices_to_keep]
        generated_pet_paths = [generated_pet_paths[i] for i in indices_to_keep]
        gan_labels = [gan_labels[i] for i in indices_to_keep]

    # Step 5: Assign labels to real PET and generated PET images
    print("Assigning labels to real PET and generated PET images...")
    real_pet_labels, generated_pet_labels = assign_labels_to_pets(real_pet_paths, generated_pet_paths, gan_labels)

    # Check if real_pet_paths is empty
    if not real_pet_paths:
        print("Error: No real PET images to process after filtering.")
        return

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
    performance_real_pet, _, _, _ = nested_crossvalidation(
        processed_real_pet, real_pet_labels, 'Real_PET_z', task
    )
    end_time = time.time()
    print(f"Real PET classification took {end_time - start_time:.2f} seconds.")

    print("\nPerforming classification on generated PET data...")
    start_time = time.time()
    performance_generated_pet, _, _, _ = nested_crossvalidation(
        processed_generated_pet, generated_pet_labels, 'Generated_PET_z', task
    )
    end_time = time.time()
    print(f"Generated PET classification took {end_time - start_time:.2f} seconds.")

    print("\nPerforming classification on original MRI data...")
    start_time = time.time()
    performance_mri, _, _, _ = nested_crossvalidation(
        processed_original_mri, original_labels, 'MRI', task
    )
    end_time = time.time()
    print(f"Original MRI classification took {end_time - start_time:.2f} seconds.")

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
