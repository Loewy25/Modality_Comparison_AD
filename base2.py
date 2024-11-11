import os
import numpy as np
import nibabel as nib
from nilearn.maskers import NiftiMasker
from scipy.ndimage import zoom
from scipy.stats import zscore
import time
import hashlib
from data_loading import generate_data_path_less, generate, binarylabel

from main import nested_crossvalidation_multi_kernel, nested_crossvalidation

def resize_and_normalize_image(image, target_shape):
    """
    Resize a 3D image to the target shape using zoom and apply z-score normalization.
    Args:
        image (numpy.ndarray): 3D image.
        target_shape (tuple): Desired shape.
    Returns:
        numpy.ndarray: Resized and normalized image.
    """
    # Resize image
    zoom_factors = [target_shape[i] / image.shape[i] for i in range(3)]
    image_resized = zoom(image, zoom_factors, order=1)
    # Apply z-score normalization
    image_normalized = zscore(image_resized, axis=None)
    return image_normalized

def compute_image_hash(image_data):
    """
    Compute a hash for the image data array.
    Args:
        image_data (numpy.ndarray): Image data array.
    Returns:
        str: Hash value of the image data.
    """
    data_bytes = image_data.tobytes()
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

    # Load the mask image
    mask_img = nib.load('/home/l.peiwang/MR-PET-Classfication/mask_gm_p4_new4.nii')

    # Create masker
    masker = NiftiMasker(mask_img=mask_img)

    # Apply binary labels
    train_label = binarylabel(train_label, task)

    return data_train, train_label, masker

def load_and_preprocess_images(image_paths, target_shape):
    """
    Load images from paths, resize, and z-score normalize.
    Args:
        image_paths (list): List of image file paths.
        target_shape (tuple): Desired shape after resizing.
    Returns:
        list: List of preprocessed images.
    """
    preprocessed_images = []
    for path in image_paths:
        img = nib.load(path).get_fdata()
        img_preprocessed = resize_and_normalize_image(img, target_shape)
        preprocessed_images.append(img_preprocessed)
    return preprocessed_images

def match_mri_hashes_and_labels(original_mri_data_preprocessed, original_labels, gan_mri_data_preprocessed):
    """
    Create a mapping from MRI data hashes to their labels and assign labels to GAN MRI data.
    Args:
        original_mri_data_preprocessed (list): List of preprocessed original MRI data arrays.
        original_labels (list): Corresponding labels.
        gan_mri_data_preprocessed (list): List of preprocessed GAN MRI data arrays.
    Returns:
        list: Assigned labels for GAN MRI data.
    """
    label_mapping = {}
    print("Creating label mapping for original MRI images using data hashes...")
    for data_array, label in zip(original_mri_data_preprocessed, original_labels):
        image_hash = compute_image_hash(data_array)
        label_mapping[image_hash] = label

    assigned_labels = []
    for gan_data in gan_mri_data_preprocessed:
        image_hash = compute_image_hash(gan_data)
        if image_hash in label_mapping:
            label = label_mapping[image_hash]
        else:
            label = -1  # Assign -1 if no match found
            print("No match found for a GAN MRI image.")
        assigned_labels.append(label)
    return assigned_labels

def resize_images_to_original(images_preprocessed, original_shape):
    """
    Resize images back to the original shape.
    Args:
        images_preprocessed (list): List of preprocessed images.
        original_shape (tuple): Original shape to resize back to.
    Returns:
        list: List of images resized back to original shape.
    """
    images_original_shape = []
    for img in images_preprocessed:
        img_resized_back = resize_image(img, original_shape)
        images_original_shape.append(img_resized_back)
    return images_original_shape

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

def main():
    task = 'cd'       # Example task identifier
    info = 'trying2'  # Example subfolder identifier

    # Define shapes
    target_shape = (128, 128, 128)  # Shape used in GAN processing
    original_shape = (91, 109, 91)  # Original image shape

    # Step 1: Load original MRI data with labels and file paths
    print("Loading original MRI data with labels and file paths...")
    original_mri_paths, original_labels, masker = loading_mask(task=task, modality='MRI')

    # Step 2: Load and preprocess original MRI data
    print("Loading and preprocessing original MRI data...")
    original_mri_data_preprocessed = load_and_preprocess_images(original_mri_paths, target_shape)

    # Step 3: Load and preprocess GAN MRI data
    print("Loading and preprocessing GAN MRI data...")
    gan_mri_dir = f'/home/l.peiwang/Modality_Comparison_AD/gan/{task}/{info}/mri'
    gan_mri_files = sorted([f for f in os.listdir(gan_mri_dir) if f.endswith('.nii.gz')])
    gan_mri_paths = [os.path.join(gan_mri_dir, f) for f in gan_mri_files]
    gan_mri_data_preprocessed = load_and_preprocess_images(gan_mri_paths, target_shape)

    # Step 4: Match GAN MRI data with original MRI data to assign labels
    print("Matching GAN MRI images with original MRI images to assign labels...")
    gan_labels = match_mri_hashes_and_labels(original_mri_data_preprocessed, original_labels, gan_mri_data_preprocessed)

    # Remove unmatched entries
    matched_indices = [i for i, label in enumerate(gan_labels) if label != -1]
    if not matched_indices:
        print("Error: No matches found between GAN MRI images and original MRI images.")
        return
    gan_mri_data_preprocessed = [gan_mri_data_preprocessed[i] for i in matched_indices]
    gan_labels = [gan_labels[i] for i in matched_indices]
    gan_mri_paths = [gan_mri_paths[i] for i in matched_indices]

    # Step 5: Load and preprocess real PET and generated PET data
    print("Loading and preprocessing real PET and generated PET data...")
    real_pet_dir = f'/home/l.peiwang/Modality_Comparison_AD/gan/{task}/{info}/real_pet'
    generated_pet_dir = f'/home/l.peiwang/Modality_Comparison_AD/gan/{task}/{info}/pet'

    real_pet_files = sorted([f for f in os.listdir(real_pet_dir) if f.endswith('.nii.gz')])
    generated_pet_files = sorted([f for f in os.listdir(generated_pet_dir) if f.endswith('.nii.gz')])

    real_pet_paths = [os.path.join(real_pet_dir, f) for f in real_pet_files]
    generated_pet_paths = [os.path.join(generated_pet_dir, f) for f in generated_pet_files]

    # Ensure paths are aligned
    real_pet_paths = [real_pet_paths[i] for i in matched_indices]
    generated_pet_paths = [generated_pet_paths[i] for i in matched_indices]

    # Load and preprocess PET data
    real_pet_data_preprocessed = load_and_preprocess_images(real_pet_paths, target_shape)
    generated_pet_data_preprocessed = load_and_preprocess_images(generated_pet_paths, target_shape)

    # Step 6: Resize all data back to original shape
    print("Resizing all data back to original shape...")
    gan_mri_data_original = resize_images_to_original(gan_mri_data_preprocessed, original_shape)
    real_pet_data_original = resize_images_to_original(real_pet_data_preprocessed, original_shape)
    generated_pet_data_original = resize_images_to_original(generated_pet_data_preprocessed, original_shape)

    # Step 7: Apply masker to data
    print("Applying masker to data...")
    # Create Nifti images with appropriate affine
    sample_img = nib.load(original_mri_paths[0])
    original_affine = sample_img.affine

    gan_mri_images = [nib.Nifti1Image(data, affine=original_affine) for data in gan_mri_data_original]
    real_pet_images = [nib.Nifti1Image(data, affine=original_affine) for data in real_pet_data_original]
    generated_pet_images = [nib.Nifti1Image(data, affine=original_affine) for data in generated_pet_data_original]

    # Fit masker on one of the datasets (e.g., GAN MRI)
    print("Fitting masker...")
    masker.fit(gan_mri_images)

    # Transform data
    print("Transforming data with masker...")
    processed_gan_mri = masker.transform(gan_mri_images)
    processed_real_pet = masker.transform(real_pet_images)
    processed_generated_pet = masker.transform(generated_pet_images)

    # Step 8: Perform classification comparisons using nested_crossvalidation
    print("Performing classification on GAN MRI data...")
    performance_mri, _, _, _ = nested_crossvalidation(
        processed_gan_mri, gan_labels, 'GAN_MRI', task
    )

    print("Performing classification on real PET data...")
    performance_real_pet, _, _, _ = nested_crossvalidation(
        processed_real_pet, gan_labels, 'Real_PET', task
    )

    print("Performing classification on generated PET data...")
    performance_generated_pet, _, _, _ = nested_crossvalidation(
        processed_generated_pet, gan_labels, 'Generated_PET', task
    )

    # Optional: Print performance metrics
    print("\nClassification Performance on GAN MRI Data:")
    print(performance_mri)

    print("\nClassification Performance on Real PET Data:")
    print(performance_real_pet)

    print("\nClassification Performance on Generated PET Data:")
    print(performance_generated_pet)

    print("\nAll steps completed successfully.")

if __name__ == "__main__":
    main()
