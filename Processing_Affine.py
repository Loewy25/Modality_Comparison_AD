import pandas as pd
import os
import glob
import subprocess

def generate_and_process_data():
    # Paths to your CSV files
    files = [
        '/scratch/jjlee/Singularity/ADNI/bids/derivatives/table_preclinical_cross-sectional.csv',
        '/scratch/jjlee/Singularity/ADNI/bids/derivatives/table_cn_cross-sectional.csv',
        '/scratch/jjlee/Singularity/ADNI/bids/derivatives/table_cdr_0p5_apos_cross-sectional.csv',
        '/scratch/jjlee/Singularity/ADNI/bids/derivatives/table_cdr_gt_0p5_apos_cross-sectional.csv'
    ]
    class_labels = ['PCN', 'CN', 'MCI', 'Dementia']

    # Update this path to the location of your MNI template
    mni_template = '/ceph/intradb/chpc_resources/old_chpc2_resources/fsl-5.0.9-custom-20170410/data/standard/MNI152_T1_1mm.nii.gz'  # Replace with the actual path

    # Output base directory where you want to save the transformed images
    output_base_directory = '/scratch/l.peiwang/derivatives_less'  # Replace with your actual directory

    # Initialize lists to store paths
    mri_paths = []
    pet_paths = []
    class_labels_out = []

    for file, class_label in zip(files, class_labels):
        df = pd.read_csv(file)
        for _, row in df.iterrows():
            # Extract sub-xxxx and ses-xxxx from original paths
            try:
                fdg_filename = row['FdgFilename']
                path_parts = fdg_filename.strip().split('/')
                sub_ses_info = '/'.join(path_parts[8:10])  # Adjust indices if needed
            except Exception as e:
                print(f"Error extracting sub-ses info from {fdg_filename}: {e}")
                continue

            # Construct source directory paths
            source_directory = os.path.join(
                '/scratch/jjlee/Singularity/ADNI/bids/derivatives',
                sub_ses_info,
                'pet'
            )

            # Find MRI files
            t1w_brain_pattern = os.path.join(source_directory, '*T1w_brain.nii.gz')
            t1w_brain_files = glob.glob(t1w_brain_pattern)
            affine_mat_pattern = os.path.join(source_directory, '*_T1w_brain_0GenericAffine.mat')
            affine_mat_files = glob.glob(affine_mat_pattern)

            # Process MRI images
            if t1w_brain_files and affine_mat_files:
                t1w_brain_file = t1w_brain_files[0]
                affine_mat_file = affine_mat_files[0]

                # Define output directory
                destination_directory = os.path.join(output_base_directory, sub_ses_info, 'anat')
                os.makedirs(destination_directory, exist_ok=True)

                # Define MRI output filename
                mri_output_file = os.path.join(
                    destination_directory,
                    os.path.basename(t1w_brain_file).replace('.nii.gz', '_MNI.nii.gz')
                )

                # Apply affine transformation to MRI
                mri_cmd = [
                    'antsApplyTransforms',
                    '-d', '3',
                    '-i', t1w_brain_file,
                    '-r', mni_template,
                    '-o', mri_output_file,
                    '-t', affine_mat_file,
                    '--interpolation', 'Linear',
                    '--verbose'
                ]

                # Execute MRI transformation
                try:
                    subprocess.run(mri_cmd, check=True)
                    mri_paths.append(mri_output_file)
                except subprocess.CalledProcessError as e:
                    print(f"Error processing MRI for {sub_ses_info}: {e}")
                    continue

                # Process PET images
                pet_on_t1w_pattern = os.path.join(source_directory, '*_pet_on_T1w.nii.gz')
                pet_on_t1w_files = glob.glob(pet_on_t1w_pattern)

                if pet_on_t1w_files:
                    pet_on_t1w_file = pet_on_t1w_files[0]

                    # Define PET output filename
                    pet_output_file = os.path.join(
                        destination_directory,
                        os.path.basename(pet_on_t1w_file).replace('.nii.gz', '_MNI.nii.gz')
                    )

                    # Apply affine transformation to PET
                    pet_cmd = [
                        'antsApplyTransforms',
                        '-d', '3',
                        '-i', pet_on_t1w_file,
                        '-r', mni_template,
                        '-o', pet_output_file,
                        '-t', affine_mat_file,
                        '--interpolation', 'Linear',
                        '--verbose'
                    ]

                    # Execute PET transformation
                    try:
                        subprocess.run(pet_cmd, check=True)
                        pet_paths.append(pet_output_file)
                        class_labels_out.append(class_label)
                        print(f"Successfully processed MRI and PET for {sub_ses_info}")
                    except subprocess.CalledProcessError as e:
                        print(f"Error processing PET for {sub_ses_info}: {e}")
                        continue
                else:
                    print(f"PET file not found for {sub_ses_info}")
                    continue
            else:
                print(f"MRI files not found for {sub_ses_info}")
                continue

    return mri_paths, pet_paths, class_labels_out

# Run the function
if __name__ == "__main__":
    mri_paths, pet_paths, class_labels_out = generate_and_process_data()
    print("Processing complete.")
