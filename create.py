def create_hdf5(task="cd"):
    """
    Create HDF5 file from MRI and PET data with random tabular data and store in the repository.
    Args:
        task (str): Task identifier (e.g., "CD" for CN vs Dementia).
    """
    # Load MRI, PET, and labels
    mri_data, pet_data, labels = load_mri_pet_data(task)

    # Generate random tabular data (6 features)
    num_samples = len(labels)
    tabular_data = np.random.rand(num_samples, 6)  # Random values for the tabular data
    tabular_data[:, 0] = np.random.randint(50, 90, size=num_samples)  # Random age
    tabular_data[:, 1] = np.random.randint(0, 2, size=num_samples)  # Random gender (0 or 1)

    # Generate random attributes
    patient_ids = np.random.randint(1000, 9999, size=num_samples)  # Random patient IDs
    visit_codes = np.random.choice(["BL", "M12", "M24"], size=num_samples)  # Random visit codes

    # Meta-information for tabular data
    columns = ["age", "gender", "education", "MMSE", "ADAS-Cog-13", "ApoE4"]
    mean_tabular = np.mean(tabular_data, axis=0)
    std_tabular = np.std(tabular_data, axis=0)

    # Define output directory and ensure it exists
    output_dir = "/scratch/l.peiwang"
    os.makedirs(output_dir, exist_ok=True)

    # Define file path
    file_path = os.path.join(output_dir, f"{task}_dataset.h5")

    # Create the HDF5 file
    with h5py.File(file_path, "w") as f:
        for i in range(num_samples):
            image_id = f"ID_{i:03d}"
            group = f.create_group(image_id)

            # Add MRI and PET data
            mri_group = group.create_group("MRI")
            mri_group.create_dataset("T1", data=mri_data[i], compression="gzip")

            pet_group = group.create_group("PET")
            pet_group.create_dataset("FDG", data=pet_data[i], compression="gzip")

            # Add tabular data
            group.create_dataset("tabular", data=tabular_data[i])

            # Add attributes
            group.attrs["DX"] = labels[i]
            group.attrs["RID"] = patient_ids[i]
            group.attrs["VISCODE"] = visit_codes[i]

        # Add stats for the tabular data
        stats_group = f.create_group("stats")
        tabular_stats = stats_group.create_group("tabular")
        tabular_stats.create_dataset("columns", data=np.array(columns, dtype='S'))
        tabular_stats.create_dataset("mean", data=mean_tabular)
        tabular_stats.create_dataset("stddev", data=std_tabular)

    print(f"HDF5 file created and saved to {file_path}")


# Run the function
create_hdf5(task="cd")
