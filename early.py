import time
from data_loading import loading_mask
from main import nested_crossvalidation
import numpy as np
from sklearn.random_projection import SparseRandomProjection


image_mri,label,masker=loading_mask('cd','MRI')
image_pet,label,masker=loading_mask('cd','PET')


# Assuming MRI_data and PET_data are your MRI and PET features respectively
combined_data = np.hstack((image_pet, image_mri))
srp = SparseRandomProjection(n_components=122597)  # Choose an appropriate number of components
reduced_data = srp.fit_transform(combined_data)


start_time = time.time()  # Capture start time

# Run your function (replace with actual arguments)
performance_dict_mri,all_y_test_mri, all_y_prob_mri, all_predictions_mri=nested_crossvalidation(reduced_data, label, 'early-min', 'cd')
end_time = time.time()  # Capture end time

elapsed_time = end_time - start_time  # Calculate elapsed time
print(f"The function took {elapsed_time:.2f} seconds to run.")

image_mri,label,masker=loading_mask('cm','MRI')
image_pet,label,masker=loading_mask('cm','PET')


# Assuming MRI_data and PET_data are your MRI and PET features respectively
combined_data = np.hstack((image_pet, image_mri))
srp = SparseRandomProjection(n_components=122597)  # Choose an appropriate number of components
reduced_data = srp.fit_transform(combined_data)


start_time = time.time()  # Capture start time

# Run your function (replace with actual arguments)
performance_dict_mri,all_y_test_mri, all_y_prob_mri, all_predictions_mri=nested_crossvalidation(reduced_data, label, 'early-min', 'cm')
end_time = time.time()  # Capture end time

elapsed_time = end_time - start_time  # Calculate elapsed time
print(f"The function took {elapsed_time:.2f} seconds to run.")

image_mri,label,masker=loading_mask('dm','MRI')
image_pet,label,masker=loading_mask('dm','PET')


# Assuming MRI_data and PET_data are your MRI and PET features respectively
combined_data = np.hstack((image_pet, image_mri))
srp = SparseRandomProjection(n_components=122597)  # Choose an appropriate number of components
reduced_data = srp.fit_transform(combined_data)


start_time = time.time()  # Capture start time

# Run your function (replace with actual arguments)
performance_dict_mri,all_y_test_mri, all_y_prob_mri, all_predictions_mri=nested_crossvalidation(reduced_data, label, 'early-min', 'dm')
end_time = time.time()  # Capture end time

elapsed_time = end_time - start_time  # Calculate elapsed time
print(f"The function took {elapsed_time:.2f} seconds to run.")

image_mri,label,masker=loading_mask('pc','MRI')
image_pet,label,masker=loading_mask('pc','PET')


# Assuming MRI_data and PET_data are your MRI and PET features respectively
combined_data = np.hstack((image_pet, image_mri))
srp = SparseRandomProjection(n_components=122597)  # Choose an appropriate number of components
reduced_data = srp.fit_transform(combined_data)


start_time = time.time()  # Capture start time

# Run your function (replace with actual arguments)
performance_dict_mri,all_y_test_mri, all_y_prob_mri, all_predictions_mri=nested_crossvalidation(reduced_data, label, 'early-min', 'pc')
end_time = time.time()  # Capture end time

elapsed_time = end_time - start_time  # Calculate elapsed time
print(f"The function took {elapsed_time:.2f} seconds to run.")
