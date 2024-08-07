import time
from data_loading import loading_mask
from main import nested_crossvalidation_multi_kernel, nested_crossvalidation




image_mri,label,masker=loading_mask('cm','MRI')
image_pet,label,masker=loading_mask('cm','PET')

start_time = time.time()  # Capture start time

performance_dict_mri,all_y_test_mri, all_y_prob_mri, all_predictions_mri=nested_crossvalidation(image_pet, label, 'PET_old', 'cm')
end_time = time.time()  # Capture end time

elapsed_time = end_time - start_time  # Calculate elapsed time
print(f"The function took {elapsed_time:.2f} seconds to run.")
