
import time
from data_loading import loading_mask
from main import nested_crossvalidation_multi_kernel, nested_crossvalidation, nested_crossvalidation_late_fusion




image_mri,label,masker=loading_mask('cd','MRI')
image_pet,label,masker=loading_mask('cd','PET')


start_time = time.time()  # Capture start time

performance_dict_mri,all_y_test_mri, all_y_prob_mri, all_predictions_mri=nested_crossvalidation_late_fusion(image_mri, image_pet, label, 'Late_z', 'cd')
end_time = time.time()  # Capture end time

elapsed_time = end_time - start_time  # Calculate elapsed time
print(f"The function took {elapsed_time:.2f} seconds to run.")

image_mri,label,masker=loading_mask('cm','MRI')
image_pet,label,masker=loading_mask('cm','PET')


start_time = time.time()  # Capture start time

performance_dict_mri,all_y_test_mri, all_y_prob_mri, all_predictions_mri=nested_crossvalidation_late_fusion(image_mri, image_pet, label, 'Late_z', 'cm')
end_time = time.time()  # Capture end time

elapsed_time = end_time - start_time  # Calculate elapsed time
print(f"The function took {elapsed_time:.2f} seconds to run.")

image_mri,label,masker=loading_mask('dm','MRI')
image_pet,label,masker=loading_mask('dm','PET')


start_time = time.time()  # Capture start time

performance_dict_mri,all_y_test_mri, all_y_prob_mri, all_predictions_mri=nested_crossvalidation_late_fusion(image_mri, image_pet, label, 'Late_z', 'dm')
end_time = time.time()  # Capture end time

elapsed_time = end_time - start_time  # Calculate elapsed time
print(f"The function took {elapsed_time:.2f} seconds to run.")

image_mri,label,masker=loading_mask('pc','MRI')
image_pet,label,masker=loading_mask('pc','PET')


start_time = time.time()  # Capture start time

performance_dict_mri,all_y_test_mri, all_y_prob_mri, all_predictions_mri=nested_crossvalidation_late_fusion(image_mri, image_pet, label, 'Late_z', 'pc')
end_time = time.time()  # Capture end time

elapsed_time = end_time - start_time  # Calculate elapsed time
print(f"The function took {elapsed_time:.2f} seconds to run.")







