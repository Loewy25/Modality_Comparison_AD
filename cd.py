import time
from data_loading import loading_mask
from main import nested_crossvalidation



image_mri,label,masker=loading_mask('cd','MRI')
#image_pet,label,masker=loading_mask('cd','PET')

start_time = time.time()  # Capture start time

performance_dict_mri,all_y_test_mri, all_y_prob_mri, all_predictions_mri=nested_crossvalidation(image_mri, label, 'MRI', 'cd')
end_time = time.time()  # Capture end time

elapsed_time = end_time - start_time  # Calculate elapsed time
print(f"The function took {elapsed_time:.2f} seconds to run.")

image_mri,label,masker=loading_mask('cd','PET')
#image_pet,label,masker=loading_mask('cd','PET')

start_time = time.time()  # Capture start time

performance_dict_mri,all_y_test_mri, all_y_prob_mri, all_predictions_mri=nested_crossvalidation(image_mri, label, 'PET', 'cd')
end_time = time.time()  # Capture end time

elapsed_time = end_time - start_time  # Calculate elapsed time
print(f"The function took {elapsed_time:.2f} seconds to run.")





#start_time = time.time()  # Capture start time

#performance_dict_mri,all_y_test_mri, all_y_prob_mri, all_predictions_mri=nested_crossvalidation(image_pet, label, 'PET', 'cd')
#end_time = time.time()  # Capture end time

#elapsed_time = end_time - start_time  # Calculate elapsed time
#print(f"The function took {elapsed_time:.2f} seconds to run.")
