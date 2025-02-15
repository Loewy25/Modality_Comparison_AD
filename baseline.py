import time
from data_loading import loading_mask
from main import nested_crossvalidation_multi_kernel, nested_crossvalidation




image_mri,label,masker=loading_mask('cd','MRI')
image_pet,label,masker=loading_mask('cd','PET')

start_time = time.time()  # Capture start time

performance_dict_mri,all_y_test_mri, all_y_prob_mri, all_predictions_mri=nested_crossvalidation(image_pet, label, 'PET_z', 'cd')
end_time = time.time()  # Capture end time

elapsed_time = end_time - start_time  # Calculate elapsed time
print(f"The function took {elapsed_time:.2f} seconds to run.")

start_time = time.time()  # Capture start time

performance_dict_mri,all_y_test_mri, all_y_prob_mri, all_predictions_mri=nested_crossvalidation(image_mri, label, 'MRI_z', 'cd')
end_time = time.time()  # Capture end time

elapsed_time = end_time - start_time  # Calculate elapsed time
print(f"The function took {elapsed_time:.2f} seconds to run.")



image_mri,label,masker=loading_mask('cm','MRI')
image_pet,label,masker=loading_mask('cm','PET')

start_time = time.time()  # Capture start time

performance_dict_mri,all_y_test_mri, all_y_prob_mri, all_predictions_mri=nested_crossvalidation(image_pet, label, 'PET_z', 'cm')
end_time = time.time()  # Capture end time

elapsed_time = end_time - start_time  # Calculate elapsed time
print(f"The function took {elapsed_time:.2f} seconds to run.")

start_time = time.time()  # Capture start time

performance_dict_mri,all_y_test_mri, all_y_prob_mri, all_predictions_mri=nested_crossvalidation(image_mri, label, 'MRI_z', 'cm')
end_time = time.time()  # Capture end time

elapsed_time = end_time - start_time  # Calculate elapsed time
print(f"The function took {elapsed_time:.2f} seconds to run.")


image_mri,label,masker=loading_mask('dm','MRI')
image_pet,label,masker=loading_mask('dm','PET')

start_time = time.time()  # Capture start time

performance_dict_mri,all_y_test_mri, all_y_prob_mri, all_predictions_mri=nested_crossvalidation(image_pet, label, 'PET_z', 'dm')
end_time = time.time()  # Capture end time

elapsed_time = end_time - start_time  # Calculate elapsed time
print(f"The function took {elapsed_time:.2f} seconds to run.")

start_time = time.time()  # Capture start time

performance_dict_mri,all_y_test_mri, all_y_prob_mri, all_predictions_mri=nested_crossvalidation(image_mri, label, 'MRI_z', 'dm')
end_time = time.time()  # Capture end time

elapsed_time = end_time - start_time  # Calculate elapsed time
print(f"The function took {elapsed_time:.2f} seconds to run.")


image_mri,label,masker=loading_mask('pc','MRI')
image_pet,label,masker=loading_mask('pc','PET')

start_time = time.time()  # Capture start time

performance_dict_mri,all_y_test_mri, all_y_prob_mri, all_predictions_mri=nested_crossvalidation(image_pet, label, 'PET_z', 'pc')
end_time = time.time()  # Capture end time

elapsed_time = end_time - start_time  # Calculate elapsed time
print(f"The function took {elapsed_time:.2f} seconds to run.")

start_time = time.time()  # Capture start time

performance_dict_mri,all_y_test_mri, all_y_prob_mri, all_predictions_mri=nested_crossvalidation(image_mri, label, 'MRI_z', 'pc')
end_time = time.time()  # Capture end time

elapsed_time = end_time - start_time  # Calculate elapsed time
print(f"The function took {elapsed_time:.2f} seconds to run.")
#start_time = time.time()  # Capture start time

#performance_dict_mri,all_y_test_mri, all_y_prob_mri, all_predictions_mri=nested_crossvalidation_multi_kernel(image_mri, image_pet, label, 'multi_kernel', 'cd')
#end_time = time.time()  # Capture end time

#elapsed_time = end_time - start_time  # Calculate elapsed time
#print(f"The function took {elapsed_time:.2f} seconds to run.")



#start_time = time.time()  # Capture start time

#performance_dict_pet,all_y_test_pet, all_y_prob_pet, all_predictions_pet=nested_crossvalidation_multi_kernel(image_pet, label, 'PET', 'cd')
#end_time = time.time()  # Capture end time

#elapsed_time = end_time - start_time  # Calculate elapsed time
#print(f"The function took {elapsed_time:.2f} seconds to run.")






