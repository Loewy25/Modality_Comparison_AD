from nilearn.input_data import NiftiMasker

from data_loading import loading_mask
from utils import normalize_features, threshold_p_values
from plot_utils import plot_glass_brain, plot_stat_map
from main import hyperparameter_tuning_visual_cov_V3



task="pc"
method="MRI"
threshold=0.05

image,label,masker=loading_mask(task,method)

average_single_weights,average_corrected_weights,average_permuted_single_weights,average_permuted_corrected_weights=hyperparameter_tuning_visual_cov_V3(image,label,[22],5,3,1000)
average_single_weights=1-average_single_weights
average_permuted_corrected_weights=1-average_permuted_corrected_weights
small_value = 1e-10
average_permuted_corrected_weights[average_permuted_corrected_weights == 0] = small_value
average_single_weights[average_single_weights == 0] = small_value
average_single_weights_5 = threshold_p_values(average_single_weights,threshold=threshold)
average_corrected_weights_5 = threshold_p_values(average_corrected_weights,threshold=threshold)
average_permuted_single_weights_5 = threshold_p_values(average_permuted_single_weights,threshold=threshold)
average_permuted_corrected_weights_5 = threshold_p_values(average_permuted_corrected_weights,threshold=threshold)
average_single_weights_5 = masker.inverse_transform(average_single_weights_5)
average_corrected_weights_5 = masker.inverse_transform(average_corrected_weights_5)
average_permuted_single_weights_5 = masker.inverse_transform(average_permuted_single_weights_5)
average_permuted_corrected_weights_5 = masker.inverse_transform(average_permuted_corrected_weights_5)
plot_glass_brain(average_single_weights_5, 'average_single_weights_masking', task, method,vmax=threshold)
plot_stat_map(average_single_weights_5, 0, 'average_single_weights_masking', task, method,vmax=threshold)
plot_glass_brain(average_corrected_weights_5, 'average_corrected_weights_masking', task, method,vmax=threshold)
plot_stat_map(average_corrected_weights_5, 0, 'average_corrected_weights_masking', task, method,vmax=threshold)
plot_glass_brain(average_permuted_single_weights_5, 'average_permuted_single_weights_masking', task, method,vmax=threshold)
plot_stat_map(average_permuted_single_weights_5, 0, 'average_permuted_single_weights_masking', task, method,vmax=threshold)
plot_glass_brain(average_permuted_corrected_weights_5, 'average_permuted_corrected_weights_masking', task, method,vmax=threshold)
plot_stat_map(average_permuted_corrected_weights_5, 0, 'average_permuted_corrected_weights_masking', task, method,vmax=threshold)
