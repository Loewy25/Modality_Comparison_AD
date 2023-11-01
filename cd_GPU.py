from cuml import SVC as cuSVC
from cuml.model_selection import GridSearchCV as cuGridSearchCV
import time
from data_loading import loading_mask
from main import nested_crossvalidation
import dicom2nifti
import glob
import math
import nibabel as nib
import nilearn as nil
import numpy as np
import os
import pandas as pd
import pickle
import random
import scipy.ndimage as ndi
import statsmodels.stats.contingency_tables as ct
import time
from collections import Counter
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from nilearn import image, plotting
from nilearn.input_data import NiftiMasker
from nilearn.masking import (apply_mask, compute_brain_mask,
                             compute_multi_brain_mask, intersect_masks, unmask)
from nilearn.plotting import plot_roi, plot_stat_map, show
from numpy import mean, std
from numpy.linalg import inv
from scipy.stats import chi2_contingency, norm
from sklearn import metrics, svm
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_curve, precision_recall_fscore_support,
                             roc_auc_score, roc_curve,balanced_accuracy_score,precision_score,recall_score,f1_score)
from sklearn.model_selection import (GridSearchCV, KFold, StratifiedKFold,
                                     cross_val_predict, cross_val_score,
                                     train_test_split)
from sklearn.preprocessing import Binarizer, label_binarize
from sklearn.svm import LinearSVC, SVC




def nested_crossvalidation1(data, label, method, task):

    train_data = data
    train_label = label
    random_states = [10]
    
    all_y_test = []
    all_y_prob = []
    all_predictions = []
    performance_dict = {}

    tasks_dict = {
        'cd': ('AD', 'CN'),
        'dm': ('AD', 'MCI'),
        'cm': ('MCI', 'CN'),
        'pc': ('Preclinical', 'CN')
    }
    
    positive, negative = tasks_dict[task]
    for rs in random_states:
        cv_outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=rs)
        
        for train_ix, test_ix in cv_outer.split(train_data, train_label):
            
            X_train, X_test = train_data[train_ix, :], train_data[test_ix, :]
            y_train, y_test = np.array(train_label)[train_ix], np.array(train_label)[test_ix]
            
            # Normalize the training data based on control indices within this fold
            control_indices_train = [i for i, label in enumerate(y_train) if label == 0]
            X_train, normalization_params = normalize_features(X_train, control_indices_train, return_params=True)
            
            # Normalize the test data using the same normalization parameters
            X_test = apply_normalization(X_test, normalization_params)
            
            # Convert data to GPU
            X_train_gpu = cudf.DataFrame.from_pandas(X_train)
            X_test_gpu = cudf.DataFrame.from_pandas(X_test)
            y_train_gpu = cudf.Series(y_train)
            y_test_gpu = cudf.Series(y_test)
            cv_inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=1)
            
            # Use cuSVC and cuGridSearchCV
            model = cuSVC(kernel="linear", class_weight='balanced', probability=True)
            space = {'C': [1, 100, 10, 0.1, 0.01, 0.001, 0.0001, 0.00001]}
            search = cuGridSearchCV(model, space, scoring='roc_auc', cv=cv_inner, refit=True)
            result = search.fit(X_train_gpu, y_train_gpu)
            best_model = result.best_estimator_
            yhat = best_model.predict_proba(X_test_gpu)
            yhat = yhat[:, 1]
            all_y_test.extend(y_test.tolist())
            all_y_prob.extend(yhat.tolist())
            all_predictions.extend(best_model.predict(X_test_gpu).tolist())

            for params, mean_score, std_score in zip(search.cv_results_['params'], 
                                                      search.cv_results_['mean_test_score'], 
                                                      search.cv_results_['std_test_score']):
                C_value = params['C']
                if C_value not in performance_dict:
                    performance_dict[C_value] = {'mean_scores': [], 'std_scores': []}
                performance_dict[C_value]['mean_scores'].append(mean_score)
                performance_dict[C_value]['std_scores'].append(std_score)

    auc = roc_auc_score(all_y_test, all_y_prob)
    accuracy = accuracy_score(all_y_test, all_predictions)
    balanced_accuracy = balanced_accuracy_score(all_y_test, all_predictions)
    
    # Handle UndefinedMetricWarning by setting zero_division=1
    precision_classwise = precision_score(all_y_test, all_predictions, average=None, zero_division=1)
    recall_classwise = recall_score(all_y_test, all_predictions, average=None)
    f1_classwise = f1_score(all_y_test, all_predictions, average=None)
    
    ppv = precision_classwise  # since they are the same, no need to compute twice
    
    cm = confusion_matrix(all_y_test, all_predictions)
    
    # Handle RuntimeWarning by checking for zero denominator
    denominator = cm[1,1] + cm[0,1]
    npv = cm[1,1] / denominator if denominator != 0 else 0

    
    # 2. Compute bootstrap confidence intervals for each of these metrics.
    confi_auc = compute_bootstrap_confi(all_y_prob, all_y_test, roc_auc_score)
    confi_accuracy = compute_bootstrap_confi(all_predictions, all_y_test, accuracy_score)
    confi_balanced_accuracy = compute_bootstrap_confi(all_predictions, all_y_test, balanced_accuracy_score)
    confi_precision = [compute_bootstrap_confi(all_predictions, all_y_test, lambda y_true, y_pred: precision_score(y_true, y_pred, average=None)[i]) for i in range(2)]
    confi_recall = [compute_bootstrap_confi(all_predictions, all_y_test, lambda y_true, y_pred: recall_score(y_true, y_pred, average=None)[i]) for i in range(2)]
    confi_f1 = [compute_bootstrap_confi(all_predictions, all_y_test, lambda y_true, y_pred: f1_score(y_true, y_pred, average=None)[i]) for i in range(2)]
    confi_ppv = [compute_bootstrap_confi(all_predictions, all_y_test, lambda y_true, y_pred: precision_score(y_true, y_pred, average=None)[i]) for i in range(2)]
    confi_npv = compute_bootstrap_confi(all_predictions, all_y_test, lambda y_true, y_pred: (confusion_matrix(y_true, y_pred)[1,1] / (confusion_matrix(y_true, y_pred)[1,1] + confusion_matrix(y_true, y_pred)[0,1])))
    
    plot_roc_curve(all_y_test, all_y_prob, method, task)
    plot_confusion_matrix(all_y_test, all_predictions, positive, negative, method, task)
    
    directory = './result'
    os.makedirs(directory, exist_ok=True)
    filename = os.path.join(directory, f'results_{method}_{task}.pickle')
    
    with open(filename, 'wb') as f:
        pickle.dump((performance_dict, all_y_test, all_y_prob, all_predictions), f)
        
    print(f"AUC: {auc} (95% CI: {confi_auc})")
    print(f"Accuracy: {accuracy} (95% CI: {confi_accuracy})")
    print(f"Balanced accuracy: {balanced_accuracy} (95% CI: {confi_balanced_accuracy})")
    print(f"Precision per class: {precision_classwise[0]} {negative} (95% CI: {confi_precision[0]}), {precision_classwise[1]} {positive} (95% CI: {confi_precision[1]})")
    print(f"Recall per class: {recall_classwise[0]} {negative} (95% CI: {confi_recall[0]}), {recall_classwise[1]} {positive} (95% CI: {confi_recall[1]})")
    print(f"F1-score per class: {f1_classwise[0]} {negative} (95% CI: {confi_f1[0]}), {f1_classwise[1]} {positive} (95% CI: {confi_f1[1]})")
    print(f"PPV per class: {ppv[0]} {negative} (95% CI: {confi_ppv[0]}), {ppv[1]} {positive} (95% CI: {confi_ppv[1]})")
    print(f"NPV: {npv} (95% CI: {confi_npv})")
    




    return performance_dict, all_y_test, all_y_prob, all_predictions




image_mri,label,masker=loading_mask('cd','MRI')
#image_pet,label,masker=loading_mask('cd','PET')

start_time = time.time()  # Capture start time

performance_dict_mri,all_y_test_mri, all_y_prob_mri, all_predictions_mri=nested_crossvalidation1(image_mri, label, 'MRI', 'cd')
end_time = time.time()  # Capture end time

elapsed_time = end_time - start_time  # Calculate elapsed time
print(f"The function took {elapsed_time:.2f} seconds to run.")
