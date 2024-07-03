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
from sklearn.metrics import auc as calculate_auc
from sklearn.preprocessing import MinMaxScaler




from utils import (
    compute_kernel_matrix,
    linear_kernel,
    min_max_normalization,
    compute_p_values,
    compute_weights_for_linear_kernel,
    compute_covariance_directly,
    compute_p_values_with_correction,
    normalize_features,
    apply_normalization,
    compute_bootstrap_confi

)

from plot_utils import (
    plot_roc_curve,
    plot_confusion_matrix
)


def compute_auprc(y_true, y_pred_probs):
    precision, recall, _ = precision_recall_curve(y_true, y_pred_probs)
    sorted_indices = np.argsort(recall)
    sorted_recall = recall[sorted_indices]
    sorted_precision = precision[sorted_indices]
    unique_recall, unique_indices = np.unique(sorted_recall, return_index=True)
    unique_precision = np.array([max(sorted_precision[:i + 1]) for i in unique_indices])
    return calculate_auc(unique_recall, unique_precision)

def normalize_features(X, indices, return_params=False):
    scaler = MinMaxScaler()
    scaler.fit(X[indices])  # Fit only to control group data
    X_scaled = scaler.transform(X)  # Apply to all data
    if return_params:
        return X_scaled, scaler
    return X_scaled

def apply_normalization(X, scaler):
    return scaler.transform(X)

def hyperparameter_tuning_visual_cov_V3(data, label, randomseed, outer, inner, num_permutations):
    train_data = data
    train_label = label
    random_states = randomseed
    
    all_aucs = []
    all_single_weights = []
    all_corrected_weights = []
    all_p_values_single = []
    all_p_values_corrected = []
    
    for rs in random_states:
        print(f"Running outer loop with random state: {rs}")
        cv_outer = StratifiedKFold(n_splits=outer, shuffle=True, random_state=rs)
        
        for train_ix, test_ix in cv_outer.split(train_data, train_label):
            X_train, X_test = train_data[train_ix, :], train_data[test_ix, :]
            y_train, y_test = np.array(train_label)[train_ix], np.array(train_label)[test_ix]
            
            # Normalize the training data based on control indices within this fold
            control_indices_train = [i for i, label in enumerate(y_train) if label == 0]
            X_train, normalization_params = normalize_features(X_train, control_indices_train, return_params=True)
            
            # Normalize the test data using the same normalization parameters
            X_test = apply_normalization(X_test, normalization_params)
            # Compute kernel matrices for both training and test data
            K_train = compute_kernel_matrix(X_train, X_train, linear_kernel)
            K_test = compute_kernel_matrix(X_test, X_train, linear_kernel)

            cv_inner = StratifiedKFold(n_splits=inner, shuffle=True, random_state=1)
            model = SVC(kernel="precomputed", class_weight='balanced', probability=True)
            space = dict()
            space['C'] = [1, 0.1, 0.01, 0.001, 0.0001]
            search = GridSearchCV(model, space, scoring='roc_auc', cv=cv_inner, refit=True)
            result = search.fit(K_train, y_train)
            best_model = result.best_estimator_

            yhat = best_model.predict_proba(K_test)
            yhat = yhat[:, 1]
            auc = roc_auc_score(y_test, yhat)
            all_aucs.append(auc)

            X_support = X_train[best_model.support_, :]
            single_weights = abs(compute_weights_for_linear_kernel(best_model, X_support))
            single_weights = min_max_normalization(single_weights)
            all_single_weights.append(single_weights)
            p_values_single = compute_p_values(X_train, K_train, y_train, best_model, num_permutations)
            all_p_values_single.append(p_values_single)

            corrected_weights = compute_covariance_directly(X_train, y_train)
            all_corrected_weights.append(corrected_weights)
            p_values_corrected = compute_p_values_with_correction(X_train, K_train,y_train, best_model, num_permutations)
            all_p_values_corrected.append(p_values_corrected)

    average_single_weights = np.mean(all_single_weights, axis=0)
    average_corrected_weights = np.mean(all_corrected_weights, axis=0)
    average_p_values_single = np.mean(all_p_values_single, axis=0)
    average_p_values_corrected = np.mean(all_p_values_corrected, axis=0)

    print(f"Average outer loop performance (AUC): {np.mean(all_aucs)}")

    return (
        average_single_weights,
        average_corrected_weights,
        average_p_values_single,
        average_p_values_corrected
    )



def nested_crossvalidation(data, label, method, task):

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
            X_train, scaler = normalize_features(X_train, control_indices_train, return_params=True)
            X_test = apply_normalization(X_test, scaler)
            K_train = compute_kernel_matrix(X_train, X_train, linear_kernel)
            K_test = compute_kernel_matrix(X_test, X_train, linear_kernel)
            cv_inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=1)
            model = SVC(kernel="precomputed", class_weight='balanced', probability=True)
            space = {'C': [1, 100, 10, 0.1, 0.01, 0.001, 0.0001, 0.00001]}
            search = GridSearchCV(model, space, scoring='roc_auc', cv=cv_inner, refit=True)
            result = search.fit(K_train, y_train)
            best_model = result.best_estimator_
            yhat = best_model.predict_proba(K_test)
            yhat = yhat[:, 1]
            
            all_y_test.extend(y_test.tolist())
            all_y_prob.extend(yhat.tolist())
            all_predictions.extend(best_model.predict(K_test).tolist())

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

    auprc = compute_auprc(all_y_test, all_y_prob)
  
    ppv = precision_classwise  # since they are the same, no need to compute twice
    
    cm = confusion_matrix(all_y_test, all_predictions)
    
    # Handle RuntimeWarning by checking for zero denominator
    denominator = cm[1,1] + cm[0,1]
    npv = cm[1,1] / denominator if denominator != 0 else 0
    specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])

    
    # 2. Compute bootstrap confidence intervals for each of these metrics.
    confi_auc = compute_bootstrap_confi(all_y_prob, all_y_test, roc_auc_score)
    confi_accuracy = compute_bootstrap_confi(all_predictions, all_y_test, accuracy_score)
    confi_balanced_accuracy = compute_bootstrap_confi(all_predictions, all_y_test, balanced_accuracy_score)
    confi_precision = [compute_bootstrap_confi(all_predictions, all_y_test, lambda y_true, y_pred: precision_score(y_true, y_pred, average=None)[i]) for i in range(2)]
    confi_recall = [compute_bootstrap_confi(all_predictions, all_y_test, lambda y_true, y_pred: recall_score(y_true, y_pred, average=None)[i]) for i in range(2)]
    confi_f1 = [compute_bootstrap_confi(all_predictions, all_y_test, lambda y_true, y_pred: f1_score(y_true, y_pred, average=None)[i]) for i in range(2)]
    confi_auprc = compute_bootstrap_confi(all_y_prob, all_y_test, lambda y_true, y_pred_probs: compute_auprc(y_true, y_pred_probs))
    confi_specificity = compute_bootstrap_confi(all_predictions, all_y_test, lambda y_true, y_pred: cm[0, 0] / (cm[0, 0] + cm[0, 1]))
    confi_ppv = [compute_bootstrap_confi(all_predictions, all_y_test, lambda y_true, y_pred: precision_score(y_true, y_pred, average=None)[i]) for i in range(2)]
    confi_npv = compute_bootstrap_confi(all_predictions, all_y_test, lambda y_true, y_pred: (confusion_matrix(y_true, y_pred)[1,1] / (confusion_matrix(y_true, y_pred)[1,1] + confusion_matrix(y_true, y_pred)[0,1])))
    
    plot_roc_curve(all_y_test, all_y_prob, method, task)
    plot_confusion_matrix(all_y_test, all_predictions, positive, negative, method, task)
    
    # Directory path
    directory = "./result"
    
    # Construct the task specific directory path
    task_directory = os.path.join(directory, task)
    
    # Ensure the task specific directory exists
    os.makedirs(task_directory, exist_ok=True)
    
    # Construct complete file path for pickle file with 'results' + task + method as the filename
    filename = f"results_{task}_{method}.pickle"
    file_path = os.path.join(task_directory, filename)
    
    # Save the pickle file in the constructed path
    with open(file_path, 'wb') as f:
        pickle.dump((performance_dict, all_y_test, all_y_prob, all_predictions), f)

        
    print(f"AUC: {auc} (95% CI: {confi_auc})")
    print(f"Accuracy: {accuracy} (95% CI: {confi_accuracy})")
    print(f"Balanced accuracy: {balanced_accuracy} (95% CI: {confi_balanced_accuracy})")
    print(f"Precision per class: {precision_classwise[0]} {negative} (95% CI: {confi_precision[0]}), {precision_classwise[1]} {positive} (95% CI: {confi_precision[1]})")
    print(f"Recall per class: {recall_classwise[0]} {negative} (95% CI: {confi_recall[0]}), {recall_classwise[1]} {positive} (95% CI: {confi_recall[1]})")
    print(f"F1-score per class: {f1_classwise[0]} {negative} (95% CI: {confi_f1[0]}), {f1_classwise[1]} {positive} (95% CI: {confi_f1[1]})")
    print(f"AUPRC: {auprc} (95% CI: {confi_auprc})")
    print(f"Specificity: {specificity} (95% CI: {confi_specificity})")
    print(f"PPV per class: {ppv[0]} {negative} (95% CI: {confi_ppv[0]}), {ppv[1]} {positive} (95% CI: {confi_ppv[1]})")
    print(f"NPV: {npv} (95% CI: {confi_npv})")
    
    
    return performance_dict, all_y_test, all_y_prob, all_predictions




def nested_crossvalidation_late_fusion(data_pet, data_mri, label, method, task):
    train_label = label
    random_states = [10]
    
    performance_dict = {}
    tasks_dict = {
        'cd': ('AD', 'CN'),
        'dm': ('AD', 'MCI'),
        'cm': ('MCI', 'CN'),
        'pc': ('Preclinical', 'CN')
    }
    
    all_y_test = []
    all_y_prob = []
    all_predictions = []
    performance_dict = {}
    best_models_pet = []
    best_models_mri = []
    best_weights_list = []
    best_auc_list = []
    
    positive, negative = tasks_dict[task]
    train_data_pet = np.array(data_pet)
    train_data_mri = np.array(data_mri)
    train_label = np.array(train_label)
    
    for rs in random_states:
        cv_outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=rs)        
        for train_ix, test_ix in cv_outer.split(train_data_pet, train_label):
            X_train_pet, X_test_pet = train_data_pet[train_ix, :], train_data_pet[test_ix, :]
            X_train_mri, X_test_mri = train_data_mri[train_ix, :], train_data_mri[test_ix, :]
            y_train, y_test = train_label[train_ix], train_label[test_ix]
        
            # Normalize the PET and MRI training data based on control indices within this fold
            control_indices_train = [i for i, label in enumerate(y_train) if label == 0]
            
            # Calculate normalization parameters from the training data
            X_train_pet, normalization_params_pet = normalize_features(X_train_pet, control_indices_train, return_params=True)
            X_train_mri, normalization_params_mri = normalize_features(X_train_mri, control_indices_train, return_params=True)
            
            # Use those normalization parameters to normalize the test data
            X_test_pet = apply_normalization(X_test_pet, normalization_params_pet)
            X_test_mri = apply_normalization(X_test_mri, normalization_params_mri)
        
            # Compute kernel matrices for PET and MRI data
            K_train_pet = compute_kernel_matrix(X_train_pet, X_train_pet, linear_kernel)
            K_test_pet = compute_kernel_matrix(X_test_pet, X_train_pet, linear_kernel)
            
            K_train_mri = compute_kernel_matrix(X_train_mri, X_train_mri, linear_kernel)
            K_test_mri = compute_kernel_matrix(X_test_mri, X_train_mri, linear_kernel)

            best_auc = 0
            best_weights = (0, 0)
            
            for w1 in np.linspace(0, 1, 51):  # 51 points for weights
                w2 = 1 - w1
                
                cv_inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=1)
                
                model_pet = SVC(kernel="precomputed", class_weight='balanced', probability=True)
                model_mri = SVC(kernel="precomputed", class_weight='balanced', probability=True)
                
                space = {'C': [1, 100, 10, 0.1, 0.01, 0.001, 0.0001, 0.00001]}
                
                search_pet = GridSearchCV(model_pet, space, scoring='roc_auc', cv=cv_inner, refit=True)
                search_mri = GridSearchCV(model_mri, space, scoring='roc_auc', cv=cv_inner, refit=True)
                
                search_pet.fit(K_train_pet, y_train)
                search_mri.fit(K_train_mri, y_train)
                
                pet_best = search_pet.best_estimator_
                mri_best = search_mri.best_estimator_
                
                pet_prob = pet_best.predict_proba(K_test_pet)[:, 1]
                mri_prob = mri_best.predict_proba(K_test_mri)[:, 1]
                
                fused_prob = w1 * pet_prob + w2 * mri_prob
                
                auc = roc_auc_score(y_test, fused_prob)
                
                if auc > best_auc:
                    best_auc = auc
                    best_weights = (w1, w2)
                    best_w1=w1
                    best_w2=w2
            print(f"Best weights for this fold: {best_weights}, AUC: {best_auc}")
            
            search_pet.fit(K_train_pet, y_train)
            search_mri.fit(K_train_mri, y_train)

            pet_best = search_pet.best_estimator_
            mri_best = search_mri.best_estimator_

            pet_prob = pet_best.predict_proba(K_test_pet)[:, 1]
            mri_prob = mri_best.predict_proba(K_test_mri)[:, 1]
            fused_prob = best_w1 * pet_prob + best_w2 * mri_prob
            yhat = fused_prob
            y_test = np.array(train_label)[test_ix]
            predictions = (yhat >= 0.5).astype(int)
            all_y_test.extend(y_test.tolist())
            all_y_prob.extend(yhat.tolist())
            all_predictions.extend(predictions.tolist())
            
            for params, mean_score, std_score in zip(search_pet.cv_results_['params'], 
                                                     search_pet.cv_results_['mean_test_score'], 
                                                     search_pet.cv_results_['std_test_score']):
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
    





def normalize_kernel(K):
    diag_elements = np.diag(K)
    if np.any(diag_elements == 0):
        raise ValueError("Zero diagonal element found in kernel matrix")
    K_normalized = K / np.sqrt(np.outer(diag_elements, diag_elements))
    return K_normalized

def normalize_test_kernel(K_test, K_train_diag, K_test_diag):
    # Check for zeros to prevent division by zero errors
    if np.any(K_train_diag == 0) or np.any(K_test_diag == 0):
        raise ValueError("Zero diagonal element found in kernel matrix")
    
    # Compute the normalization factors: should result in a matrix of shape (len(K_train_diag), len(K_test_diag))
    normalization_matrix = np.sqrt(np.outer(K_train_diag, K_test_diag))
    
    # Correctly reshape the normalization matrix to match K_test dimensions
    # This involves transposing the matrix since np.outer produces (train, test) and we need (test, train)
    normalization_matrix = normalization_matrix.T
    
    # Perform element-wise division
    K_test_normalized = K_test / normalization_matrix
    return K_test_normalized

import numpy as np

# Assuming K_train_pet, K_test_pet, K_train_mri, K_test_mri are your matrices

# Function to check and print NaN indices in a matrix
def print_nan_indices(matrix, matrix_name):
    nan_indices = np.argwhere(np.isnan(matrix))
    if nan_indices.size > 0:
        print(f"NaN values found in {matrix_name} at indices:")
        for index in nan_indices:
            print(f"Row: {index[0]}, Column: {index[1]}")
    else:
        print(f"No NaN values found in {matrix_name}")



def nested_crossvalidation_multi_kernel(data_pet, data_mri, label, method, task):
    train_label = label
    random_states = [10]
    
    performance_dict = {}
    tasks_dict = {
        'cd': ('AD', 'CN'),
        'dm': ('AD', 'MCI'),
        'cm': ('MCI', 'CN'),
        'pc': ('Preclinical', 'CN')
    }
    
    all_y_test = []
    all_y_prob = []
    all_predictions = []
    performance_dict = {}
    best_models_pet = []
    best_models_mri = []
    best_weights_list = []
    best_auc_list = []
    
    positive, negative = tasks_dict[task]
    train_data_pet = np.array(data_pet)
    train_data_mri = np.array(data_mri)
    train_label = np.array(train_label)
    
    for rs in random_states:
        cv_outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=rs)        
        for train_ix, test_ix in cv_outer.split(train_data_pet, train_label):
            X_train_pet, X_test_pet = train_data_pet[train_ix, :], train_data_pet[test_ix, :]
            X_train_mri, X_test_mri = train_data_mri[train_ix, :], train_data_mri[test_ix, :]
            y_train, y_test = train_label[train_ix], train_label[test_ix]
            print("like it!!!!!!!!!!!!!!!!!!!!!!!")
            print(X_train_pet)
            print(X_train_mri)
            print(X_test_pet)
            print(X_test_mri)
            print(y_train)
            print(y_test)
            # Normalize the PET and MRI training data based on control indices within this fold
            control_indices_train = [i for i, label in enumerate(y_train) if label == 0]
            
            X_train_pet, scaler_pet = normalize_features(X_train_pet, control_indices_train, return_params=True)
            X_train_mri, scaler_mri = normalize_features(X_train_mri, control_indices_train, return_params=True)
            
            X_test_pet = apply_normalization(X_test_pet, scaler_pet)
            X_test_mri = apply_normalization(X_test_mri, scaler_mri)
            print("first normalization")
            print(X_train_pet)
            print(X_train_mri)
            print(X_test_pet)
            print(X_test_mri)
            print_nan_indices(X_train_pet, "PET TRAIN kernel matrices")
            print_nan_indices(X_test_pet, "PET TEST kernel matrices")
            print_nan_indices(X_train_mri, "MRI TRAIN kernel matrices")
            print_nan_indices(X_test_mri, "MRI TEST kernel matrices")

            # Compute kernel matrices for PET and MRI data
            K_train_pet = compute_kernel_matrix(X_train_pet, X_train_pet, linear_kernel)
            K_test_pet = compute_kernel_matrix(X_test_pet, X_train_pet, linear_kernel)
            
            K_train_mri = compute_kernel_matrix(X_train_mri, X_train_mri, linear_kernel)
            K_test_mri = compute_kernel_matrix(X_test_mri, X_train_mri, linear_kernel)
            print("kernelxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
            print(K_train_pet)
            print(K_train_mri)
            print(K_test_pet)
            print(K_test_mri)

            # Extract diagonals before normalizing
            K_train_mri_diag = np.diag(K_train_mri)
            K_train_pet_diag = np.diag(K_train_pet)
            K_test_mri_diag = np.diag(K_test_mri)
            K_test_pet_diag = np.diag(K_test_pet)
            print("diagxxhfujewifjiwejfpiwejffjiowejfipwejfipwejfipwejf")
            print(K_train_mri_diag)
            print(K_train_pet_diag)
            print(K_test_mri_diag)
            print(K_test_pet_diag)


            print_nan_indices(K_train_pet, "PET TRAIN kernel matrices")
            print_nan_indices(K_test_pet, "PET TEST kernel matrices")
            print_nan_indices(K_train_mri, "MRI TRAIN kernel matrices")
            print_nan_indices(K_test_mri, "MRI TEST kernel matrices")

            # Normalize training kernels
            K_train_mri = normalize_kernel(K_train_mri)
            K_train_pet = normalize_kernel(K_train_pet)
            
            
            # Normalize test kernels using the original (unnormalized) training kernel diagonals
            K_test_mri = normalize_test_kernel(K_test_mri, K_train_mri_diag, K_test_mri_diag)
            K_test_pet = normalize_test_kernel(K_test_pet, K_train_pet_diag, K_test_pet_diag)
            print("second normalization")
            print(K_train_pet)
            print(K_train_mri)
            print(K_test_pet)
            print(K_test_mri)          

            # Check for NaN values after normalization
            # Check and print NaN indices for each matrix
            print_nan_indices(K_train_pet, "PET TRAIN kernel matrices")
            print_nan_indices(K_test_pet, "PET TEST kernel matrices")
            print_nan_indices(K_train_mri, "MRI TRAIN kernel matrices")
            print_nan_indices(K_test_mri, "MRI TEST kernel matrices")


            best_auc = 0
            best_weights = (0, 0)
            
            for w1 in np.linspace(0, 1, 51):  # 51 points for weights
                w2 = 1 - w1
                
                # Combine kernels using weighted sum
                K_train_combined = w1 * K_train_pet + w2 * K_train_mri
                K_test_combined = w1 * K_test_pet + w2 * K_test_mri
                
                cv_inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=1)
                
                model = SVC(kernel="precomputed", class_weight='balanced', probability=True)
                
                space = {'C': [1, 100, 10, 0.1, 0.01, 0.001, 0.0001, 0.00001]}
                
                search = GridSearchCV(model, space, scoring='roc_auc', cv=cv_inner, refit=True)
                
                search.fit(K_train_combined, y_train)
                
                best_model = search.best_estimator_
                
                # Predict probabilities on the test set
                test_prob = best_model.predict_proba(K_test_combined)[:, 1]
                
                # Calculate AUC
                auc = roc_auc_score(y_test, test_prob)
                
                # Track the best AUC and corresponding weights
                if auc > best_auc:
                    best_auc = auc
                    best_weights = (w1, w2)
                    best_w1 = w1
                    best_w2 = w2
            
            print(f"Best weights for this fold: {best_weights}, AUC: {best_auc}")
            
            # Re-train the best model with the best weights
            K_train_combined = best_w1 * K_train_pet + best_w2 * K_train_mri
            K_test_combined = best_w1 * K_test_pet + best_w2 * K_test_mri

            search.fit(K_train_combined, y_train)
            best_model = search.best_estimator_

            # Predict probabilities on the test set
            yhat = best_model.predict_proba(K_test_combined)[:, 1]
            predictions = (yhat >= 0.5).astype(int)

            all_y_test.extend(y_test.tolist())
            all_y_prob.extend(yhat.tolist())
            all_predictions.extend(predictions.tolist())
            
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

    
    # Compute bootstrap confidence intervals for each of these metrics
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

# The normalize_features, apply_normalization, compute_kernel_matrix, linear_kernel, compute_bootstrap_confi, plot_roc_curve, and plot_confusion_matrix functions are assumed to be defined elsewhere.
