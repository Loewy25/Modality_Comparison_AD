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
    normalize_features_z,
    apply_normalization_z,
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
    print("min-max")
    scaler = MinMaxScaler()
    scaler.fit(X[indices])  # Fit only to control group data
    X_scaled = scaler.transform(X)  # Apply to all data
    if return_params:
        return X_scaled, scaler
    return X_scaled

def apply_normalization(X, scaler):
    print("min-max")
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
            X_train, scaler = normalize_features_z(X_train, control_indices_train, return_params=True)
            X_test = apply_normalization_z(X_test, scaler)
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
    f1 = f1_score(all_y_test, all_predictions, average='binary')  
    cm = confusion_matrix(all_y_test, all_predictions) 
    specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    sensitivity = recall_score(all_y_test, all_predictions, average='binary')
    npv = cm[0, 0] / (cm[0, 0] + cm[1, 0]) if (cm[0, 0] + cm[1, 0]) != 0 else 0
    ppv = precision_score(all_y_test, all_predictions, average='binary', zero_division=1)
    auprc = compute_auprc(all_y_test, all_y_prob)
    accuracy = accuracy_score(all_y_test, all_predictions)
    balanced_accuracy = balanced_accuracy_score(all_y_test, all_predictions)

    confi_auc = compute_bootstrap_confi(all_y_prob, all_y_test, roc_auc_score)
    confi_f1 = compute_bootstrap_confi(all_predictions, all_y_test, f1_score)
    confi_specificity = compute_bootstrap_confi(all_predictions, all_y_test, lambda y_true, y_pred: confusion_matrix(y_true, y_pred)[0, 0] / (confusion_matrix(y_true, y_pred)[0, 0] + confusion_matrix(y_true, y_pred)[0, 1]))
    confi_sensitivity = compute_bootstrap_confi(all_predictions, all_y_test, recall_score)
    confi_npv = compute_bootstrap_confi(all_predictions, all_y_test, lambda y_true, y_pred: confusion_matrix(y_true, y_pred)[0, 0] / (confusion_matrix(y_true, y_pred)[0, 0] + confusion_matrix(y_true, y_pred)[1, 0]))
    confi_ppv = compute_bootstrap_confi(all_predictions, all_y_test, precision_score)
    confi_auprc = compute_bootstrap_confi(all_y_prob, all_y_test, compute_auprc)
    confi_accuracy = compute_bootstrap_confi(all_predictions, all_y_test, accuracy_score)
    confi_balanced_accuracy = compute_bootstrap_confi(all_predictions, all_y_test, balanced_accuracy_score)

    plot_roc_curve(all_y_test, all_y_prob, method, task)
    plot_confusion_matrix(all_y_test, all_predictions, positive, negative, method, task)
    
    directory = "./result"
    task_directory = os.path.join(directory, task)
    os.makedirs(task_directory, exist_ok=True)
    
    filename = f"results_{task}_{method}.pickle"
    file_path = os.path.join(task_directory, filename)
    
    with open(file_path, 'wb') as f:
        pickle.dump((performance_dict, all_y_test, all_y_prob, all_predictions), f)

    print(f"AUC: {auc} (95% CI: {confi_auc})")
    print(f"F1-score: {f1} (95% CI: {confi_f1})")
    print(f"Specificity: {specificity} (95% CI: {confi_specificity})")
    print(f"Sensitivity: {sensitivity} (95% CI: {confi_sensitivity})")
    print(f"NPV: {npv} (95% CI: {confi_npv})")
    print(f"PPV (Precision): {ppv} (95% CI: {confi_ppv})")
    print(f"AUPRC: {auprc} (95% CI: {confi_auprc})")
    print(f"Accuracy: {accuracy} (95% CI: {confi_accuracy})")
    print(f"Balanced accuracy: {balanced_accuracy} (95% CI: {confi_balanced_accuracy})")
    
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
            X_train_pet, normalization_params_pet = normalize_features_z(X_train_pet, control_indices_train, return_params=True)
            X_train_mri, normalization_params_mri = normalize_features_z(X_train_mri, control_indices_train, return_params=True)
            
            # Use those normalization parameters to normalize the test data
            X_test_pet = apply_normalization_z(X_test_pet, normalization_params_pet)
            X_test_mri = apply_normalization_z(X_test_mri, normalization_params_mri)
        
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
    f1 = f1_score(all_y_test, all_predictions, average='binary')  
    cm = confusion_matrix(all_y_test, all_predictions) 
    specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    sensitivity = recall_score(all_y_test, all_predictions, average='binary')
    npv = cm[0, 0] / (cm[0, 0] + cm[1, 0]) if (cm[0, 0] + cm[1, 0]) != 0 else 0
    ppv = precision_score(all_y_test, all_predictions, average='binary', zero_division=1)
    auprc = compute_auprc(all_y_test, all_y_prob)
    accuracy = accuracy_score(all_y_test, all_predictions)
    balanced_accuracy = balanced_accuracy_score(all_y_test, all_predictions)

    confi_auc = compute_bootstrap_confi(all_y_prob, all_y_test, roc_auc_score)
    confi_f1 = compute_bootstrap_confi(all_predictions, all_y_test, f1_score)
    confi_specificity = compute_bootstrap_confi(all_predictions, all_y_test, lambda y_true, y_pred: confusion_matrix(y_true, y_pred)[0, 0] / (confusion_matrix(y_true, y_pred)[0, 0] + confusion_matrix(y_true, y_pred)[0, 1]))
    confi_sensitivity = compute_bootstrap_confi(all_predictions, all_y_test, recall_score)
    confi_npv = compute_bootstrap_confi(all_predictions, all_y_test, lambda y_true, y_pred: confusion_matrix(y_true, y_pred)[0, 0] / (confusion_matrix(y_true, y_pred)[0, 0] + confusion_matrix(y_true, y_pred)[1, 0]))
    confi_ppv = compute_bootstrap_confi(all_predictions, all_y_test, precision_score)
    confi_auprc = compute_bootstrap_confi(all_y_prob, all_y_test, compute_auprc)
    confi_accuracy = compute_bootstrap_confi(all_predictions, all_y_test, accuracy_score)
    confi_balanced_accuracy = compute_bootstrap_confi(all_predictions, all_y_test, balanced_accuracy_score)

    plot_roc_curve(all_y_test, all_y_prob, method, task)
    plot_confusion_matrix(all_y_test, all_predictions, positive, negative, method, task)
    
    directory = "./result"
    task_directory = os.path.join(directory, task)
    os.makedirs(task_directory, exist_ok=True)
    
    filename = f"results_{task}_{method}.pickle"
    file_path = os.path.join(task_directory, filename)
    
    with open(file_path, 'wb') as f:
        pickle.dump((performance_dict, all_y_test, all_y_prob, all_predictions), f)

    print(f"AUC: {auc} (95% CI: {confi_auc})")
    print(f"F1-score: {f1} (95% CI: {confi_f1})")
    print(f"Specificity: {specificity} (95% CI: {confi_specificity})")
    print(f"Sensitivity: {sensitivity} (95% CI: {confi_sensitivity})")
    print(f"NPV: {npv} (95% CI: {confi_npv})")
    print(f"PPV (Precision): {ppv} (95% CI: {confi_ppv})")
    print(f"AUPRC: {auprc} (95% CI: {confi_auprc})")
    print(f"Accuracy: {accuracy} (95% CI: {confi_accuracy})")
    print(f"Balanced accuracy: {balanced_accuracy} (95% CI: {confi_balanced_accuracy})")
    
    return performance_dict, all_y_test, all_y_prob, all_predictions




def normalize_kernel(K):
    diag_elements = np.diag(K)
    if np.any(diag_elements == 0):
        raise ValueError("Zero diagonal element found in kernel matrix")
    K_normalized = K / np.sqrt(np.outer(diag_elements, diag_elements))
    return K_normalized



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

def remove_nan_subjects(K_train_pet, K_train_mri, K_test_pet, K_test_mri, y_train, y_test):
    # Find indices of subjects with any NaN values
    nan_indices_train_pet = np.any(np.isnan(K_train_pet), axis=1)
    nan_indices_train_mri = np.any(np.isnan(K_train_mri), axis=1)
    nan_indices_test_pet = np.any(np.isnan(K_test_pet), axis=1)
    nan_indices_test_mri = np.any(np.isnan(K_test_mri), axis=1)
    
    # Combine indices to find subjects with NaNs in any matrix
    nan_indices_train = nan_indices_train_pet | nan_indices_train_mri
    nan_indices_test = nan_indices_test_pet | nan_indices_test_mri
    
    # Count number of subjects with NaNs
    num_subjects_with_nans_train = np.sum(nan_indices_train)
    num_subjects_with_nans_test = np.sum(nan_indices_test)
    
    # Remove subjects with NaNs from training data and labels
    K_train_pet_clean = K_train_pet[~nan_indices_train]
    K_train_mri_clean = K_train_mri[~nan_indices_train]
    y_train_clean = y_train[~nan_indices_train]
    
    # Remove subjects with NaNs from test data and labels
    K_test_pet_clean = K_test_pet[~nan_indices_test]
    K_test_mri_clean = K_test_mri[~nan_indices_test]
    y_test_clean = y_test[~nan_indices_test]
    
    print(f"Number of subjects with NaN values in training data: {num_subjects_with_nans_train}")
    print(f"Number of subjects with NaN values in test data: {num_subjects_with_nans_test}")
    
    return K_train_pet_clean, K_train_mri_clean, K_test_pet_clean, K_test_mri_clean, y_train_clean, y_test_clean


def normalize_kernel(K):
    diag_elements = np.diag(K)
    if np.any(diag_elements == 0):
        raise ValueError("Zero diagonal element found in kernel matrix")
    K_normalized = K / np.sqrt(np.outer(diag_elements, diag_elements))
    return K_normalized

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
            
            # Normalize the PET and MRI training data based on control indices within this fold
            control_indices_train = [i for i, label in enumerate(y_train) if label == 0]
            
            X_train_pet, scaler_pet = normalize_features_z(X_train_pet, control_indices_train, return_params=True)
            X_train_mri, scaler_mri = normalize_features_z(X_train_mri, control_indices_train, return_params=True)
            
            X_test_pet = apply_normalization_z(X_test_pet, scaler_pet)
            X_test_mri = apply_normalization_z(X_test_mri, scaler_mri)
            
            # Combine training and test data
            X_combined_pet = np.vstack((X_train_pet, X_test_pet))
            X_combined_mri = np.vstack((X_train_mri, X_test_mri))
            
            # Compute the combined kernel matrices
            K_combined_pet = np.dot(X_combined_pet, X_combined_pet.T)
            K_combined_mri = np.dot(X_combined_mri, X_combined_mri.T)
            
            # Normalize the combined kernel matrices
            K_combined_pet_normalized = normalize_kernel(K_combined_pet)
            K_combined_mri_normalized = normalize_kernel(K_combined_mri)
            
            # Extract the training and test kernel matrices
            K_train_pet = K_combined_pet_normalized[:len(train_ix), :len(train_ix)]
            K_test_pet = K_combined_pet_normalized[len(train_ix):, :len(train_ix)]
            # K_train_pet = K_combined_pet[:len(train_ix), :len(train_ix)]
            # K_test_pet = K_combined_pet[len(train_ix):, :len(train_ix)]
            
            K_train_mri = K_combined_mri_normalized[:len(train_ix), :len(train_ix)]
            K_test_mri = K_combined_mri_normalized[len(train_ix):, :len(train_ix)]
            
            cv_inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=1)
            model = SVC(kernel="precomputed", class_weight='balanced', probability=True)
            space = {'C': [1, 100, 10, 0.1, 0.01, 0.001, 0.0001, 0.00001]}
            
            best_auc = 0
            best_weights = (0, 0)
            
            for w1 in np.linspace(0, 1, 51):  # 51 points for weights
                w2 = 1 - w1
                
                # Combine kernels using weighted sum
                K_train_combined = w1 * K_train_pet + w2 * K_train_mri
                K_test_combined = w1 * K_test_pet + w2 * K_test_mri
                
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
    f1 = f1_score(all_y_test, all_predictions, average='binary')  
    cm = confusion_matrix(all_y_test, all_predictions) 
    specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    sensitivity = recall_score(all_y_test, all_predictions, average='binary')
    npv = cm[0, 0] / (cm[0, 0] + cm[1, 0]) if (cm[0, 0] + cm[1, 0]) != 0 else 0
    ppv = precision_score(all_y_test, all_predictions, average='binary', zero_division=1)
    auprc = compute_auprc(all_y_test, all_y_prob)
    accuracy = accuracy_score(all_y_test, all_predictions)
    balanced_accuracy = balanced_accuracy_score(all_y_test, all_predictions)

    confi_auc = compute_bootstrap_confi(all_y_prob, all_y_test, roc_auc_score)
    confi_f1 = compute_bootstrap_confi(all_predictions, all_y_test, f1_score)
    confi_specificity = compute_bootstrap_confi(all_predictions, all_y_test, lambda y_true, y_pred: confusion_matrix(y_true, y_pred)[0, 0] / (confusion_matrix(y_true, y_pred)[0, 0] + confusion_matrix(y_true, y_pred)[0, 1]))
    confi_sensitivity = compute_bootstrap_confi(all_predictions, all_y_test, recall_score)
    confi_npv = compute_bootstrap_confi(all_predictions, all_y_test, lambda y_true, y_pred: confusion_matrix(y_true, y_pred)[0, 0] / (confusion_matrix(y_true, y_pred)[0, 0] + confusion_matrix(y_true, y_pred)[1, 0]))
    confi_ppv = compute_bootstrap_confi(all_predictions, all_y_test, precision_score)
    confi_auprc = compute_bootstrap_confi(all_y_prob, all_y_test, compute_auprc)
    confi_accuracy = compute_bootstrap_confi(all_predictions, all_y_test, accuracy_score)
    confi_balanced_accuracy = compute_bootstrap_confi(all_predictions, all_y_test, balanced_accuracy_score)

    plot_roc_curve(all_y_test, all_y_prob, method, task)
    plot_confusion_matrix(all_y_test, all_predictions, positive, negative, method, task)
    
    directory = "./result"
    task_directory = os.path.join(directory, task)
    os.makedirs(task_directory, exist_ok=True)
    
    filename = f"results_{task}_{method}.pickle"
    file_path = os.path.join(task_directory, filename)
    
    with open(file_path, 'wb') as f:
        pickle.dump((performance_dict, all_y_test, all_y_prob, all_predictions), f)

    print(f"AUC: {auc} (95% CI: {confi_auc})")
    print(f"F1-score: {f1} (95% CI: {confi_f1})")
    print(f"Specificity: {specificity} (95% CI: {confi_specificity})")
    print(f"Sensitivity: {sensitivity} (95% CI: {confi_sensitivity})")
    print(f"NPV: {npv} (95% CI: {confi_npv})")
    print(f"PPV (Precision): {ppv} (95% CI: {confi_ppv})")
    print(f"AUPRC: {auprc} (95% CI: {confi_auprc})")
    print(f"Accuracy: {accuracy} (95% CI: {confi_accuracy})")
    print(f"Balanced accuracy: {balanced_accuracy} (95% CI: {confi_balanced_accuracy})")
    
    return performance_dict, all_y_test, all_y_prob, all_predictions


# The normalize_features, apply_normalization, compute_kernel_matrix, linear_kernel, compute_bootstrap_confi, plot_roc_curve, and plot_confusion_matrix functions are assumed to be defined elsewhere.
