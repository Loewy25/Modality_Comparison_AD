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
                             roc_auc_score, roc_curve)
from sklearn.model_selection import (GridSearchCV, KFold, StratifiedKFold,
                                     cross_val_predict, cross_val_score,
                                     train_test_split)
from sklearn.preprocessing import Binarizer, label_binarize
from sklearn.svm import LinearSVC, SVC



def generate_Mask(imgs,threshold):
    count=0
    final=np.zeros((91,109,91))
    binarizer = Binarizer(threshold=0.5)
    binarizer_ave = Binarizer(threshold=threshold)
    for i in imgs:
        temp=[]
        img=nib.load(i)
        img_data=img.get_fdata()
        for n in img_data:
            n=binarizer.fit_transform(n)
            temp.append(n)
        final=np.array(final)
        temp=np.array(temp)
        final+=temp
        #final=list(final)
        count+=1
    final=np.array(final)
    final/= count
    #final=list(final)
    final_mask=[]
    masks=[]
    for m in final:
        m=binarizer_ave.fit_transform(m)
        final_mask.append(m)
    final_mask=np.array(final_mask)
    return final_mask


def mcnemar_test(y_true, model1_preds, model2_preds):
    a=0
    b=0
    c=0
    d=0
    for i in range(len(y_true)):
        if y_true[i]==model1_preds[i]:
            if y_true[i]==model2_preds[i]:
                a+=1
            else:
                b+=1
        elif (y_true[i] == model2_preds[i]):
            c+=1
        else:
            d+=1

    # Construct the contingency table
    table = np.array([[a, b], [c, d]])
    print(a)
    print(b)
    print(c)
    print(d)
    # Perform the exact McNemar test
    result = ct.mcnemar(table, exact=False, correction = False)

    # Print the results
    print(f"Test statistic: {result.statistic:.2f}")
    print(f"P-value: {result.pvalue:.9f}")


    
def save_array_to_file(arr, task, method, path="/scratch/l.peiwang/arrays"):
    """
    Saves a numpy array to a file. The filename is derived from the task and method parameters.
    
    Parameters:
    - arr: The numpy array to save.
    - task: A string representing the task.
    - method: A string representing the method.
    - path: The path where the file will be saved. Default is "/scratch/l.peiwang/arrays".
    """
    # Make sure the path exists
    os.makedirs(path, exist_ok=True) 
    # Create the filename
    filename = f"{path}/{task}_{method}.txt"  
    # Save the array
    np.savetxt(filename, arr)


def write_list_to_csv(input_list, filename):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for item in input_list:
            writer.writerow([item])

def compute_kernel_matrix(X1, X2, kernel_function):
    n_samples1, n_samples2 = X1.shape[0], X2.shape[0]
    kernel_matrix = np.zeros((n_samples1, n_samples2))

    for i in range(n_samples1):
        for j in range(n_samples2):
            kernel_matrix[i, j] = kernel_function(X1[i], X2[j])

    return kernel_matrix


def linear_kernel(x1, x2):
    return np.dot(x1, x2)

def interpret_backward2forward(X_train,y_train,weight):
    cov_matrix_x = np.cov(X_train.T)
    len_x=cov_matrix_x.shape[0]
    W=weight.reshape(len_x,1)
    len_y=y_train.shape[0]
    y_train=y_train.reshape(1,len_y)
    cov_matrix_y = np.cov(y_train)
    A_inv = np.array([[1/cov_matrix_y]])
    temp1 = np.dot(cov_matrix_x, W)
    temp2=np.dot(temp1,A_inv)
    activation_pattern=temp2.reshape(1,122597)
    return activation_pattern


def ensure_directory_exists(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)




def min_max_normalization(arr):
    min_val = np.min(arr)
    max_val = np.max(arr)
    normalized_arr = (arr - min_val) / (max_val - min_val)
    return normalized_arr


def normalize_features_z(data, control_indices, return_params=False):
    """
    Normalize features using the control group data.
    
    data: ndarray, shape (n_samples, n_features)
        The data to be normalized.
    control_indices: list
        The indices of the control samples in the data.
    return_params: bool, optional (default=False)
        If True, returns normalization parameters alongside normalized data.
        
    Returns: ndarray, (and tuple if return_params=True)
        The normalized data and normalization parameters (control_mean, control_std).
    """
    # Select the control group data
    control_data = data[control_indices, :]
    # Calculate the mean and standard deviation for each feature from the control group
    control_mean = np.mean(control_data, axis=0)
    control_std = np.std(control_data, axis=0)
    
    # Normalize the features for all samples
    normalized_data = (data - control_mean) / control_std

    if return_params:
        return normalized_data, (control_mean, control_std)
    else:
        return normalized_data

def apply_normalization_z(data, params):
    """
    Apply normalization to the data using provided parameters.
    
    data: ndarray, shape (n_samples, n_features)
        The data to be normalized.
    params: tuple
        The normalization parameters in the form (mean, std).
        
    Returns: ndarray
        The normalized data.
    """
    mean, std = params
    return (data - mean) / std



def compute_bootstrap_confi(predictions, ground_truth, scoring_func, n_iterations=1000):
    scores = []
    
    for _ in range(n_iterations):
        indices = np.random.choice(len(ground_truth), len(ground_truth), replace=True)
        sample_true = np.array(ground_truth)[indices]
        sample_pred = np.array(predictions)[indices]

        score = scoring_func(sample_true, sample_pred)
        scores.append(score)

    lower = np.percentile(scores, 2.5)
    upper = np.percentile(scores, 97.5)
    
    return lower, upper


def compute_weights_for_linear_kernel(svm, X_support):
    """Compute SVM weights for a linear kernel.

    Parameters:
    - svm: Trained SVM model with a precomputed linear kernel.
    - X_support: Support vectors.

    Returns:
    - Weights in the original feature space.
    """
    alpha_times_y = svm.dual_coef_[0]  # This is alpha_i * y_i for all support vectors
    weights = np.dot(alpha_times_y, X_support)
    return weights


def compute_p_values_with_correction(X, K, y, model, num_permutations):
    # Initialize array to hold weights from permuted datasets
    permuted_weights = np.zeros((X.shape[1], num_permutations))

    # Shuffle labels and train model for each permutation
    for i in range(num_permutations):
        y_permuted = y.copy()
        np.random.shuffle(y_permuted)
        model.fit(K, y_permuted)
        # Correct the permuted weights
        permuted_weights[:, i] = compute_covariance_directly(X, y_permuted)

    # Train model on original data
    model.fit(K, y)
    # Correct the original weights
    original_weights = compute_covariance_directly(X, y)

    # Compute p-values
    p_values = np.empty(X.shape[1])
    for feature in range(X.shape[1]):
        p_values[feature] = (np.abs(permuted_weights[feature]) >= np.abs(original_weights[feature])).mean()

    return p_values


def apply_covariance_correction(features, target, model_weights):
    """
    Apply a covariance-based correction to model weights.

    Parameters
    ----------
    features : numpy.array, shape (n_samples, n_features)
        Array of feature data.
    target : numpy.array, shape (n_samples,)
        Array of target labels.
    model_weights : numpy.array, shape (n_features,)
        Array of model weights.

    Returns
    -------
    corrected_weights : numpy.array, shape (n_features,)
        Corrected model weights.

    The function computes the covariance matrices of features and target labels,
    then scales the product of the features' covariance matrix and the model weights
    by the inverse of the labels' variance.
    """

    # Compute covariance matrices
    features_cov_matrix = np.cov(features.T)
    target_variance = np.cov(target)

    # Reshape weights to be a column vector
    reshaped_weights = model_weights.reshape(-1, 1)

    # Compute inverse of target variance
    inverse_target_variance = 1/target_variance

    # Apply covariance correction
    weight_scaling_factor = np.dot(features_cov_matrix, reshaped_weights)
    corrected_weights = np.dot(weight_scaling_factor, inverse_target_variance).flatten()
    
    # Apply Min-Max Normalization
    corrected_weights = (corrected_weights - corrected_weights.min()) / (corrected_weights.max() - corrected_weights.min())
    
    return corrected_weights


def compute_covariance_directly(X_train, y_train):
    # Initialize array to hold covariances
    covariances = np.zeros(X_train.shape[1])
    
    # Compute covariance between each column of X_train and y_train
    for i in range(X_train.shape[1]):
        covariances[i] = np.cov(X_train[:, i], y_train)[0, 1]

    # Min-Max Normalization
    min_val = np.min(covariances)
    max_val = np.max(covariances)
    normalized_covariances = (covariances - min_val) / (max_val - min_val)

    return normalized_covariances


def compute_p_values(X, K, y, model, num_permutations):
    permuted_weights = np.zeros((X.shape[1], num_permutations))
    for i in range(num_permutations):
        y_permuted = y.copy()
        np.random.shuffle(y_permuted)
        model.fit(K, y_permuted)
        X_support = X[model.support_, :]
        permuted_weights[:, i]  = abs(compute_weights_for_linear_kernel(model, X_support))

    model.fit(K, y)
    X_support = X[model.support_, :]
    original_weights = abs(compute_weights_for_linear_kernel(model, X_support))
    p_values = np.empty(X.shape[1])
    for feature in range(X.shape[1]):
        p_values[feature] = (np.abs(permuted_weights[feature]) >= np.abs(original_weights[feature])).mean()

    return p_values


def fdr_correction_masked(p_values, alpha=0.05):
    """
    Perform Benjamini-Hochberg FDR correction on a list of p-values and return a masked array.
    
    Parameters:
    - p_values (array-like): List of p-values to correct.
    - alpha (float): Desired FDR control level.
    
    Returns:
    - Masked p-values where non-significant values are set to 1.
    """
    
    p_values = np.array(p_values)
    m = len(p_values)
    ranks = np.argsort(p_values)
    sorted_p_values = p_values[ranks]
    
    threshold_values = [(i+1)/m * alpha for i in range(m)]
    below_threshold = sorted_p_values <= threshold_values
    if below_threshold.any():
        max_p_value = sorted_p_values[below_threshold].max()
    else:
        max_p_value = 0.0

    # Create the masked p-values
    masked_p_values = np.where(p_values <= max_p_value, p_values, 1)
    
    return masked_p_values



def threshold_p_values(p_values, threshold=0.05):
    """
    Threshold p-values at a specified significance level.
    
    Parameters:
    - p_values (array-like): List of p-values to threshold.
    - threshold (float): Desired significance level, default is 0.05.
    
    Returns:
    - Masked p-values where non-significant values are set to 1.
    """
    
    p_values = np.array(p_values)
    masked_p_values = np.where(p_values <= threshold, p_values, 0)
    
    return masked_p_values


def create_cmap():
    cmap = LinearSegmentedColormap.from_list(
    "my_cmap", [(1, 1, 0.8), (0.6, 0, 0)], N=256
    )
    return cmap

