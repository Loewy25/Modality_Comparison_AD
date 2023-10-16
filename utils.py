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


def plot_confusion_matrix(true_y, y_prob, positive, negative, method, task):
    """
    Plots the confusion matrix based on the probabilities.
    """
    confusion_matrix = metrics.confusion_matrix(true_y, y_prob)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [negative, positive])
    cm_display.plot()
    
    # Directory path
    directory = "/result/"

    # Construct the task specific directory path
    task_directory = os.path.join(directory, task)

    # Ensure directory exists
    os.makedirs(task_directory, exist_ok=True)

    # Construct complete file path with 'cm' + task + method as the filename
    filename = f"confusion_matrix_{task}_{method}.png"
    file_path = os.path.join(task_directory, filename)

    # Save the plot
    plt.savefig(file_path)
    plt.show()


def plot_roc_curve(true_y, y_prob, method, task):


    fpr, tpr, thresholds = roc_curve(true_y, y_prob)
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    # Directory path
    directory = "/scratch/l.peiwang"
    
    # Construct the task specific directory path
    task_directory = os.path.join(directory, task)

    # Ensure directory exists
    os.makedirs(task_directory, exist_ok=True)

    # Construct complete file path with 'auc' + task + method as the filename
    filename = f"auc_{task}_{method}.png"
    file_path = os.path.join(task_directory, filename)
    
    # Save the plot
    plt.savefig(file_path)
    
    # Show the plot
    plt.show()
    time.sleep(2.5)
    print(f"Overall ROC AUC for all data: {roc_auc_score(true_y, y_prob)}")




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


  def ensure_directory_exists(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    return activation_pattern


def plot_glass_brain(feature_importance_map_3d, method, task, modality):
    base_path = os.getcwd() # Get the current working directory
    result_path = os.path.join(base_path, 'result')
    ensure_directory_exists(result_path)
    output_path = os.path.join(result_path, f'glass_brain_{method}_{task}_{modality}.png')
    
    cmap = create_cmap()
    plotting.plot_glass_brain(feature_importance_map_3d, colorbar=True, plot_abs=True, cmap='jet', output_file=output_path, vmin=0, vmax=1)
    print(f'Glass brain plot saved at {output_path}')

def plot_stat_map(weight_img, threshold, method, task, modality):
    base_path = os.getcwd() # Get the current working directory
    result_path = os.path.join(base_path, 'result')
    ensure_directory_exists(result_path)
    output_path = os.path.join(result_path, f'stat_map_{method}_{task}_{modality}.png')
    
    cmap = create_cmap()
    plotting.plot_stat_map(weight_img, display_mode='x', threshold=threshold, cut_coords=range(0, 51, 5), title='Slices', cmap='jet', output_file=output_path, vmax=1)
    print(f'Stat map plot saved at {output_path}')


def min_max_normalization(arr):
    min_val = np.min(arr)
    max_val = np.max(arr)
    normalized_arr = (arr - min_val) / (max_val - min_val)
    return normalized_arr


def normalize_features(data, control_indices):
    """
    Normalize features using the control group data.
    
    data: ndarray, shape (n_samples, n_features)
        The data to be normalized.
    control_indices: list
        The indices of the control samples in the data.
        
    Returns: ndarray
        The normalized data.
    """
    # Select the control group data
    control_data = data[control_indices, :]
    # Calculate the mean and standard deviation for each feature from the control group
    control_mean = np.mean(control_data, axis=0)
    control_std = np.std(control_data, axis=0)
    
    # Normalize the features for all samples
    normalized_data = (data - control_mean) / control_std
       
    return normalized_data


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

