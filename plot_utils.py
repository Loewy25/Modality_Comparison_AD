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
