def hyperparameter_tuning_visual_cov_V3(data, label, randomseed, outer, inner, num_permutations):
    train_data = data
    train_label = label
    random_states = randomseed

    all_aucs = []
    all_single_weights = []
    all_corrected_weights = []
    all_p_values_single = []
    all_p_values_corrected = []


    full_kernel_matrix = compute_kernel_matrix(train_data, train_data, linear_kernel)
    for rs in random_states:
        print(f"Running outer loop with random state: {rs}")
        cv_outer = StratifiedKFold(n_splits=outer, shuffle=True, random_state=rs)
        
        for train_ix, test_ix in cv_outer.split(train_data, train_label):
            X_train, X_test = train_data[train_ix, :], train_data[test_ix, :]
            y_train, y_test = np.array(train_label)[train_ix], np.array(train_label)[test_ix]

            K_train = full_kernel_matrix[train_ix][:, train_ix]
            K_test = full_kernel_matrix[test_ix][:, train_ix]

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
    from sklearn.model_selection import StratifiedKFold, GridSearchCV
    from sklearn.metrics import (roc_auc_score, precision_recall_fscore_support,
                                 accuracy_score, balanced_accuracy_score, precision_score, confusion_matrix)
    from sklearn.svm import SVC
    import numpy as np
    import os
    import pickle

    train_data = data
    train_label = label
    control_indices = [i for i, label in enumerate(train_label) if label == 0]
    train_data = normalize_features(train_data, control_indices)
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

    full_kernel_matrix = compute_kernel_matrix(train_data, train_data, linear_kernel)
    for rs in random_states:
        cv_outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=rs)
        
        for train_ix, test_ix in cv_outer.split(train_data, train_label):
            X_train, X_test = train_data[train_ix, :], train_data[test_ix, :]
            y_train, y_test = np.array(train_label)[train_ix], np.array(train_label)[test_ix]
            
            K_train = full_kernel_matrix[train_ix][:, train_ix]
            K_test = full_kernel_matrix[test_ix][:, train_ix]
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

    precision_classwise = precision_score(all_y_test, all_predictions, average=None)
    recall_classwise = recall_score(all_y_test, all_predictions, average=None)
    f1_classwise = f1_score(all_y_test, all_predictions, average=None)

    ppv = precision_score(all_y_test, all_predictions, average=None)
    cm = confusion_matrix(all_y_test, all_predictions)
    npv = cm[1,1] / (cm[1,1] + cm[0,1])
    
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
    
    directory = '/scratch/l.peiwang/results'
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




def nested_crossvalidation_late_fusion(data_pet, data_mri, label, method, task):
    train_label = label
    control_indices = [i for i, label in enumerate(train_label) if label == 0]
    train_data_pet = normalize_features(data_pet, control_indices)
    train_data_mri = normalize_features(data_mri, control_indices)
    
    full_kernel_matrix_pet = compute_kernel_matrix(train_data_pet, train_data_pet,linear_kernel)
    full_kernel_matrix_mri = compute_kernel_matrix(train_data_mri, train_data_mri,linear_kernel)
    
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
    train_data_pet=np.array(train_data_pet)
    train_data_mri=np.array(train_data_mri)
    train_label=np.array(train_label)
    for rs in random_states:
        cv_outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=rs)
        
        for train_ix, test_ix in cv_outer.split(train_data_pet, train_label):
            y_train, y_test = np.array(train_label)[train_ix], np.array(train_label)[test_ix]

            K_train_pet = full_kernel_matrix_pet[train_ix][:, train_ix]
            K_test_pet = full_kernel_matrix_pet[test_ix][:, train_ix]
            
            K_train_mri = full_kernel_matrix_mri[train_ix][:, train_ix]
            K_test_mri = full_kernel_matrix_mri[test_ix][:, train_ix]
            
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
    precision_classwise = precision_score(all_y_test, all_predictions, average=None)
    recall_classwise = recall_score(all_y_test, all_predictions, average=None)
    f1_classwise = f1_score(all_y_test, all_predictions, average=None)
    ppv = precision_score(all_y_test, all_predictions, average=None)
    cm = confusion_matrix(all_y_test, all_predictions)
    npv = cm[1,1] / (cm[1,1] + cm[0,1])
    
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
    
    directory = '/scratch/l.peiwang/results'
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
    
