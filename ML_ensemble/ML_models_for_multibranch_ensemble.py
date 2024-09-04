import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import os
import pickle
import torch
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from torchmetrics import AUROC, Accuracy

from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_sample_weight

import global_config
from utils import ensure_dir

global_config.suppress_warnings()

# Assuming X contains the sigmoid probabilities or logits and y contains the target labels
# X.shape should be (num_samples, 13, num_classes)
# => Here: X is a stack of BranchNets outputs followed by the Multibranch output
# y.shape should be (num_samples, num_classes)

def _evaluate_ML_model(y_pred, y_pred_probs, y_test, target_names, save_path, path_suffix=None):

    save_path = os.path.join(save_path, path_suffix) if path_suffix is not None else save_path
    ensure_dir(save_path)

    # Evaluate the performance of the model
    eval_dict = classification_report(y_test, y_pred,
                                      target_names=target_names,
                                      output_dict=True)
    df_sklearn_summary = pd.DataFrame.from_dict(eval_dict)

    # Values need to be within [0,1] because otherwise, torchmetrics will apply a Sigmoid function!
    num_labels = len(target_names)
    torch_roc_auc = AUROC(task='multilabel', num_labels=num_labels, average=None)
    torch_roc_auc_scores = torch_roc_auc(preds=torch.tensor(y_pred_probs), target=y_test)
    macro_torch_roc_auc = AUROC(task='multilabel', num_labels=num_labels, average="macro")
    macro_torch_roc_auc_score = macro_torch_roc_auc(preds=torch.tensor(y_pred_probs), target=y_test)
    weighted_torch_roc_auc = AUROC(task='multilabel', num_labels=num_labels, average="weighted")
    weighted_torch_roc_auc_score = weighted_torch_roc_auc(preds=torch.tensor(y_pred_probs), target=y_test)

    torch_acc = Accuracy(task='multilabel', num_labels=num_labels, average=None)
    torch_acc_scores = torch_acc(preds=torch.tensor(y_pred_probs), target=y_test)
    macro_torch_acc = AUROC(task='multilabel', num_labels=num_labels, average="macro")
    macro_torch_acc_score = macro_torch_acc(preds=torch.tensor(y_pred_probs), target=y_test)
    weighted_torch_acc = AUROC(task='multilabel', num_labels=num_labels, average="weighted")
    weighted_torch_acc_score = weighted_torch_acc(preds=torch.tensor(y_pred_probs), target=y_test)

    # Class_wise_torch_accuracy
    subset_acc = accuracy_score(y_true=y_test, y_pred=y_pred)

    # Save the evaluation results to a file
    columns = [*target_names,*['macro avg', 'weighted avg']]
    df_class_wise_results = pd.DataFrame(columns=columns)
    df_class_wise_results = pd.concat([df_class_wise_results, df_sklearn_summary[columns]])

    # Append the roc_auc and acc scores to the dataframe
    df_class_wise_results.loc["torch_roc_auc"] = torch.cat(
        (torch_roc_auc_scores, macro_torch_roc_auc_score.unsqueeze(0),
         weighted_torch_roc_auc_score.unsqueeze(0))).numpy()

    df_class_wise_results.loc["torch_accuracy"] = torch.cat(
        (torch_acc_scores, macro_torch_acc_score.unsqueeze(0),
         weighted_torch_acc_score.unsqueeze(0))).numpy()

    # Reorder the class columns of the dataframe  if the used dataset is CPSC 2018
    if target_names ==  ["IAVB", "AF", "LBBB", "PAC", "RBBB", "SNR", "STD", "STE", "VEB"]:
        desired_col_order = ['SNR', 'AF', 'IAVB', 'LBBB', 'RBBB', 'PAC', 'VEB', 'STD', 'STE',
                             'macro avg', 'weighted avg']
        df_class_wise_results = df_class_wise_results[desired_col_order]

    df_single_metric_results = pd.DataFrame(columns=['sk_subset_accuracy'])
    # Append the subset accuracy to the dataframe
    df_single_metric_results = df_single_metric_results.append({'sk_subset_accuracy': subset_acc}, ignore_index=True)
    df_single_metric_results.rename(index={0: 'value'}, inplace=True)

    with open(os.path.join(save_path, 'eval_class_wise.p'), 'wb') as file:
        pickle.dump(df_class_wise_results, file)

    with open(os.path.join(save_path, 'eval_single_metrics.p'), 'wb') as file:
        pickle.dump(df_single_metric_results, file)

    with open(os.path.join(save_path, 'eval_results.tex'), 'w') as file:
        df_class_wise_results.to_latex(buf=file, index=True, bold_rows=True, float_format="{:0.3f}".format)
        file.write(f"\n\n\n\n")
        df_single_metric_results.to_latex(buf=file, index=True, bold_rows=True, float_format="{:0.3f}".format)

    return df_class_wise_results, df_single_metric_results


def train_ML_model(X_train, X_valid, X_test, y_train, y_valid, y_test, save_path,
                   strategy=None, individual_features=False, reduced_individual_features=False,
                   target_names=["IAVB", "AF", "LBBB", "PAC", "RBBB", "SNR", "STD", "STE", "VEB"],
                   use_SMOTE=False, use_class_weights=False):

    # Note: X_valid is only used for hyperparameter tuning of the GradientBoostingClassifier and
    #  the AdaBoostClassifier with all features right now
    methods_with_feature_importance = ["decision_tree", "gradient_boosting", "ada_boost"]
    num_classes = y_train.shape[1]
    assert len(target_names) == num_classes, "Loaded target names do NOT match the number of classes"

    y_preds = []
    y_pred_probs = []

    # -------------------------------- ONLY NEEDED FOR DECISION TREE ------------------------------
    if strategy in methods_with_feature_importance:
        df_feature_importance = _prepare_df_for_feature_importance(individual_features, reduced_individual_features,
                                                                   target_names)

    # -------------------------------- END ONLY NEEDED FOR DECISION TREE ------------------------------

    # Store the original X_train, X_valid and X_test for later use
    # Always use it as the basis to selct X_train, X_valid and X_test for the current class
    if strategy == "gradient_boosting" or strategy == "ada_boost":
        # Fuse the train and validation set to fine-tune the hyperparameters
        X_train_valid = np.concatenate((X_train, X_valid), axis=0)
        y_train_valid = np.concatenate((y_train, y_valid), axis=0)
    else:
        X_train_orig = X_train
        X_valid_orig = X_valid
    X_test_orig = X_test

    # Fit one classifier per target
    for class_index in range(0, num_classes):

        if strategy in ["gradient_boosting", "ada_boost"]:

            if individual_features:
                if not reduced_individual_features:
                    # Use only the features of the current class for training (incl multibranch output)
                    X_train = X_train_valid[:, :, class_index]
                    X_test = X_test_orig[:, :, class_index]
                else:
                    # Use only the features of the current class for training (without multibranch output)
                    X_train = X_train_valid[:, :12, class_index]
                    X_test = X_test_orig[:, :12, class_index]
            else:
                X_train = X_train_valid.reshape((X_train_valid.shape[0], -1))
                X_test = X_test_orig.reshape((X_test.shape[0], -1))

            classifier = fine_tune_gradient_boosting_or_ada_boost(X_train=X_train,
                                                                  y_train=y_train_valid[:, class_index],
                                                                  class_index=class_index,
                                                                  save_path=save_path,
                                                                  target_names=target_names,
                                                                  strategy=strategy,
                                                                  use_SMOTE=use_SMOTE,
                                                                  use_class_weights=use_class_weights)


            # Also save the test set to a file
            with open(os.path.join(save_path, f'X_test_{target_names[class_index]}.p'), 'wb') as file:
                pickle.dump(X_test, file)
            with open(os.path.join(save_path, f'y_test_{target_names[class_index]}.p'), 'wb') as file:
                pickle.dump(y_test[:, class_index], file)

            y_pred = classifier.predict(X_test)
            y_pred_prob = classifier.predict_proba(X_test)[:, 1]

        else:

            # Determine the train, valid and test set for X depending on the feature selection
            if individual_features:
                if not reduced_individual_features:
                    # Use only the features of the current class for training (incl multibranch output)
                    X_train = X_train_orig[:, :, class_index]
                    X_valid = X_valid_orig[:, :, class_index]
                    X_test = X_test_orig[:, :, class_index]
                else:
                    # Use only the features of the current class for training (without multibranch output)
                    X_train = X_train_orig[:, :12, class_index]
                    X_valid = X_valid_orig[:, :12, class_index]
                    X_test = X_test_orig[:, :12, class_index]
            else:
                # Use all features (across all classes)
                # Reshape X to (num_samples, 13 * num_classes)
                X_train = X_train_orig.reshape((X_train.shape[0], -1))
                X_valid = X_valid_orig.reshape((X_valid.shape[0], -1))
                X_test = X_test_orig.reshape((X_test.shape[0], -1))

            # Fit the model without fine-tuning
            classifier = get_classifier(strategy)
            classifier.fit(X_train, y_train[:, class_index])
            y_pred = classifier.predict(X_test)
            # Shape: 2D array of shape (num_samples, 2)
            # with the probabilities for the negative and positive class, respectively (classes_ is [0, 1])
            y_pred_prob = classifier.predict_proba(X_test)[:, 1]

        y_preds.append(y_pred)
        y_pred_probs.append(y_pred_prob)

        if strategy in methods_with_feature_importance:
            df_feature_importance.loc[
                f"class_{target_names[class_index]}"] = classifier.feature_importances_

    # Stack the predictions and prediction probabilities
    y_preds = np.stack(y_preds, axis=1)
    y_pred_probs = np.stack(y_pred_probs, axis=1)

    # -------------------------------- ONLY NEEDED FOR DECISION TREE ------------------------------
    if strategy in methods_with_feature_importance:
        with open(os.path.join(save_path, 'feature_importance.p'), 'wb') as file:
            pickle.dump(df_feature_importance, file)
        # Also store it more human-readable as csv
        df_feature_importance.to_csv(os.path.join(save_path, 'feature_importance.csv'))

    # -------------------------------- END ONLY NEEDED FOR DECISION TREE ------------------------------

    # Evaluate the performance of the model
    df_class_wise_results, df_single_metric_results = _evaluate_ML_model(y_pred=y_preds,
                                                                         y_pred_probs=y_pred_probs,
                                                                         y_test=y_test,
                                                                         target_names=target_names,
                                                                         save_path=save_path)

    return df_class_wise_results, df_single_metric_results


def _prepare_df_for_feature_importance(individual_features, reduced_individual_features, target_names):
    if individual_features:
        if not reduced_individual_features:
            df_feature_importance = pd.DataFrame(
                columns=[f"branchNet_{i + 1}" for i in range(12)] + (["multibranch"]))
        else:
            df_feature_importance = pd.DataFrame(columns=[f"branchNet_{i + 1}" for i in range(12)])
    else:
        column_names = [f"BrN{branchnet + 1}_c{target_names[class_idx]}"
                        for branchnet in range(12)
                        for class_idx in range(len(target_names))] \
                       + [f"MuB_c{target_names[class_idx]}" for class_idx in range(len(target_names))]
        df_feature_importance = pd.DataFrame(columns=column_names)
    return df_feature_importance


def get_classifier(strategy):
    match strategy:
        case "decision_tree":
            classifier = DecisionTreeClassifier(random_state=global_config.SEED)
        case "ada_boost":
            classifier = AdaBoostClassifier(random_state=global_config.SEED)
        case "ridge":
            # Fit one ridge classifier per target
            # Replace Ridge with LogisiticRegression since it supports predict_proba
            # Should be the same optimization problem as Ridge according to
            # https://stackoverflow.com/questions/66334612/plotting-roc-curve-for-ridgeclassifier-in-python
            classifier = LogisticRegression(penalty='l2', random_state=global_config.SEED)
        case "lasso":
            classifier = LogisticRegression(penalty='l1', solver='saga', random_state=global_config.SEED)
        case "gradient_boosting":
            classifier = GradientBoostingClassifier(random_state=global_config.SEED)
        case "elastic_net":
            # From https://stats.stackexchange.com/questions/304931/can-scikit-learns-elasticnetcv-be-used-for-classification-problems
            classifier = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5,
                                            max_iter=1000, random_state=global_config.SEED)
        case _:
            raise ValueError("Invalid strategy")
    return classifier


def fine_tune_gradient_boosting_or_ada_boost(X_train, y_train, class_index, save_path, target_names,
                                             strategy="gradient_boosting", use_SMOTE=False, use_class_weights=False):

    match strategy:
        case "gradient_boosting":
            # Fine-tune the hyperparameters of the GradientBoostingClassifier
            param_grid = {
                'learning_rate': [0.05, 0.1, 0.2],
                'max_depth': [2, 3, 4, 5],
                'n_estimators': [100, 150, 200],
                'min_samples_split': [2, 4, 6],
                'min_samples_leaf': [1, 2, 3],
                'subsample': [0.8, 0.9, 1],
                'max_features': [None, 'sqrt', 'log2']
                # 'learning_rate': [0.1],
                # 'max_depth': [2],
                # 'n_estimators': [100],
                # 'min_samples_split': [2],
                # 'min_samples_leaf': [1],
                # 'subsample': [0.8],
                # 'max_features': [None]
            }
            # Initialize the base classifier
            base_classifier = GradientBoostingClassifier(random_state=global_config.SEED)

        case "ada_boost":
            # Fine-tune the hyperparameters of the GradientBoostingClassifier
            # Estimator could also be LogisticRegressor, SVC, etc.
            param_grid = {
                'estimator': [DecisionTreeClassifier(max_depth=i) for i in range(1, 11)],
                'n_estimators': [10, 50, 100, 200, 500],
                'learning_rate': [0.001, 0.01, 0.1, 0.5, 0.7, 1, 1.5]
            }
            # Initialize the base classifier
            base_classifier = AdaBoostClassifier(random_state=global_config.SEED)
        case _:
            raise ValueError("Invalid strategy")

    # Initialize GridSearchCV and perform grid search on the training set
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=global_config.SEED)
    grid_search = GridSearchCV(estimator=base_classifier, param_grid=param_grid,
                               cv=cv, n_jobs=24, scoring='f1')
    print(f"Fine-tuning the hyperparameters for class {target_names[class_index]} using {strategy}."
          f" This may take a while...")

    assert not use_SMOTE or not use_class_weights, "Please do not use the combination of both"
    if use_SMOTE:
        print(f"Using SMOTE for resampling the data before fitting the classifier.")
        smote = SMOTE(random_state=global_config.SEED)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        # Fit the model on the resampled data
        grid_search.fit(X_resampled, y_resampled)
    elif use_class_weights:
        print("Using sample weights to penalize the model for misclassifying the minority class")
        # Compute sample weights to penalize the model for misclassifying the minority class
        sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
        # Fit the model using sample weights
        grid_search.fit(X_train, y_train, sample_weight=sample_weights)
    else:
        grid_search.fit(X_train, y_train)

    # Print and save the best parameters found
    print("Best Parameters:", grid_search.best_params_)
    # If the class index is 0, overwrite the file, otherwise append to it
    if class_index == 0:
        with open(os.path.join(save_path, 'best_params.txt'), 'w') as file:
            file.write(f"Class {target_names[class_index]}:\n")
            file.write(str(grid_search.best_params_) + "\n")
    else:
        with open(os.path.join(save_path, 'best_params.txt'), 'a') as file:
            file.write(f"Class {target_names[class_index]}:\n")
            file.write(str(grid_search.best_params_) + "\n")

    # Save the best parameter dict to a pickle file
    with open(os.path.join(save_path, f'best_params_{target_names[class_index]}.p'), 'wb') as file:
        pickle.dump(grid_search.best_params_, file)

    # Log the cv results to a file as csv and as pickle
    cv_result_df = pd.DataFrame(grid_search.cv_results_)
    with open(os.path.join(save_path,
                           f'{target_names[class_index]}_cv_results.p'), 'wb') as file:
        pickle.dump(cv_result_df, file)
    cv_result_df.to_csv(os.path.join(save_path, f'{target_names[class_index]}_cv_results.csv'))

    best_model = grid_search.best_estimator_  # No need for re-fitting the model, just use the best estimator
    # Save the best model to a file
    with open(os.path.join(save_path, f'best_model_{target_names[class_index]}.p'), 'wb') as file:
        pickle.dump(best_model, file)

    return best_model


