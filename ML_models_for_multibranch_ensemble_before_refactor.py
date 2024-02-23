import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import Ridge, RidgeClassifier, LogisticRegression
from sklearn.metrics import classification_report, f1_score, accuracy_score
import os
import pickle
import torch
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.extmath import softmax
from torchmetrics import AUROC, Accuracy

from utils import ensure_dir


# def predict_proba_ridge(classifier, X):
#     # TODO This is not correct yet, cf https://stackoverflow.com/questions/66334612/plotting-roc-curve-for-ridgeclassifier-in-python
#     # From https://stackoverflow.com/questions/22538080/scikit-learn-ridge-classifier-extracting-class-probabilities
#     d = classifier.decision_function(X)
#     if len(d.shape) == 1:
#         d = np.c_[-d, d]
#     return softmax(d)

# Assuming X contains the sigmoid probabilities or logits and y contains the target labels
# X.shape should be (num_samples, 13, num_classes)
# => Here: X is a stack of BranchNets outputs followed by the Multibranch output
# y.shape should be (num_samples, num_classes)

def _get_class_name_by_index(class_index):
    return ["IAVB", "AF", "LBBB", "PAC", "RBBB", "SNR", "STD", "STE", "VEB"][class_index]


def _evaluate_ML_model(y_pred, y_pred_probs, y_test, save_path, path_suffix=None):
    save_path = os.path.join(save_path, path_suffix) if path_suffix is not None else save_path
    ensure_dir(save_path)

    # Evaluate the performance of the model
    eval_dict = classification_report(y_test, y_pred,
                                      target_names=["IAVB", "AF", "LBBB", "PAC", "RBBB", "SNR", "STD", "STE", "VEB"],
                                      output_dict=True)
    df_sklearn_summary = pd.DataFrame.from_dict(eval_dict)

    # Values need to be within [0,1] because otherwise, torchmetrics will apply a Sigmoid function!
    torch_roc_auc = AUROC(task='multilabel', num_labels=9, average=None)
    torch_roc_auc_scores = torch_roc_auc(preds=torch.tensor(y_pred_probs), target=y_test)
    macro_torch_roc_auc = AUROC(task='multilabel', num_labels=9, average="macro")
    macro_torch_roc_auc_score = macro_torch_roc_auc(preds=torch.tensor(y_pred_probs), target=y_test)
    weighted_torch_roc_auc = AUROC(task='multilabel', num_labels=9, average="weighted")
    weighted_torch_roc_auc_score = weighted_torch_roc_auc(preds=torch.tensor(y_pred_probs), target=y_test)

    torch_acc = Accuracy(task='multilabel', num_labels=9, average=None)
    torch_acc_scores = torch_acc(preds=torch.tensor(y_pred_probs), target=y_test)
    macro_torch_acc = AUROC(task='multilabel', num_labels=9, average="macro")
    macro_torch_acc_score = macro_torch_acc(preds=torch.tensor(y_pred_probs), target=y_test)
    weighted_torch_acc = AUROC(task='multilabel', num_labels=9, average="weighted")
    weighted_torch_acc_score = weighted_torch_acc(preds=torch.tensor(y_pred_probs), target=y_test)

    # Class_wise_torch_accuracy
    subset_acc = accuracy_score(y_true=y_test, y_pred=y_pred)

    # Save the evaluation results to a file
    df_class_wise_results = pd.DataFrame(
        columns=['IAVB', 'AF', 'LBBB', 'PAC', 'RBBB', 'SNR', 'STD', 'STE', 'VEB', 'macro avg', 'weighted avg'])
    df_class_wise_results = pd.concat([df_class_wise_results, df_sklearn_summary[
        ['IAVB', 'AF', 'LBBB', 'PAC', 'RBBB', 'SNR', 'STD', 'STE', 'VEB', 'macro avg', 'weighted avg']]])

    # Append the roc_auc and acc scores to the dataframe
    df_class_wise_results.loc["torch_roc_auc"] = torch.cat(
        (torch_roc_auc_scores, macro_torch_roc_auc_score.unsqueeze(0),
         weighted_torch_roc_auc_score.unsqueeze(0))).numpy()

    df_class_wise_results.loc["torch_accuracy"] = torch.cat(
        (torch_acc_scores, macro_torch_acc_score.unsqueeze(0),
         weighted_torch_acc_score.unsqueeze(0))).numpy()

    # Reorder the class columns of the dataframe to match the one used in the
    desired_col_order = ['SNR', 'AF', 'IAVB', 'LBBB', 'RBBB', 'PAC', 'VEB', 'STD', 'STE', 'macro avg',
                         'weighted avg']
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
                   strategy=None, individual_features=False, reduced_individual_features=False):
    # TODO X_valid is only used for hyperparameter tuning of the GradientBoostingClassifier with all features right now
    methods_with_feature_importance = ["decision_tree", "gradient_boosting"]
    num_classes = y_train.shape[1]
    y_preds = []
    y_pred_probs = []

    # -------------------------------- ONLY NEEDED FOR DECISION TREE ------------------------------
    if strategy in methods_with_feature_importance:
        if individual_features:
            if not reduced_individual_features:
                df_feature_importance = pd.DataFrame(
                    columns=[f"branchNet_{i + 1}" for i in range(12)] + (["multibranch"]))
            else:
                df_feature_importance = pd.DataFrame(columns=[f"branchNet_{i + 1}" for i in range(12)])
        else:
            column_names = [f"BrN{branchnet + 1}_c{_get_class_name_by_index(class_idx)}"
                            for branchnet in range(12)
                            for class_idx in range(9)] \
                           + [f"MuB_c{_get_class_name_by_index(class_idx)}" for class_idx in range(9)]
            df_feature_importance = pd.DataFrame(columns=column_names)

    # -------------------------------- END ONLY NEEDED FOR DECISION TREE ------------------------------

    for class_index in range(0, num_classes):
        # Fit one decision tree classifier per target
        match strategy:
            case "decision_tree":
                classifier = DecisionTreeClassifier()
            case "ridge":
                # Fit one ridge classifier per target
                # Replace Ridge with LogisiticRegression since it supports predict_proba
                # Should be the same optimization problem as Ridge according to
                # https://stackoverflow.com/questions/66334612/plotting-roc-curve-for-ridgeclassifier-in-python
                classifier = LogisticRegression(penalty='l2')
            case "lasso":
                classifier = LogisticRegression(penalty='l1', solver='saga')
            case "gradient_boosting":
                classifier = GradientBoostingClassifier()
            case "elastic_net":
                # From https://stats.stackexchange.com/questions/304931/can-scikit-learns-elasticnetcv-be-used-for-classification-problems
                classifier = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, max_iter=1000)
            case _:
                raise ValueError("Invalid strategy")

        if individual_features:

            if not reduced_individual_features:

                if strategy == "gradient_boosting":
                    # Fuse the train and validation set to fine-tune the hyperparameters
                    X_train_valid = np.concatenate((X_train, X_valid), axis=0)
                    y_train_valid = np.concatenate((y_train, y_valid), axis=0)

                    classifier = fine_tune_gradient_boosting(X_train=X_train_valid[:, :, class_index],
                                                             y_train=y_train_valid[:, class_index],
                                                             class_index=class_index,
                                                             save_path=save_path)

                    # Also save the test set to a file
                    with open(os.path.join(save_path, f'X_test_{_get_class_name_by_index(class_index)}.p'),'wb') as file:
                        pickle.dump(X_test[:, :, class_index], file)
                    with open(os.path.join(save_path, f'y_test_{_get_class_name_by_index(class_index)}.p'),'wb') as file:
                        pickle.dump(y_test[:, class_index], file)

                    y_pred = classifier.predict(X_test[:, :, class_index])
                    y_pred_prob = classifier.predict_proba(X_test[:, :, class_index])[:, 1]

                else:
                    # Use only the features of the current class for training (incl multibranch output)
                    classifier.fit(X_train[:, :, class_index], y_train[:, class_index])
                    y_pred = classifier.predict(X_test[:, :, class_index])
                    # Shape: 2D array of shape (num_samples, 2)
                    # with the probabilities for the negative and positive class, respectively (classes_ is [0, 1])
                    y_pred_prob = classifier.predict_proba(X_test[:, :, class_index])[:, 1]
            else:
                if strategy == "gradient_boosting":
                    # Fuse the train and validation set to fine-tune the hyperparameters
                    X_train_valid = np.concatenate((X_train, X_valid), axis=0)
                    y_train_valid = np.concatenate((y_train, y_valid), axis=0)

                    classifier = fine_tune_gradient_boosting(X_train=X_train_valid[:, :12, class_index],
                                                             y_train=y_train_valid[:, class_index],
                                                             class_index=class_index,
                                                             save_path=save_path)

                    # Also save the test set to a file
                    with open(os.path.join(save_path, f'X_test_{_get_class_name_by_index(class_index)}.p'),'wb') as file:
                        pickle.dump(X_test[:, :, class_index], file)
                    with open(os.path.join(save_path, f'y_test_{_get_class_name_by_index(class_index)}.p'),'wb') as file:
                        pickle.dump(y_test[:, class_index], file)

                    y_pred = classifier.predict(X_test[:, :12, class_index])
                    y_pred_prob = classifier.predict_proba(X_test[:, :12, class_index])[:, 1]
                else:
                    # Use only the features of the current class for training (without multibranch output)
                    classifier.fit(X_train[:, :12, class_index], y_train[:, class_index])
                    y_pred = classifier.predict(X_test[:, :12, class_index])
                    y_pred_prob = classifier.predict_proba(X_test[:, :12, class_index])[:, 1]

        else:
            # Use all features (across all classes)
            # Reshape X to (num_samples, 13 * num_classes)
            X_train_all = X_train.reshape((X_train.shape[0], -1))
            X_valid_all = X_valid.reshape((X_valid.shape[0], -1))
            X_test_all = X_test.reshape((X_test.shape[0], -1))

            if strategy == "gradient_boosting":
                # Fuse the train and validation set to fine-tune the hyperparameters
                X_train_valid = np.concatenate((X_train_all, X_valid_all), axis=0)
                y_train_valid = np.concatenate((y_train, y_valid), axis=0)

                classifier = fine_tune_gradient_boosting(X_train=X_train_valid,
                                                         y_train=y_train_valid[:, class_index],
                                                         class_index=class_index,
                                                         save_path=save_path)

                # Also save the test set to a file
                with open(os.path.join(save_path, f'X_test_{_get_class_name_by_index(class_index)}.p'), 'wb') as file:
                    pickle.dump(X_test_all, file)
                with open(os.path.join(save_path, f'y_test_{_get_class_name_by_index(class_index)}.p'), 'wb') as file:
                    pickle.dump(y_test[:, class_index], file)

                y_pred = classifier.predict(X_test_all)
                y_pred_prob = classifier.predict_proba(X_test_all)[:, 1]
            else:
                # Fit the model without fine-tuning
                classifier.fit(X_train_all, y_train[:, class_index])
                y_pred = classifier.predict(X_test_all)
                y_pred_prob = classifier.predict_proba(X_test_all)[:, 1]

        y_preds.append(y_pred)
        y_pred_probs.append(y_pred_prob)

        if strategy in methods_with_feature_importance:
            df_feature_importance.loc[
                f"class_{_get_class_name_by_index(class_index)}"] = classifier.feature_importances_

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
                                                                         save_path=save_path)

    return df_class_wise_results, df_single_metric_results


def fine_tune_gradient_boosting(X_train, y_train, class_index, save_path):
    # Fine-tune the hyperparameters of the GradientBoostingClassifier
    param_grid = {
        'learning_rate': [0.05, 0.1, 0.2],
        'max_depth': [3, 4, 5],
        'n_estimators': [100, 150, 200]
    }
    # Initialize the base classifier
    base_classifier = GradientBoostingClassifier()

    # Initialize GridSearchCV and perform grid search on the training set
    grid_search = GridSearchCV(estimator=base_classifier, param_grid=param_grid,
                               cv=3, n_jobs=48, scoring='f1')
    grid_search.fit(X_train, y_train)

    # Print the best parameters found
    print("Best Parameters:", grid_search.best_params_)
    cv_result_df = pd.DataFrame(grid_search.cv_results_)

    # Log the best parameters and to a file
    with open(os.path.join(save_path, 'best_params.txt'), 'a') as file:
        file.write(f"Class {_get_class_name_by_index(class_index)}:\n")
        file.write(str(grid_search.best_params_) + "\n")

    # Log the cv results to a file as csv and as pickle
    with open(os.path.join(save_path,
                           f'{_get_class_name_by_index(class_index)}_cv_results.p'), 'wb') as file:
        pickle.dump(cv_result_df, file)
    cv_result_df.to_csv(os.path.join(save_path, f'{_get_class_name_by_index(class_index)}_cv_results.csv'))

    # # Initialize the classifier with the best parameters
    # classifier = GradientBoostingClassifier(**grid_search.best_params_)
    # classifier.fit(X_train, y_train)
    # # Save the best model to a file
    # with open(os.path.join(save_path, f'best_model_{_get_class_name_by_index(class_index)}.p'), 'wb') as file:
    #     pickle.dump(classifier, file)

    best_model = grid_search.best_estimator_  # No need for re-fitting the model, just use the best estimator
    # Save the best model to a file
    with open(os.path.join(save_path, f'best_model_{_get_class_name_by_index(class_index)}.p'), 'wb') as file:
        pickle.dump(best_model, file)

    return best_model


def _optimize_ridge_alpha(X_train, X_valid, y_train, y_valid):
    # List of alpha values to try
    alpha_values = [0.1, 1.0, 10.0, 100.0]  # alphas = np.logspace(-6, 6, 13)

    best_alpha = None
    best_score = 0.0  # Assuming a higher score is better, adjust based on your metric

    # Loop over alpha values
    for alpha in alpha_values:
        # Initialize Ridge regression model
        ridge_model = Ridge(alpha=alpha)

        # Train the model on the training set
        ridge_model.fit(X_train, y_train)

        # Make predictions on the validation set
        y_pred = ridge_model.predict(X_valid)
        # Ridge regression does not enforce the output to be within [0, 1], and the predictions can be any real number.
        # If the target labels are binary (0 or 1), round the predictions
        y_pred_binary = np.round(y_pred)

        # Evaluate performance
        current_score = f1_score(y_valid, y_pred_binary, average='macro')

        print(f"Alpha: {alpha}, Score: {current_score}")

        # Update the best hyperparameter if needed
        if current_score > best_score:
            best_score = current_score
            best_alpha = alpha

    print(f"Best Alpha: {best_alpha}, Best Score on validation set: {best_score}")
    return best_alpha
