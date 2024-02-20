import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import classification_report, f1_score, accuracy_score
import os
import pickle
import torch

from torchmetrics import AUROC, Accuracy


# Assuming X contains the sigmoid probabilities and y contains the target labels
# X.shape should be (num_samples, 13, num_classes)
# y.shape should be (num_samples, num_classes)
def train_ridge_model(X_train, X_valid, X_test, y_train, y_valid, y_test, save_path):
    # Reshape X to (num_samples, 13 * num_classes)
    X_train = X_train.reshape((X_train.shape[0], -1))
    X_valid = X_valid.reshape((X_valid.shape[0], -1))
    X_test = X_test.reshape((X_test.shape[0], -1))

    # Optimizing the regularization strength alpha based on the validation set
    best_alpha = _optimize_alpha(X_train, X_valid, y_train, y_valid)

    # Initialize Ridge regression model with the best alpha
    ridge_model = Ridge(alpha=best_alpha)

    # Train the model on the training set
    ridge_model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = ridge_model.predict(X_test)
    y_pred_binary = np.round(y_pred)

    # Evaluate the performance of the model
    eval_dict = classification_report(y_test, y_pred_binary,
                                      target_names=["IAVB", "AF", "LBBB", "PAC", "RBBB", "SNR", "STD", "STE", "VEB"],
                                      output_dict=True)
    df_sklearn_summary = pd.DataFrame.from_dict(eval_dict)

    # TODO check if this really is correct
    # Values need to be within [0,1] because otherwise, torchmetrics will apply a Sigmoid function!
    y_pred_for_torchmetrics = np.clip(y_pred, 0, 1)
    torch_roc_auc = AUROC(task='multilabel',num_labels=9, average=None)
    torch_roc_auc_scores = torch_roc_auc(preds=torch.tensor(y_pred_for_torchmetrics), target=y_test)
    macro_torch_roc_auc = AUROC(task='multilabel', num_labels=9, average="macro")
    macro_torch_roc_auc_score = macro_torch_roc_auc(preds=torch.tensor(y_pred_for_torchmetrics), target=y_test)
    weighted_torch_roc_auc = AUROC(task='multilabel', num_labels=9, average="weighted")
    weighted_torch_roc_auc_score = weighted_torch_roc_auc(preds=torch.tensor(y_pred_for_torchmetrics), target=y_test)

    torch_acc = Accuracy(task='multilabel', num_labels=9, average=None)
    torch_acc_scores = torch_acc(preds=torch.tensor(y_pred_for_torchmetrics), target=y_test)
    macro_torch_acc = AUROC(task='multilabel', num_labels=9, average="macro")
    macro_torch_acc_score = macro_torch_acc(preds=torch.tensor(y_pred_for_torchmetrics), target=y_test)
    weighted_torch_acc = AUROC(task='multilabel', num_labels=9, average="weighted")
    weighted_torch_acc_score = weighted_torch_acc(preds=torch.tensor(y_pred_for_torchmetrics), target=y_test)

    # 'class_wise_torch_accuracy
    subset_acc = accuracy_score(y_true=y_test, y_pred=y_pred_binary)

    # Save the evaluation results to a file
    df_class_wise_results = pd.DataFrame(
        columns=['IAVB', 'AF', 'LBBB', 'PAC', 'RBBB', 'SNR', 'STD', 'STE', 'VEB', 'macro avg', 'weighted avg'])
    df_class_wise_results = pd.concat([df_class_wise_results, df_sklearn_summary[
        ['IAVB', 'AF', 'LBBB', 'PAC', 'RBBB', 'SNR', 'STD', 'STE', 'VEB', 'macro avg', 'weighted avg']]])

    # Append the roc_auc and acc scores to the dataframe
    df_class_wise_results.loc["torch_roc_auc"] = torch.cat((torch_roc_auc_scores,macro_torch_roc_auc_score.unsqueeze(0),
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
        # file.write(f"Subset Accuracy: {subset_acc:.3f}\n")
        file.write(f"\n\n\nBest Alpha: {best_alpha}\n")

    return df_class_wise_results, df_single_metric_results


def _optimize_alpha(X_train, X_valid, y_train, y_valid):
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
