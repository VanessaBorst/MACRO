import argparse
import copy
import pickle
import pandas as pd
from pathlib import Path

import torch

import global_config
from ML_models_for_multibranch_ensemble import train_ML_model
from logger import update_logging_setup_for_tune_or_cross_valid

from retrieve_detached_cross_fold_tensors import load_config_and_setup_paths, setup_cross_fold, prepare_result_data_structures
from utils import ensure_dir


# Needed for working with SSH Interpreter...
import os

os.environ["CUDA_VISIBLE_DEVICES"] = global_config.CUDA_VISIBLE_DEVICES


def _convert_logits_to_prediction(logits, thresholds=None):
    if isinstance(thresholds, dict):
        thresholds = list(thresholds.values())
    if thresholds is not None:
        ts = torch.tensor(thresholds).unsqueeze(0)
    else:
        # Default threshold: 0.5
        ts = torch.tensor([0.5] * logits.shape[1]).unsqueeze(0)
    sigmoid_probs = torch.sigmoid(logits)
    return torch.where(sigmoid_probs > ts, 1, 0)


def get_all_binary_predictions(det_outputs, det_single_lead_outputs):
    # Shape: (num_samples, 1, num_classes)
    multibranch_prediction = _convert_logits_to_prediction(det_outputs).unsqueeze(1)

    # Shape: (num_samples, 12, num_classes)
    single_branch_predictions = [_convert_logits_to_prediction(det_single_lead_output)
                                 for det_single_lead_output in det_single_lead_outputs]
    single_branch_predictions = torch.stack(single_branch_predictions, dim=0)

    # Shape: (num_samples, 13, num_classes)
    all_predictions = torch.cat([single_branch_predictions, multibranch_prediction], dim=1)
    return all_predictions


def get_all_predictions_as_probs(det_outputs, det_single_lead_outputs):
    # Shape: (num_samples, 1, num_classes)
    multibranch_prediction = torch.sigmoid(det_outputs).unsqueeze(1)

    # Shape: (num_samples, 12, num_classes)
    single_branch_predictions = [torch.sigmoid(det_single_lead_output)
                                 for det_single_lead_output in det_single_lead_outputs]
    single_branch_predictions = torch.stack(single_branch_predictions, dim=0)

    # Shape: (num_samples, 13, num_classes)
    all_predictions = torch.cat([single_branch_predictions, multibranch_prediction], dim=1)
    return all_predictions


def get_all_predictions_as_logits(det_outputs, det_single_lead_outputs):
    return torch.cat([det_single_lead_outputs,det_outputs.unsqueeze(1)], dim=1)


def train_ML_models_on_cross_fold_data(main_path, strategy=None, use_logits=False, individual_features=False,
                                       reduced_individual_features=False):
    assert strategy in ["decision_tree", "ridge", "lasso","gradient_boosting","elastic_net","ada_boost"], \
        "The given strategy is not supported for training ML models!"
    assert not reduced_individual_features or individual_features, \
        "Reduced individual features can only be used if individual features are used!"

    strategy_name = strategy if not use_logits else strategy + " with logits"
    strategy_name = strategy_name + "_individual_features" if individual_features else strategy_name
    strategy_name = strategy_name + "_reduced" if reduced_individual_features else strategy_name

    config = load_config_and_setup_paths(main_path, sub_dir=os.path.join("ML models",strategy_name))
    assert config['arch']['type'] == 'FinalModelMultiBranch', \
        "The ML models can only be trained for multi-branch models!"

    base_config, base_log_dir, base_save_dir, data_dir, dataset, fold_data, total_num_folds = setup_cross_fold(config)

    # Save the results of each run
    class_wise_metrics, folds, test_results_class_wise, test_results_single_metrics = \
        prepare_result_data_structures(total_num_folds)

    print(f"Starting with {total_num_folds}-fold cross validation for {strategy_name} classifier")
    if use_logits:
        print("Using the raw logits for the training of the ML models")
    valid_fold_index = total_num_folds - 2
    test_fold_index = total_num_folds - 1

    for k in range(total_num_folds):
        if k in [0,1,2,3,4,5,6]:
            # Skip
            valid_fold_index = (valid_fold_index + 1) % total_num_folds
            test_fold_index = (test_fold_index + 1) % total_num_folds
            config = copy.deepcopy(base_config)
            continue

        print("Starting fold " + str(k + 1))

        # Adapt the log and save paths for the current fold
        config.save_dir = Path(os.path.join(base_save_dir, "Fold_" + str(k + 1)))
        config.log_dir = Path(os.path.join(base_log_dir, "Fold_" + str(k + 1)))
        ensure_dir(config.save_dir)
        ensure_dir(config.log_dir)
        update_logging_setup_for_tune_or_cross_valid(config.log_dir)

        # Load the detached outputs to train the LASSO model
        detached_storage_path = os.path.join(config.save_dir.parent.parent.parent, "output_logging", "Fold_" + str(k + 1))
        with open(os.path.join(detached_storage_path, f"train_det_outputs.p"), 'rb') as file:
            det_outputs_train = pickle.load(file)
        with open(os.path.join(detached_storage_path, f"train_det_targets.p"), 'rb') as file:
            det_targets_train = pickle.load(file)
        with open(os.path.join(detached_storage_path, f"train_det_single_lead_outputs.p"), 'rb') as file:
            det_single_lead_outputs_train = pickle.load(file)
        with open(os.path.join(detached_storage_path, "valid_det_outputs.p"), 'rb') as file:
            det_outputs_valid = pickle.load(file)
        with open(os.path.join(detached_storage_path, "valid_det_targets.p"), 'rb') as file:
            det_targets_valid = pickle.load(file)
        with open(os.path.join(detached_storage_path, "valid_det_single_lead_outputs.p"), 'rb') as file:
            det_single_lead_outputs_valid = pickle.load(file)
        with open(os.path.join(detached_storage_path, "test_det_outputs.p"), 'rb') as file:
            det_outputs_test = pickle.load(file)
        with open(os.path.join(detached_storage_path, "test_det_targets.p"), 'rb') as file:
            det_targets_test = pickle.load(file)
        with open(os.path.join(detached_storage_path, "test_det_single_lead_outputs.p"), 'rb') as file:
            det_single_lead_outputs_test = pickle.load(file)

        # Train the ML models
        predicted_train = get_all_predictions_as_probs(det_outputs_train, det_single_lead_outputs_train) \
            if not use_logits else get_all_predictions_as_logits(det_outputs_train, det_single_lead_outputs_train)
        predicted_valid = get_all_predictions_as_probs(det_outputs_valid, det_single_lead_outputs_valid) \
            if not use_logits else get_all_predictions_as_logits(det_outputs_valid, det_single_lead_outputs_valid)
        predicted_test = get_all_predictions_as_probs(det_outputs_test, det_single_lead_outputs_test) \
            if not use_logits else get_all_predictions_as_logits(det_outputs_test, det_single_lead_outputs_test)

        fold_eval_class_wise, fold_eval_single_metrics = train_ML_model(X_train=predicted_train,
                                                                        X_valid=predicted_valid,
                                                                        X_test=predicted_test,
                                                                        # Ground Truth
                                                                        y_train=det_targets_train,
                                                                        y_valid=det_targets_valid,
                                                                        y_test=det_targets_test,
                                                                        # Save Path
                                                                        save_path=config.save_dir,
                                                                        # Stratgey
                                                                        strategy=strategy,
                                                                        individual_features=individual_features,
                                                                        reduced_individual_features=reduced_individual_features)

        # Class-Wise Metrics
        test_results_class_wise.loc[(folds[k], fold_eval_class_wise.index), fold_eval_class_wise.columns] = \
            fold_eval_class_wise.values
        # Single Metrics
        pd_series = fold_eval_single_metrics.loc['value']
        pd_series.name = folds[k]
        test_results_single_metrics = test_results_single_metrics.append(pd_series)

        # Update the indices and reset the config (including resume!)
        valid_fold_index = (valid_fold_index + 1) % total_num_folds
        test_fold_index = (test_fold_index + 1) % total_num_folds
        config = copy.deepcopy(base_config)

    # Summarize the results of the cross validation and write everything to file
    # --------------------------- Test Class-Wise Metrics ---------------------------
    iterables_summary = [["mean", "median"], class_wise_metrics]
    multi_index = pd.MultiIndex.from_product(iterables_summary, names=["merging", "metric"])
    test_results_class_wise_summary = pd.DataFrame(
        columns=['SNR', 'AF', 'IAVB', 'LBBB', 'RBBB', 'PAC', 'VEB', 'STD', 'STE',
                 'macro avg', 'weighted avg'], index=multi_index)
    for metric in class_wise_metrics:
        test_results_class_wise_summary.loc[('mean', metric)] = test_results_class_wise.xs(metric, level=1).mean()
        test_results_class_wise_summary.loc[('median', metric)] = test_results_class_wise.xs(metric,
                                                                                             level=1).median()

    path = os.path.join(base_save_dir, "test_results_class_wise.p")
    with open(path, 'wb') as file:
        pickle.dump(test_results_class_wise, file)
    path = os.path.join(base_save_dir, "test_results_class_wise_summary.p")
    with open(path, 'wb') as file:
        pickle.dump(test_results_class_wise_summary, file)

    # --------------------------- Test Single Metrics ---------------------------
    test_results_single_metrics.loc['mean'] = test_results_single_metrics.mean()
    test_results_single_metrics.loc['median'] = test_results_single_metrics[:][:-1].median()

    path = os.path.join(base_save_dir, "test_results_single_metrics.p")
    with open(path, 'wb') as file:
        pickle.dump(test_results_single_metrics, file)

    #  --------------------------- Omit Train Result ---------------------------

    # --------------------------- Test Metrics To Latex---------------------------
    with open(os.path.join(base_save_dir, 'cross_validation_results.tex'), 'w') as file:
        file.write("Class-Wise Summary:\n\n")
        test_results_class_wise_summary.to_latex(buf=file, index=False, float_format="{:0.3f}".format,
                                                 escape=False)
        file.write("\n\n\nSingle Metrics:\n\n")
        test_results_single_metrics.to_latex(buf=file, index=False, float_format="{:0.3f}".format,
                                             escape=False)
    print(f"Finished additional run of cross-fold-validation to train {strategy_name} models")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MACRO Paper: Strategy-based Evaluation')
    parser.add_argument('-p', '--path', default=None, type=str,
                        help='main path to CV runs(default: None)')
    parser.add_argument('--strategy', default=None, type=str,
                        help='strategy to use for final model prediction (default: None)')
    parser.add_argument('--use_logits', action='store_true',
                        help='Use the raw logits for the training of the ML models')
    parser.add_argument('--individual_features', action='store_true',
                        help='Use the individual features per class for the training of the ML models')
    parser.add_argument('--reduced_individual_features', action='store_true',
                        help='Use the reduced individual features per class for the training of the ML models '
                             'but omit the multibranch features')
    args = parser.parse_args()
    train_ML_models_on_cross_fold_data(args.path, args.strategy, args.use_logits,
                                       args.individual_features, args.reduced_individual_features)
