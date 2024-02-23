import argparse
import copy
import pickle
import pandas as pd
from pathlib import Path

import torch

import global_config
from logger import update_logging_setup_for_tune_or_cross_valid

from retrieve_detached_cross_fold_tensors import prepare_model_for_inference, load_config_and_setup_paths, \
    setup_cross_fold, prepare_result_data_structures
from train_ML_models_with_detached_tensors import _convert_logits_to_prediction, get_all_binary_predictions
from utils import ensure_dir

from sklearn.metrics import multilabel_confusion_matrix, \
    accuracy_score, roc_auc_score, f1_score, precision_score, \
    recall_score, classification_report

# Needed for working with SSH Interpreter...
import os

os.environ["CUDA_VISIBLE_DEVICES"] = global_config.CUDA_VISIBLE_DEVICES


def determine_final_model_output(det_outputs, det_single_lead_outputs, strategy=None):
    if det_single_lead_outputs is not None:
        match strategy:
            case "simple_majority_vote":
                # Build a simple majority vote for the single lead outputs and the final model output
                # For this, convert the logits to predictions for each single lead
                # and then take the majority vote
                all_predictions = get_all_binary_predictions(det_outputs, det_single_lead_outputs)
                # Calculate the majority vote
                majority_vote, _ = torch.mode(all_predictions, dim=1)
                return majority_vote

                # Notes: This approach does not seem to work well for mintory classes like STE.
                # Example:
                # STE_class_labels = det_targets[:, -2]
                # STE_label_entries_with_one = torch.where(STE_class_labels==1) liefert
                # tensor([9, 19, 38, 64, 70, 78, 105, 114, 134, 149, 164, 203, 221, 260, 383, 511, 579, 603, 607, 610])
                # ABER:
                # Für das erste STE Sample mit Num 9:
                # single_branch_probs[9][7] => [0.0062, 0.0008, 0.0289, 0.0005, 0.0260, 0.1655, 0.3347, 0.4444, 0.0011]
                # multibranch_probs[9][7] => 0.8440

            case "confidence_weighted_majority_vote":
                # Build a confidence weighted majority vote for the single lead outputs and the final model output
                pass
            case "LASSO_regression":
                # Train a LASSO model to predict the final model output based on the single lead and final outputs
                # Check if the LASSO model is already trained
                # PASS
                # Otherwise, train the LASSO model

                pass
            case None:
                # Default: No strategy given, just use the final model output for predictions
                # This means, the outputs of the BranchNets are ignored
                return _convert_logits_to_prediction(det_outputs)
    else:
        # Default: No multi-branched model, just use the final model output for predictions
        return _convert_logits_to_prediction(det_outputs)


def run_evaluation_on_given_fold(config,
                                 dataset,
                                 strategy=None,
                                 k_fold=None,
                                 data_type=None):
    if data_type is not None:
        assert data_type in ["train", "valid", "test"], "Data type must be one of 'train', 'valid' or 'test'!"

    assert config["metrics"]["additional_metrics_args"].get("logits", False), \
        "The detached tensors should be raw lagits!"

    if config['arch']['args']['multi_label_training']:
        import evaluation.multi_label_metrics as module_metric
    else:
        raise NotImplementedError(
            "Single label metrics haven't been checked after the Python update! Do not use them!")
        import evaluation.single_label_metrics as module_metric

    logger = config.get_logger('run_evaluation_on_fold_' + str(k_fold) + f'_{data_type}')
    device, model = prepare_model_for_inference(config, logger)

    class_labels = dataset.class_labels

    # Store potential parameters needed for metrics
    _param_dict = {
        "labels": class_labels,
        "device": device,
        "sigmoid_probs": config["metrics"]["additional_metrics_args"].get("sigmoid_probs", False),
        "log_probs": config["metrics"]["additional_metrics_args"].get("log_probs", False),
        "logits": config["metrics"]["additional_metrics_args"].get("logits", False),
        # "pos_weights": data_loader.dataset.get_ml_pos_weights(
        #     idx_list=list(range(len(data_loader.sampler))),
        #     mode='test',
        #     cross_valid_active=True),
        # "class_weights": data_loader.dataset.get_inverse_class_frequency(
        #     idx_list=list(range(len(data_loader.sampler))),
        #     multi_label_training=multi_label_training,
        #     mode='test',
        #     cross_valid_active=True),
        "lambda_balance": config["loss"]["add_args"].get("lambda_balance", 1),
        "gamma_neg": config["loss"]["add_args"].get("gamma_neg", 4),
        "gamma_pos": config["loss"]["add_args"].get("gamma_pos", 1),
        "clip": config["loss"]["add_args"].get("clip", 0.05),
    }

    # Load the tensors for further evaluation
    with open(os.path.join(config.save_dir, f"{data_type}_det_outputs.p"), 'rb') as file:
        det_outputs = pickle.load(file)
    with open(os.path.join(config.save_dir, f"{data_type}_det_targets.p"), 'rb') as file:
        det_targets = pickle.load(file)
    if config['arch']['type'] == 'FinalModelMultiBranch':
        with open(os.path.join(config.save_dir, f"{data_type}_det_single_lead_outputs.p"), 'rb') as file:
            det_single_lead_outputs = pickle.load(file)
    else:
        det_single_lead_outputs = None

    # Transfer the detached tensors to a final model prediction based on the given strategy
    model_outputs = determine_final_model_output(det_outputs, det_single_lead_outputs, strategy)

    # ------------ Metrics can be calculated here ------------------------------------
    # Metrics of interest: Subset Accuracy, ROC AUC
    # F1, ROC AUC, Accuracy
    # Eventually: Precision, Recall

    # ------------ ROC Plots can be created here------------------------------------

    # ------------ Confusion matrices can be determined here------------------------

    # ------------------- Summary Report can be created here -------------------
    summary_dict = classification_report(y_true=det_targets, y_pred=model_outputs,
                                         labels=_param_dict["labels"],
                                         digits=3,
                                         target_names=["IAVB", "AF", "LBBB", "PAC", "RBBB", "SNR", "STD", "STE", "VEB"],
                                         output_dict=True)

    summary_dict = module_metric.sk_classification_summary(output=det_outputs, target=det_targets,
                                                           sigmoid_probs=_param_dict["sigmoid_probs"],
                                                           logits=_param_dict["logits"],
                                                           labels=_param_dict["labels"],
                                                           output_dict=True)

    df_sklearn_summary = pd.DataFrame.from_dict(summary_dict)
    #
    # df_class_wise_results = pd.DataFrame(
    #     columns=['IAVB', 'AF', 'LBBB', 'PAC', 'RBBB', 'SNR', 'STD', 'STE', 'VEB', 'macro avg', 'weighted avg'])
    # df_class_wise_results = pd.concat([df_class_wise_results, df_sklearn_summary[
    #     ['IAVB', 'AF', 'LBBB', 'PAC', 'RBBB', 'SNR', 'STD', 'STE', 'VEB', 'macro avg', 'weighted avg']]])
    #
    # # Reorder the class columns of the dataframe to match the one used in the
    # desired_col_order = ['SNR', 'AF', 'IAVB', 'LBBB', 'RBBB', 'PAC', 'VEB', 'STD', 'STE', 'macro avg',
    #                      'weighted avg']
    # df_class_wise_results = df_class_wise_results[desired_col_order]

    # with open(os.path.join(config.test_output_dir, 'eval_results.tex'), 'w') as file:
    #     df_class_wise_results.to_latex(buf=file, index=True, bold_rows=True, float_format="{:0.3f}".format)
    # df_single_metric_results.to_latex(buf=file, index=True, bold_rows=True, float_format="{:0.3f}".format)

    print("Finished fold " + str(k_fold) + " " + data_type + " evaluation")


def run_evaluation_on_cross_fold_data(main_path, strategy=None):
    config = load_config_and_setup_paths(main_path, sub_dir="output_logging")

    base_config, base_log_dir, base_save_dir, data_dir, dataset, fold_data, total_num_folds = setup_cross_fold(config)

    # Save the results of each run
    class_wise_metrics, folds, test_results_class_wise, test_results_single_metrics = \
        prepare_result_data_structures(total_num_folds)

    print("Starting with " + str(total_num_folds) + "-fold cross validation")
    valid_fold_index = total_num_folds - 2
    test_fold_index = total_num_folds - 1

    for k in range(total_num_folds):
        print("Starting fold " + str(k + 1))

        # Adapt the log and save paths for the current fold
        config.save_dir = Path(os.path.join(base_save_dir, "Fold_" + str(k + 1)))
        config.log_dir = Path(os.path.join(base_log_dir, "Fold_" + str(k + 1)))
        ensure_dir(config.save_dir)
        ensure_dir(config.log_dir)
        update_logging_setup_for_tune_or_cross_valid(config.log_dir)

        # Skip the training and load the trained model
        config.resume = os.path.join(main_path, "Fold_" + str(k + 1), "model_best.pth")

        # ---------------------------------- CALCULATE METRICS ----------------------------------
        config.test_output_dir = Path(os.path.join(config.save_dir, 'test_output_fold_' + str(k + 1)))
        ensure_dir(config.test_output_dir)
        fold_eval_class_wise, fold_eval_single_metrics = run_evaluation_on_given_fold(config=config,
                                                                                      dataset=dataset,
                                                                                      strategy=strategy,
                                                                                      k_fold=k,
                                                                                      data_type="test")

        # Class-Wise Metrics
        fold_eval_class_wise = fold_eval_class_wise.rename(index={'torch_acc': 'torch_accuracy'})
        test_results_class_wise.loc[(folds[k], fold_eval_class_wise.index), fold_eval_class_wise.columns] = \
            fold_eval_class_wise.values
        # Single Metrics
        pd_series = fold_eval_single_metrics.loc['value']
        pd_series.name = folds[k]
        test_results_single_metrics = test_results_single_metrics.append(pd_series)

        # ----------------------------------END OF METRICS CALCULATION ---------------------------

        # Update the indices and reset the config (including resume!)
        valid_fold_index = (valid_fold_index + 1) % total_num_folds
        test_fold_index = (test_fold_index + 1) % total_num_folds
        config = copy.deepcopy(base_config)

    # Summarize the results of the cross validation and write everything to files

    # --------------------------- Test Class-Wise Metrics ---------------------------
    iterables_summary = [["mean", "median"], class_wise_metrics]
    multi_index = pd.MultiIndex.from_product(iterables_summary, names=["merging", "metric"])
    test_results_class_wise_summary = pd.DataFrame(
        columns=['SNR', 'AF', 'IAVB', 'LBBB', 'RBBB', 'PAC', 'VEB', 'STD', 'STE',
                 'macro avg', 'weighted avg'], index=multi_index)
    for metric in class_wise_metrics:
        test_results_class_wise_summary.loc[('mean', metric)] = test_results_class_wise.xs(metric, level=1).mean()
        test_results_class_wise_summary.loc[('median', metric)] = test_results_class_wise.xs(metric, level=1).median()

    ensure_dir(os.path.join(base_save_dir, "additional_eval"))
    path = os.path.join(base_save_dir, "additional_eval", "test_results_class_wise.p")
    with open(path, 'wb') as file:
        pickle.dump(test_results_class_wise, file)
    path = os.path.join(base_save_dir, "additional_eval", "test_results_class_wise_summary.p")
    with open(path, 'wb') as file:
        pickle.dump(test_results_class_wise_summary, file)

    # --------------------------- Test Single Metrics ---------------------------
    test_results_single_metrics.loc['mean'] = test_results_single_metrics.mean()
    test_results_single_metrics.loc['median'] = test_results_single_metrics[:][:-1].median()

    path = os.path.join(base_save_dir, "additional_eval", "test_results_single_metrics.p")
    with open(path, 'wb') as file:
        pickle.dump(test_results_single_metrics, file)

    #  --------------------------- Omit Train Result ---------------------------

    # --------------------------- Test Metrics To Latex---------------------------
    with open(os.path.join(base_save_dir, "additional_eval", 'cross_validation_results.tex'), 'w') as file:
        file.write("Class-Wise Summary:\n\n")
        test_results_class_wise_summary.to_latex(buf=file, index=False, float_format="{:0.3f}".format,
                                                 escape=False)
        file.write("\n\n\nSingle Metrics:\n\n")
        test_results_single_metrics.to_latex(buf=file, index=False, float_format="{:0.3f}".format,
                                             escape=False)

    print("Finished additional run of cross-fold-validation to do the evaluation")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MACRO Paper: Strategy-based Evaluation')
    parser.add_argument('-p', '--path', default=None, type=str,
                        help='main path to CV runs(default: None)')
    parser.add_argument('--strategy', default=None, type=str,
                        help='strategy to use for final model prediction (default: None)')
    parser.add_argument('--use_logits', action='store_true',
                        help='Use the raw logits for the training of the ML models')
    parser.add_argument('--individual_features', action='store_true',
                        help='Use the raw logits for the training of the ML models')
    args = parser.parse_args()
    run_evaluation_on_cross_fold_data(args.path, args.strategy)
