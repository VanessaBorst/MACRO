import argparse
import collections
import copy

import time
import pickle
import pandas as pd
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import LabelEncoder
from collections import Counter

import global_config
from data_loader.ecg_data_set import ECGDataset
from logger import update_logging_setup_for_tune_or_cross_valid

from parse_config import ConfigParser
from test import test_model
from train import train_model, _set_seed
from utils import ensure_dir, read_json
# Needed for working with SSH Interpreter...
import os

os.environ["CUDA_VISIBLE_DEVICES"] = global_config.CUDA_VISIBLE_DEVICES


def test_fold(config, data_dir, test_idx, k_fold, total_num_folds):
    df_class_wise_results, df_single_metric_results = test_model(config, tune_config=None, cv_active=True,
                                                                 cv_data_dir=data_dir, test_idx=test_idx,
                                                                 k_fold=k_fold, total_num_folds=total_num_folds)
    return df_class_wise_results, df_single_metric_results
#
#
# def create_roc_curve_report(X, y, model, model_name):
#     try:
#         tprs = []
#         aucs = []
#         mean_fpr = np.linspace(0, 1, 100)
#         i = 0
#         fig, ax = plt.subplots()
#         for i, (train, test) in enumerate(cv.split(X, y)):
#             model.fit(X[train], y[train])
#             viz = plot_roc_curve(model, X[test], y[test],
#                                  name='ROC fold {}'.format(i),
#                                  alpha=0.3, lw=1, ax=ax)
#             interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
#             interp_tpr[0] = 0.0
#             tprs.append(interp_tpr)
#             aucs.append(viz.roc_auc)
#
#         ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
#                 label='Chance', alpha=.8)
#
#         mean_tpr = np.mean(tprs, axis=0)
#         mean_tpr[-1] = 1.0
#         mean_auc = auc(mean_fpr, mean_tpr)
#         std_auc = np.std(aucs)
#         ax.plot(mean_fpr, mean_tpr, color='b',
#                 label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
#                 lw=2, alpha=.8)
#
#         std_tpr = np.std(tprs, axis=0)
#         tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
#         tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
#         ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
#                         label=r'$\pm$ 1 std. dev.')
#
#         ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
#                title=model_name)
#         ax.legend(loc="lower right")
#         plt.savefig(
#             "../results/plots/test/roc_%s_%0i_features_%0i_test.pdf" % (
#                 model_name, feature_size, len(X)),
#             dpi=100, facecolor='w', edgecolor='b', orientation='portrait', transparent=False, bbox_inches=None,
#             pad_inches=0.1)
#         print("Created %s ROC figure" % model_name)
#         plt.close()
#     except (AttributeError, OverflowError) as detail:
#         print(model_name + " Failed due to ", detail)

def evaluate_cross_validation(main_path):
    # Load the main config
    config = read_json(os.path.join(main_path, "config.json"))
    config = ConfigParser(config=config, create_save_log_dir=False)
    # Update paths to old paths (otherwise set to current date)
    config.save_dir = Path(main_path)
    config.log_dir = Path(main_path)

    assert config["data_loader"]["cross_valid"]["enabled"], "Cross-valid should be enabled when running this script"
    # fix random seeds for reproducibility
    global_config.SEED = config.config.get("SEED", global_config.SEED)
    _set_seed(global_config.SEED)

    total_num_folds = config["data_loader"]["cross_valid"]["k_fold"]
    data_dir = config["data_loader"]["cross_valid"]["data_dir"]

    # Get the main config and the dirs for logging and saving checkpoints
    base_config = copy.deepcopy(config)
    base_save_dir = config.save_dir
    base_log_dir = config.log_dir

    # Run k-fold-cross-validation for evaluation
    # Save the results of each run
    class_wise_metrics = ["precision", "recall", "f1-score", "torch_roc_auc", "torch_accuracy", "support"]
    folds = ['fold_' + str(i) for i in range(1, total_num_folds + 1)]

    iterables = [folds, class_wise_metrics]
    multi_index = pd.MultiIndex.from_product(iterables, names=["fold", "metric"])

    test_results_class_wise = pd.DataFrame(columns=['SNR', 'AF', 'IAVB', 'LBBB', 'RBBB', 'PAC', 'VEB', 'STD', 'STE',
                                                    'macro avg', 'weighted avg'], index=multi_index)

    test_results_single_metrics = pd.DataFrame(columns=['loss', 'sk_subset_accuracy'])

    print("Starting with " + str(total_num_folds) + "-fold cross validation")

    for k in range(total_num_folds):
        print("Starting fold " + str(k))

        # Adapt the log and save paths for the current fold
        # config.save_dir = Path(os.path.join(base_save_dir, "Fold_" + str(k + 1), "additional_eval"))
        # config.log_dir = Path(os.path.join(base_log_dir, "Fold_" + str(k + 1), "additional_eval"))
        # ensure_dir(config.save_dir)
        # ensure_dir(config.log_dir)
        # update_logging_setup_for_tune_or_cross_valid(config.log_dir)

        # Load the data split
        with open(os.path.join(base_save_dir, "Fold_" + str(k + 1), "data_split.csv"), "r") as file:
            data = pd.read_csv(file, header=1)
            dict = {
                "train_records": data[0],
                "valid_records": data[1],
                "test_records": data[2]
            }
        test_idx = dict["test_records"]

        # Skip the training and load the trained model
        config.resume = os.path.join(base_save_dir, "Fold_" + str(k + 1), "model_best.pth")
        # config.resume = Path(os.path.join(config.save_dir, "model_best.pth"))
        config.test_output_dir = Path(os.path.join(config.resume.parent, 'additional_test_output_fold_' + str(k + 1)))
        ensure_dir(config.test_output_dir)
        fold_eval_class_wise, fold_eval_single_metrics = test_fold(config, data_dir=data_dir, test_idx=test_idx,
                                                                   k_fold=k, total_num_folds=total_num_folds)

        # Class-Wise Metrics
        test_results_class_wise.loc[(folds[k], fold_eval_class_wise.index), fold_eval_class_wise.columns] = \
            fold_eval_class_wise.values
        # Single Metrics
        pd_series = fold_eval_single_metrics.loc['value']
        pd_series.name = folds[k]
        test_results_single_metrics = test_results_single_metrics.append(pd_series)

        # Reset the config (including resume!)
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

    #  --------------------------- Train Result ---------------------------
    valid_results['mean'] = valid_results.mean(axis=1)
    valid_results['median'] = valid_results.median(axis=1)

    path = os.path.join(base_save_dir, "cross_validation_valid_results.p")
    with open(path, 'wb') as file:
        pickle.dump(valid_results, file)

    # --------------------------- Test Metrics To Latex---------------------------
    with open(os.path.join(base_save_dir, 'cross_validation_results.tex'), 'w') as file:
        file.write("Class-Wise Summary:\n\n")
        test_results_class_wise_summary.to_latex(buf=file, index=False, float_format="{:0.3f}".format,
                                                 escape=False)
        file.write("\n\n\nSingle Metrics:\n\n")
        test_results_single_metrics.to_latex(buf=file, index=False, float_format="{:0.3f}".format,
                                             escape=False)

    # Finish everything
    end = time.time()
    ty_res = time.gmtime(end - start)
    res = time.strftime("%H hours, %M minutes, %S seconds", ty_res)

    print("Finished cross-fold-validation")
    print("Consuming time: " + str(res))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MACRO Paper: Cross-Validation Evaluation')
    parser.add_argument('-p', '--path', default=None, type=str,
                        help='main path to CV runs(default: None)')
    args = parser.parse_args()
    evaluate_cross_validation(args.path)
