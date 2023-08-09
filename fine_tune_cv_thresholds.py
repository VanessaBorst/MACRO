import argparse
import collections
import copy
import csv

import time
import os
import pickle
import random
import pandas as pd
from pathlib import Path

import numpy as np

from data_loader.ecg_data_set import ECGDataset
from fine_tune_thresholds_on_valid_set import fine_tune_thresholds
from logger import update_logging_setup_for_tune_or_cross_valid

from parse_config import ConfigParser
from test import test_model_with_threshold


import torch

from utils import ensure_dir


def _set_seed(SEED):
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    # VB: Replaced by use_deterministic_algorithms, which will make more PyTorch operations behave deterministically
    # See https://pytorch.org/docs/stable/notes/randomness.html
    torch.backends.cudnn.deterministic = True
    # torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False

    random.seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    # os.environ['PYTHONHASHSEED'] = str(SEED)


# fix random seeds for reproducibility
SEED = 123
_set_seed(SEED)


def test_fold_with_thresholds(config, data_dir, test_idx, k_fold, thresholds):
    df_class_wise_results, df_single_metric_results = test_model_with_threshold(config, cv_data_dir=data_dir,
                                                                                test_idx=test_idx, k_fold=k_fold,
                                                                                thresholds=thresholds)
    return df_class_wise_results, df_single_metric_results



def fine_tune_thresholds_cross_validation(config):
    k_fold = config["data_loader"]["cross_valid"]["k_fold"]
    data_dir = config["data_loader"]["cross_valid"]["data_dir"]
    # Update data dir in Dataloader ARGs!
    config["data_loader"]["args"]["data_dir"] = data_dir
    dataset = ECGDataset(data_dir)
    n_samples = len(dataset)
    idx_full = np.arange(n_samples)
    np.random.shuffle(idx_full)

    # Get the main config and the dirs for logging and saving checkpoints
    base_config = copy.deepcopy(config)
    base_log_dir = config.log_dir
    base_save_dir = config.save_dir

    # Path("savedVM_v2/models/FinalModel/cross_validation_rerun_withFC_0.3_32_16")
    # Path("savedVM_v2/models/FinalModel/final_cross_validation")
    # Path("savedVM_v2/models/FinalModel/cross_validation_rerun_withFC_0.2_12_8")
    # Path("savedVM_v2/models/FinalModel/cross_validation_rerun_noFC_0.2_24_8")
    # Path("savedVM_v2/models/FinalModel/final_cross_validation_sqrtT")
    # models_path = Path("savedVM_v2/models/FinalModel/final_cross_validation_sqrtT")

    models_path = Path("savedVM/models/FinalModel_MACRO/0804_171418_ml_bs64_cross_validation_MACRO_withFC_0.3_32_16")

    # Divide the samples into k distinct sets
    fold_size = n_samples // k_fold
    fold_data = []
    lower_limit = 0
    for i in range(0, k_fold):
        if i != k_fold - 1:
            fold_data.append(idx_full[lower_limit:lower_limit + fold_size])
            lower_limit = lower_limit + fold_size
        else:
            # Last fold may be a bit larger
            fold_data.append(idx_full[lower_limit:n_samples])

    # Run k-fold-cross-validation
    # Each time, one of the subset functions as valid and another as test set

    # Save the results of each run
    class_wise_metrics = ["precision", "recall", "f1-score", "torch_roc_auc", "torch_acc", "support"]
    folds = ['fold_' + str(i) for i in range(1, k_fold + 1)]

    iterables = [folds, class_wise_metrics]
    multi_index = pd.MultiIndex.from_product(iterables, names=["fold", "metric"])

    valid_results = pd.DataFrame(columns=folds)
    test_results_class_wise = pd.DataFrame(columns=['SNR', 'AF', 'IAVB', 'LBBB', 'RBBB', 'PAC', 'VEB', 'STD', 'STE',
                                                    'macro avg', 'weighted avg'], index=multi_index)

    test_results_single_metrics = pd.DataFrame(columns=['loss', 'sk_subset_accuracy', 'cpsc_F1', 'cpsc_Faf',
                                                        'cpsc_Fblock', 'cpsc_Fpc', 'cpsc_Fst'])

    print("Finetuning " + str(k_fold) + "-fold cross validation")
    start = time.time()
    valid_fold_index = k_fold - 2
    test_fold_index = k_fold - 1
    for k in range(k_fold):
        # Get the idx for valid and test samples, train idx not needed
        train_sets = [fold for id, fold in enumerate(fold_data)
                      if id != valid_fold_index and id != test_fold_index]
        train_idx = np.concatenate(train_sets, axis=0)
        # valid_idx = fold_data[valid_fold_index]
        valid_idx = np.concatenate([fold_data[valid_fold_index], train_idx], axis=0)
        test_idx = fold_data[test_fold_index]

        print("Starting fold " + str(k))
        print("Valid Set: " + str(valid_fold_index) + ", Test Set: " + str(test_fold_index))

        # Adapt the log and save paths for the current fold
        config.save_dir = Path(os.path.join(base_save_dir, "Fold_" + str(k + 1)))
        config.log_dir = Path(os.path.join(base_log_dir, "Fold_" + str(k + 1)))
        ensure_dir(config.save_dir)
        ensure_dir(config.log_dir)
        update_logging_setup_for_tune_or_cross_valid(config.log_dir)

        # Write record names to pickle for reproducing single folds
        dict = {
            "valid_records": np.array(dataset.records)[valid_idx],
            "test_records": np.array(dataset.records)[test_idx]
        }
        with open(os.path.join(config.save_dir, "data_split.csv"), "w") as file:
            pd.DataFrame.from_dict(data=dict, orient='index').to_csv(file, header=False)

        # Skip training!

        # Find the best thresholds on the validation set
        config.resume = Path(os.path.join(models_path, "Fold_" + str(k + 1), "model_best.pth"))
        thresholds = fine_tune_thresholds(config, cv_data_dir=data_dir, valid_idx=valid_idx, k_fold=k)
        # [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        with open(os.path.join(config.save_dir, "thresholds.csv"), "w") as file:
            wr = csv.writer(file, quoting=csv.QUOTE_ALL)
            wr.writerow(thresholds)
            # pd.DataFrame.from_dict(data=thresholds, orient='index').to_csv(file, header=False)


        # Do the testing with the fine_tuned thresholds and add the fold results to the dfs
        config.test_output_dir = Path(os.path.join(config.save_dir, 'test_output_fold_' + str(k + 1)))
        ensure_dir(config.test_output_dir)
        fold_eval_class_wise, fold_eval_single_metrics = test_fold_with_thresholds(config, data_dir=data_dir,
                                                                                   test_idx=test_idx,
                                                                                   k_fold=k, thresholds=thresholds)
        # Class-Wise Metrics
        test_results_class_wise.loc[(folds[k], fold_eval_class_wise.index), fold_eval_class_wise.columns] = \
            fold_eval_class_wise.values
        # Single Metrics
        pd_series = fold_eval_single_metrics.loc['value']
        pd_series.name = folds[k]
        test_results_single_metrics = test_results_single_metrics.append(pd_series)

        # Update the indices and reset the config (including resume!)
        valid_fold_index = (valid_fold_index + 1) % (k_fold)
        test_fold_index = (test_fold_index + 1) % (k_fold)
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
    args = argparse.ArgumentParser(description='MA Vanessa')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-t', '--tune', action='store_true', help='Use to enable tuning')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
        # options added here can be modified by command line flags.
    ]
    config = ConfigParser.from_args(args=args, options=options)
    assert config["data_loader"]["cross_valid"]["enabled"], "Cross-valid should be enabled when running this script"
    fine_tune_thresholds_cross_validation(config)
