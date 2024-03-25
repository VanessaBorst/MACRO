import argparse
import copy

import time
import pickle
import pandas as pd
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

import global_config
from data_loader.ecg_data_set import ECGDataset
from logger import update_logging_setup_for_tune_or_cross_valid

from parse_config import ConfigParser
from test import test_model
from train import _set_seed
from utils import ensure_dir, read_json
import evaluation.multi_label_metrics as module_metric

import data_loader.data_loaders as module_data

# Needed for working with SSH Interpreter...
import os
import torch
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = global_config.CUDA_VISIBLE_DEVICES

def run_inference_on_given_fold_data(config, cv_data_dir=None,
                                     test_idx=None,
                                     k_fold=None,
                                     data_type=None):


    logger = config.get_logger('run_inference_on_fold_' + str(k_fold) + f'_{data_type}')

    data_loader = getattr(module_data, config['data_loader']['type'])(
        cv_data_dir,
        batch_size=64,
        shuffle=False,
        validation_split=0.0,
        num_workers=4,
        cross_valid=True,
        test_idx=test_idx,
        cv_train_mode=False,
        fold_id=k_fold
    )

    device, model = prepare_model_for_inference(config, logger)

    with torch.no_grad():
        # Store the intermediate targets. Always store the output scores
        outputs_list = []
        targets_list = []
        single_lead_outputs_list = []

        for batch_idx, (padded_records, _, first_labels, labels_one_hot, record_names) in \
                enumerate(tqdm(data_loader)):
            data, target = padded_records.to(device), labels_one_hot.to(device)
            data = data.permute(0, 2, 1)  # switch seq_len and feature_size (12 = #leads)
            output = model(data)
            multi_lead_branch_active = False
            if type(output) is tuple:
                if isinstance(output[1], list):
                    # multi-branch network
                    # first element is the overall network output, the second one a list of the single lead branches
                    multi_lead_branch_active = True
                    output, single_lead_outputs = output
                    detached_single_lead_outputs = torch.stack(single_lead_outputs).detach().cpu()  # Shape: [12, 64, 9]
                    single_lead_outputs_list.append(detached_single_lead_outputs)
                else:
                    # single-branch network
                    output, attention_weights = output

            # Detach tensors needed for further tracing and metrics calculation to remove them from the graph
            detached_output = output.detach().cpu()
            detached_target = target.detach().cpu()
            outputs_list.append(detached_output)
            targets_list.append(detached_target)

            # INNER_EPOCH METRICS COULD BE CALCULATED WITH DETACHED TENSORS HERE

    # Get detached tensors from the list for further evaluation
    # For this, create a tensor from the dynamically filled list
    det_outputs = torch.cat(outputs_list).detach().cpu()
    det_targets = torch.cat(targets_list).detach().cpu()
    det_single_lead_outputs = torch.cat(([branch_outputs for branch_outputs in single_lead_outputs_list]), dim=1)\
        .permute(1,0,2).detach().cpu() if multi_lead_branch_active else None

    # Store the tensors for further evaluation
    with open(os.path.join(config.save_dir, f"{data_type}_det_outputs.p"), 'wb') as file:
        pickle.dump(det_outputs, file)
    with open(os.path.join(config.save_dir, f"{data_type}_det_targets.p"), 'wb') as file:
        pickle.dump(det_targets, file)
    if det_single_lead_outputs is not None:
        with open(os.path.join(config.save_dir, f"{data_type}_det_single_lead_outputs.p"), 'wb') as file:
            pickle.dump(det_single_lead_outputs, file)

    # ------------ Metrics could be calculated here ------------------------------------

    # ------------ ROC Plots could be created here------------------------------------

    # ------------ Confusion matrices could be determined here------------------------

    # ------------------- Summary Report -------------------
    # summary_dict = module_metric.sk_classification_summary(output=det_outputs, target=det_targets,
    #                                                        logits=_param_dict["logits"],
    #                                                        labels=_param_dict["labels"],
    #                                                        output_dict=True)
    #
    # df_sklearn_summary = pd.DataFrame.from_dict(summary_dict)

    print("Finished fold " + str(k_fold+1) + " " + data_type + " inference")


def prepare_model_for_inference(config, logger):

    # Conditional inputs depending on the config
    if config['arch']['type'] == 'BaselineModel':
        import model.baseline_model as module_arch
    elif config['arch']['type'] == 'BaselineModelWithMHAttentionV2':
        import model.baseline_model_with_MHAttention_v2 as module_arch
    elif config['arch']['type'] == 'BaselineModelWithSkipConnectionsAndNormV2PreActivation':
        import model.baseline_model_with_skips_and_norm_v2_pre_activation_design as module_arch
    elif config['arch']['type'] == 'FinalModel':
        import model.final_model as module_arch
    elif config['arch']['type'] == 'FinalModelMultiBranch':
        import model.final_model_multibranch as module_arch

    model = config.init_obj('arch', module_arch)
    logger.info(model)
    # Load the model from the checkpoint
    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume,
                            map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)
    # Prepare the model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    return device, model


def retrieve_detached_cross_fold_tensors_for_further_evaluation(main_path):
    config = load_config_and_setup_paths(main_path, sub_dir="output_logging")

    base_config, base_log_dir, base_save_dir, data_dir, dataset, fold_data, total_num_folds = setup_cross_fold(config)

    print("Starting with " + str(total_num_folds) + "-fold cross validation")
    valid_fold_index = total_num_folds - 2
    test_fold_index = total_num_folds - 1

    for k in range(total_num_folds):
        print("Starting fold " + str(k + 1))
        # Get the idx for valid and test samples, train idx not needed
        train_idx, valid_idx, test_idx = get_train_valid_test_indices(main_path=main_path,
                                                                      dataset=dataset,
                                                                      fold_data=fold_data,
                                                                      k=k,
                                                                      test_fold_index=test_fold_index,
                                                                      valid_fold_index=valid_fold_index,
                                                                      merge_train_into_valid_idx=False)

        # Adapt the log and save paths for the current fold
        config.save_dir = Path(os.path.join(base_save_dir, "Fold_" + str(k + 1)))
        config.log_dir = Path(os.path.join(base_log_dir, "Fold_" + str(k + 1)))
        ensure_dir(config.save_dir)
        ensure_dir(config.log_dir)
        update_logging_setup_for_tune_or_cross_valid(config.log_dir)

        # Skip the training and load the trained model
        config.resume = os.path.join(main_path, "Fold_" + str(k + 1), "model_best.pth")


        run_inference_on_given_fold_data(config=config,
                                         cv_data_dir=data_dir,
                                         test_idx=train_idx,
                                         k_fold=k,
                                         data_type="train")

        run_inference_on_given_fold_data(config=config,
                                         cv_data_dir=data_dir,
                                         test_idx=valid_idx,
                                         k_fold=k,
                                         data_type="valid")

        run_inference_on_given_fold_data(config=config,
                                         cv_data_dir=data_dir,
                                         test_idx=test_idx,
                                         k_fold=k,
                                         data_type="test")

        # Update the indices and reset the config (including resume!)
        valid_fold_index = (valid_fold_index + 1) % (total_num_folds)
        test_fold_index = (test_fold_index + 1) % (total_num_folds)
        config = copy.deepcopy(base_config)

    print("Finished additional run of cross-fold-validation to retrieve detached tensors for further evaluation")


def setup_cross_fold(config):
    # Get the number of folds and the data dir
    total_num_folds = config["data_loader"]["cross_valid"]["k_fold"]
    data_dir = config["data_loader"]["cross_valid"]["data_dir"]

    # Update data dir in Dataloader ARGs!
    config["data_loader"]["args"]["data_dir"] = data_dir
    dataset = ECGDataset(data_dir)
    n_samples = len(dataset)

    # Get the main config and the dirs for logging and saving checkpoints
    base_config = copy.deepcopy(config)
    base_save_dir = config.save_dir
    base_log_dir = config.log_dir

    # Divide the samples into k distinct sets
    fold_data = split_dataset_into_folds(n_samples=n_samples, total_num_folds=total_num_folds)

    # Return everything
    return base_config, base_log_dir, base_save_dir, data_dir, dataset, fold_data, total_num_folds


def prepare_result_data_structures(total_num_folds, include_valid_results=False):
    class_wise_metrics = ["precision", "recall", "f1-score", "torch_roc_auc", "torch_accuracy", "support"]
    folds = ['fold_' + str(i) for i in range(1, total_num_folds + 1)]
    iterables = [folds, class_wise_metrics]
    multi_index = pd.MultiIndex.from_product(iterables, names=["fold", "metric"])
    if include_valid_results:
        valid_results = pd.DataFrame(columns=folds)
    test_results_class_wise = pd.DataFrame(columns=['SNR', 'AF', 'IAVB', 'LBBB', 'RBBB', 'PAC', 'VEB', 'STD', 'STE',
                                                    'macro avg', 'weighted avg'], index=multi_index)
    test_results_single_metrics = pd.DataFrame(columns=['loss', 'sk_subset_accuracy'])
    if not include_valid_results:
        return class_wise_metrics, folds, test_results_class_wise, test_results_single_metrics
    else:
        return class_wise_metrics, folds, test_results_class_wise, test_results_single_metrics, valid_results


def load_config_and_setup_paths(main_path, sub_dir=None):
    # Load the main config
    config = read_json(os.path.join(main_path, "config.json"))
    config = ConfigParser(config=config, create_save_log_dir=False)

    # Update paths to old paths (otherwise set to current date)
    config.save_dir = Path(main_path) if sub_dir is None else os.path.join(Path(main_path), sub_dir)
    log_dir = Path(main_path.replace("models", "log"))
    config.log_dir = log_dir if sub_dir is None else os.path.join(log_dir, sub_dir)

    assert config["data_loader"]["cross_valid"]["enabled"], "Cross-valid should be enabled when running this script"

    # fix random seeds for reproducibility
    global_config.SEED = config.config.get("SEED", global_config.SEED)
    _set_seed(global_config.SEED)

    return config


def get_train_valid_test_indices(main_path, dataset, fold_data, k, test_fold_index, valid_fold_index,
                                 merge_train_into_valid_idx=False):
    train_sets = [fold for id, fold in enumerate(fold_data)
                  if id != valid_fold_index and id != test_fold_index]
    train_idx = np.concatenate(train_sets, axis=0)
    valid_idx = fold_data[valid_fold_index]
    test_idx = fold_data[test_fold_index]
    # Load the old data split for sanity check
    with open(os.path.join(main_path, "Fold_" + str(k + 1), "data_split.csv"), "r") as file:
        data = pd.read_csv(file, index_col=0, header=None)
        dict = {
            "train_records": data.loc["train_records"],
            "valid_records": data.loc["valid_records"].dropna(),
            "test_records": data.loc["test_records"].dropna()
        }
    assert np.array(dataset.records)[train_idx].tolist() == dict["train_records"].tolist(), \
        "Train indices do not match the old data split"
    assert np.array(dataset.records)[valid_idx].tolist() == dict["valid_records"].tolist(), \
        "Valid indices do not match the old data split"
    assert np.array(dataset.records)[test_idx].tolist() == dict["test_records"].tolist(), \
        "Test indices do not match the old data split"

    if merge_train_into_valid_idx:
        valid_idx = np.concatenate([fold_data[valid_fold_index], train_idx], axis=0)

    return train_idx, valid_idx, test_idx


def split_dataset_into_folds(n_samples, total_num_folds):
    idx_full = np.arange(n_samples)
    np.random.shuffle(idx_full)

    fold_size = n_samples // total_num_folds
    fold_data = []
    lower_limit = 0
    for i in range(0, total_num_folds):
        if i != total_num_folds - 1:
            fold_data.append(idx_full[lower_limit:lower_limit + fold_size])
            lower_limit = lower_limit + fold_size
        else:
            # Last fold may be a bit larger
            fold_data.append(idx_full[lower_limit:n_samples])
    return fold_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MACRO Paper: Cross-Validation Evaluation')
    parser.add_argument('-p', '--path', default=None, type=str,
                        help='main path to CV runs(default: None)')
    args = parser.parse_args()
    retrieve_detached_cross_fold_tensors_for_further_evaluation(args.path)
