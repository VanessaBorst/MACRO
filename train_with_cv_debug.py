import argparse
import collections
import copy

import time
import os
import pickle
import random
import pandas as pd
from pathlib import Path

import numpy as np

import global_config
from data_loader.ecg_data_set import ECGDataset
from logger import update_logging_setup_for_tune_or_cross_valid

from parse_config import ConfigParser
from test import test_model
from train import train_model, _set_seed
from train_with_cv import train_fold, test_fold
from utils import ensure_dir
# Needed for working with SSH Interpreter...
import os


os.environ["CUDA_VISIBLE_DEVICES"] = global_config.CUDA_VISIBLE_DEVICES
PROBLEM_FOLD_ID = 4
CROSS_FOLD_LOG_PATH = "/home/vab30xh/projects/2023-macro-paper-3.10/cross_fold_log"


def run_cross_validation_debug(config):
    data_dir = config["data_loader"]["cross_valid"]["data_dir"]

    # Update data dir in Dataloader ARGs!
    config["data_loader"]["args"]["data_dir"] = data_dir
    dataset = ECGDataset(data_dir)
    n_samples = len(dataset)
    idx_full = np.arange(n_samples)
    np.random.shuffle(idx_full)

    valid_idx = []
    with open(os.path.join(CROSS_FOLD_LOG_PATH, "cross_validation_valid_{}.txt".format(PROBLEM_FOLD_ID)), "rb") as file:
        for line in file:
            valid_idx.append(int(line.strip()))
    valid_idx = np.array(valid_idx)

    test_idx = []
    with open(os.path.join(CROSS_FOLD_LOG_PATH, "cross_validation_test_{}.txt".format(PROBLEM_FOLD_ID)), "rb") as file:
        for line in file:
            test_idx.append(int(line.strip()))
    test_idx = np.array(test_idx)

    # Retrieve train_idx as all the files in the base data path that are neither in valid_idx nor test_idx
    train_idx = np.array([idx for idx in idx_full if idx not in valid_idx and idx not in test_idx])

    # Get the main config and the dirs for logging and saving checkpoints
    base_log_dir = config.log_dir
    base_save_dir = config.save_dir

    k = PROBLEM_FOLD_ID - 1
    print("Starting fold " + str(k + 1))

    # Adapt the log and save paths for the current fold
    config.save_dir = Path(os.path.join(base_save_dir, "Fold_" + str(k + 1)))
    config.log_dir = Path(os.path.join(base_log_dir, "Fold_" + str(k + 1)))
    ensure_dir(config.save_dir)
    ensure_dir(config.log_dir)
    update_logging_setup_for_tune_or_cross_valid(config.log_dir)

    # Train with the debug fold
    fold_train_model_best = train_fold(config, train_idx=train_idx, valid_idx=valid_idx, k_fold=k)

    # Do the testing and add the fold results to the dfs
    config.resume = Path(os.path.join(config.save_dir, "model_best.pth"))
    config.test_output_dir = Path(os.path.join(config.resume.parent, 'test_output_fold_' + str(k + 1)))
    ensure_dir(config.test_output_dir)
    fold_eval_class_wise, fold_eval_single_metrics = test_fold(config, data_dir=data_dir, test_idx=test_idx,
                                                               k_fold=k)

    # Finish everything
    print("Finished")


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='MACRO Paper: Cross-Validation')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-t', '--tune', action='store_true', help='Use to enable tuning')
    args.add_argument('--seed', type=int, default=123, help='Random seed')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
        # options added here can be modified by command line flags.
    ]
    config = ConfigParser.from_args(args=args, options=options)
    assert config["data_loader"]["cross_valid"]["enabled"], "Cross-valid should be enabled when running this script"

    # fix random seeds for reproducibility
    global_config.SEED = config.config.get("SEED", global_config.SEED)
    _set_seed(global_config.SEED)

    run_cross_validation_debug(config)
