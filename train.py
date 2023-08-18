import argparse
import collections
from datetime import datetime
import os
import pickle
import random
from pathlib import Path

import ray
import numpy as np
from ray.tune.search import BasicVariantGenerator
from ray.tune.web_server import TuneClient
from ray import tune
from ray.tune import CLIReporter, Callback


import data_loader.data_loaders as module_data_loader
import model.loss as module_loss
from logger import update_logging_setup_for_tune_or_cross_valid
from parse_config import ConfigParser
from trainer.ecg_trainer import ECGTrainer
from utils import prepare_device, get_project_root

# Needed for working with SSH Interpreter...
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "MIG-11c29e81-e611-50b5-b5ef-609c0a0fe58b"
import torch



def _set_seed(SEED):
    # OLD VERSION MA
    # np.random.seed(SEED)
    # torch.manual_seed(SEED)
    # # VB: Replaced by use_deterministic_algorithms, which will make more PyTorch operations behave deterministically
    # # See https://pytorch.org/docs/stable/notes/randomness.html
    # torch.backends.cudnn.deterministic = True
    # # torch.use_deterministic_algorithms(True)
    # torch.backends.cudnn.benchmark = False
    #
    # random.seed(SEED)
    # torch.cuda.manual_seed_all(SEED)
    # # os.environ['PYTHONHASHSEED'] = str(SEED)

    # NEW VERSION
    # https://pytorch.org/docs/stable/notes/randomness.html
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    random.seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# fix random seeds for reproducibility
SEED = 123
_set_seed(SEED)


# Define custom functions for conditional Tune Search Spaces
# Problem: Does not work with Optuna or Ax, only when not passing an search algo.
# ==> Currently tune.sample_from() only works with random search (the default).
# (https://github.com/ray-project/ray/issues/13614)

# # 1) The total amount of conv blocks should not exceed 7
# def _get_poss_num_second_conv_blocks(spec):
#     poss_nums = [1, 2, 3, 4, 5, 6]
#     poss_nums = [item for item in poss_nums if 8 >= item + spec.config.num_first_conv_blocks >= 4]
#     return np.random.choice(poss_nums)
#
#
# # 2) The depth/amount of channels should stay the same or increase, but it should not decrease
# def _get_poss_second_conv_blocks_depth(spec):
#     poss_depths = [12, 24, 32, 64]  # , 128]
#     poss_depths = [item for item in poss_depths if item >= spec.config.out_channel_first_conv_blocks]
#     return np.random.choice(poss_depths)
#
#
# # 3) With different amounts of in and out channels in a block, pooling can not be used for aligning the residual tensor
# def _get_poss_down_samples(spec):
#     if spec.config.out_channel_first_conv_blocks != 12 or \
#             spec.config.out_channel_first_conv_blocks != spec.config.out_channel_second_conv_blocks:
#         return "conv"
#     else:
#         return np.random.choice(["conv", "max_pool", "avg_pool"])
#
#
# # Sample from can not be used with SearchAlgo other than the default
# def restricted_tuning_params(name):
#     if name == "BaselineModelWithSkipConnections":
#         return {
#             "num_first_conv_blocks": tune.randint(2, 6),
#             "num_second_conv_blocks": tune.sample_from(_get_poss_num_second_conv_blocks),
#             "drop_out_first_conv_blocks": tune.choice([0.2, 0.3, 0.4]),
#             "drop_out_second_conv_blocks": tune.choice([0.2, 0.3, 0.4]),
#             "out_channel_first_conv_blocks": tune.choice([12, 24, 32, 64]),
#             "out_channel_second_conv_blocks": tune.sample_from(_get_poss_second_conv_blocks_depth),
#             "mid_kernel_size_first_conv_blocks": tune.choice([3, 5, 7]),
#             "mid_kernel_size_second_conv_blocks": tune.choice([3, 5, 7]),
#             "last_kernel_size_first_conv_blocks": tune.randint(10, 24),
#             "last_kernel_size_second_conv_blocks": tune.randint(20, 50),
#             "stride_first_conv_blocks": 2,  # tune.randint(1, 2),
#             "stride_second_conv_blocks": 2,  # tune.randint(1, 2),
#             "down_sample": tune.sample_from(_get_poss_down_samples)
#         }
#     else:
#         return None
#
#
# def axsearch_tuning_params(name):
#     if name == "BaselineModelWithSkipConnections":
#         return {
#             "num_first_conv_blocks": tune.randint(1, 6),
#             "num_second_conv_blocks": tune.randint(1, 6),
#             "drop_out_first_conv_blocks": tune.choice([0.2, 0.3, 0.4]),
#             "drop_out_second_conv_blocks": tune.choice([0.2, 0.3, 0.4]),
#             "out_channel_first_conv_blocks": tune.randint(12, 64),
#             "out_channel_second_conv_blocks": tune.randint(12, 64),
#             "mid_kernel_size_first_conv_blocks": tune.randint(3, 7),
#             "mid_kernel_size_second_conv_blocks": tune.randint(3, 7),
#             "last_kernel_size_first_conv_blocks": tune.randint(9, 39),
#             "last_kernel_size_second_conv_blocks": tune.randint(33, 63),
#             "stride_first_conv_blocks": tune.randint(1, 2),
#             "stride_second_conv_blocks": tune.randint(1, 2),
#             "down_sample": "conv",
#             # "dilation_first_conv_blocks": tune.randint(1, 3),
#             # "dilation_second_conv_blocks": tune.randint(1, 3),
#             # "expansion_first_conv_blocks": tune.choice(["1", "mul 2", "add 32"]),
#             # "expansion_second_conv_blocks": tune.choice(["1", "mul 2", "add 32"]),
#         }
#     else:
#         return None
#

def _get_mid_kernel_size_second_conv_blocks(spec):
    # Choose the same size for the second block as well to reduce the amount of hyperparams
    return spec.config.mid_kernel_size_first_conv_blocks


def tuning_params(name):
    if name == "BaselineModelWithSkipConnections" or name == "BaselineModelWithSkipConnectionsAndInstanceNorm":
        return {
            "mid_kernel_size_first_conv_blocks": tune.grid_search([3, 5, 7]),
            "mid_kernel_size_second_conv_blocks": tune.sample_from(_get_mid_kernel_size_second_conv_blocks),
            "last_kernel_size_first_conv_blocks": tune.grid_search([21, 24, 27]),
            "last_kernel_size_second_conv_blocks": tune.grid_search([45, 48, 51]),
            "down_sample": tune.grid_search(["conv", "max_pool"])
        }
    elif name == "BaselineModelWithMHAttention":
        return {
            "dropout_attention": tune.grid_search([0.2, 0.3, 0.4]),
            "heads": tune.grid_search([3, 5, 8, 16, 32]),
            "gru_units": tune.grid_search([12, 24, 32]),
            "discard_FC_before_MH": tune.grid_search([True, False])
        }
    elif name == "BaselineModelWithSkipConnectionsAndNorm":
        return {
            "mid_kernel_size_first_conv_blocks": 7,
            "mid_kernel_size_second_conv_blocks": 7,
            "last_kernel_size_first_conv_blocks": 21,
            "last_kernel_size_second_conv_blocks": 48,
            "down_sample": "conv",
            "norm_type": tune.grid_search(["BN", "IN", "LN"]),
            "norm_pos": tune.grid_search(["all", "last"]),
            "norm_before_act": tune.grid_search([True, False])
        }
    elif name == "BaselineModelWithSkipConnectionsV2":
        return {
            "down_sample": tune.grid_search(["conv", "max_pool"]),
            "vary_channels": tune.grid_search([True, False]),
            "pos_skip": tune.grid_search(["all", "not_last", "not_first"])
        }
    elif name == "BaselineModelWithSkipConnectionsAndNormV2":
        return {
            "down_sample": "conv", #tune.grid_search(["conv", "max_pool"]),
            "vary_channels": True,
            "pos_skip": "not_first",  #tune.grid_search(["all", "not_last", "not_first"]),
            "norm_type": "BN", #tune.grid_search(["BN", "IN", "LN"]),
            "norm_pos": "all" # tune.grid_search(["all", "last"])
            # Tested TOP3 from first experiment, i.e., the following:
            # Conv + True + not_last,
            # MaxPool + True + all,
            # MaxPool + True + not_first
        }
    elif name == "BaselineModelWithSkipConnectionsAndNormV2PreActivation":
        # Run 1
        # return {
        #     "down_sample": tune.grid_search(["conv", "max_pool"]),
        #     "vary_channels": True,
        #     "pos_skip": tune.grid_search(["all", "not_last"]),
        #     "norm_type": "BN",
        #     "norm_pos": "all",  #tune.grid_search(["all", "last"]),
        #     "norm_before_act": tune.grid_search([True, False]),
        #     # Tested TOP3 from second experiment, i.e., the following:
        #     # Conv + True + not_last + BN + all
        #     # Conv + True + all + BN + all
        #     # Pool + True + all + BN + all
        # }


        # Run 2: Only best one of run 1, but this time for different kernel sizes
        return {
            "down_sample": "conv",
            "vary_channels": True,
            "pos_skip": "all",
            "norm_type": "BN",
            "norm_pos": "all",
            "norm_before_act": True,
            "use_pre_conv": True,
            "pre_conv_kernel": tune.grid_search([3, 8, 12, 20, 24])
        }
    elif name == "FinalModel":
        # # Variant with additional FC for Attention
        # return {
        #     "down_sample": "conv",
        #     "vary_channels": True,
        #     "pos_skip": "all",
        #     "norm_type": "BN",
        #     "norm_pos": "all",
        #     "norm_before_act": True,
        #     "use_pre_activation_design": tune.grid_search([True, False]),
        #     "use_pre_conv": True,       # only has a meaning when used with pre-activation design
        #     "dropout_attention": tune.grid_search([0.2, 0.4]),  # see table, but also visualization (mark areas)
        #     "heads": tune.grid_search([5, 8, 16, 32]),  # No clear direction, 8 and 32 most promising according to plot
        #     "gru_units": tune.grid_search([12, 18, 24]),  # See graphical visualization and TOP 5 Table
        #     "discard_FC_before_MH": False
        # }
        #
        # # Variant without additional FC for Attention
        # return {
        #     "down_sample": "conv",
        #     "vary_channels": True,
        #     "pos_skip": "all",
        #     "norm_type": "BN",
        #     "norm_pos": "all",
        #     "norm_before_act": True,
        #     "use_pre_activation_design": True, # tune.grid_search([True, False]),
        #     "use_pre_conv": True,  # only has a meaning when used with pre-activation design
        #     "dropout_attention": tune.grid_search([0.3, 0.4]),  # Majority in TOP5
        #     "heads": tune.grid_search([5, 8, 12, 16]),    # Runs mit 3 und 32 bis auf eine Ausnahme nicht gut
        #     "gru_units": tune.grid_search([24, 28, 32]),    # See graphical visualization
        #     "discard_FC_before_MH": True
        # }

        #Rerun without FC
        return {
            "down_sample": "conv",
            "vary_channels": True,
            "pos_skip": "all",
            "norm_type": "BN",
            "norm_pos": "all",
            "norm_before_act": True,
            "use_pre_activation_design": True, # tune.grid_search([True, False]),
            "use_pre_conv": True,  # only has a meaning when used with pre-activation design
            "dropout_attention": tune.grid_search([0.2, 0.3, 0.4]),
            "heads": tune.grid_search([3, 5, 8, 32]),  # See visualization, 16 does not work well (was trained nevertheless)
            "gru_units": tune.grid_search([12, 24, 32]),  # Eventually add 18
            "discard_FC_before_MH": True
        }

        # # Rerun with FC
        # return {
        #     "down_sample": "conv",
        #     "vary_channels": True,
        #     "pos_skip": "all",
        #     "norm_type": "BN",
        #     "norm_pos": "all",
        #     "norm_before_act": True,
        #     "use_pre_activation_design": True,  # tune.grid_search([True, False]),
        #     "use_pre_conv": True,  # only has a meaning when used with pre-activation design
        #     "dropout_attention": tune.grid_search([0.2, 0.3, 0.4]),
        #     "heads": tune.grid_search([3, 8, 16, 32]),  # See visualization, 5 does not work well in comparison
        #     "gru_units": tune.grid_search([12, 24, 32]),  # Eventually add 18
        #     "discard_FC_before_MH": False
        # }
    elif name == "FinalModelMultiBranch":
        return {
            # "lr"
            "multi_branch_gru_units": tune.grid_search([12, 24, 32]),
            "multi_branch_heads": tune.grid_search([1, 4, 8, 16]),


        }
    else:
        return None


class MyTuneCallback(Callback):

    def __init__(self):
        self.already_seen = set()
        self.manager = TuneClient(tune_address="127.0.0.1", port_forward=4321)

    def setup(self, ):
        # Experiment 3: Final Model
        # seen_configs = [
        #     {
        #         "down_sample": "conv",
        #         "vary_channels": True,
        #         "pos_skip": "all",
        #         "norm_type": "BN",
        #         "norm_pos": "all",
        #         "norm_before_act": True,
        #         "use_pre_activation_design": True,
        #         "use_pre_conv": True,
        #         "dropout_attention": 0.3,
        #         "heads": 5,
        #         "gru_units": 24,
        #         "discard_FC_before_MH": True
        #     },
        #     {
        #         "down_sample": "conv",
        #         "vary_channels": True,
        #         "pos_skip": "all",
        #         "norm_type": "BN",
        #         "norm_pos": "all",
        #         "norm_before_act": True,
        #         "use_pre_activation_design": True,
        #         "use_pre_conv": True,
        #         "dropout_attention": 0.4,
        #         "heads": 5,
        #         "gru_units": 24,
        #         "discard_FC_before_MH": True
        #     },
        #     {
        #         "down_sample": "conv",
        #         "vary_channels": True,
        #         "pos_skip": "all",
        #         "norm_type": "BN",
        #         "norm_pos": "all",
        #         "norm_before_act": True,
        #         "use_pre_activation_design": True,
        #         "use_pre_conv": True,
        #         "dropout_attention": 0.3,
        #         "heads": 5,
        #         "gru_units": 28,
        #         "discard_FC_before_MH": True
        #     },
        #     {
        #         "down_sample": "conv",
        #         "vary_channels": True,
        #         "pos_skip": "all",
        #         "norm_type": "BN",
        #         "norm_pos": "all",
        #         "norm_before_act": True,
        #         "use_pre_activation_design": True,
        #         "use_pre_conv": True,
        #         "dropout_attention": 0.4,
        #         "heads": 5,
        #         "gru_units": 28,
        #         "discard_FC_before_MH": True
        #     },
        #     {
        #         "down_sample": "conv",
        #         "vary_channels": True,
        #         "pos_skip": "all",
        #         "norm_type": "BN",
        #         "norm_pos": "all",
        #         "norm_before_act": True,
        #         "use_pre_activation_design": True,
        #         "use_pre_conv": True,
        #         "dropout_attention": 0.3,
        #         "heads": 5,
        #         "gru_units": 32,
        #         "discard_FC_before_MH": True
        #     },
        #     {
        #         "down_sample": "conv",
        #         "vary_channels": True,
        #         "pos_skip": "all",
        #         "norm_type": "BN",
        #         "norm_pos": "all",
        #         "norm_before_act": True,
        #         "use_pre_activation_design": True,
        #         "use_pre_conv": True,
        #         "dropout_attention": 0.4,
        #         "heads": 5,
        #         "gru_units": 32,
        #         "discard_FC_before_MH": True
        #     },
        #     {
        #         "down_sample": "conv",
        #         "vary_channels": True,
        #         "pos_skip": "all",
        #         "norm_type": "BN",
        #         "norm_pos": "all",
        #         "norm_before_act": True,
        #         "use_pre_activation_design": True,
        #         "use_pre_conv": True,
        #         "dropout_attention": 0.3,
        #         "heads": 8,
        #         "gru_units": 24,
        #         "discard_FC_before_MH": True
        #     },
        #     {
        #         "down_sample": "conv",
        #         "vary_channels": True,
        #         "pos_skip": "all",
        #         "norm_type": "BN",
        #         "norm_pos": "all",
        #         "norm_before_act": True,
        #         "use_pre_activation_design": True,
        #         "use_pre_conv": True,
        #         "dropout_attention": 0.4,
        #         "heads": 8,
        #         "gru_units": 24,
        #         "discard_FC_before_MH": True
        #     },
        #     {
        #         "down_sample": "conv",
        #         "vary_channels": True,
        #         "pos_skip": "all",
        #         "norm_type": "BN",
        #         "norm_pos": "all",
        #         "norm_before_act": True,
        #         "use_pre_activation_design": True,
        #         "use_pre_conv": True,
        #         "dropout_attention": 0.3,
        #         "heads": 8,
        #         "gru_units": 28,
        #         "discard_FC_before_MH": True
        #     },
        #
        # ]
        seen_configs = []
        for config in seen_configs:
            self.already_seen.add(str(config))

    def on_trial_start(self, iteration, trials, trial, **info):
        # Experiment 1_2
        # Tested TOP3 from first experiment, i.e., the following:
        # Conv + True + not_last
        # MaxPool + True + all
        # MaxPool + True + not_first
        # unwanted_combination = (trial.config["down_sample"] == "conv" and trial.config["pos_skip"] == "all") or \
        #     (trial.config["down_sample"] == "conv" and trial.config["pos_skip"] == "not_first") or \
        #     (trial.config["down_sample"] == "max_pool" and trial.config["pos_skip"] == "not_last")

        # Experiment 1_3: Wanted
        # Conv + True + not_last + BN + all
        # Conv + True + all + BN + all
        # Pool + True + all + BN + all
        # unwanted_combination = (trial.config["down_sample"] == "max_pool" and trial.config["pos_skip"] == "not_last")

        if str(trial.config) in self.already_seen: # or unwanted_combination:
            print("Stop trial with id " + str(trial.trial_id))
            self.manager.stop_trial(trial.trial_id)
        else:
            self.already_seen.add(str(trial.config))


def hyper_study(main_config, tune_config, num_tune_samples=1):
    def name_trial(trial):
        file_name = f""
        for key in tune_config.keys():
            # file_name += f"{key[:8] if len(key) > 8 else key}={trial.config[key]}_"
            file_name += f"{trial.config[key]}_"
        if len(file_name) > 240:
            file_name = file_name[:240]
        file_name += datetime.now().strftime('%H-%M-%S.%f')
        return file_name

    data_dir = main_config['data_loader']['args']['data_dir']
    full_data_dir = os.path.join(str(get_project_root()), data_dir)
    data_loader = main_config.init_obj('data_loader', module_data_loader, data_dir=full_data_dir,
                                       single_batch=config['data_loader'].get('overfit_single_batch', False))
    valid_data_loader = data_loader.split_validation()

    def train_fn(config, checkpoint_dir=None):
        train_model(config=main_config, tune_config=config, train_dl=data_loader, valid_dl=valid_data_loader,
                    checkpoint_dir=checkpoint_dir, use_tune=True)

    ray.init(_temp_dir=os.path.join(get_project_root(), 'ray_tmp'))

    trainer = main_config['trainer']
    early_stop = trainer.get('monitor', 'off')
    if early_stop != 'off':
        mnt_mode, mnt_metric = early_stop.split()
    else:
        mnt_mode = "min"
        mnt_metric = "val_loss"

    # The idea behind SHA (Algorithm 1) is simple: allocate a small budget to each configuration, evaluate all
    # configurations and keep the top 1/reduction_factor, increase the budget per configuration by a factor of
    # reduction_factor, and repeat until the maximum per-configuration budget of R is reached
    # scheduler = ASHAScheduler(
    #     time_attr='training_iteration',
    #     metric=mnt_metric,  # The training result objective value attribute. Stopping procedures will use this.
    #     mode=mnt_mode,
    #     max_t=100,  # trainer.get('epochs', 120),  # Trials will be stopped after max_t time units (based on time_attr)
    #     grace_period=1,  # Only stop trials at least this old in time.
    #     reduction_factor=2)
    # # Problem:  the ASHAScheduler will aggressively terminate low-performing trials

    if main_config["arch"]["type"] == "BaselineModelWithSkipConnections" \
            or main_config["arch"]["type"] == "BaselineModelWithSkipConnectionsAndInstanceNorm":
        reporter = CLIReporter(
            parameter_columns={
                "mid_kernel_size_first_conv_blocks": "Mid kernel 1st",
                "mid_kernel_size_second_conv_blocks": "Mid kernel 2nd",
                "last_kernel_size_first_conv_blocks": "Last kernel 1st",
                "last_kernel_size_second_conv_blocks": "Last kernel 2nd",
                "down_sample": "Downsampling"
            },
            metric_columns=["loss", "val_loss",
                            "val_weighted_sk_f1",
                            "val_cpsc_F1",
                            "val_cpsc_Faf",
                            "val_cpsc_Fblock",
                            "val_cpsc_Fpc",
                            "val_cpsc_Fst",
                            "training_iteration"])
    elif main_config["arch"]["type"] == "BaselineModelWithMHAttention":
        reporter = CLIReporter(
            parameter_columns={
                "dropout_attention": "Droput MH Attention",
                "heads": "Num Heads",
                "gru_units": "Num Units GRU"
            },
            metric_columns=["loss", "val_loss",
                            "val_weighted_sk_f1",
                            "val_cpsc_F1",
                            "val_cpsc_Faf",
                            "val_cpsc_Fblock",
                            "val_cpsc_Fpc",
                            "val_cpsc_Fst",
                            "training_iteration"])
    elif main_config["arch"]["type"] == "BaselineModelWithSkipConnectionsAndNorm":
        reporter = CLIReporter(
            parameter_columns={
                "norm_type": "Type",
                "norm_pos": "Position",
                "norm_before_act": "Before L-ReLU"
            },
            metric_columns=["loss", "val_loss",
                            "val_weighted_sk_f1",
                            "val_cpsc_F1",
                            "val_cpsc_Faf",
                            "val_cpsc_Fblock",
                            "val_cpsc_Fpc",
                            "val_cpsc_Fst",
                            "training_iteration"])
    elif main_config["arch"]["type"] == "BaselineModelWithSkipConnectionsV2":
        reporter = CLIReporter(
            parameter_columns={
                "down_sample": "Downsampling",
                "vary_channels": "Varied channels",
                "pos_skip": "Skip Pos"
            },
            metric_columns=["loss", "val_loss",
                            "val_weighted_sk_f1",
                            "val_cpsc_F1",
                            "val_cpsc_Faf",
                            "val_cpsc_Fblock",
                            "val_cpsc_Fpc",
                            "val_cpsc_Fst",
                            "training_iteration"])
    elif main_config["arch"]["type"] == "BaselineModelWithSkipConnectionsAndNormV2":
        reporter = CLIReporter(
            parameter_columns={
                "down_sample": "Downsampling",
                "vary_channels": "Varied channels",
                "pos_skip": "Skip Pos",
                "norm_type": "Type",
                "norm_pos": "Position",
                "pre_conv_kernel": "1st Conv Kernel"
            },
            metric_columns=["loss", "val_loss",
                            "val_weighted_sk_f1",
                            "val_cpsc_F1",
                            "val_cpsc_Faf",
                            "val_cpsc_Fblock",
                            "val_cpsc_Fpc",
                            "val_cpsc_Fst",
                            "training_iteration"])
    elif main_config["arch"]["type"] == "BaselineModelWithSkipConnectionsAndNormV2PreActivation":
        reporter = CLIReporter(
            parameter_columns={
                "down_sample": "Downsampling",
                "vary_channels": "Varied channels",
                "pos_skip": "Skip Pos",
                "norm_type": "Type",
                "norm_pos": "Norm pos",
                "norm_before_act": "NormBefAct"
            },
            metric_columns=["loss", "val_loss",
                            "val_weighted_sk_f1",
                            "val_cpsc_F1",
                            "val_cpsc_Faf",
                            "val_cpsc_Fblock",
                            "val_cpsc_Fpc",
                            "val_cpsc_Fst",
                            "training_iteration"])
    elif main_config["arch"]["type"] == "FinalModel":
        reporter = CLIReporter(
            parameter_columns={
                "down_sample": "Downsampling",
                "vary_channels": "Varied channels",
                "pos_skip": "Skip Pos",
                "norm_type": "Type",
                "norm_pos": "Norm pos",
                "norm_before_act": "NormBefAct",
                "use_pre_activation_design": "PreActDesign",
                "dropout_attention": "DP Att",
                "heads": "H",
                "gru_units": "GRU",
                "discard_FC_before_MH": "WithoutFC"
            },
            metric_columns=["loss", "val_loss",
                            "val_weighted_sk_f1",
                            "val_cpsc_F1",
                            "val_cpsc_Faf",
                            "val_cpsc_Fblock",
                            "val_cpsc_Fpc",
                            "val_cpsc_Fst",
                            "training_iteration"])


    # experiment_stopper = ExperimentPlateauStopper(
    #     metric=mnt_metric,
    #     mode=mnt_mode,
    #     patience=trainer.get('early_stop', 5),  # Number of epochs to wait for a change in the top models
    #     top=10  # The number of best models to consider.
    # )
    #
    # trial_stopper = TrialPlateauStopper(
    #     metric=mnt_metric,
    #     mode=mnt_mode,
    #     std=0.01,  # Maximum metric standard deviation to decide if a trial plateaued.
    #     num_results=20,  # trainer.get('early_stop', 5),  # Number of results to consider for stdev calculation.
    #     grace_period=1,  # Minimum number of timesteps before a trial can be early stopped
    # )

    # initial_param_suggestions = [
    #     # Original setting
    #     {
    #         "num_first_conv_blocks": 4,
    #         "num_second_conv_blocks": 1,
    #         "drop_out_first_conv_blocks": 0.2,
    #         "drop_out_second_conv_blocks": 0.2,
    #         "out_channel_first_conv_blocks": 12,
    #         "out_channel_second_conv_blocks": 12,
    #         "mid_kernel_size_first_conv_blocks": 3,
    #         "mid_kernel_size_second_conv_blocks": 3,
    #         "last_kernel_size_first_conv_blocks": 24,
    #         "last_kernel_size_second_conv_blocks": 48,
    #         "stride_first_conv_blocks": 2,
    #         "stride_second_conv_blocks": 2,
    #         "down_sample": "conv"
    #     },
    #     # Good configurations found by first manual runs
    #     {
    #         "num_first_conv_blocks": 3,
    #         "num_second_conv_blocks": 1,
    #         "drop_out_first_conv_blocks": 0.2,
    #         "drop_out_second_conv_blocks": 0.2,
    #         "out_channel_first_conv_blocks": 12,
    #         "out_channel_second_conv_blocks": 12,
    #         "mid_kernel_size_first_conv_blocks": 3,
    #         "mid_kernel_size_second_conv_blocks": 3,
    #         "last_kernel_size_first_conv_blocks": 24,
    #         "last_kernel_size_second_conv_blocks": 48,
    #         "stride_first_conv_blocks": 2,
    #         "stride_second_conv_blocks": 2,
    #         "down_sample": "conv"
    #     },
    #     # Good configurations found by first RandomSearch
    #     {
    #         "num_first_conv_blocks": 3,
    #         "num_second_conv_blocks": 3,
    #         "drop_out_first_conv_blocks": 0.3,
    #         "drop_out_second_conv_blocks": 0.2,
    #         "out_channel_first_conv_blocks": 32,
    #         "out_channel_second_conv_blocks": 32,
    #         "mid_kernel_size_first_conv_blocks": 3,
    #         "mid_kernel_size_second_conv_blocks": 3,
    #         "last_kernel_size_first_conv_blocks": 13,
    #         "last_kernel_size_second_conv_blocks": 48,
    #         "stride_first_conv_blocks": 2,
    #         "stride_second_conv_blocks": 2,
    #         "down_sample": "conv"
    #     },
    #     {
    #         "num_first_conv_blocks": 4,
    #         "num_second_conv_blocks": 2,
    #         "drop_out_first_conv_blocks": 0.2,
    #         "drop_out_second_conv_blocks": 0.2,
    #         "out_channel_first_conv_blocks": 24,
    #         "out_channel_second_conv_blocks": 24,
    #         "mid_kernel_size_first_conv_blocks": 5,
    #         "mid_kernel_size_second_conv_blocks": 5,
    #         "last_kernel_size_first_conv_blocks": 11,
    #         "last_kernel_size_second_conv_blocks": 38,
    #         "stride_first_conv_blocks": 2,
    #         "stride_second_conv_blocks": 2,
    #         "down_sample": "conv"
    #     },
    #     {
    #         "num_first_conv_blocks": 2,
    #         "num_second_conv_blocks": 3,
    #         "drop_out_first_conv_blocks": 0.2,
    #         "drop_out_second_conv_blocks": 0.2,
    #         "out_channel_first_conv_blocks": 12,
    #         "out_channel_second_conv_blocks": 24,
    #         "mid_kernel_size_first_conv_blocks": 7,
    #         "mid_kernel_size_second_conv_blocks": 3,
    #         "last_kernel_size_first_conv_blocks": 22,
    #         "last_kernel_size_second_conv_blocks": 37,
    #         "stride_first_conv_blocks": 2,
    #         "stride_second_conv_blocks": 2,
    #         "down_sample": "conv"
    #     },
    # ]

    # optuna_searcher = OptunaSearch(metric=mnt_metric, mode=mnt_mode,
    #                                points_to_evaluate=initial_param_suggestions,
    #                                sampler=TPESampler(seed=SEED))
    #
    # # ax = AxClient(enforce_sequential_optimization=False)  # (https://ax.dev/tutorials/raytune_pytorch_cnn.html)
    # # ax = AxClient(random_seed=SEED)
    # ax_searcher = AxSearch(metric=mnt_metric, mode=mnt_mode,
    #                        points_to_evaluate=initial_param_suggestions,
    #                        parameter_constraints=["num_first_conv_blocks + num_second_conv_blocks <= 8",
    #                                               "num_first_conv_blocks + num_second_conv_blocks >= 4",
    #                                               # at least in one block size reduction
    #                                               "stride_first_conv_blocks + stride_second_conv_blocks >= 3",
    #                                               "out_channel_first_conv_blocks <= out_channel_second_conv_blocks"])
    # ax_searcher = ConcurrencyLimiter(ax_searcher, max_concurrent=2)

    num_gpu = 0.5 if not main_config["arch"]["type"] == "BaselineModelWithSkipConnectionsV2" and not \
        main_config["arch"]["type"] == "BaselineModelWithSkipConnectionsAndNormV2" and not \
        main_config["arch"]["type"] == "BaselineModelWithSkipConnectionsAndNormV2PreActivation" and not \
        main_config["arch"]["type"] == "FinalModel" else 1
    analysis = tune.run(
        run_or_experiment=train_fn,
        num_samples=num_tune_samples,
        name=str(main_config.save_dir),  # experiment_name
        trial_dirname_creator=name_trial,  # trial_name
        local_dir=str(main_config.save_dir),

        #  scheduler=scheduler,             # Do not use any scheduler, early stopping can be configured in the Config!
        metric=mnt_metric,
        mode=mnt_mode,
        # stop=CombinedStopper(experiment_stopper, trial_stopper),
        # keep_checkpoints_num=10,
        # checkpoint_score_attr=f"{mnt_mode}-{mnt_metric}",

        # search_alg=ax_searcher,           # Just use Random Grid Search instead of an advanced search algo
        search_alg=BasicVariantGenerator(),  # points_to_evaluate=initial_param_suggestions, max_concurrent=3),
        config={**tune_config},
        resources_per_trial={"cpu": 5 if torch.cuda.is_available() else 1,
                             "gpu": num_gpu if torch.cuda.is_available() else 0},

        max_failures=2,  # retry when error, e.g. OutOfMemory, default is 0
        raise_on_failed_trial=False,  # Failed trials are expected due to assertion errors
        verbose=1,
        progress_reporter=reporter,
        log_to_file=True,

        callbacks=[MyTuneCallback()],
        server_port=4321
    )

    print("Best hyperparameters found were: ", analysis.best_config)
    print("Best Trials best checkpoint: " + str(analysis.best_checkpoint))

    # Get a dataframe for the max CPSC f1 score seen for each trial
    df = analysis.dataframe(metric=mnt_metric, mode=mnt_mode)
    with open(os.path.join(main_config.save_dir, "best_per_trial.p"), "wb") as file:
        pickle.dump(df, file)
    with open(os.path.join(main_config.save_dir, "analysis.p"), "wb") as file:
        pickle.dump(analysis, file)


def train_model(config, tune_config=None, train_dl=None, valid_dl=None, checkpoint_dir=None, use_tune=False,
                train_idx=None, valid_idx=None, k_fold=None, cv_active=False):
    # config: type: ConfigParser -> can be used as usual
    # tune_config: type: Dict -> contains the tune params with the samples values,
    #               e.g. {'num_first_conv_blocks': 8, 'num_second_conv_blocks': 9, ...}
    import torch  # Needed to work with asych. tune workers as well

    assert use_tune is False or cv_active is False, "Cross Validation does not work with active tuning!"

    if use_tune:
        # When using Ray Tune, this is distributed among worker processes, which requires seeding within the function
        # Otherwise the same config may to different results -> not reproducible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        random.seed(SEED)
        torch.cuda.manual_seed_all(SEED)

    # Conditional inputs depending on the config
    if config['arch']['type'] == 'BaselineModelWoRnnWoAttention':
        import model.baseline_model_woRNN_woAttention as module_arch
    elif config['arch']['type'] == 'BaselineModel':
        import model.baseline_model as module_arch
    elif config['arch']['type'] == 'BaselineModelWithSkipConnections':
        import model.old.baseline_model_with_skips as module_arch
    elif config['arch']['type'] == 'BaselineModelWithSkipConnectionsAndInstanceNorm':
        import model.old.baseline_model_with_skips_and_InstNorm as module_arch
    elif config['arch']['type'] == "BaselineModelWithSkipConnectionsAndNorm":
        import model.old.baseline_model_with_skips_and_norm as module_arch
    elif config['arch']['type'] == "BaselineModelWithSkipConnectionsV2":
        import model.baseline_model_with_skips_v2 as module_arch
    elif config['arch']['type'] == "BaselineModelWithSkipConnectionsAndNormV2":
        import model.baseline_model_with_skips_and_norm_v2 as module_arch
    elif config['arch']['type'] == 'BaselineModelWithMHAttention':
        import model.baseline_model_with_MHAttention as module_arch
    elif config['arch']['type'] == 'BaselineModelWithSkipConnectionsAndNormV2PreActivation':
        import model.baseline_model_with_skips_and_norm_v2_pre_activation_design as module_arch
    elif config['arch']['type'] == 'FinalModel':
        import model.final_model as module_arch
    elif config['arch']['type'] == 'FinalModelMultiBranch':
        import model.final_model_multibranch as module_arch

    if config['arch']['args']['multi_label_training']:
        import evaluation.multi_label_metrics as module_metric
    else:
        raise NotImplementedError("Single label metrics haven't been checked after the Python update! Do not use them!")
        import evaluation.single_label_metrics as module_metric

    if use_tune:
        # Adapt the save path for the logging since it differs from trial to trial
        log_dir = Path(tune.get_trial_dir().replace('/models/', '/log/'))
        log_dir.mkdir(parents=True, exist_ok=True)
        update_logging_setup_for_tune_or_cross_valid(log_dir)
        # Update the config if a checkpoint is passed by Tune
        if checkpoint_dir is not None:
            config.resume = checkpoint_dir

    # config is of type parse_config.ConfigParser
    if k_fold is None:
        logger = config.get_logger('train')
    else:
        logger = config.get_logger('train_fold_' + str(k_fold))

    # setup data_loader instances if not already done because use_tune is enabled
    if use_tune:
        data_loader = train_dl
        valid_data_loader = valid_dl
    elif cv_active:
        # Setup data_loader instances for current the cross validation run
        data_loader = config.init_obj('data_loader', module_data_loader,
                                      cross_valid=True, train_idx=train_idx, valid_idx=valid_idx, cv_train_mode=True,
                                      fold_id=k_fold,
                                      single_batch=False)
        valid_data_loader = data_loader.split_validation()
    else:
        data_loader = config.init_obj('data_loader', module_data_loader,
                                      single_batch=config['data_loader'].get('overfit_single_batch', False))
        valid_data_loader = data_loader.split_validation()

    # build model architecture, then print to console
    if tune_config is None:
        model = config.init_obj('arch', module_arch)
    else:
        model = config.init_obj('arch', module_arch, **tune_config)
    logger.info(model)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config['n_gpu'])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # Get function handles of loss and metrics
    # Important: The method config['loss'] must exist in the loss module (<module 'model.loss' >)
    # Equivalently, all metrics specified in the context must exist in the metrics modul
    criterion = getattr(module_loss, config['loss']['type'])
    if config['arch']['args']['multi_label_training']:
        metrics_iter = [getattr(module_metric, met) for met in config['metrics']['ml']['per_iteration'].keys()]
        metrics_epoch = [getattr(module_metric, met) for met in config['metrics']['ml']['per_epoch']]
        metrics_epoch_class_wise = [getattr(module_metric, met) for met in
                                    config['metrics']['ml']['per_epoch_class_wise']]
    else:
        metrics_iter = [getattr(module_metric, met) for met in config['metrics']['sl']['per_iteration'].keys()]
        metrics_epoch = [getattr(module_metric, met) for met in config['metrics']['sl']['per_epoch']]
        metrics_epoch_class_wise = [getattr(module_metric, met) for met in
                                    config['metrics']['sl']['per_epoch_class_wise']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
    if config['lr_scheduler']['active']:
        lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)
    else:
        lr_scheduler = None

    trainer = ECGTrainer(model=model,
                         criterion=criterion,
                         metric_ftns_iter=metrics_iter,
                         metric_ftns_epoch=metrics_epoch,
                         metric_ftns_epoch_class_wise=metrics_epoch_class_wise,
                         optimizer=optimizer,
                         config=config,
                         device=device,
                         data_loader=data_loader,
                         valid_data_loader=valid_data_loader,
                         lr_scheduler=lr_scheduler,
                         use_tune=use_tune,
                         cross_valid_active=cv_active)

    log_best = trainer.train()
    if use_tune:
        path = os.path.join(Path(tune.get_trial_dir().replace('/models/', '/log/')), "model_best_metrics.p")
    else:
        path = os.path.join(config.log_dir, "model_best_metrics.p")
    with open(path, 'wb') as file:
        pickle.dump(log_best, file)
    return log_best


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
    if config.use_tune:
        tuning_params = tuning_params(name=config["arch"]["type"])
        # With grid search, only 1 times !
        hyper_study(main_config=config, tune_config=tuning_params, num_tune_samples=1)
    else:
        train_model(config)
