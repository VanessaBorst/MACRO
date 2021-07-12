import argparse
import collections
import json
from datetime import datetime
import os
import pickle
import random
from pathlib import Path

import ray
import torch
import numpy as np
from ax.service.ax_client import AxClient
from jinja2.nodes import List
from optuna.samplers import TPESampler
from ray import tune
from ray.tune import CLIReporter, Callback
from ray.tune.schedulers import ASHAScheduler
from ray.tune.suggest import ConcurrencyLimiter
from ray.tune.suggest.ax import AxSearch
from ray.tune.suggest.optuna import OptunaSearch
from ray.tune.stopper import ExperimentPlateauStopper, TrialPlateauStopper, CombinedStopper, FunctionStopper

import data_loader.data_loaders as module_data_loader
import model.loss as module_loss
from logger import update_logging_setup_for_tune, setup_logging
from parse_config import ConfigParser
from trainer.ecg_trainer import ECGTrainer
from utils import prepare_device, get_project_root, write_json


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


# Define custom functions for conditional Tune Search Spaces
# Problem: Does not work with Optuna or Ax, only when not passing an search algo.
# ==> Currently tune.sample_from() only works with random search (the default).
# (https://github.com/ray-project/ray/issues/13614)

# 1) The total amount of conv blocks should not exceed 7
def _get_poss_num_second_conv_blocks(spec):
    poss_nums = [1, 2, 3, 4, 5, 6]
    poss_nums = [item for item in poss_nums if 8 >= item + spec.config.num_first_conv_blocks >= 4]
    return np.random.choice(poss_nums)


# 2) The depth/amount of channels should stay the same or increase, but it should not decrease
def _get_poss_second_conv_blocks_depth(spec):
    poss_depths = [12, 24, 32, 64]  # , 128]
    poss_depths = [item for item in poss_depths if item >= spec.config.out_channel_first_conv_blocks]
    return np.random.choice(poss_depths)


# 3) With different amounts of in and out channels in a block, pooling can not be used for aligning the residual tensor
def _get_poss_down_samples(spec):
    if spec.config.out_channel_first_conv_blocks != 12 or \
            spec.config.out_channel_first_conv_blocks != spec.config.out_channel_second_conv_blocks:
        return "conv"
    else:
        return np.random.choice(["conv", "max_pool", "avg_pool"])


# Sample from can not be used with SearchAlgo other than the default
def random_search_tuning_params(name):
    if name == "BaselineModelWithSkipConnections":
        return {
            "num_first_conv_blocks": tune.randint(2, 6),
            "num_second_conv_blocks": tune.sample_from(_get_poss_num_second_conv_blocks),
            "drop_out_first_conv_blocks": tune.choice([0.2, 0.3, 0.4]),
            "drop_out_second_conv_blocks": tune.choice([0.2, 0.3, 0.4]),
            "out_channel_first_conv_blocks": tune.choice([12, 24, 32, 64]),
            "out_channel_second_conv_blocks": tune.sample_from(_get_poss_second_conv_blocks_depth),
            "mid_kernel_size_first_conv_blocks": tune.choice([3, 5, 7]),
            "mid_kernel_size_second_conv_blocks": tune.choice([3, 5, 7]),
            "last_kernel_size_first_conv_blocks": tune.randint(10, 24),
            "last_kernel_size_second_conv_blocks": tune.randint(20, 50),
            "stride_first_conv_blocks": 2,  # tune.randint(1, 2),
            "stride_second_conv_blocks": 2,  # tune.randint(1, 2),
            "down_sample": tune.sample_from(_get_poss_down_samples)
        }
    else:
        return None


def axsearch_tuning_params(name):
    if name == "BaselineModelWithSkipConnections":
        return {
            "num_first_conv_blocks": tune.randint(1, 6),
            "num_second_conv_blocks": tune.randint(1, 6),
            "drop_out_first_conv_blocks": tune.choice([0.2, 0.3, 0.4]),
            "drop_out_second_conv_blocks": tune.choice([0.2, 0.3, 0.4]),
            "out_channel_first_conv_blocks": tune.randint(12, 64),
            "out_channel_second_conv_blocks": tune.randint(12, 64),
            "mid_kernel_size_first_conv_blocks": tune.randint(3, 7),
            "mid_kernel_size_second_conv_blocks": tune.randint(3, 7),
            "last_kernel_size_first_conv_blocks": tune.randint(9, 39),
            "last_kernel_size_second_conv_blocks": tune.randint(33, 63),
            "stride_first_conv_blocks": tune.randint(1, 2),
            "stride_second_conv_blocks": tune.randint(1, 2),
            "down_sample": "conv",
            # "dilation_first_conv_blocks": tune.randint(1, 3),
            # "dilation_second_conv_blocks": tune.randint(1, 3),
            # "expansion_first_conv_blocks": tune.choice(["1", "mul 2", "add 32"]),
            # "expansion_second_conv_blocks": tune.choice(["1", "mul 2", "add 32"]),
        }
    else:
        return None

def optuna_params(name):
    if name == "BaselineModelWithSkipConnections":
        return {
            "num_first_conv_blocks": tune.randint(1, 6),
            "num_second_conv_blocks": tune.randint(1, 6),
            "drop_out_first_conv_blocks": tune.choice([0.2, 0.3, 0.4]),
            "drop_out_second_conv_blocks": tune.choice([0.2, 0.3, 0.4]),
            "out_channel_first_conv_blocks": tune.randint(12, 64),
            "out_channel_second_conv_blocks": tune.randint(12, 64),
            "mid_kernel_size_first_conv_blocks": tune.randint(3, 7),
            "mid_kernel_size_second_conv_blocks": tune.randint(3, 7),
            "last_kernel_size_first_conv_blocks": tune.randint(9, 39),
            "last_kernel_size_second_conv_blocks": tune.randint(33, 63),
            "stride_first_conv_blocks": tune.randint(1, 2),
            "stride_second_conv_blocks": tune.randint(1, 2),
            "down_sample":  tune.choice(["conv", "avg_pool", "max_pool"]),
            # "dilation_first_conv_blocks": tune.randint(1, 3),
            # "dilation_second_conv_blocks": tune.randint(1, 3),
            # "expansion_first_conv_blocks": tune.choice(["1", "mul 2", "add 32"]),
            # "expansion_second_conv_blocks": tune.choice(["1", "mul 2", "add 32"]),
        }
    else:
        return None


def hyper_study(main_config, tune_config, num_tune_samples):
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
        # model = getattr(models, config["model_name"])(**config)
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
    scheduler = ASHAScheduler(
        time_attr='training_iteration',
        metric=mnt_metric,  # The training result objective value attribute. Stopping procedures will use this.
        mode=mnt_mode,
        max_t=100, # trainer.get('epochs', 120),  # Trials will be stopped after max_t time units (based on time_attr)
        grace_period=1,  # Only stop trials at least this old in time.
        reduction_factor=2)
    # # Problem:  the ASHAScheduler will aggressively terminate low-performing trials

    reporter = CLIReporter(
        parameter_columns={
            "num_first_conv_blocks": "Num 1st",
            "num_second_conv_blocks": "Num 2nd",
            "drop_out_first_conv_blocks": "Drop 1st",
            "drop_out_second_conv_blocks": "Drop 2nd",
            "out_channel_first_conv_blocks": "Out 1st",
            "out_channel_second_conv_blocks": "Out 2nd",
            "mid_kernel_size_first_conv_blocks": "Mid kernel 1st",
            "mid_kernel_size_second_conv_blocks": "Mid kernel 2nd",
            "last_kernel_size_first_conv_blocks": "Last kernel 1st",
            "last_kernel_size_second_conv_blocks": "Last kernel 2nd",
            "stride_first_conv_blocks": "S 1st",
            "stride_second_conv_blocks": "S 2nd",
            "down_sample": "ds"
        },
        metric_columns=["val_loss", "val_cpsc_F1", "training_iteration"])

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

    initial_param_suggestions = [
        # Original setting
        {
            "num_first_conv_blocks": 4,
            "num_second_conv_blocks": 1,
            "drop_out_first_conv_blocks": 0.2,
            "drop_out_second_conv_blocks": 0.2,
            "out_channel_first_conv_blocks": 12,
            "out_channel_second_conv_blocks": 12,
            "mid_kernel_size_first_conv_blocks": 3,
            "mid_kernel_size_second_conv_blocks": 3,
            "last_kernel_size_first_conv_blocks": 24,
            "last_kernel_size_second_conv_blocks": 48,
            "stride_first_conv_blocks": 2,
            "stride_second_conv_blocks": 2,
            "down_sample": "conv"
        },
        # Good configurations found by first manual runs
        {
            "num_first_conv_blocks": 3,
            "num_second_conv_blocks": 1,
            "drop_out_first_conv_blocks": 0.2,
            "drop_out_second_conv_blocks": 0.2,
            "out_channel_first_conv_blocks": 12,
            "out_channel_second_conv_blocks": 12,
            "mid_kernel_size_first_conv_blocks": 3,
            "mid_kernel_size_second_conv_blocks": 3,
            "last_kernel_size_first_conv_blocks": 24,
            "last_kernel_size_second_conv_blocks": 48,
            "stride_first_conv_blocks": 2,
            "stride_second_conv_blocks": 2,
            "down_sample": "conv"
        },
        # Good configurations found by first RandomSearch
        {
            "num_first_conv_blocks": 3,
            "num_second_conv_blocks": 3,
            "drop_out_first_conv_blocks": 0.3,
            "drop_out_second_conv_blocks": 0.2,
            "out_channel_first_conv_blocks": 32,
            "out_channel_second_conv_blocks": 32,
            "mid_kernel_size_first_conv_blocks": 3,
            "mid_kernel_size_second_conv_blocks": 3,
            "last_kernel_size_first_conv_blocks": 13,
            "last_kernel_size_second_conv_blocks": 48,
            "stride_first_conv_blocks": 2,
            "stride_second_conv_blocks": 2,
            "down_sample": "conv"
        },
        {
            "num_first_conv_blocks": 4,
            "num_second_conv_blocks": 2,
            "drop_out_first_conv_blocks": 0.2,
            "drop_out_second_conv_blocks": 0.2,
            "out_channel_first_conv_blocks": 24,
            "out_channel_second_conv_blocks": 24,
            "mid_kernel_size_first_conv_blocks": 5,
            "mid_kernel_size_second_conv_blocks": 5,
            "last_kernel_size_first_conv_blocks": 11,
            "last_kernel_size_second_conv_blocks": 38,
            "stride_first_conv_blocks": 2,
            "stride_second_conv_blocks": 2,
            "down_sample": "conv"
        },
        {
            "num_first_conv_blocks": 2,
            "num_second_conv_blocks": 3,
            "drop_out_first_conv_blocks": 0.2,
            "drop_out_second_conv_blocks": 0.2,
            "out_channel_first_conv_blocks": 12,
            "out_channel_second_conv_blocks": 24,
            "mid_kernel_size_first_conv_blocks": 7,
            "mid_kernel_size_second_conv_blocks": 3,
            "last_kernel_size_first_conv_blocks": 22,
            "last_kernel_size_second_conv_blocks": 37,
            "stride_first_conv_blocks": 2,
            "stride_second_conv_blocks": 2,
            "down_sample": "conv"
        },
    ]

    optuna_searcher = OptunaSearch(metric=mnt_metric, mode=mnt_mode,
                                   points_to_evaluate=initial_param_suggestions,
                                   sampler=TPESampler(seed=SEED))

    # ax = AxClient(enforce_sequential_optimization=False)  # (https://ax.dev/tutorials/raytune_pytorch_cnn.html)
    # ax = AxClient(random_seed=SEED)
    ax_searcher = AxSearch(metric=mnt_metric, mode=mnt_mode,
                           points_to_evaluate=initial_param_suggestions,
                           parameter_constraints=["num_first_conv_blocks + num_second_conv_blocks <= 8",
                                                  "num_first_conv_blocks + num_second_conv_blocks >= 4",
                                                  # at least in one block size reduction
                                                  "stride_first_conv_blocks + stride_second_conv_blocks >= 3",
                                                  "out_channel_first_conv_blocks <= out_channel_second_conv_blocks"])
    ax_searcher = ConcurrencyLimiter(ax_searcher, max_concurrent=2)

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

        search_alg=ax_searcher,
        config={**tune_config},
        resources_per_trial={"cpu": 8 if torch.cuda.is_available() else 1,
                             "gpu": 0.5 if torch.cuda.is_available() else 0},

        max_failures=2,  # retry when error, e.g. OutOfMemory, default is 0
        raise_on_failed_trial=False,  # Failed trials are expected due to assertion errors
        verbose=1,
        progress_reporter=reporter,
        log_to_file=True
    )

    print("Best hyperparameters found were: ", analysis.best_config)
    print("Best Trials best checkpoint: " + str(analysis.best_checkpoint))

    # Get a dataframe for the max CPSC f1 score seen for each trial
    df = analysis.dataframe(metric=mnt_metric, mode=mnt_mode)
    with open(os.path.join(main_config.save_dir, "best_per_trial.p"), "wb") as file:
        pickle.dump(df, file)
    with open(os.path.join(main_config.save_dir, "analysis.p"), "wb") as file:
        pickle.dump(analysis, file)


def train_model(config, tune_config=None, train_dl=None, valid_dl=None, checkpoint_dir=None, use_tune=False):
    # config: type: ConfigParser -> can be used as usual
    # tune_config: type: Dict -> contains the tune params with the samples values,
    #               e.g. {'num_first_conv_blocks': 8, 'num_second_conv_blocks': 9, ...}

    # Conditional inputs depending on the config
    if config['arch']['type'] == 'BaselineModelWoRnnWoAttention':
        import model.baseline_model_woRNN_woAttention as module_arch
    elif config['arch']['type'] == 'BaselineModel':
        import model.baseline_model as module_arch
    elif config['arch']['type'] == 'BaselineModelWithSkipConnections':
        import model.baseline_model_with_skips as module_arch
    else:
        import model.baseline_model_variableConvs as module_arch

    if config['arch']['args']['multi_label_training']:
        import evaluation.multi_label_metrics as module_metric
    else:
        import evaluation.single_label_metrics as module_metric

    if use_tune:
        # Adapt the save path for the logging since it differs from trial to trial
        log_dir = Path(tune.get_trial_dir().replace('/models/', '/log/'))
        log_dir.mkdir(parents=True, exist_ok=True)
        update_logging_setup_for_tune(log_dir)
        # Update the config if a checkpoint is passed by Tune
        if checkpoint_dir is not None:
            config.resume = checkpoint_dir

    # config is of type parse_config.ConfigParser
    logger = config.get_logger('train')

    # setup data_loader instances if not already done because use_tune is enabled
    if use_tune:
        data_loader = train_dl
        valid_data_loader = valid_dl
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
                         use_tune=use_tune)

    log_best = trainer.train()
    if use_tune:
        path = os.path.join(Path(tune.get_trial_dir().replace('/models/', '/log/')), "model_best_metrics.p")
    else:
        path = os.path.join(config.log_dir, "model_best_metrics.p")
    with open(path, 'wb') as file:
        pickle.dump(log_best, file)


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
        tuning_params = axsearch_tuning_params(name=config["arch"]["type"])
        hyper_study(main_config=config, tune_config=tuning_params, num_tune_samples=100)
    else:
        dummy_params = {
          "down_sample": "conv",
          "drop_out_first_conv_blocks": 0.2,
          "drop_out_second_conv_blocks": 0.2,
          "last_kernel_size_first_conv_blocks": 23,
          "last_kernel_size_second_conv_blocks": 49,
          "mid_kernel_size_first_conv_blocks": 5,
          "mid_kernel_size_second_conv_blocks": 4,
          "num_first_conv_blocks": 1,
          "num_second_conv_blocks": 3,
          "out_channel_first_conv_blocks": 31,
          "out_channel_second_conv_blocks": 39,
          "stride_first_conv_blocks": 2,
          "stride_second_conv_blocks": 1
        }
        train_model(config)
