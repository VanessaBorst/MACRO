import argparse
import collections
from datetime import datetime
import os
import pickle
import random
from pathlib import Path

import torch
import numpy as np
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.tune.suggest.optuna import OptunaSearch

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


def tuning_params(name):
    if name == "BaselineModelWithSkipConnections":
        return {
            "down_sample": tune.choice(["conv", "max_pool", "avg_pool"]),
            "num_blocks": tune.randint(3, 7),
        }
        # return {
        #     "num_first_conv_blocks": tune.randint(1, 10),
        #     "num_second_conv_blocks": tune.randint(1, 10),
        #     "mid_kernel_size_first_conv_blocks":  tune.randint(3, 6),
        #     "mid_kernel_size_first_conv_blocks": tune.randint(15, 30),
        #     "mid_kernel_size_second_conv_blocks": tune.randint(3, 6),
        #     "mid_kernel_size_second_conv_blocks": tune.randint(45, 60),
        #     "stride_first_conv_blocks": tune.randint(1, 4),
        #     "stride_second_conv_blocks": tune.randint(1, 4),
        #     "dilation_first_conv_blocks": tune.randint(1, 3),
        #     "dilation_second_conv_blocks": tune.randint(1, 3),
        #     "expansion_first_conv_blocks": tune.choice(["1", "mul 2", "add 32"]),
        #     "expansion_second_conv_blocks": tune.choice(["1", "mul 2", "add 32"]),
        #     "down_sample": tune.choice(["conv", "max_pool", "avg_pool"])
        # }
    else:
        return None


def hyper_study(main_config, tune_config, num_tune_samples):
    def name_dir(trial):
        file_name = f""
        for key in tune_config.keys():
            file_name += f"{key[:8] if len(key) > 8 else key}={trial.config[key]}_"
        if len(file_name) > 240:
            file_name = file_name[:240]
        file_name += datetime.now().strftime('%H-%M-%S.%f')
        return file_name

    def train_fn(config, checkpoint_dir=None):
        # model = getattr(models, config["model_name"])(**config)
        train_model(config=main_config, tune_config=config, checkpoint_dir=checkpoint_dir, use_tune=True)

    mode = "max"
    metric = "val_cpsc_F1"

    scheduler = ASHAScheduler(
        metric="val_cpsc_F1",
        mode="max",
        max_t=500,
        grace_period=4,
        reduction_factor=2)
    # reporter = CLIReporter(
    #     # parameter_columns=["l1", "l2", "lr", "batch_size"],
    #     metric_columns=["val_loss", "val_cpsc_F1", "training_iteration"])

    #stopping_criteria = {"training_iteration": 300}
    analysis = tune.run(
        train_fn,
        num_samples=num_tune_samples,
        local_dir="ray_results",
        name=main_config.save_dir,
        trial_dirname_creator=name_dir,

        metric=metric,
        mode=mode,
        keep_checkpoints_num=10,
        checkpoint_score_attr=f"{mode}-{metric}",

        config={**tune_config},
        resources_per_trial={"gpu": 1},
        # search_alg=HyperOptSearch(),
        search_alg=OptunaSearch(),
        # scheduler=ASHAScheduler(grace_period=const_config["epochs"]),
        max_failures=5,  # retry when error, e.g. OutOfMemory
        verbose=1,
        #stop=stopping_criteria
        #progress_reporter=reporter
    )
    print("Best hyperparameters found were: ", analysis.best_config)


def train_model(config, tune_config=None, checkpoint_dir=None, use_tune=False):
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
        log_dir.mkdir(parents=True)
        update_logging_setup_for_tune(log_dir)
        # Update the config if a checkpoint is passed by Tune
        if checkpoint_dir is not None:
            config.resume = checkpoint_dir

    # config is of type parse_config.ConfigParser
    logger = config.get_logger('train')

    # setup data_loader instances if not already done because use_tune is enabled
    if use_tune:
        data_dir = config['data_loader']['args']['data_dir']
        full_data_dir = os.path.join(str(get_project_root()), data_dir)
        data_loader = config.init_obj('data_loader', module_data_loader, data_dir = full_data_dir,
                                      single_batch=config['data_loader'].get('overfit_single_batch', False))
    else:
        data_loader = config.init_obj('data_loader', module_data_loader,
                                      single_batch=config['data_loader'].get('overfit_single_batch', False))
    valid_data_loader = data_loader.split_validation()

    # build model architecture, then print to console
    if not use_tune:
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
    args.add_argument('-t', '--tune', action='store_true', help='Use with True to enable tuning')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
        # options added here can be modified by command line flags.
    ]
    config = ConfigParser.from_args(args=args, options=options)
    if config._use_tune:
        tuning_params = tuning_params(name=config["arch"]["type"])
        hyper_study(main_config=config, tune_config=tuning_params, num_tune_samples=5)
    else:
        train_model(config)
