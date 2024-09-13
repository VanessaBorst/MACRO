import argparse
import collections
from datetime import datetime
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
import global_config
import loss.loss as module_loss
from logger import update_logging_setup_for_tune_or_cross_valid
from parse_config import ConfigParser
from trainer.ecg_trainer import ECGTrainer
from utils import prepare_device, get_project_root, ensure_dir

# Needed for working with SSH Interpreter...
import os
import torch

global_config.suppress_warnings()
os.environ["CUDA_VISIBLE_DEVICES"] = global_config.CUDA_VISIBLE_DEVICES
TUNE_TEMP_DIR = global_config.TUNE_TEMP_DIR


def _set_seed(SEED):
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    random.seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    # Note: If sparse or entmax are not used at the end, warn only can be set to false again!
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _get_mid_kernel_size_second_conv_blocks(spec):
    # Choose the same size for the second block as well to reduce the amount of hyperparams
    return spec.config.mid_kernel_size_first_conv_blocks


def tuning_params(name):
    # The parameters for the tuning can be specified here according to the example scheme below
    if name == "BaselineModelWithMHAttention":
        return {
            "dropout_attention": tune.grid_search([0.2, 0.3]),
            "heads": tune.grid_search([6]),
            "gru_units": tune.grid_search([12])
            # "dropout_attention": tune.grid_search([0.2, 0.3]),
            # "heads": tune.grid_search([6, 8, 12]),
            # "gru_units": tune.grid_search([12, 24, 36])
        }
    elif name == "FinalModel":
        return {
            "dropout_attention": tune.grid_search([0.2, 0.3, 0.4]),
            "heads": tune.grid_search([6, 8, 12])
            #"gru_units": tune.grid_search([12, 24]),
        }
    elif name == "FinalModelMultiBranch":
        return {
            # BranchNet specifics
            "branchNet_reduce_channels": tune.grid_search([True, False]),
            "branchNet_heads": tune.grid_search([6, 8]),
            "branchNet_attention_dropout": tune.grid_search([0.2, 0.4]),
            # Multibranch specifics
            "multi_branch_heads": tune.grid_search([12, 24]),
            "multi_branch_attention_dropout": tune.grid_search([0.2, 0.4]),
            "use_conv_reduction_block": True,
            "conv_reduction_first_kernel_size": tune.grid_search([3, 8, 16]),
            "conv_reduction_second_kernel_size": tune.grid_search([3, 8, 16]),
            "conv_reduction_third_kernel_size": tune.grid_search([3, 8, 16]),
        }
    else:
        return None


class MyTuneCallback(Callback):

    def __init__(self):
        self.already_seen = set()
        self.manager = TuneClient(tune_address="127.0.0.1", port_forward=4321)

    def setup(self):
        # Load the already seen configurations to avoid that the experiments are repeated here
        # Useful, if training was interrupted and the same experiment should be continued
        seen_configs = []
        for config in seen_configs:
            self.already_seen.add(str(config))

    def on_trial_start(self, iteration, trials, trial, **info):
        if str(trial.config) in self.already_seen:
            print("Stop trial with id " + str(trial.trial_id))
            self.manager.stop_trial(trial.trial_id)
        else:
            self.already_seen.add(str(trial.config))


def hyper_study(main_config, tune_config, num_tune_samples=1):
    def name_trial(trial):
        file_name = f""
        for key in tune_config.keys():
            file_name += f"{trial.config[key]}_"
        if len(file_name) > 240:
            file_name = file_name[:240]
        file_name += datetime.now().strftime('%H-%M-%S.%f')
        return file_name

    data_dir = main_config['data_loader']['args']['data_dir']
    full_data_dir = os.path.join(str(get_project_root()), data_dir)

    validation_split = main_config['data_loader']['args']['validation_split']
    if isinstance(validation_split, str):
        validation_split = os.path.join(str(get_project_root()), validation_split)

    def train_fn(config, checkpoint_dir=None):
        global_config.suppress_warnings()

        # Without this, the valid split varies from worker to worker!
        np.random.seed(global_config.SEED)
        torch.manual_seed(global_config.SEED)
        random.seed(global_config.SEED)
        torch.cuda.manual_seed_all(global_config.SEED)
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
        # Since entmax is used, warn only needs to be set to True
        torch.use_deterministic_algorithms(True, warn_only=True)
        # Each of the following two aspects would lead to a TypeError !
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

        data_loader = main_config.init_obj('data_loader', module_data_loader,
                                           data_dir=full_data_dir, validation_split=validation_split)
        valid_data_loader = data_loader.split_validation()

        print(f"Len Dataloader {len(data_loader.dataset)}")
        print(f"Len Validloader {len(valid_data_loader.dataset)}")

        # Check data splitting by record name -> should be equal across all workers and tune runs
        valid_records = []
        for idx in valid_data_loader.sampler.indices:
            valid_records.append(valid_data_loader.dataset[idx][4])
        valid_records.sort()

        project_root = get_project_root()
        ensure_dir(os.path.join(project_root, 'data_loader', 'tune_log'))

        dataset = data_dir.split("/")[1]
        print(f"Tune run on dataset {dataset}")
        suffix = f"{dataset}" if dataset == "CinC_CPSC" \
            else f"{dataset}_{data_dir.split('/')[2].split('_')[0]}"

        if Path(os.path.join(project_root, 'data_loader', 'tune_log', f'valid_records_tune_train_fn_{suffix}.txt')).is_file():
            # If the file already exists, check if the records are the same
            with open(os.path.join(project_root, 'data_loader', 'tune_log', f'valid_records_tune_train_fn_{suffix}.txt'),
                      "r") as txt_file:
                lines = txt_file.read().splitlines()
                assert valid_records == lines, "Data Split Error! Check this again!"
        else:
            with open(os.path.join(project_root, 'data_loader', 'tune_log', f'valid_records_tune_train_fn_{suffix}.txt'),
                      "w+") as txt_file:
                for line in valid_records:
                    txt_file.write("".join(line) + "\n")

        train_model(config=main_config, tune_config=config, train_dl=data_loader, valid_dl=valid_data_loader,
                    checkpoint_dir=checkpoint_dir, use_tune=True)

    ray.init(_temp_dir=os.path.join(TUNE_TEMP_DIR, 'ray_tmp'))

    trainer = main_config['trainer']
    early_stop = trainer.get('monitor', 'off')
    if early_stop != 'off':
        mnt_mode, mnt_metric = early_stop.split()
    else:
        mnt_mode = "min"
        mnt_metric = "val_loss"

    # The reporter is used to print the results of the tuning to the console
    # It can be configured here according to the example scheme below
    if main_config["arch"]["type"] == "BaselineModelWithMHAttention":
        reporter = CLIReporter(
            parameter_columns={
                "dropout_attention": "Droput MH Attention",
                "heads": "Num Heads",
                "gru_units": "Num Units GRU"
            },
            metric_columns=["loss", "val_loss",
                            "val_macro_sk_f1",
                            "val_weighted_sk_f1",
                            "training_iteration"])
    elif main_config["arch"]["type"] == "FinalModel":
        reporter = CLIReporter(
            # MACRO
            parameter_columns={
                "dropout_attention": "Droput MH Attention",
                "heads": "H",
                "gru_units": "GRU"
            },
            metric_columns=["loss", "val_loss",
                            "val_macro_sk_f1",
                            "val_weighted_sk_f1",
                            "training_iteration"])
    elif main_config["arch"]["type"] == "FinalModelMultiBranch":
        reporter = CLIReporter(
            parameter_columns={
                "branchNet_reduce_channels": "BN_Rdc",
                "branchNet_heads": "BN_H",
                "branchNet_attention_dropout": "BN_DP",
                # Multibranch specifics
                "multi_branch_heads": "MB_H",
                "multi_branch_attention_dropout": "MB_DP",
                "use_conv_reduction_block": "MB_ConvRed",
                "conv_reduction_first_kernel_size": "ConvRed_1st",
                "conv_reduction_second_kernel_size": "ConvRed_2nd",
                "conv_reduction_third_kernel_size": "ConvRed_3rd"
            },
            metric_columns=["loss", "val_loss",
                            "val_macro_sk_f1",
                            "val_weighted_sk_f1",
                            "training_iteration"])

    # The number of GPUs to use depends on the architecture
    match main_config["arch"]["type"]:
        case "BaselineModelWithMHAttention":
            # Six trials in parallel
            num_gpu = 0.16
        case "FinalModel":
            # Five trials in parallel
            num_gpu = 0.2
        case "FinalModelMultiBranch":
            # Two trials in parallel
            num_gpu = 0.5
        case "BaselineModelWithSkipConnectionsV2" | "BaselineModelWithSkipConnectionsAndNormV2" | \
             "BaselineModelWithSkipConnectionsAndNormPreActivation":
            # One trial at a time
            num_gpu = 1
        case _:
            # Default: Four trials in parallel
            num_gpu = 0.25

    analysis = tune.run(
        run_or_experiment=train_fn,
        num_samples=num_tune_samples,
        name=str(main_config.save_dir),  # experiment_name
        trial_dirname_creator=name_trial,  # trial_name
        storage_path=str(main_config.save_dir),
        sync_config=tune.SyncConfig(
            syncer="auto",
            # Sync approximately every minute rather than on every checkpoint
            sync_on_checkpoint=False,
            sync_period=60,
        ),

        metric=mnt_metric,
        mode=mnt_mode,


        search_alg=BasicVariantGenerator(),
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

    # Get a dataframe for the max metric score seen for each trial
    df = analysis.dataframe(metric=mnt_metric, mode=mnt_mode)
    with open(os.path.join(main_config.save_dir, "best_per_trial.p"), "wb") as file:
        pickle.dump(df, file)



def train_model(config, tune_config=None, train_dl=None, valid_dl=None, checkpoint_dir=None, use_tune=False,
                train_idx=None, valid_idx=None, k_fold=None, total_num_folds=None, cv_active=False):
    """
      Train the model based on the provided configuration.
       Args:
           config (ConfigParser): Can be used as usual
           tune_config (dict, optional): Configuration dictionary for tuning
                                        (contains the tune params with the samples values). Defaults to None.
           train_dl (DataLoader, optional): Training data loader. Defaults to None.
           valid_dl (DataLoader, optional): Validation data loader. Defaults to None.
           checkpoint_dir (str, optional): Directory containing checkpoint for resuming training/testing. Defaults to None.
           use_tune (bool, optional): Whether to use tuning. Defaults to False.
           train_idx (list, optional): Indices for training data in cross-validation. Defaults to None.
           valid_idx (list, optional): Indices for validation data in cross-validation. Defaults to None.
           k_fold (int, optional): Current fold in cross-validation. Defaults to None.
           total_num_folds (int, optional): Total number of folds in cross-validation. Defaults to None.
           cv_active (bool, optional): Whether cross-validation is active. Defaults to False.

       Returns:
           dict: Dictionary containing the best metrics achieved during training.
   """

    import torch  # Needed to work with asych. tune workers as well

    assert use_tune is False or cv_active is False, "Cross Validation does not work with active tuning!"

    if use_tune:
        # When using Ray Tune, this is distributed among worker processes, which requires seeding within the function
        # Otherwise the same config may to different results -> not reproducible
        np.random.seed(global_config.SEED)
        torch.manual_seed(global_config.SEED)
        random.seed(global_config.SEED)
        torch.cuda.manual_seed_all(global_config.SEED)
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
        # Since entmax is used, warn only needs to be set to True!
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        os.environ["CUDA_VISIBLE_DEVICES"] = global_config.CUDA_VISIBLE_DEVICES

    # Conditional inputs depending on the config
    if config['arch']['type'] == 'BaselineModel':
        import model.baseline_model as module_arch
    elif config['arch']['type'] == 'BaselineModelWithMHAttention':
        import model.baseline_model_with_MHAttention as module_arch
    elif config['arch']['type'] == 'BaselineModelWithSkipConnectionsAndNormPreActivation':
        import model.baseline_model_with_skips_and_norm_pre_activation_design as module_arch
    elif config['arch']['type'] == 'FinalModel':
        import model.final_model as module_arch
    elif config['arch']['type'] == 'FinalModelMultiBranch':
        import model.final_model_multibranch as module_arch

    if config['arch']['args']['multi_label_training']:
        import evaluation.multi_label_metrics as module_metric
    else:
        # raise NotImplementedError("Single label metrics haven't been checked after the Python update! Do not use them!")
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
                                      fold_id=k_fold, total_num_folds=total_num_folds)
        valid_data_loader = data_loader.split_validation()
    else:
        data_loader = config.init_obj('data_loader', module_data_loader)
        valid_data_loader = data_loader.split_validation()

    # build model architecture, then print to console
    if tune_config is None:
        model = config.init_obj('arch', module_arch)
    else:
        model = config.init_obj('arch', module_arch, **tune_config)
    logger.info(model)

    if config.config.get("resume", None) is not None:
        # Load the model from the checkpoint
        logger.info('Loading checkpoint: {} ...'.format(config.config.get("resume")))
        checkpoint = torch.load(config.config["resume"],
                            map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        state_dict = checkpoint['state_dict']
        desired_dict = model.state_dict()
        pre_train_dict={k:v for k,v in state_dict.items() if k in desired_dict.keys()
                        and not k.endswith(("_fcn.weight","_fcn.bias"))}
        desired_dict.update(pre_train_dict)
        model.load_state_dict(desired_dict)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config['n_gpu'])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # Get function handles of loss and metrics
    # Important: The method config['loss'] must exist in the loss module (<module 'loss.loss' >)
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
    args = argparse.ArgumentParser(description='MACRO Paper: Single Training Run')
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

    # fix random seeds for reproducibility
    global_config.SEED = config.config.get("SEED", global_config.SEED)
    _set_seed(global_config.SEED)

    if config.use_tune:
        tuning_params = tuning_params(name=config["arch"]["type"])
        # With grid search, only 1 times ! -> # Set num_samples to 1, as grid search generates all combination
        hyper_study(main_config=config, tune_config=tuning_params, num_tune_samples=1)
    else:
        train_model(config)
