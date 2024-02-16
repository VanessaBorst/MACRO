import inspect
from matplotlib import pyplot as plt
from utils.tracker import ConfusionMatrixTracker, MetricTracker

import argparse
import copy
import csv

import time
import os
import pickle
import pandas as pd
from pathlib import Path
import numpy as np

import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import global_config
from evaluate_cv_folds import split_dataset_into_folds, get_train_valid_test_indices, load_config_and_setup_paths, \
    prepare_result_data_structures, setup_cross_fold

# fix random seeds for reproducibility
from train import _set_seed
from utils.optimize_thresholds import optimize_ts, optimize_ts_based_on_roc_auc, optimize_ts_based_on_f1, \
    optimize_ts_manual

from data_loader.ecg_data_set import ECGDataset
from logger import update_logging_setup_for_tune_or_cross_valid

from utils import ensure_dir


# Should only be used with cv_active
def test_fold_with_thresholds(config,
                              cv_data_dir=None,
                              test_idx=None,
                              k_fold=None,
                              total_num_folds=None,
                              thresholds=None):
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

    if config['arch']['args']['multi_label_training']:
        import evaluation.multi_label_metrics_variedThreshold as module_metric
    else:
        raise NotImplementedError("Single label metrics haven't been checked after the Python update! Do not use them!")
        import evaluation.single_label_metrics as module_metric

    logger = config.get_logger('test_fold_' + str(k_fold))

    stratified_k_fold = config.config.get("data_loader", {}).get("cross_valid", {}).get("stratified_k_fold", False)
    data_loader = getattr(module_data, config['data_loader']['type'])(
        cv_data_dir,
        batch_size=64,
        shuffle=False,
        validation_split=0.0,
        num_workers=4,
        cross_valid=True,
        test_idx=test_idx,
        cv_train_mode=False,
        fold_id=k_fold,
        total_num_folds=total_num_folds,
        stratified_k_fold=stratified_k_fold
    )

    # build model architecture
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # Load the model from the checkpoint
    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # Prepare the model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    loss_fn = getattr(module_loss, config['loss']['type'])

    metrics_iter = [getattr(module_metric, met) for met in ['sk_subset_accuracy']]
    metrics_epoch = [getattr(module_metric, met) for met in ['cpsc_score',
                                                             'weighted_torch_roc_auc',
                                                             'weighted_torch_acc',
                                                             'macro_torch_roc_auc',
                                                             'macro_torch_acc']]
    metrics_epoch_class_wise = [getattr(module_metric, met) for met in ['class_wise_torch_roc_auc',
                                                                        'class_wise_torch_acc']]

    multi_label_training = config['arch']['args']['multi_label_training']
    class_labels = data_loader.dataset.class_labels

    # Store potential parameters needed for metrics
    _param_dict = {
        "thresholds": thresholds,
        "labels": class_labels,
        "device": device,
        "sigmoid_probs": config["metrics"]["additional_metrics_args"].get("sigmoid_probs", False),
        "log_probs": config["metrics"]["additional_metrics_args"].get("log_probs", False),
        "logits": config["metrics"]["additional_metrics_args"].get("logits", False),
        "pos_weights": data_loader.dataset.get_ml_pos_weights(
            idx_list=list(range(len(data_loader.sampler))),
            mode='test',
            cross_valid_active=True),
        "class_weights": data_loader.dataset.get_inverse_class_frequency(
            idx_list=list(range(len(data_loader.sampler))),
            multi_label_training=multi_label_training,
            mode='test',
            cross_valid_active=True),
        "class_freqs": data_loader.dataset.get_class_frequency(
            idx_list=list(range(len(data_loader.sampler))),
            multi_label_training=multi_label_training,
            mode='test',
            cross_valid_active=True),
        "lambda_balance": config["loss"]["add_args"].get("lambda_balance", 1)
    }

    # Setup visualization writer instance
    # writer = TensorboardWriter(config.test_output_dir, logger, config['trainer']['tensorboard'])

    # Set up confusion matrices tracker
    cm_tracker = ConfusionMatrixTracker(*class_labels, writer=None,
                                        multi_label_training=multi_label_training)

    # Set up metric tracker
    keys_iter = [m.__name__ for m in metrics_iter]
    keys_epoch = [m.__name__ for m in metrics_epoch]
    keys_epoch_class_wise = [m.__name__ for m in metrics_epoch_class_wise]
    metric_tracker = MetricTracker(keys_iter=['loss'] + keys_iter, keys_epoch=keys_epoch,
                                   keys_epoch_class_wise=keys_epoch_class_wise,
                                   labels=class_labels,
                                   writer=None)

    with torch.no_grad():

        # Store the intermediate targets. Always store the output scores
        outputs_list = []
        targets_list = []
        targets_all_labels_list = [] if not multi_label_training else None

        start = time.time()

        for batch_idx, (padded_records, _, first_labels, labels_one_hot, record_names) in \
                enumerate(tqdm(data_loader)):
            if multi_label_training:
                data, target = padded_records.to(device), labels_one_hot.to(device)
            else:
                # target contains the first GT label, target_all_labels contains all labels in 1-hot-encoding
                data, target, target_all_labels = padded_records.to(device), first_labels.to(device), \
                    labels_one_hot.to(device)

            data = data.permute(0, 2, 1)  # switch seq_len and feature_size (12 = #leads)
            output = model(data)

            multi_lead_branch_active = False
            if type(output) is tuple:
                if isinstance(output[1], list):
                    # multi-branch network
                    # first element is the overall network output, the second one a list of the single lead branches
                    multi_lead_branch_active = True
                    output, single_lead_outputs = output
                    # detached_single_lead_outputs = torch.stack(single_lead_outputs).detach().cpu()
                else:
                    # single-branch network
                    output, attention_weights = output

            # Detach tensors needed for further tracing and metrics calculation to remove them from the graph
            detached_output = output.detach().cpu()
            detached_target = target.detach().cpu()
            if not multi_label_training:
                detached_target_all_labels = target_all_labels.detach().cpu()

            outputs_list.append(detached_output)
            targets_list.append(detached_target)
            if not multi_label_training:
                targets_all_labels_list.append(detached_target_all_labels)

            # Compute the loss on the test set
            args = inspect.signature(loss_fn).parameters.values()
            # Output and target are needed for all metrics! Only consider other args WITHOUT default
            additional_args = [arg.name for arg in args
                               if arg.name not in ('output', 'target') and arg.default is arg.empty]

            if not multi_lead_branch_active:
                additional_kwargs = {
                    param_name: _param_dict[param_name] for param_name in additional_args
                }
                loss = loss_fn(output=output, target=target, **additional_kwargs)
            else:
                # Ensure that self.criterion is a function, namely multi_branch_BCE_with_logits
                assert callable(loss_fn) and loss_fn.__name__ == "multi_branch_BCE_with_logits", \
                    "For the multibranch network, the multibranch BCE with logits loss function has to be used!"

                assert additional_args == ['single_lead_outputs', 'lambda_balance'], \
                    "Something went wrong with the kwargs"

                additional_kwargs = {
                    "lambda_balance": _param_dict["lambda_balance"]
                }

                # Calculate the joint loss of each single lead branch and the overall network
                loss = loss_fn(target=target, output=output,
                               single_lead_outputs=single_lead_outputs,
                               **additional_kwargs)

            # batch_size = data.shape[0]
            # total_loss += loss.item() * batch_size
            metric_tracker.iter_update('loss', loss.item(), n=output.shape[0])

            # Compute the the iteration-based metrics on test set
            for i, met in enumerate(metrics_iter):
                args = inspect.signature(met).parameters.values()
                # Output and target are needed for all metrics! Only consider other args WITHOUT default
                additional_args = [arg.name for arg in args
                                   if arg.name not in ('output', 'target') and arg.default is arg.empty]
                additional_kwargs = {
                    param_name: _param_dict[param_name] for param_name in additional_args
                }
                metric_tracker.iter_update(met.__name__, met(target=detached_target, output=detached_output,
                                                             **additional_kwargs), n=output.shape[0])
                # total_metrics_iter[i] += met(output=detached_output, target=detached_target,
                #                              **additional_kwargs) * batch_size

    # Get detached tensors from the list for further evaluation
    # For this, create a tensor from the dynamically filled list
    det_outputs = torch.cat(outputs_list).detach().cpu()
    det_targets = torch.cat(targets_list).detach().cpu()
    det_targets_all_labels = torch.cat(
        targets_all_labels_list).detach().cpu() if not multi_label_training else None

    # ------------ Metrics ------------------------------------
    if len(metrics_epoch) > 0 or len(metrics_epoch_class_wise) > 0:
        # Finally, the epoch-based metrics need to be updated
        # For this, calculate both, the normal epoch-based metrics as well as the class-wise epoch-based metrics
        for met in metrics_epoch:
            args = inspect.signature(met).parameters.values()
            # Output and target are needed for all metrics! Only consider other args WITHOUT default
            additional_args = [arg.name for arg in args
                               if arg.name not in ('output', 'target') and arg.default is arg.empty]
            additional_kwargs = {
                param_name: _param_dict[param_name] for param_name in additional_args
            }
            if not multi_label_training and met.__name__ == 'cpsc_score':
                # Consider all labels for evaluation, even in the single label case
                metric_tracker.epoch_update(met.__name__, met(target=det_targets_all_labels, output=det_outputs,
                                                              **additional_kwargs))
            else:
                metric_tracker.epoch_update(met.__name__, met(target=det_targets, output=det_outputs,
                                                              **additional_kwargs))

        # This holds for the class-wise, epoch-based metrics as well
        for met in metrics_epoch_class_wise:
            args = inspect.signature(met).parameters.values()
            # Output and target are needed for all metrics! Only consider other args WITHOUT default
            additional_args = [arg.name for arg in args
                               if arg.name not in ('output', 'target') and arg.default is arg.empty]
            additional_kwargs = {
                param_name: _param_dict[param_name] for param_name in additional_args
            }
            metric_tracker.class_wise_epoch_update(met.__name__, met(target=det_targets, output=det_outputs,
                                                                     **additional_kwargs))

    # ------------ ROC Plots ------------------------------------
    if config['arch']['args']['multi_label_training']:
        fpr, tpr, thresholds = module_metric.torch_roc(output=det_outputs, target=det_targets,
                                                       sigmoid_probs=_param_dict["sigmoid_probs"],
                                                       logits=_param_dict["logits"], labels=_param_dict["labels"])
        roc_auc_scores = module_metric.class_wise_torch_roc_auc(output=det_outputs, target=det_targets,
                                                                sigmoid_probs=_param_dict["sigmoid_probs"],
                                                                logits=_param_dict["logits"],
                                                                labels=_param_dict["labels"])
    else:
        fpr, tpr, thresholds = module_metric.torch_roc(output=det_outputs, target=det_targets,
                                                       log_probs=_param_dict["log_probs"],
                                                       logits=_param_dict["logits"], labels=_param_dict["labels"])
        roc_auc_scores = module_metric.class_wise_torch_roc_auc(output=det_outputs, target=det_targets,
                                                                log_probs=_param_dict["log_probs"],
                                                                logits=_param_dict["logits"],
                                                                labels=_param_dict["labels"])

    fig, axs = plt.subplots(3, 3, figsize=(15, 10))
    axis_0 = 0
    axis_1 = 0
    line_width = 2
    target_names = ["IAVB", "AF", "LBBB", "PAC", "RBBB", "SNR", "STD", "STE", "VEB"]
    desired_order = ['SNR', 'AF', 'IAVB', 'LBBB', 'RBBB', 'PAC', 'VEB', 'STD', 'STE']
    class_shares = {'SNR': ' (12.5%)',
                    'AF': ' (16.6%)',
                    'IAVB': ' (9.8%)',
                    'LBBB': ' (3.2%)',
                    'RBBB': ' (25.2%)',
                    'PAC': ' (8.4%)',
                    'PVC': ' (9.5%)',
                    'STD': ' (11.8%)',
                    'STE': ' (3.0%)'}
    for i in range(0, 9):
        desired_class = desired_order[i]
        idx = target_names.index(desired_class)
        fpr_class_i = fpr[idx].numpy()
        tpr_class_i = tpr[idx].numpy()
        # Scale values by a factor of 1000 to better match the cpsc raw values
        axs[axis_0, axis_1].plot(fpr_class_i, tpr_class_i, color='darkorange', lw=line_width,
                                 label='ROC curve (area = %0.3f)' % roc_auc_scores[idx])
        axs[axis_0, axis_1].plot([0, 1], [0, 1], color='navy', lw=line_width, linestyle='--')
        axs[axis_0, axis_1].set_xlabel('False Positive Rate')
        axs[axis_0, axis_1].set_ylabel('True Positive Rate')
        axs[axis_0, axis_1].set_xlim([0.0, 1.0])
        axs[axis_0, axis_1].set_ylim([0.0, 1.05])
        axs[axis_0, axis_1].legend(loc="lower right")

        class_name = str(target_names[idx]).replace('VEB', 'PVC')
        axs[axis_0, axis_1].set_title('ROC curve for class ' + class_name + str(class_shares[class_name]))
        # Also save the single plots per class
        file_name = 'roc_curve_' + class_name + '.pdf'
        extent = axs[axis_0, axis_1].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        # Pad the saved area by 30% in the x-direction and 35% in the y-direction
        fig.savefig(config.test_output_dir / file_name, bbox_inches=extent.expanded(1.3, 1.35))
        axis_1 = (axis_1 + 1) % 3
        if axis_1 == 0:
            axis_0 += 1
    plt.tight_layout()
    plt.savefig(config.test_output_dir / "roc_curves.pdf")

    # ------------ Confusion matrices ------------------------
    # Dump the confusion matrices into a pickle file and write figures of them to file
    # 1) Update the confusion matrices maintained by the ClassificationTracker
    upd_class_wise_cms = module_metric.class_wise_confusion_matrices_multi_label_sk(output=det_outputs,
                                                                                    target=det_targets,
                                                                                    sigmoid_probs=_param_dict[
                                                                                        'sigmoid_probs'],
                                                                                    logits=_param_dict['logits'],
                                                                                    labels=_param_dict['labels'],
                                                                                    thresholds=_param_dict[
                                                                                        'thresholds'])
    cm_tracker.update_class_wise_cms(upd_class_wise_cms)
    # 2) Explicitly write a plot of the confusion matrices to a file
    cm_tracker.save_result_cms_to_file(config.test_output_dir)
    # Moreover, save them as pickle
    path_name = os.path.join(config.test_output_dir, "cms_test_model.p")
    with open(path_name, 'wb') as cm_file:
        all_cms = [cm_tracker.cm, cm_tracker.class_wise_cms]
        pickle.dump(all_cms, cm_file)

    # ------------------- Summary Report -------------------
    summary_dict = module_metric.sk_classification_summary(output=det_outputs, target=det_targets,
                                                                                 sigmoid_probs=_param_dict[
                                                                                     "sigmoid_probs"],
                                                                                 logits=_param_dict["logits"],
                                                                                 labels=_param_dict["labels"],
                                                                                 output_dict=True,
                                                                                 thresholds=_param_dict['thresholds'])

    # ------------------------------------Final Test Steps ---------------------------------------------
    df_sklearn_summary = pd.DataFrame.from_dict(summary_dict)
    df_metric_results = metric_tracker.result(include_epoch_metrics=True)

    df_class_wise_results = pd.DataFrame(
        columns=['IAVB', 'AF', 'LBBB', 'PAC', 'RBBB', 'SNR', 'STD', 'STE', 'VEB', 'macro avg', 'weighted avg'])
    df_class_wise_results = pd.concat([df_class_wise_results, df_sklearn_summary[
        ['IAVB', 'AF', 'LBBB', 'PAC', 'RBBB', 'SNR', 'STD', 'STE', 'VEB', 'macro avg', 'weighted avg']]])

    df_class_wise_metrics = df_metric_results.loc[df_metric_results.index.str.startswith(
        ('class_wise', 'weighted', 'macro'))]['mean'].to_frame()
    df_class_wise_metrics.index = df_class_wise_metrics.index.set_names('metric')
    df_class_wise_metrics.reset_index(inplace=True)
    # df_class_wise_metrics['metric'] = df_class_wise_metrics['metric'].apply(lambda x: str(x).replace("class_wise_", "").replace("_class", ""))

    metric_names = [met.__name__.replace("class_wise_", "") for met in metrics_epoch_class_wise]
    for metric_name in metric_names:
        df_temp = df_class_wise_metrics.loc[df_class_wise_metrics.metric.str.contains(metric_name)].transpose()
        df_temp = df_temp.rename(columns=df_temp.iloc[0]).drop(df_temp.index[0])
        df_temp.rename(index={'mean': metric_name}, inplace=True)
        cols = df_temp.columns.tolist()
        # Reorder the dataframe
        desired_order = []
        for i in range(0, 9):
            desired_order.append('class_wise_' + metric_name + '_class_' + str(i))
        if 'macro_' + metric_name in cols:
            desired_order.append('macro_' + metric_name)
        if 'weighted_' + metric_name in cols:
            desired_order.append('weighted_' + metric_name)
        # if 'micro' + metric_name in cols:
        #     desired_order.append('micro' + metric_name)
        df_temp = df_temp[desired_order]
        df_temp.columns = df_class_wise_results.columns
        df_class_wise_results = pd.concat([df_class_wise_results, df_temp], axis=0)

    idx = df_class_wise_results.index.drop('support').tolist() + ['support']
    df_class_wise_results = df_class_wise_results.reindex(idx)
    df_class_wise_results.loc['support'] = df_class_wise_results.loc['support'].apply(int)

    # Reorder the class columns of the dataframe to match the one used in the
    desired_col_order = ['SNR', 'AF', 'IAVB', 'LBBB', 'RBBB', 'PAC', 'VEB', 'STD', 'STE', 'macro avg', 'weighted avg']
    df_class_wise_results = df_class_wise_results[desired_col_order]

    df_single_metric_results = df_metric_results.loc[~df_metric_results.index.str.startswith(
        ('class_wise', 'weighted', 'macro'))]['mean'].to_frame().transpose()
    df_single_metric_results.rename(index={'mean': 'value'}, inplace=True)

    with open(os.path.join(config.test_output_dir, 'eval_class_wise.p'), 'wb') as file:
        pickle.dump(df_class_wise_results, file)

    with open(os.path.join(config.test_output_dir, 'eval_single_metrics.p'), 'wb') as file:
        pickle.dump(df_single_metric_results, file)

    with open(os.path.join(config.test_output_dir, 'eval_results.tex'), 'w') as file:
        df_class_wise_results.to_latex(buf=file, index=True, bold_rows=True, float_format="{:0.3f}".format)
        df_single_metric_results.to_latex(buf=file, index=True, bold_rows=True, float_format="{:0.3f}".format)

    end = time.time()
    ty_res = time.gmtime(end - start)
    res = time.strftime("%H hours, %M minutes, %S seconds", ty_res)

    eval_log = {'Runtime': res}
    eval_info_single_metrics = ', '.join(
        f"{key}: {str(value).split('Name')[0].split('value')[1]}" for key, value in df_single_metric_results.items())
    eval_info_class_wise_metrics = ', '.join(f"{key}: {value}" for key, value in df_class_wise_results.items())
    logger_info = f"{eval_log}\n{eval_info_single_metrics}\n{eval_info_class_wise_metrics}\n"
    logger.info(logger_info)

    return df_class_wise_results, df_single_metric_results


def fine_tune_thresholds_on_valid_set(config, cv_data_dir=None, valid_idx=None, k_fold=None, strategy=None):
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

    _set_seed(global_config.SEED)

    logger = config.get_logger('fine_tune_threshold_fold_' + str(k_fold))

    data_loader = getattr(module_data, config['data_loader']['type'])(
        cv_data_dir,
        batch_size=64,
        shuffle=False,
        validation_split=0.0,
        num_workers=4,
        cross_valid=True,
        test_idx=valid_idx,
        cv_train_mode=False,
        fold_id=k_fold
    )

    class_labels = data_loader.dataset.class_labels

    # build model architecture
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # Load the model from the checkpoint
    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # Prepare the model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    # Inference: Compute the raw network scores on the validation data
    with torch.no_grad():

        # Store the intermediate targets. Always store the output scores
        outputs_list = []
        targets_list = []

        for batch_idx, (padded_records, _, first_labels, labels_one_hot, record_names) in \
                enumerate(tqdm(data_loader)):
            data, target = padded_records.to(device), labels_one_hot.to(device)

            data = data.permute(0, 2, 1)  # switch seq_len and feature_size (12 = #leads)
            output = model(data)

            if type(output) is tuple:
                output, attention_weights = output

            # Detach tensors needed for further tracing and metrics calculation to remove them from the graph
            detached_output = output.detach().cpu()
            detached_target = target.detach().cpu()

            outputs_list.append(detached_output)
            targets_list.append(detached_target)

    # Get detached tensors from the list for fine-tuning the thresholds
    # For this, create a tensor from the dynamically filled list
    det_outputs = torch.cat(outputs_list).detach().cpu()
    det_targets = torch.cat(targets_list).detach().cpu()

    # ------------ Fine tune thresholds ------------------------------------
    match strategy:
        case "manual":
            thresholds = optimize_ts_manual(logits=det_outputs, target=det_targets, labels=class_labels)
        case "bayesianOptimization":
            # raise ValueError("The strategy for threshold optimization is not tested yet!")
            thresholds = optimize_ts(logits=det_outputs, target=det_targets, labels=class_labels)
        case "roc_auc":
            raise ValueError("The strategy for threshold optimization is not tested yet!")
            thresholds = optimize_ts_based_on_roc_auc(logits=det_outputs, target=det_targets, labels=class_labels)
        case "f1":
            raise ValueError("The strategy for threshold optimization is not tested yet!")
            thresholds = optimize_ts_based_on_f1(logits=det_outputs, target=det_targets, labels=class_labels)
        case _:
            # Should not occur
            raise ValueError("The strategy for threshold optimization is not valid!")

    return thresholds


def fine_tune_thresholds_cross_validation(main_path, strategy=None, includeTrain=False):
    sub_dir = f"threshold_tuning_{strategy}" if not includeTrain else f"threshold_tuning_{strategy}_includeTrain"
    config = load_config_and_setup_paths(main_path=main_path, sub_dir=sub_dir)

    base_config, base_log_dir, base_save_dir, data_dir, dataset, fold_data, total_num_folds = setup_cross_fold(config)

    # Save the results of each run
    class_wise_metrics, folds, test_results_class_wise, test_results_single_metrics, valid_results = \
        prepare_result_data_structures(total_num_folds, include_valid_results=True)

    print("Finetuning " + str(total_num_folds) + "-fold cross validation")
    start = time.time()
    valid_fold_index = total_num_folds - 2
    test_fold_index = total_num_folds - 1

    for k in range(total_num_folds):
        print("Starting fold " + str(k+1))
        # Get the idx for valid and test samples, train idx not needed
        train_idx, valid_idx, test_idx = get_train_valid_test_indices(main_path=main_path,
                                                                      dataset=dataset,
                                                                      fold_data=fold_data,
                                                                      k=k,
                                                                      test_fold_index=test_fold_index,
                                                                      valid_fold_index=valid_fold_index,
                                                                      merge_train_into_valid_idx=includeTrain)

        # Adapt the log and save paths for the current fold
        config.save_dir = Path(os.path.join(base_save_dir, "Fold_" + str(k + 1)))
        config.log_dir = Path(os.path.join(base_log_dir, "Fold_" + str(k + 1)))
        ensure_dir(config.save_dir)
        ensure_dir(config.log_dir)
        update_logging_setup_for_tune_or_cross_valid(config.log_dir)

        # Skip training and find the best thresholds on the validation set
        config.resume = Path(os.path.join(main_path, "Fold_" + str(k + 1), "model_best.pth"))

        ############################### THRESHOLD TUNING ###############################
        thresholds = fine_tune_thresholds_on_valid_set(config, cv_data_dir=data_dir,
                                                       valid_idx=valid_idx, k_fold=k,
                                                       strategy=strategy) \
            if strategy is not None else [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]   # Default
        with open(os.path.join(config.save_dir, "thresholds.csv"), "w") as file:
            wr = csv.writer(file, quoting=csv.QUOTE_ALL)
            wr.writerow(thresholds)
        ############################### END THRESHOLD TUNING ###############################

        # Do the testing with the fine_tuned thresholds and add the fold results to the dfs
        config.test_output_dir = Path(os.path.join(config.save_dir, 'test_output_fold_' + str(k + 1)))
        ensure_dir(config.test_output_dir)
        fold_eval_class_wise, fold_eval_single_metrics = test_fold_with_thresholds(config,
                                                                                   cv_data_dir=data_dir,
                                                                                   test_idx=test_idx,
                                                                                   k_fold=k,
                                                                                   total_num_folds=total_num_folds,
                                                                                   thresholds=thresholds)
        # Class-Wise Metrics
        fold_eval_class_wise = fold_eval_class_wise.rename(index={'torch_acc': 'torch_accuracy'})
        test_results_class_wise.loc[(folds[k], fold_eval_class_wise.index), fold_eval_class_wise.columns] = \
            fold_eval_class_wise.values
        # Single Metrics
        pd_series = fold_eval_single_metrics.loc['value']
        pd_series.name = folds[k]
        test_results_single_metrics = test_results_single_metrics.append(pd_series)

        # Update the indices and reset the config (including resume!)
        valid_fold_index = (valid_fold_index + 1) % (total_num_folds)
        test_fold_index = (test_fold_index + 1) % (total_num_folds)
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

    print("Finished cross-fold-validation threshold fine-tuning")
    print("Consuming time: " + str(res))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MACRO Paper: Threshold Finetuning for CV')
    parser.add_argument('-p', '--path', default=None, type=str,
                        help='main path to CV runs(default: None)')
    parser.add_argument('--strategy', default=None, type=str,
                        help='strategy to use for threshold optimization (default: None)')
    parser.add_argument('--includeTrain',  action='store_true',
                        help='Set this flag if the training data should be included for threshold optimization '
                             '(default: False)')
    args = parser.parse_args()
    fine_tune_thresholds_cross_validation(args.path, args.strategy, args.includeTrain)
