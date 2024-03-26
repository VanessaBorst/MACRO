import argparse
import collections
import inspect
import json
import pickle
import time
from pathlib import Path

import pandas as pd

# Needed for working with SSH Interpreter...
import os

import global_config
import torch

from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from tqdm import tqdm

import data_loader.data_loaders as module_data
import loss.loss as module_loss
from evaluation import multi_label_metrics, single_label_metrics
from evaluation.multi_label_metrics import class_wise_confusion_matrices_multi_label_sk, THRESHOLD
from evaluation.single_label_metrics import overall_confusion_matrix_sk, class_wise_confusion_matrices_single_label_sk
from parse_config import ConfigParser

from train import _set_seed
from utils.tracker import ConfusionMatrixTracker, MetricTracker

os.environ["CUDA_VISIBLE_DEVICES"] = global_config.CUDA_VISIBLE_DEVICES


def test_model(config, tune_config=None, cv_active=False, cv_data_dir=None,
               test_idx=None, k_fold=None, total_num_folds= None):
    assert not cv_active or tune_config is None, "For cross validation, please specifiy ALL params in the main config"

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
        import evaluation.multi_label_metrics as module_metric
    else:
        raise NotImplementedError("Single label metrics haven't been checked after the Python update! Do not use them!")
        import evaluation.single_label_metrics as module_metric

    if not cv_active:
        logger = config.get_logger('test')
    else:
        logger = config.get_logger('test_fold_' + str(k_fold))

    if not cv_active:
        # setup data_loader instances
        data_loader = getattr(module_data, config['data_loader']['type'])(
            config['data_loader']['test_dir'],
            batch_size=64,
            shuffle=False,
            validation_split=0.0,
            num_workers=4,
            cross_valid=False,
            test_idx=None,
            cv_train_mode=False,
            fold_id=None
        )
    else:
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
            total_num_folds=total_num_folds
        )

    # build model architecture
    if tune_config is None:
        model = config.init_obj('arch', module_arch)
    else:
        model = config.init_obj('arch', module_arch, **tune_config)
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

    # get function handles of loss
    # Important: The method config['loss'] must exist in the loss module (<module 'model.loss' >)
    loss_fn = getattr(module_loss, config['loss']['type'])

    # HARD-CODE the metrics to calc here
    # The sk-summary report is used automatically
    # Thus, the support and class-wise, w-avg and micro-avg for Precision, Recall, F1 are always contained and don't
    # need to be specified below
    # !!!!! IMPORTANT!!!!!
    # For each additional metric specified in "metrics_epoch_class_wise", there should be two corresponding entries
    # in "metrics_epoch" containing the macro and the weighted average (otherwise results can not be merged)
    if config['arch']['args']['multi_label_training']:
        metrics_iter = [getattr(module_metric, met) for met in ['sk_subset_accuracy']]
        metrics_epoch = [getattr(module_metric, met) for met in ['weighted_torch_roc_auc',
                                                                 'weighted_torch_accuracy',
                                                                 'macro_torch_roc_auc',
                                                                 'macro_torch_accuracy']]
        metrics_epoch_class_wise = [getattr(module_metric, met) for met in ['class_wise_torch_roc_auc',
                                                                            'class_wise_torch_accuracy']]
    else:
        metrics_iter = [getattr(module_metric, met) for met in ['sk_accuracy']]
        metrics_epoch = [getattr(module_metric, met) for met in ['weighted_torch_roc_auc',
                                                                 'weighted_torch_accuracy',
                                                                 'macro_torch_roc_auc',
                                                                 'macro_torch_accuracy']]
        metrics_epoch_class_wise = [getattr(module_metric, met) for met in ['class_wise_torch_roc_auc',
                                                                            'class_wise_torch_accuracy']]

    multi_label_training = config['arch']['args']['multi_label_training']
    class_labels = data_loader.dataset.class_labels


    # Store potential parameters needed for metrics
    _param_dict = {
        "labels": class_labels,
        "device": device,
        "log_probs": config["arch"]["args"].get("apply_final_activation", False),
        "logits": not config["arch"]["args"].get("apply_final_activation", False),
        "pos_weights": data_loader.dataset.get_ml_pos_weights(
            idx_list=list(range(len(data_loader.sampler))),
            mode='test',
            cross_valid_active=cv_active),
        "class_weights": data_loader.dataset.get_inverse_class_frequency(
            idx_list=list(range(len(data_loader.sampler))),
            multi_label_training=multi_label_training,
            mode='test',
            cross_valid_active=cv_active),
        "lambda_balance": config["loss"]["add_args"].get("lambda_balance", 1),
        "gamma_neg": config["loss"]["add_args"].get("gamma_neg", 4),
        "gamma_pos": config["loss"]["add_args"].get("gamma_pos", 1),
        "clip": config["loss"]["add_args"].get("clip", 0.05),
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
                assert callable(loss_fn) and loss_fn.__name__ \
                       in ["multi_branch_BCE_with_logits", "multi_branch_asymmetric_loss_with_logits"],\
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

            # Compute the iteration-based metrics on test set
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

    # n_samples = len(data_loader.sampler)
    # log = {'loss': total_loss / n_samples}
    # log.update({
    #     met.__name__: total_metrics_iter[i].item() / n_samples for i, met in enumerate(metrics_iter)
    # })

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
                                                       logits=_param_dict["logits"], labels=_param_dict["labels"])
        roc_auc_scores = module_metric.class_wise_torch_roc_auc(output=det_outputs, target=det_targets,
                                                                logits=_param_dict["logits"],
                                                                labels=_param_dict["labels"])

        # TODO can be removed later again
        if "additional" in config.test_output_dir.name:
            # Save det_outputs and det_targets to file for further analysis
            with open(config.test_output_dir / "det_outputs.p", 'wb') as file:
                pickle.dump(det_outputs, file)
            with open(config.test_output_dir / "det_targets.p", 'wb') as file:
                pickle.dump(det_targets, file)
    else:
        fpr, tpr, thresholds = module_metric.torch_roc(output=det_outputs, target=det_targets,
                                                       log_probs=_param_dict["log_probs"],
                                                       logits=_param_dict["logits"], labels=_param_dict["labels"])
        roc_auc_scores = module_metric.class_wise_torch_roc_auc(output=det_outputs, target=det_targets,
                                                                log_probs=_param_dict["log_probs"],
                                                                logits=_param_dict["logits"],
                                                                labels=_param_dict["labels"])

    fig, axs = plt.subplots(3, 3, figsize=(18, 10))
    axis_0 = 0
    axis_1 = 0
    line_width = 2
    target_names = ["IAVB", "AF", "LBBB", "PAC", "RBBB", "SNR", "STD", "STE", "VEB"]
    desired_order = ['SNR', 'AF', 'IAVB', 'LBBB', 'RBBB', 'PAC', 'VEB', 'STD', 'STE']
    class_shares = {'SNR': ' (12.5%)',
                    'AF': ' (16.6%)',
                    'I-AVB': ' (9.8%)',
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
                                 label='ROC curve (AUC = %0.3f)' % roc_auc_scores[idx])
        axs[axis_0, axis_1].plot([0, 1], [0, 1], color='navy', lw=line_width, linestyle='--')
        axs[axis_0, axis_1].tick_params(axis='both', which='major', labelsize=20)
        axs[axis_0, axis_1].set_yticks([0.25, 0.5, 0.75, 1])

        if axis_0 == 2:
            axs[axis_0, axis_1].set_xlabel('False Positive Rate', fontsize=20)
        if axis_1 == 0:
            axs[axis_0, axis_1].set_ylabel('True Positive Rate', fontsize=20)

        axs[axis_0, axis_1].set_xlim([0.0, 1.0])
        axs[axis_0, axis_1].set_ylim([0.0, 1.05])
        axs[axis_0, axis_1].legend(loc="lower right", fontsize=20)

        class_name = str(target_names[idx]).replace('IAVB', 'I-AVB').replace('VEB', 'PVC')
        axs[axis_0, axis_1].set_title('Class ' + class_name + class_shares[class_name], fontsize=20)
        # Also save the single plots per class
        file_name = 'roc_curve_' + target_names[idx] + '.pdf'
        # 'ROC curve for class ' + str(target_names[idx]) + str())
        extent = axs[axis_0, axis_1].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        # Pad the saved area by 30% in the x-direction and 35% in the y-direction
        fig.savefig(config.test_output_dir / file_name, bbox_inches=extent.expanded(1.3, 1.35))
        axis_1 = (axis_1 + 1) % 3
        if axis_1 == 0:
            axis_0 += 1
    plt.tight_layout(pad=2, h_pad=3, w_pad=1.5)
    plt.savefig(config.test_output_dir / "roc_curves_with_shares.pdf")

    # ------------ Confusion matrices ------------------------
    # Dump the confusion matrices into a pickle file and write figures of them to file
    # 1) Update the confusion matrices maintained by the ClassificationTracker
    if not multi_label_training:
        upd_cm = overall_confusion_matrix_sk(output=det_outputs,
                                             target=det_targets,
                                             log_probs=_param_dict['log_probs'],
                                             logits=_param_dict['logits'],
                                             labels=_param_dict['labels'])
        cm_tracker.update_cm(upd_cm)
        upd_class_wise_cms = class_wise_confusion_matrices_single_label_sk(output=det_outputs,
                                                                           target=det_targets,
                                                                           log_probs=_param_dict['log_probs'],
                                                                           logits=_param_dict['logits'],
                                                                           labels=_param_dict['labels'])
    else:
        upd_class_wise_cms = class_wise_confusion_matrices_multi_label_sk(output=det_outputs,
                                                                          target=det_targets,
                                                                          logits=_param_dict['logits'],
                                                                          labels=_param_dict['labels'])
    cm_tracker.update_class_wise_cms(upd_class_wise_cms)
    # 2) Explicitly write a plot of the confusion matrices to a file
    cm_tracker.save_result_cms_to_file(config.test_output_dir)
    # Moreover, save them as pickle
    path_name = os.path.join(config.test_output_dir, "cms_test_model.p")
    with open(path_name, 'wb') as cm_file:
        all_cms = [cm_tracker.cm, cm_tracker.class_wise_cms]
        pickle.dump(all_cms, cm_file)


    # ------------------- Predicted Scores and Classes -------------------
    str_mode = "Test" if 'test' in str(config.test_output_dir).lower() else "Validation"
    if multi_label_training:
        if _param_dict['logits']:
            probs = torch.sigmoid(det_outputs)
            classes = torch.where(probs > THRESHOLD, 1, 0)
        else:
            probs = det_outputs
            classes = torch.where(det_outputs > THRESHOLD, 1, 0)
    else:
        if _param_dict['logits']:
            probs = torch.softmax(det_outputs, dim=1)
        else:
            probs = det_outputs
        # Use the argmax (doesn't matter if the outputs are probs or logits)
        pred_classes = torch.argmax(det_outputs, dim=1)
        classes = torch.nn.functional.one_hot(pred_classes, len(class_labels))

    # 1) Predicted Probabilities
    fig_output_probs, ax = plt.subplots(figsize=(10, 20))
    sns.heatmap(data=probs.detach().cpu(), ax=ax)
    ax.set_xlabel("Class ID")
    ax.set_ylabel(str(str_mode).capitalize() + " Sample ID")
    fig_output_probs.savefig(os.path.join(config.test_output_dir, "Predicted_probs.pdf"))
    fig_output_probs.clear()
    plt.close(fig_output_probs)
    # 2) Predicted Classes
    # Create the figure and write it to a file
    fig_output_classes, ax = plt.subplots(figsize=(10, 20))
    # Define the colors
    colors = ["lightgray", "gray"]
    cmap = LinearSegmentedColormap.from_list('Custom', colors, len(colors))
    # Classes should be one-hot like [[1, 0, 0, 0, 0], [0, 1, 1, 0, 0]]
    sns.heatmap(data=classes.detach().numpy(), cmap=cmap, ax=ax)
    # Set the Colorbar labels
    colorbar = ax.collections[0].colorbar
    colorbar.set_ticks([0.25, 0.75])
    colorbar.set_ticklabels(['0', '1'])
    ax.set_xlabel("Class ID")
    ax.set_ylabel(str(str_mode).capitalize() + " Sample ID")
    fig_output_classes.savefig(os.path.join(config.test_output_dir, "Predicted_classes.pdf"))
    fig_output_classes.clear()
    plt.close(fig_output_classes)

    # ------------------- Summary Report -------------------
    if multi_label_training:
        summary_dict = multi_label_metrics.sk_classification_summary(output=det_outputs, target=det_targets,
                                                                     logits=_param_dict["logits"],
                                                                     labels=_param_dict["labels"],
                                                                     output_dict=True)
    else:
        summary_dict = single_label_metrics.sk_classification_summary(output=det_outputs, target=det_targets,
                                                                      log_probs=_param_dict["log_probs"],
                                                                      logits=_param_dict["logits"],
                                                                      labels=_param_dict["labels"],
                                                                      output_dict=True)

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


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='MA Vanessa')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-t', '--tune', action='store_true', help='Use when model was derived during tuning')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--test_dir', '--td'], type=Path, target='data_loader;test_dir')
        # options added here can be modified by command line flags.
    ]

    config = ConfigParser.from_args(args=args, options=options, mode='test')

    # fix random seeds for reproducibility
    _set_seed(global_config.SEED)

    if config.use_tune:
        tune_config_path = os.path.join(config.test_output_dir, "../params.json")
        with open(tune_config_path, 'r') as file:
            tune_config = json.load(file)
    else:
        tune_config = None
    test_model(config, tune_config)
