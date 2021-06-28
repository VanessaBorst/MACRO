import inspect
import time

import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap

from base import BaseTrainer
from model.multi_label_metrics import class_wise_confusion_matrices_multi_label, THRESHOLD
from model.single_label_metrics import class_wise_confusion_matrices_single_label, overall_confusion_matrix
from utils import inf_loop, plot_grad_flow_lines, plot_grad_flow_bars
from utils.tracker import MetricTracker, ConfusionMatrixTracker
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class ECGTrainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(self, model, criterion, metric_ftns_iter, metric_ftns_epoch, metric_ftns_epoch_class_wise, optimizer,
                 config, device, data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, optimizer, config)
        self.config = config
        self.device = device
        self.metrics_iter = metric_ftns_iter
        self.metrics_epoch = metric_ftns_epoch
        self.metrics_epoch_class_wise = metric_ftns_epoch_class_wise
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self._class_labels = self.data_loader.dataset.class_labels

        # metrics
        keys_iter = [m.__name__ for m in self.metrics_iter]
        keys_epoch = [m.__name__ for m in self.metrics_epoch]
        keys_epoch_class_wise = [m.__name__ for m in self.metrics_epoch_class_wise]
        self.train_metrics = MetricTracker(keys_iter=['loss'] + keys_iter, keys_epoch=keys_epoch,
                                           keys_epoch_class_wise=keys_epoch_class_wise, labels=self._class_labels,
                                           writer=self.writer)
        self.valid_metrics = MetricTracker(keys_iter=['loss'] + keys_iter, keys_epoch=keys_epoch,
                                           keys_epoch_class_wise=keys_epoch_class_wise, labels=self._class_labels,
                                           writer=self.writer)

        # Store potential parameters needed for metrics
        # TODO Find out which shape is needed by the weights - inverse or normal?
        # val_class_weights = self.data_loader.dataset.get_target_distribution(
        #     idx_list=self.data_loader.valid_sampler.indices, multi_labels=self.multi_label_training) \
        #     if not self.overfit_single_batch else None
        val_pos_weights = self.data_loader.dataset.get_ml_pos_weights(
                idx_list=self.data_loader.valid_sampler.indices) \
            if not self.overfit_single_batch else None
        self._param_dict = {
            "labels": self._class_labels,

            "sigmoid_probs": config["metrics"]["additional_metrics_args"].get("sigmoid_probs", False),
            "log_probs": config["metrics"]["additional_metrics_args"].get("log_probs", False),
            "logits": config["metrics"]["additional_metrics_args"].get("logits", False),

            # "train_class_weights": self.data_loader.dataset.get_target_distribution(
            #     idx_list=self.data_loader.batch_sampler.sampler.indices, multi_labels=self.multi_label_training),
            "train_pos_weights": self.data_loader.dataset.get_ml_pos_weights(
                idx_list=self.data_loader.batch_sampler.sampler.indices),

            # "valid_class_weights": val_class_weights,
            "valid_pos_weights": val_pos_weights
        }

        # confusion matrices
        self.train_cms = ConfusionMatrixTracker(*self.data_loader.dataset.class_labels,
                                                writer=self.writer, multi_label_training=self.multi_label_training)
        # The ECGDataset is the same for data_loader and valid_loader as it is realized with SubSamplers
        self.valid_cms = ConfusionMatrixTracker(*self.data_loader.dataset.class_labels, writer=self.writer,
                                                multi_label_training=self.multi_label_training)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log (as dateframe) that contains the loss, the metrics and confusion matrices of this epoch.
        """
        self.model.train()
        # Reset the trackers
        self.train_metrics.reset()
        self.train_cms.reset()
        # If there are epoch-based metrics, store the intermediate targets. Always store the output scores
        outputs = torch.FloatTensor().to(self.device)
        if len(self.metrics_epoch) > 0:
            targets = torch.FloatTensor().to(self.device)
            if not self.multi_label_training:
                targets_all_labels = torch.FloatTensor().to(self.device)

        # Set the writer object to training mode
        self.writer.set_mode('train')

        # # Create a figure for the average gradient flows of the epoch
        # fig_gradient_flow_lines = plt.figure(figsize=(8,8))
        # fig_gradient_flow_bars = plt.figure(figsize=(8,8))

        start = time.time()
        for batch_idx, (padded_records, labels, labels_one_hot, lengths, record_names) in enumerate(self.data_loader):
            if self.multi_label_training:
                data, target = padded_records.to(self.device), labels_one_hot.to(self.device)
            else:
                # target contains the first GT label, target_all_labels contains all labels in 1-hot-encoding
                data, target, target_all_labels = padded_records.to(self.device), labels.to(self.device), \
                                                  labels_one_hot.to(self.device)
            data = data.permute(0, 2, 1)  # switch seq_len and feature_size (12 = #leads)

            self.optimizer.zero_grad()
            # data has shape [batch_size, 12, seq_len]
            output = self.model(data)

            # if type(output) is tuple:
            #     output, attention_weights = output
            #     fig_attention_weights = plt.figure(figsize=(8,8))
            #     sns.heatmap(data=attention_weights.detach().numpy()[:, :, 0], ax=fig_attention_weights.add_subplot())
            #     self.writer.add_figure("Attention weights for batch " + str(batch_idx),
            #                            fig_attention_weights, global_step=epoch)
            #     plt.close(fig_attention_weights)

            outputs = torch.cat((outputs, output))
            if len(self.metrics_epoch) > 0:
                targets = torch.cat((targets, target))
                if not self.multi_label_training:
                    targets_all_labels = torch.cat((targets_all_labels, target_all_labels))

            args, _, _, _, _, _, _ = inspect.getfullargspec(self.criterion)
            # Output and target are needed for all metrics!
            additional_args = [arg for arg in args if arg not in ('output', 'target')]
            additional_kwargs = {
                param_name: self._param_dict[param_name] if not param_name == 'pos_weights'
                else self._param_dict['train_pos_weights'] for param_name in additional_args
            }
            loss = self.criterion(target=target, output=output, **additional_kwargs)
            loss.backward()

            # # Add the average gradient of the current batch to the respective figure to
            # # record the average gradients per layer in every training iteration
            # plot_grad_flow_lines(self.model.named_parameters(), fig_gradient_flow_lines)
            # plot_grad_flow_bars(self.model.named_parameters(), fig_gradient_flow_bars)

            self.optimizer.step()


            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)

            # Iteratively update the loss and the metrics with the MetricTracker
            # To this end, the Tracker internally maintains a dataframe containing different columns:
            # columns=['current', 'sum', 'square_sum','counts', 'mean', 'square_avg', 'std']
            # The loss and each metric are updated in a separate row for each of them
            self.train_metrics.iter_update('loss', loss.item(), n=output.shape[0])
            for met in self.metrics_iter:
                args = inspect.signature(met).parameters.values()
                # Output and target are needed for all metrics! Only consider other args WITHOUT default
                additional_args = [arg.name for arg in args
                                   if arg.name not in ('output', 'target') and arg.default is arg.empty]
                additional_kwargs = {
                    param_name: self._param_dict[param_name] for param_name in additional_args
                }
                self.train_metrics.iter_update(met.__name__, met(target=target, output=output, **additional_kwargs),
                                               n=output.shape[0])

            # Update the confusion matrices maintained by the ClassificationTracker
            if not self.multi_label_training:
                upd_cm = overall_confusion_matrix(output=output, target=target,
                                                  log_probs=self._param_dict['log_probs'],
                                                  logits=self._param_dict['logits'],
                                                  labels=self._param_dict['labels'])
                self.train_cms.update_cm(upd_cm)
                upd_class_wise_cms = class_wise_confusion_matrices_single_label(output=output, target=target,
                                                                                log_probs=self._param_dict['log_probs'],
                                                                                logits=self._param_dict['logits'],
                                                                                labels=self._param_dict['labels'])
            else:
                upd_class_wise_cms = class_wise_confusion_matrices_multi_label(output=output, target=target,
                                                                               sigmoid_probs=self._param_dict[
                                                                                   'sigmoid_probs'],
                                                                               logits=self._param_dict['logits'],
                                                                               labels=self._param_dict['labels'])
            self.train_cms.update_class_wise_cms(upd_class_wise_cms)

            if batch_idx % self.log_step == 0:
                epoch_debug = f"Train Epoch: {epoch} {self._progress(batch_idx)} "
                current_metrics = self.train_metrics.current()
                metrics_debug = ", ".join(f"{key}: {value:.6f}" for key, value in current_metrics.items())
                self.logger.debug(epoch_debug + metrics_debug)

            if batch_idx == self.len_epoch:  # or self.overfit_single_batch:
                break

        # # Send the gradient flows of the current epoch to the TensorboardWriter
        # self.writer.add_figure("Gradient flow as lines", fig_gradient_flow_lines, global_step=epoch)
        # self.writer.add_figure("Gradient flow as bars", fig_gradient_flow_bars, global_step=epoch)
        # fig_gradient_flow_lines.clf()
        # fig_gradient_flow_bars.clf()

        # At the end of each epoch, explicitly send the confusion matrices to the SummaryWriter/TensorboardWriter
        self.train_cms.send_cms_to_writer(epoch=epoch)

        # Also set the epoch number for the MetricTracker
        self.train_metrics.epoch = epoch

        # When overfitting a single batch, only a small amount of data is used and hence, not all clases may be present
        # In such case, not all metrics are defined, so skip updating the metrics in that case
        if not self.overfit_single_batch:
            # Moreover, the epoch-based metrics need to be updated
            for met in self.metrics_epoch:
                args = inspect.signature(met).parameters.values()
                # Output and target are needed for all metrics! Only consider other args WITHOUT default
                additional_args = [arg.name for arg in args
                                   if arg.name not in ('output', 'target') and arg.default is arg.empty]
                additional_kwargs = {
                    param_name: self._param_dict[param_name] for param_name in additional_args
                }
                if not self.multi_label_training and met.__name__=='cpsc_score_adapted':
                    # Consider all labels for evaluation, even in the single label case
                    self.train_metrics.epoch_update(met.__name__,
                                                    met(target=targets_all_labels, output=outputs, **additional_kwargs))
                else:
                    self.train_metrics.epoch_update(met.__name__,
                                                    met(target=targets, output=outputs, **additional_kwargs))

            # This holds for the class-wise, epoch-based metrics as well
            for met in self.metrics_epoch_class_wise:
                args = inspect.signature(met).parameters.values()
                # Output and target are needed for all metrics! Only consider other args WITHOUT default
                additional_args = [arg.name for arg in args
                                   if arg.name not in ('output', 'target') and arg.default is arg.empty]
                additional_kwargs = {
                    param_name: self._param_dict[param_name] for param_name in additional_args
                }
                self.train_metrics.class_wise_epoch_update(met.__name__, met(target=targets, output=outputs,
                                                                             **additional_kwargs))

        # # Plot heatmaps of the predicted scores for each training sample to verify if they change
        # self._send_pred_scores_to_writer(epoch, outputs, 'training')
        # # Plot heatmaps of the predicted classes for easier interpretability as well
        # self._send_pred_classes_to_writer(epoch, outputs, 'training')

        # Contains only NaNs for all non-iteration-based metrics when overfit_single_batch is True
        train_log = self.train_metrics.result()

        if self.do_validation:
            # log.update({'Note': '-------------Start of Validation-------------'})
            valid_log, valid_cm_information = self._valid_epoch(epoch)
            valid_log.set_index('val_' + valid_log.index.astype(str), inplace=True)
            # log.update(**{'val_' + k: v for k, v in val_log.items()})  # Extends the dict by the val loss and metrics

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        log = pd.concat([train_log, valid_log]) if self.do_validation else train_log
        end = time.time()
        ty_res = time.gmtime(end - start)
        res = time.strftime("%H hours, %M minutes, %S seconds", ty_res)
        epoch_log = {'epochs': epoch,
                     'iterations': self.len_epoch * epoch,
                     'Runtime': res}
        epoch_info = ', '.join(f"{key}: {value}" for key, value in epoch_log.items())
        logger_info = f"{epoch_info}\n{log}"
        self.logger.info(logger_info)

        # Directly log the confusion matrix-related information to a dict and send it to the logger
        # In the returned dataframe, the confusion matrices are not contained!
        cm_information = dict({"overall_cm": "\n" + str(self.train_cms.cm)},
                              **{"Confusion matrix for class " + str(class_cm.columns.name): "\n" + str(class_cm)
                                 for _, class_cm in enumerate(self.train_cms.class_wise_cms)})
        self.logger.info("------------------Confusion Matrices (train) for epoch " + str(epoch) + "------------------")
        for key, value in cm_information.items():
            self.logger.info('    {:15s}: {}'.format(str(key), value))

        if self.do_validation:
            # Do the same with the cm-related dict from the validation step
            self.logger.info(
                "------------------Confusion Matrices (valid) for epoch " + str(epoch) + "------------------")
            for key, value in valid_cm_information.items():
                self.logger.info('    {:15s}: {}'.format(str(key), value))

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation (as dataframe)
        """
        self.model.eval()
        # Reset the trackers
        self.valid_metrics.reset()
        self.valid_cms.reset()

        # Set the writer object to validation mode
        self.writer.set_mode('valid')

        with torch.no_grad():
            # If there are epoch-based metrics, store the intermediate targets. Always store the outputs
            outputs = torch.FloatTensor().to(self.device)
            if len(self.metrics_epoch) > 0:
                targets = torch.FloatTensor().to(self.device)
                if not self.multi_label_training:
                    targets_all_labels = torch.FloatTensor().to(self.device)

            for batch_idx, (padded_records, labels, labels_one_hot, lengths, record_names) in enumerate(
                    self.valid_data_loader):

                if self.multi_label_training:
                    data, target = padded_records.to(self.device), labels_one_hot.to(self.device)
                else:
                    # target contains the first GT label, target_all_labels contains all labels in 1-hot-encoding
                    data, target, target_all_labels = padded_records.to(self.device), labels.to(self.device), \
                                                      labels_one_hot.to(self.device)

                data = data.permute(0, 2, 1)  # switch seq_len and feature_size (12 = #leads)

                output = self.model(data)
                # if type(output) is tuple:
                #     output, attention_weights = output
                #     fig_attention_weights_valid = plt.figure(figsize=(8, 8))
                #     sns.heatmap(data=attention_weights.detach().numpy()[:, :, 0],
                #                 ax=fig_attention_weights_valid.add_subplot())
                #     self.writer.add_figure("Attention weights for validation batch " + str(batch_idx),
                #                            fig_attention_weights_valid, global_step=epoch)
                #     plt.close(fig_attention_weights_valid)

                outputs = torch.cat((outputs, output))
                if len(self.metrics_epoch) > 0:
                    targets = torch.cat((targets, target))
                    if not self.multi_label_training:
                        targets_all_labels = torch.cat((targets_all_labels, target_all_labels))

                args, _, _, _, _, _, _ = inspect.getfullargspec(self.criterion)
                # Output and target are needed for all metrics!
                additional_args = [arg for arg in args if arg not in ('output', 'target')]
                additional_kwargs = {
                    param_name: self._param_dict[param_name] if not param_name == 'pos_weights'
                    else self._param_dict['valid_pos_weights'] for param_name in additional_args
                }
                loss = self.criterion(target=target, output=output, **additional_kwargs)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx)

                self.valid_metrics.iter_update('loss', loss.item(), n=output.shape[0])
                for met in self.metrics_iter:
                    args = inspect.signature(met).parameters.values()
                    # Output and target are needed for all metrics! Only consider other args WITHOUT default
                    additional_args = [arg.name for arg in args
                                       if arg.name not in ('output', 'target') and arg.default is arg.empty]
                    additional_kwargs = {
                        param_name: self._param_dict[param_name] for param_name in additional_args
                    }
                    self.valid_metrics.iter_update(met.__name__, met(target=target, output=output, **additional_kwargs),
                                                   n=output.shape[0])

                # Update the confusion matrices maintained by the ClassificationTracker
                if not self.multi_label_training:
                    upd_cm = overall_confusion_matrix(output=output, target=target,
                                                      log_probs=self._param_dict['log_probs'],
                                                      logits=self._param_dict['logits'],
                                                      labels=self._param_dict['labels'])
                    self.valid_cms.update_cm(upd_cm)
                    upd_class_wise_cms = class_wise_confusion_matrices_single_label(output=output, target=target,
                                                                                    log_probs=self._param_dict[
                                                                                        'log_probs'],
                                                                                    logits=self._param_dict['logits'],
                                                                                    labels=self._param_dict['labels'])
                else:
                    upd_class_wise_cms = class_wise_confusion_matrices_multi_label(output=output, target=target,
                                                                                   sigmoid_probs=self._param_dict[
                                                                                       'sigmoid_probs'],
                                                                                   logits=self._param_dict['logits'],
                                                                                   labels=self._param_dict['labels'])

                self.valid_cms.update_class_wise_cms(upd_class_wise_cms)

        # At the end of each epoch, explicitly send the confusion matrices to the SummaryWriter/TensorboardWriter
        self.valid_cms.send_cms_to_writer(epoch=epoch)

        # Also set the epoch number for the MetricTracker
        self.valid_metrics.epoch = epoch

        # When overfitting a single batch, only a small amount of data is used and hence, not all clases may be present
        # In such case, not all metrics are defined, so skip updating the metrics in that case
        if not self.overfit_single_batch:
            # Moreover, the epoch-based metrics need to be updated
            for met in self.metrics_epoch:
                args = inspect.signature(met).parameters.values()
                # Output and target are needed for all metrics! Only consider other args WITHOUT default
                additional_args = [arg.name for arg in args
                                   if arg.name not in ('output', 'target') and arg.default is arg.empty]
                additional_kwargs = {
                    param_name: self._param_dict[param_name] for param_name in additional_args
                }
                if not self.multi_label_training and met.__name__=='cpsc_score_adapted':
                    # Consider all labels for evaluation, even in the single label case
                    self.valid_metrics.epoch_update(met.__name__,
                                                    met(target=targets_all_labels, output=outputs, **additional_kwargs))
                else:
                    self.valid_metrics.epoch_update(met.__name__,
                                                    met(target=targets, output=outputs, **additional_kwargs))

            # This holds for the class-wise, epoch-based metrics as well
            for met in self.metrics_epoch_class_wise:
                args = inspect.signature(met).parameters.values()
                # Output and target are needed for all metrics! Only consider other args WITHOUT default
                additional_args = [arg.name for arg in args
                                   if arg.name not in ('output', 'target') and arg.default is arg.empty]
                additional_kwargs = {
                    param_name: self._param_dict[param_name] for param_name in additional_args
                }
                self.valid_metrics.class_wise_epoch_update(met.__name__, met(target=targets, output=outputs,
                                                                             **additional_kwargs))

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')

        # # Plot heatmaps of the predicted scores for each validation sample to verify if they change
        # self._send_pred_scores_to_writer(epoch, outputs, 'validation')
        #
        # # Plot heatmaps of the predicted classes for easier interpretability as well
        # self._send_pred_classes_to_writer(epoch, outputs, 'validation')

        valid_log = self.valid_metrics.result()

        # Also log the confusion matrix-related information to the valid log
        valid_cm_information = dict({"overall_cm": "\n" + str(self.valid_cms.cm)},
                                    **{"Confusion matrix for class " + str(class_cm.columns.name): "\n" + str(class_cm)
                                       for _, class_cm in enumerate(self.valid_cms.class_wise_cms)})

        return valid_log, valid_cm_information

    def _send_pred_scores_to_writer(self, epoch, outputs, str_mode):
        """
                :param epoch: Current epoch
                :param outputs: All outputs of the current train/validation session
                :param str_mode: Should either be 'training' or 'validation'
                :return:
                """
        fig_output_scores = sns.heatmap(data=outputs.detach().numpy()).get_figure()
        plt.xlabel("Class ID")
        plt.ylabel(str(str_mode).capitalize() + " Sample ID")
        plt.close(fig_output_scores)  # close the current figure
        self.writer.add_figure("Predicted output scores per " + str(str_mode).lower() + " sample",
                               fig_output_scores, global_step=epoch)

    def _send_pred_classes_to_writer(self, epoch, outputs, str_mode):
        """
        :param epoch: Current epoch
        :param outputs: All outputs of the current train/validation session
        :param str_mode: Should either be 'training' or 'validation'
        :return:
        """
        if self.multi_label_training:
            if self._param_dict['logits']:
                sigmoid_probs = torch.sigmoid(outputs)
                classes = torch.where(sigmoid_probs > THRESHOLD, 1, 0)
            else:
                classes = torch.where(outputs > THRESHOLD, 1, 0)
        else:
            # Use the argmax (doesn't matter if the outputs are probs or logits)
            pred_classes = torch.argmax(outputs, dim=1)
            classes = torch.nn.functional.one_hot(pred_classes, len(self._class_labels))
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
        self.writer.add_figure("Predicted output class(es) per " + str(str_mode).lower() + " sample",
                               ax.get_figure(), global_step=epoch)

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
