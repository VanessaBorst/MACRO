import inspect
import time

import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from model.multi_label_metrics import class_wise_confusion_matrices_multi_label
from model.single_label_metrics import class_wise_confusion_matrices_single_label, overall_confusion_matrix
from utils import inf_loop
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

        class_labels = self.data_loader.dataset.class_labels

        # metrics
        keys_iter = [m.__name__ for m in self.metrics_iter]
        keys_epoch = [m.__name__ for m in self.metrics_epoch]
        keys_epoch_class_wise = [m.__name__ for m in self.metrics_epoch_class_wise]
        self.train_metrics = MetricTracker(keys_iter=['loss'] + keys_iter, keys_epoch=keys_epoch,
                                           keys_epoch_class_wise=keys_epoch_class_wise, labels=class_labels,
                                           writer=self.writer)
        self.valid_metrics = MetricTracker(keys_iter=['loss'] + keys_iter, keys_epoch=keys_epoch,
                                           keys_epoch_class_wise=keys_epoch_class_wise, labels=class_labels,
                                           writer=self.writer)

        # Store potential parameters needed for metrics
        # TODO Find out which shape is needed by the weights - inverse or normal?
        val_class_weights = self.data_loader.dataset.get_target_distribution(
            idx_list=self.data_loader.valid_sampler.indices, multi_labels=self.multi_label_training) \
            if not self.overfit_single_batch else None
        self._param_dict = {
            "labels": class_labels,
            "sigmoid_probs": config["metrics"]["additional_metrics_args"].get("sigmoid_probs", False),
            "log_probs": config["metrics"]["additional_metrics_args"].get("log_probs", False),
            "train_class_weights": self.data_loader.dataset.get_target_distribution(
                idx_list=self.data_loader.batch_sampler.sampler.indices, multi_labels=self.multi_label_training),
            "val_class_weights": val_class_weights
        }

        # confusion matrices
        self.train_cms = ConfusionMatrixTracker(*self.data_loader.dataset.class_labels, writer=self.writer)
        # The ECGDataset is the same for data_loader and valid_loader as it is realized with SubSamplers
        self.valid_cms = ConfusionMatrixTracker(*self.data_loader.dataset.class_labels, writer=self.writer)

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
        # If there are epoch-based metrics, store the intermediate outputs and targets
        if len(self.metrics_epoch) > 0:
            outputs = torch.FloatTensor().to(self.device)
            targets = torch.FloatTensor().to(self.device)

        # Set the writer object to training mode
        self.writer.set_mode('train')

        start = time.time()
        for batch_idx, (padded_records, labels, labels_one_hot, lengths, record_names) in enumerate(self.data_loader):
            data, target = padded_records.to(self.device), labels_one_hot.to(self.device)
            data = data.permute(0, 2, 1)  # switch seq_len and feature_size (12 = #leads)

            self.optimizer.zero_grad()
            # data has shape [batch_size, 12, seq_len]
            output = self.model(data)

            if len(output) == 2:
                output, attention_weights = output
                fig = sns.heatmap(data=attention_weights.detach().numpy()[:, :, 0], ).get_figure()
                # plt.show()
                plt.close(fig)  # close the curren figure
                self.writer.add_figure("Attention weights for batch " + str(batch_idx),
                                       fig, global_step=epoch)

            if len(self.metrics_epoch) > 0:
                outputs = torch.cat((outputs, output))
                targets = torch.cat((targets, target))
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)

            # Iteratively update the loss and the metrics with the MetricTracker
            # To this end, the Tracker internally maintains a dataframe containing different columns:
            # columns=['current', 'sum', 'square_sum','counts', 'mean', 'square_avg', 'std']
            # The loss and each metric are updated in a separate row for each of them
            self.train_metrics.iter_update('loss', loss.item(), n=output.shape[0])
            for met in self.metrics_iter:
                args, _, _, _ = inspect.getargspec(met)
                # Output and target are needed for all metrics!
                additional_args = [arg for arg in args if arg not in ('output', 'target')]
                additional_kwargs = {
                    param_name: self._param_dict[param_name] for param_name in additional_args
                }
                self.train_metrics.iter_update(met.__name__, met(target=target, output=output, **additional_kwargs),
                                               n=output.shape[0])

            # Update the confusion matrices maintained by the ClassificationTracker
            # TODO set log probs/sigmoid probs in config
            if not self.multi_label_training:
                upd_cm = overall_confusion_matrix(output=output, target=target,
                                                  log_probs=self._param_dict['log_probs'],
                                                  labels=self._param_dict['labels'])
                self.train_cms.update_cm(upd_cm)
                upd_class_wise_cms = class_wise_confusion_matrices_single_label(output=output, target=target,
                                                                                log_probs=self._param_dict['log_probs'],
                                                                                labels=self._param_dict['labels'])
            else:
                upd_class_wise_cms = class_wise_confusion_matrices_multi_label(output=output, target=target,
                                                                               sigmoid_probs=self._param_dict[
                                                                                   'sigmoid_probs'],
                                                                               labels=self._param_dict['labels'])
            self.train_cms.update_class_wise_cms(upd_class_wise_cms)

            if batch_idx % self.log_step == 0:
                epoch_debug = f"Train Epoch: {epoch} {self._progress(batch_idx)} "
                current_metrics = self.train_metrics.current()
                metrics_debug = ", ".join(f"{key}: {value:.6f}" for key, value in current_metrics.items())
                self.logger.debug(epoch_debug + metrics_debug)

            if batch_idx == self.len_epoch: #or self.overfit_single_batch:
                break

        # At the end of each epoch, explicitly send the confusion matrices to the SummaryWriter/TensorboardWriter
        self.train_cms.send_cms_to_writer(epoch=epoch)

        # When overfitting a single batch, only a small amount of data is used and hence, not all clases may be present
        # In such case, not all metrics are defined, so skip updating the metrics in that case
        if not self.overfit_single_batch:
            # Moreover, the epoch-based metrics need to be updated
            for met in self.metrics_epoch:
                args, _, _, _ = inspect.getargspec(met)
                # Output and target are needed for all metrics!
                additional_args = [arg for arg in args if arg not in ('output', 'target')]
                additional_kwargs = {
                    param_name: self._param_dict[param_name] for param_name in additional_args
                }
                self.train_metrics.epoch_update(met.__name__, met(target=targets, output=outputs, **additional_kwargs),
                                                epoch)
            # This holds for the class-wise, epoch-based metrics as well
            for met in self.metrics_epoch_class_wise:
                args, _, _, _ = inspect.getargspec(met)
                # Output and target are needed for all metrics!
                additional_args = [arg for arg in args if arg not in ('output', 'target')]
                additional_kwargs = {
                    param_name: self._param_dict[param_name] for param_name in additional_args
                }
                self.train_metrics.class_wise_epoch_update(met.__name__, met(target=targets, output=outputs,
                                                                             **additional_kwargs), epoch)

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
            # If there are epoch-based metrics, store the intermediate outputs and targets
            if len(self.metrics_epoch) > 0:
                outputs = torch.FloatTensor().to(self.device)
                targets = torch.FloatTensor().to(self.device)

            for batch_idx, (padded_records, labels, labels_one_hot, lengths, record_names) in enumerate(
                    self.valid_data_loader):
                data, target = padded_records.to(self.device), labels_one_hot.to(self.device)
                data = data.permute(0, 2, 1)  # switch seq_len and feature_size (12 = #leads)

                output = self.model(data)
                if len(self.metrics_epoch) > 0:
                    outputs = torch.cat((outputs, output))
                    targets = torch.cat((targets, target))
                loss = self.criterion(output, target)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx)
                self.valid_metrics.iter_update('loss', loss.item(), n=output.shape[0])
                for met in self.metrics_iter:
                    args, _, _, _ = inspect.getargspec(met)
                    # Output and target are needed for all metrics!
                    additional_args = [arg for arg in args if arg not in ('output', 'target')]
                    additional_kwargs = {
                        param_name: self._param_dict[param_name] for param_name in additional_args
                    }
                    self.valid_metrics.iter_update(met.__name__, met(target=target, output=output, **additional_kwargs),
                                                   n=output.shape[0])

                # Update the confusion matrices maintained by the ClassificationTracker
                if not self.multi_label_training:
                    upd_cm = overall_confusion_matrix(output=output, target=target,
                                                      log_probs=self._param_dict['log_probs'],
                                                      labels=self._param_dict['labels'])
                    self.valid_cms.update_cm(upd_cm)
                    upd_class_wise_cms = class_wise_confusion_matrices_single_label(output=output, target=target,
                                                                                    log_probs=self._param_dict[
                                                                                        'log_probs'],
                                                                                    labels=self._param_dict['labels'])
                else:
                    upd_class_wise_cms = class_wise_confusion_matrices_multi_label(output=output, target=target,
                                                                                   sigmoid_probs=self._param_dict[
                                                                                       'sigmoid_probs'],
                                                                                   labels=self._param_dict['labels'])

                self.valid_cms.update_class_wise_cms(upd_class_wise_cms)

        # At the end of each epoch, explicitly send the confusion matrices to the SummaryWriter/TensorboardWriter
        self.valid_cms.send_cms_to_writer(epoch=epoch)

        # Moreover, the epoch-based metrics need to be updated
        for met in self.metrics_epoch:
            args, _, _, _ = inspect.getargspec(met)
            # Output and target are needed for all metrics!
            additional_args = [arg for arg in args if arg not in ('output', 'target')]
            additional_kwargs = {
                param_name: self._param_dict[param_name] for param_name in additional_args
            }
            self.valid_metrics.epoch_update(met.__name__, met(target=targets, output=outputs, **additional_kwargs),
                                            epoch)

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')

        valid_log = self.valid_metrics.result()

        # Also log the confusion matrix-related information to the valid log
        valid_cm_information = dict({"overall_cm": "\n" + str(self.valid_cms.cm)},
                                    **{"Confusion matrix for class " + str(class_cm.columns.name): "\n" + str(class_cm)
                                       for _, class_cm in enumerate(self.valid_cms.class_wise_cms)})

        return valid_log, valid_cm_information

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
