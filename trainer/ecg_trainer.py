import inspect
import os
import pickle
import time
from contextlib import nullcontext
from pathlib import Path

import pandas as pd
import torch

from torch._C._profiler import ProfilerActivity
from torch.profiler import tensorboard_trace_handler

from base import BaseTrainer
from evaluation import multi_label_metrics, single_label_metrics
from evaluation.multi_label_metrics import class_wise_confusion_matrices_multi_label_sk, THRESHOLD
from evaluation.single_label_metrics import class_wise_confusion_matrices_single_label_sk, overall_confusion_matrix_sk
from utils import inf_loop, plot_grad_flow_lines, plot_grad_flow_bars
from utils.tracker import MetricTracker, ConfusionMatrixTracker

from ray import tune


class ECGTrainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(self, model, criterion, metric_ftns_iter, metric_ftns_epoch, metric_ftns_epoch_class_wise, optimizer,
                 config, device, data_loader, valid_data_loader=None, lr_scheduler=None, use_tune=False, len_epoch=None,
                 cross_valid_active=False):
        super().__init__(model, criterion, optimizer, config, use_tune)
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

        self.batch_log_step = int(
            512 / self.data_loader.batch_size)  # used for logging frequency during the training loop
        self.epoch_log_step_train = 25  # used for epoch metrics and confusion matrix handling
        self.epoch_log_step_valid = 15  # Usually used for confusion matrix handling, at the moment disabled

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

        # If a dedicated valid set was provided, the underlying ECG-Dataset differs, if not it is identical to the
        # Train Dataset => Use valid_data_loader.dataset here, to be sure to handle both cases correctly!
        # Since the valid_data_loader has no "valid sampler" attribute, stick to data_loader.valid_sampler, which
        # contains the needed idx for both cases
        val_class_weights = self.valid_data_loader.dataset.get_inverse_class_frequency(
            idx_list=self.data_loader.valid_sampler.indices, multi_label_training=self.multi_label_training,
            mode='valid', cross_valid_active=cross_valid_active)

        val_pos_weights = self.valid_data_loader.dataset.get_ml_pos_weights(
            idx_list=self.data_loader.valid_sampler.indices, mode='valid', cross_valid_active=cross_valid_active)

        self._param_dict = {
            "labels": self._class_labels,
            "device": self.device,
            "log_probs": config["arch"]["args"].get("apply_final_activation", False),
            "logits": not config["arch"]["args"].get("apply_final_activation", False),
            "train_pos_weights": self.data_loader.dataset.get_ml_pos_weights(
                idx_list=self.data_loader.batch_sampler.sampler.indices,
                mode='train',
                cross_valid_active=cross_valid_active),
            "train_class_weights": self.data_loader.dataset.get_inverse_class_frequency(
                idx_list=self.data_loader.batch_sampler.sampler.indices,
                multi_label_training=self.multi_label_training,
                mode='train',
                cross_valid_active=cross_valid_active),
            "valid_pos_weights": val_pos_weights,
            "valid_class_weights": val_class_weights,
            "lambda_balance": config["loss"]["add_args"].get("lambda_balance", 1),
            "gamma_neg": config["loss"]["add_args"].get("gamma_neg", 4),
            "gamma_pos": config["loss"]["add_args"].get("gamma_pos", 1),
            "clip": config["loss"]["add_args"].get("clip", 0.05),
        }

        # confusion matrices
        self.train_cms = ConfusionMatrixTracker(*self.data_loader.dataset.class_labels,
                                                writer=self.writer, multi_label_training=self.multi_label_training)
        # The ECGDataset is the same for data_loader and valid_loader as it is realized with SubSamplers
        self.valid_cms = ConfusionMatrixTracker(*self.valid_data_loader.dataset.class_labels, writer=self.writer,
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

        if epoch == 1 or epoch % self.epoch_log_step_train == 0:
            # If there are epoch-based metrics, store the intermediate targets and the output scores
            outputs_list = []
            targets_list = []
            targets_all_labels_list = [] if not self.multi_label_training else None

        # Set the writer object to training mode
        if self.writer is not None:
            self.writer.set_mode('train')
        # Moreover, set the epoch number for the MetricTracker
        self.train_metrics.epoch = epoch

        start = time.time()

        if self.profiler_active:
            main_path = self.config.log_dir if not self._use_tune else \
                Path(tune.get_trial_dir().replace('/models/', '/log/'))
            context_manager = torch.profiler.profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(
                    wait=40,
                    warmup=2,
                    active=10,
                    repeat=2),
                on_trace_ready=tensorboard_trace_handler(main_path),
                with_stack=True
            )
        else:
            context_manager = nullcontext()

        with context_manager as profiler:
            for batch_idx, (padded_records, _, first_labels, labels_one_hot, record_names) in enumerate(
                    self.data_loader):

                if self.multi_label_training:
                    data, target = padded_records.to(self.device), labels_one_hot.to(self.device)
                else:
                    # target contains the first GT label, target_all_labels contains all labels in 1-hot-encoding
                    data, target, target_all_labels = padded_records.to(self.device), first_labels.to(self.device), \
                        labels_one_hot.to(self.device)
                data = data.permute(0, 2, 1)  # switch seq_len and feature_size (12 = #leads)

                self.optimizer.zero_grad()
                # data has shape [batch_size, 12, seq_len]
                output = self.model(data)

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
                if not self.multi_label_training:
                    detached_target_all_labels = target_all_labels.detach().cpu()

                if epoch == 1 or epoch % self.epoch_log_step_train == 0:
                    # TODO Maybe also track the single lead outputs here?
                    outputs_list.append(detached_output)
                    targets_list.append(detached_target)
                    if not self.multi_label_training:
                        targets_all_labels_list.append(detached_target_all_labels)

                # Calculate the loss, here gradients are nedded!
                additional_args = self.config['loss']['add_args']
                additional_kwargs = {
                    param_name: self._param_dict[param_name.replace('pos_weights', 'train_pos_weights').
                    replace('class_weights', 'train_class_weights')] for param_name in additional_args
                }

                if not multi_lead_branch_active:
                    loss = self.criterion(target=target, output=output, **additional_kwargs)
                else:
                    # Ensure that self.criterion is a function, namely multi_branch_BCE_with_logits
                    assert callable(self.criterion) and self.criterion.__name__ \
                           in ["multi_branch_BCE_with_logits", "multi_branch_asymmetric_loss_with_logits"], \
                        "For the multibranch network, the multibranch BCE with logits loss function has to be used!"
                    # Calculate the joint loss of each single lead branch and the overall network
                    loss = self.criterion(target=target, output=output,
                                          single_lead_outputs = single_lead_outputs,
                                          **additional_kwargs)


                loss.backward()

                self.optimizer.step()
                if self.writer is not None:
                    self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)

                # Iteratively update the loss and the metrics with the MetricTracker for each batch
                # TODO Maybe also track the single lead outputs here?
                self._do_iter_update(metric_tracker=self.train_metrics, output=detached_output,
                                     target=detached_target, loss_val=loss.item())

                if batch_idx % self.batch_log_step == 0:
                    epoch_debug = f"Train Epoch: {epoch} {self._progress(batch_idx)} "
                    current_metrics = self.train_metrics.current()
                    metrics_debug = ", ".join(f"{key}: {value:.6f}" for key, value in current_metrics.items())
                    self.logger.debug(epoch_debug + metrics_debug)

                if self.profiler_active:
                    profiler.step()

                if batch_idx == self.len_epoch:
                    break

        # At the end of each epoch, explicitly handle the tracking of metrics and confusion matrices by means of
        # the SummaryWriter/TensorboardWriter, but only each epoch_log_step steps
        if epoch == 1 or epoch % self.epoch_log_step_train == 0:
            # Update the cms and metrics
            # TODO Maybe also track the single lead outputs here?
            summary_str = self._handle_tracking_at_epoch_end(metric_tracker=self.train_metrics, epoch=epoch,
                                                             outputs=outputs_list, targets=targets_list,
                                                             targets_all_labels=targets_all_labels_list,
                                                             mode='train', track_cms=False)
        else:
            summary_str = "Not calc."

        # Moreover, the train metrics are only contained each epoch_log_step times
        train_log = self.train_metrics.result(
            include_epoch_metrics=(epoch == 1 or epoch % self.epoch_log_step_train == 0))

        if self.do_validation:
            # log.update({'Note': '-------------Start of Validation-------------'})
            valid_log, valid_summary_str = self._valid_epoch(epoch)
            valid_log.set_index('val_' + valid_log.index.astype(str), inplace=True)
            # log.update(**{'val_' + k: v for k, v in val_log.items()})  # Extends the dict by the val loss and metrics

        if self.lr_scheduler is not None:
            # TODO adapt later
            assert self.lr_scheduler is None, "LR Scheduler support not yet finished"
            pass
            # self.lr_scheduler.step()
            # self.lr_scheduler.step(valid_log['mean']["val_weighted_sk_f1"])

        log = pd.concat([train_log, valid_log]) if self.do_validation else train_log
        summary = "Training summary:\n" + summary_str + "\nValidation summary:\n" + valid_summary_str \
            if self.do_validation else "Training summary:\n" + summary_str
        end = time.time()
        ty_res = time.gmtime(end - start)
        res = time.strftime("%H hours, %M minutes, %S seconds", ty_res)
        epoch_log = {'epochs': epoch,
                     'iterations': self.len_epoch * epoch,
                     'Runtime': res}
        epoch_info = ', '.join(f"{key}: {value}" for key, value in epoch_log.items())
        logger_info = f"{epoch_info}\n{log}\n{summary}"
        self.logger.info(logger_info)

        # # Directly log the confusion matrix-related information to a dict and send it to the logger
        # # In the returned dataframe, the confusion matrices are not contained!
        # cm_information = dict({"overall_cm": "\n" + str(self.train_cms.cm)},
        #                       **{"Confusion matrix for class " + str(class_cm.columns.name): "\n" + str(class_cm)
        #                          for _, class_cm in enumerate(self.train_cms.class_wise_cms)})
        # self.logger.info("------------------Confusion Matrices (train) for epoch " + str(epoch) + "------------------")
        # for key, value in cm_information.items():
        #     self.logger.info('    {:15s}: {}'.format(str(key), value))
        #
        # if self.do_validation:
        #     # Do the same with the cm-related dict from the validation step
        #     self.logger.info(
        #         "------------------Confusion Matrices (valid) for epoch " + str(epoch) + "------------------")
        #     for key, value in valid_cm_information.items():
        #         self.logger.info('    {:15s}: {}'.format(str(key), value))

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

        # If there are epoch-based metrics, store the intermediate targets. Always store the output scores
        outputs_list = []
        targets_list = []
        targets_all_labels_list = [] if not self.multi_label_training else None

        # Set the writer object to validation mode
        if self.writer is not None:
            self.writer.set_mode('valid')
        # Moreover, set the epoch number for the MetricTracker
        self.valid_metrics.epoch = epoch

        with torch.no_grad():
            for batch_idx, (padded_records, _, first_labels, labels_one_hot, record_names) in \
                    enumerate(self.valid_data_loader):
                if self.multi_label_training:
                    data, target = padded_records.to(self.device), labels_one_hot.to(self.device)
                else:
                    # target contains the first GT label, target_all_labels contains all labels in 1-hot-encoding
                    data, target, target_all_labels = padded_records.to(self.device), first_labels.to(self.device), \
                        labels_one_hot.to(self.device)

                data = data.permute(0, 2, 1)  # switch seq_len and feature_size (12 = #leads)

                output = self.model(data)

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
                if not self.multi_label_training:
                    detached_target_all_labels = target_all_labels.detach().cpu()

                # TODO Maybe also track the single lead outputs here?
                outputs_list.append(detached_output)
                targets_list.append(detached_target)
                if not self.multi_label_training:
                    targets_all_labels_list.append(detached_target_all_labels)

                additional_args = self.config['loss']['add_args']
                additional_kwargs = {
                    param_name: self._param_dict[param_name.replace('pos_weights', 'train_pos_weights').
                    replace('class_weights', 'train_class_weights')] for param_name in additional_args
                }

                if not multi_lead_branch_active:
                    loss = self.criterion(target=target, output=output, **additional_kwargs)
                else:
                    # Ensure that self.criterion is a function, namely multi_branch_BCE_with_logits
                    assert callable(self.criterion) and self.criterion.__name__ \
                           in ["multi_branch_BCE_with_logits", "multi_branch_asymmetric_loss_with_logits"], \
                        "For the multibranch network, the multibranch BCE with logits loss function has to be used!"
                    # Calculate the joint loss of each single lead branch and the overall network
                    loss = self.criterion(target=target, output=output,
                                          single_lead_outputs=single_lead_outputs,
                                          **additional_kwargs)

                if self.writer is not None:
                    self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx)

                # Iteratively update the loss and the metrics with the MetricTracker
                # TODO Maybe also track the single lead outputs here?
                self._do_iter_update(metric_tracker=self.valid_metrics, output=detached_output,
                                     target=detached_target, loss_val=loss.item())

                # if batch_idx % self.batch_log_step == 0:
                #     epoch_debug = f"Valid Epoch: {epoch} {self._progress(batch_idx, valid=True)} "
                #     current_metrics = self.valid_metrics.current()
                #     metrics_debug = ", ".join(f"{key}: {value:.6f}" for key, value in current_metrics.items())
                #     self.logger.debug(epoch_debug + metrics_debug)

        # At the end of each epoch, explicitly handle the tracking of confusion matrices and metrics by means of
        # the SummaryWriter/TensorboardWriter
        # For validation, metrics are calculated each epoch
        # Do not calculate and track the confusion matrices each time, ony each few epochs
        # TODO Maybe also track the single lead outputs here?
        valid_sum_str = self._handle_tracking_at_epoch_end(metric_tracker=self.valid_metrics, epoch=epoch,
                                                           outputs=outputs_list, targets=targets_list,
                                                           targets_all_labels=targets_all_labels_list, mode='valid',
                                                           track_cms=False)


        # For validation, the metrics are contained each epoch
        valid_log = self.valid_metrics.result(include_epoch_metrics=True)

        if self._use_tune:
            # Report some metrics back to Ray Tune. Specifically, we send the validation loss and CPSC F1 score back to
            # Ray Tune. Ray Tune can then use these metrics to decide which hyperparameter configuration lead to the
            # best results. These metrics can also be used to stop bad performing trials early
            tune.report(val_loss=valid_log['mean']["loss"],
                        val_macro_sk_f1=valid_log['mean']["macro_sk_f1"],
                        val_weighted_sk_f1=valid_log['mean']["weighted_sk_f1"])

        # # Also log the confusion matrix-related information to the valid log
        # valid_cm_information = dict({"overall_cm": "\n" + str(self.valid_cms.cm)},
        #                             **{"Confusion matrix for class " + str(class_cm.columns.name): "\n" + str(class_cm)
        #                                for _, class_cm in enumerate(self.valid_cms.class_wise_cms)})

        return valid_log, valid_sum_str

    def _do_iter_update(self, metric_tracker, output, target, loss_val):
        # Iteratively update the loss and the metrics with the MetricTracker
        # To this end, the Tracker internally maintains a dataframe containing different columns:
        # columns=['current', 'sum', 'square_sum','counts', 'mean', 'square_avg', 'std']
        # The loss and each metric are updated in a separate row for each of them
        metric_tracker.iter_update('loss', loss_val, n=output.shape[0])
        for met in self.metrics_iter:
            if self.multi_label_training:
                additional_args = self.config['metrics']['ml']['per_iteration'][met.__name__]
            else:
                additional_args = self.config['metrics']['sl']['per_iteration'][met.__name__]
            additional_kwargs = {
                param_name: self._param_dict[param_name] for param_name in additional_args
            }
            metric_tracker.iter_update(met.__name__, met(target=target, output=output, **additional_kwargs),
                                       n=output.shape[0])

    def _handle_tracking_at_epoch_end(self, metric_tracker, epoch, outputs, targets,
                                      targets_all_labels, mode, track_cms):
        # Get detached tensors from the list for further evaluation
        # For this, create a tensor from the dynamically filled list
        det_outputs = torch.cat(outputs).detach().cpu()
        det_targets = torch.cat(targets).detach().cpu()
        det_targets_all_labels = torch.cat(targets_all_labels).detach().cpu() if not self.multi_label_training else None

        # ------------ Confusion matrices ------------------------
        if track_cms:
            cm_tracker = self.train_cms if mode == 'train' else self.valid_cms
            # Dump the confusion matrices into a pickle file each few epochs and send them to the writer
            self._handle_cm_at_epoch_end(cm_tracker=cm_tracker, epoch=epoch, det_outputs=det_outputs,
                                         det_targets=det_targets, str_mode=mode)

        # ------------ Metrics ------------------------------------
        # Finally, the epoch-based metrics need to be updated
        # For this, calculate both, the normal epoch-based metrics as well as the class-wise epoch-based metrics
        for met in self.metrics_epoch:
            args = inspect.signature(met).parameters.values()
            # Output and target are needed for all metrics! Only consider other args WITHOUT default
            additional_args = [arg.name for arg in args
                               if arg.name not in ('output', 'target') and arg.default is arg.empty]
            additional_kwargs = {
                param_name: self._param_dict[param_name] for param_name in additional_args
            }
            metric_tracker.epoch_update(met.__name__, met(target=det_targets, output=det_outputs,
                                                              **additional_kwargs))

        # This holds for the class-wise, epoch-based metrics as well
        for met in self.metrics_epoch_class_wise:
            args = inspect.signature(met).parameters.values()
            # Output and target are needed for all metrics! Only consider other args WITHOUT default
            additional_args = [arg.name for arg in args
                               if arg.name not in ('output', 'target') and arg.default is arg.empty]
            additional_kwargs = {
                param_name: self._param_dict[param_name] for param_name in additional_args
            }
            metric_tracker.class_wise_epoch_update(met.__name__, met(target=det_targets, output=det_outputs,
                                                                     **additional_kwargs))

        # ------------------- Predicted Scores and Classes -------------------
        # Create a summary for each call, dump the dict and return the string
        # Retrieve the nested paths for cross_valid and args data_dir
        cross_valid_data_dir = self.config.config.get("data_loader", {}).get("cross_valid", {}).get("data_dir", "")
        args_data_dir = self.config.config.get("data_loader", {}).get("args", {}).get("data_dir", "")
        if cross_valid_data_dir:
            specified_data_dir = cross_valid_data_dir
        elif args_data_dir:
            specified_data_dir = args_data_dir
        else:
            raise ValueError("A data dir must be specified for either CV or normal training!")
        summary_str = self._create_report_summary(det_outputs=det_outputs, det_targets=det_targets, epoch=epoch,
                                                  data_dir=specified_data_dir)
        return summary_str

    def _do_cm_updates(self, cm_tracker, output, target):
        if not self.multi_label_training:
            upd_cm = overall_confusion_matrix_sk(output=output,
                                                 target=target,
                                                 log_probs=self._param_dict['log_probs'],
                                                 logits=self._param_dict['logits'],
                                                 labels=self._param_dict['labels'])
            cm_tracker.update_cm(upd_cm)
            upd_class_wise_cms = class_wise_confusion_matrices_single_label_sk(output=output,
                                                                               target=target,
                                                                               log_probs=self._param_dict['log_probs'],
                                                                               logits=self._param_dict['logits'],
                                                                               labels=self._param_dict['labels'])
        else:
            upd_class_wise_cms = class_wise_confusion_matrices_multi_label_sk(output=output,
                                                                              target=target,
                                                                              logits=self._param_dict['logits'],
                                                                              labels=self._param_dict['labels'])
        cm_tracker.update_class_wise_cms(upd_class_wise_cms)

    def _handle_cm_at_epoch_end(self, cm_tracker, epoch, det_outputs, det_targets, str_mode):
        # Update the confusion matrices maintained by the ClassificationTracker
        self._do_cm_updates(cm_tracker=cm_tracker, output=det_outputs, target=det_targets)
        # At the end of each epoch, explicitly send the confusion matrices to the SummaryWriter/TensorboardWriter
        cm_tracker.send_cms_to_writer(epoch=epoch)
        # Moreover, save them as pickle
        if not self._use_tune:
            main_path = self.config.log_dir
        else:
            main_path = Path(tune.get_trial_dir().replace('/models/', '/log/'))
        with open(os.path.join(main_path, "cms_" + str(str_mode) + "_epoch_" + str(epoch) + ".p"), 'wb') as cm_file:
            all_cms = [self.valid_cms.cm, self.valid_cms.class_wise_cms]
            pickle.dump(all_cms, cm_file)
            # Can later be loaded as follows:
            # with open(os.path.join(self.config.log_dir, "cms" + str(epoch) + ".p"), "rb") as file:
            #     test = pickle.load(file)

    def _create_report_summary(self, det_outputs, det_targets, epoch, data_dir):
        if self.multi_label_training:
            summary_dict = multi_label_metrics.sk_classification_summary(output=det_outputs, target=det_targets,
                                                                         logits=self._param_dict["logits"],
                                                                         labels=self._param_dict["labels"],
                                                                         output_dict=True,
                                                                         data_dir=data_dir)
        else:
            summary_dict = single_label_metrics.sk_classification_summary(output=det_outputs, target=det_targets,
                                                                          log_probs=self._param_dict["log_probs"],
                                                                          logits=self._param_dict["logits"],
                                                                          labels=self._param_dict["labels"],
                                                                          output_dict=True,
                                                                          data_dir=data_dir)

        return "Summary Report for Epoch " + str(epoch) + ":\n" + pd.DataFrame(summary_dict).to_string()



    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        data_loader = self.data_loader  # if not valid else self.valid_data_loader
        if hasattr(data_loader, 'n_samples'):
            current = batch_idx * data_loader.batch_size
            total = data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
