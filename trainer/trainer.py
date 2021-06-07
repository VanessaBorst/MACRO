import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from model.metric import get_confusion_matrix, get_class_wise_confusion_matrix
from utils import inf_loop
from utils.tracker import MetricTracker, ClassificationTracker


class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.device = device
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

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.train_cms = ClassificationTracker(*self.data_loader.dataset.class_labels, writer=self.writer)
        # The ECGDataset is the same for data_loader and valid_loader as it is realized with SubSamplers
        self.valid_cms = ClassificationTracker(*self.data_loader.dataset.class_labels, writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log (as dict) that contains average loss and metrics in this epoch.
            Example: {'loss': 0.6532998728738012, 'accuracy': 0.7919082176709545, 'top_k_acc': 0.9288835054163845}
        """
        self.model.train()
        # Reset the trackers
        self.train_metrics.reset()
        self.train_cms.reset()
        # Set the writer object to training mode
        self.writer.set_mode('train')

        for batch_idx, (padded_records, labels, lengths, record_names) in enumerate(self.data_loader):
            data, target = padded_records.to(self.device), labels.to(self.device)
            data = data.permute(0, 2, 1)  # switch seq_len and feature_size (12 = #leads)

            self.optimizer.zero_grad()
            # data has shape [batch_size, 12, seq_len]
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            # Iteratively update the loss and the metrics with the MetricTracker
            # To this end, the Tracker internally maintains a dataframe containing the following thee columns:
            # columns=['total', 'counts', 'average']
            # The loss and each metric are updated in a separate row for each of them
            self.train_metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output, target))

            # Update the confusion matrices maintained by the ClassificationTracker
            upd_cm = get_confusion_matrix(output=output, log_probs=True, target=target,
                                          labels=self.data_loader.dataset.class_labels)
            upd_class_wise_cms = get_class_wise_confusion_matrix(output=output, log_probs=True, target=target,
                                          labels=self.data_loader.dataset.class_labels)
            self.train_cms.update_cms(upd_cm, upd_class_wise_cms)

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))
                # cpu() moves the tensor to the CPU, because some operations cannot be performed on cuda tensors
                # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

            if batch_idx == self.len_epoch:
                break

        # At the end of each epoch, explicitly send the confusion matrices to the SummaryWriter/TensorboardWriter
        self.train_cms.send_cms_to_writer(epoch=epoch)

        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})    # Extends the dict by the val loss and metrics

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation (as dictionary)
        """
        self.model.eval()
        # Reset the trackers
        self.valid_metrics.reset()
        self.valid_cms.reset()
        # Set the writer object to validation mode
        self.writer.set_mode('valid')

        with torch.no_grad():
            for batch_idx, (padded_records, labels, lengths, record_names) in enumerate(self.valid_data_loader):
                data, target = padded_records.to(self.device), labels.to(self.device)
                data = data.permute(0, 2, 1)  # switch seq_len and feature_size (12 = #leads)

                output = self.model(data)
                loss = self.criterion(output, target)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx)
                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, target))
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

                # Update the confusion matrices maintained by the ClassificationTracker
                upd_cm = get_confusion_matrix(output=output, log_probs=True, target=target,
                                              labels=self.data_loader.dataset.class_labels)
                upd_class_wise_cms = get_class_wise_confusion_matrix(output=output, log_probs=True, target=target,
                                                                     labels=self.data_loader.dataset.class_labels)
                self.valid_cms.update_cms(upd_cm, upd_class_wise_cms)

        # At the end of each epoch, explicitly send the confusion matrices to the SummaryWriter/TensorboardWriter
        self.valid_cms.send_cms_to_writer(epoch=epoch)

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)


