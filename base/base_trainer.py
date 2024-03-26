import os
import pickle
import random
from abc import abstractmethod
from pathlib import Path

import numpy.random
import torch
from numpy import inf

from logger import TensorboardWriter
from ray import tune

class BaseTrainer:
    """
    Base class for all trainers:
    Handles checkpoint saving/resuming, training process logging, and more (including early stopping)
    """
    def __init__(self, model, criterion, optimizer, config, use_tune=False):
        self.config = config
        # create a logger with name "trainer" and the verbosity specified in the config.json
        self.logger = config.get_logger('trainer', config['trainer']['verbosity'])

        self.multi_label_training = config['arch']['args']['multi_label_training']

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

        cfg_trainer = config['trainer']
        self.epochs = cfg_trainer['epochs']
        self.save_period = cfg_trainer['save_period']
        self.monitor = cfg_trainer.get('monitor', 'off')   # 'off' is returned if the key doesn't exist in the dict
        self.profiler_active = cfg_trainer['profiler_active']

        self._use_tune = use_tune

        # configuration to monitor model performance and save best
        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']

            self.mnt_best = inf if self.mnt_mode == 'min' else -inf
            self.early_stop = cfg_trainer.get('early_stop', inf)
            if self.early_stop <= 0:
                self.early_stop = inf
            self.minimum_diff_for_improvement = cfg_trainer.get('minimum_diff_for_improvement', None)

        self.start_epoch = 1

        if not self._use_tune:
            self.checkpoint_dir = Path(config.save_dir)
        else:
            self.checkpoint_dir = Path(tune.get_trial_dir())

        if not self._use_tune:
            # setup visualization writer instance
            self.writer = TensorboardWriter(config.log_dir, self.logger, cfg_trainer['tensorboard'])
        else:
            self.writer = None

        if config.resume is not None:
            self._resume_checkpoint(config.resume)

    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def train(self):
        """
        Full training logic
        """
        not_improved_count = 0
        epoch_idx_best = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            # train_log is a df
            train_log = self._train_epoch(epoch)
            log_mean = train_log['mean']

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if self.mnt_mode != 'off':
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    if self.minimum_diff_for_improvement is None:
                        # Every improvement is considered as an improvement, no matter how small it is
                        improved = (self.mnt_mode == 'min' and log_mean[self.mnt_metric] < self.mnt_best) or \
                                   (self.mnt_mode == 'max' and log_mean[self.mnt_metric] > self.mnt_best)
                    else:
                        # Improvement is considered only if it is larger than a minimum difference
                        improved = (self.mnt_mode == 'min' and
                                    log_mean[self.mnt_metric] < self.mnt_best - self.minimum_diff_for_improvement) \
                                   or (self.mnt_mode == 'max' and
                                       log_mean[self.mnt_metric] > self.mnt_best + self.minimum_diff_for_improvement)
                except KeyError:
                    self.logger.warning("Warning: Metric '{}' is not found. "
                                        "Model performance monitoring is disabled.".format(self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = log_mean[self.mnt_metric]
                    log_best = log_mean

                    if self._use_tune:
                        # Save in case the training is stopped by a scheduler before return log best,
                        # which only is returned when the training finishes regularly
                        path = os.path.join(self.checkpoint_dir, "model_best_metrics.p")
                        with open(path, 'wb') as file:
                            pickle.dump(log_best, file)

                    not_improved_count = 0
                    best = True
                    epoch_idx_best = epoch
                else:
                    not_improved_count += 1
                    self.logger.info("Not improved for " + str(not_improved_count) + " epochs")

                if not_improved_count > self.early_stop:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                     "Training stops.".format(self.early_stop))
                    self.logger.info("Best Validation performance achieved in epoch {}.".format(epoch_idx_best))
                    break

            if epoch % self.save_period == 0 or best:
                # Save checkpoint if best in any case
                self.logger.info("Best {}: {:.6f}".format(self.mnt_metric, self.mnt_best))
                self._save_checkpoint(epoch, save_best=best)

        return log_best

    def _save_checkpoint(self, epoch, save_best=False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best,
            'config': self.config,
            'torch_rng_state': torch.get_rng_state(),
            'np_rng_state': numpy.random.get_state(),
            'random_state': random.getstate(),
            'torch_cuda_rng_states': torch.cuda.get_rng_state_all()
        }
        if self._use_tune:
            with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
                filename = str(os.path.join(checkpoint_dir, 'checkpoint-epoch{}.pth'.format(epoch)))
        else:
            filename = str(self.checkpoint_dir / 'checkpoint-epoch{}.pth'.format(epoch))

        torch.save(state, filename)
        self.logger.info("Saving checkpoint: {} ...".format(filename))

        if save_best:
            best_path = str(self.checkpoint_dir / 'model_best.pth')
            torch.save(state, best_path)
            self.logger.info("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']

        # Set the RNGs to ensure that the same data subset is used in the epoch as if there was no interruption
        torch.set_rng_state(checkpoint['torch_rng_state'])
        numpy.random.set_state(checkpoint['np_rng_state'])
        random.setstate(checkpoint['random_state'])
        torch.cuda.set_rng_state_all(checkpoint['torch_cuda_rng_states'])

        # load architecture params from checkpoint.
        if checkpoint['config']['arch'] != self.config['arch']:
            self.logger.warning("Warning: Architecture configuration given in config file is different from that of "
                                "checkpoint. This may yield an exception while state_dict is being loaded.")
        self.model.load_state_dict(checkpoint['state_dict'])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
            self.logger.warning("Warning: Optimizer type given in config file is different from that of checkpoint. "
                                "Optimizer parameters not being resumed.")
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))
