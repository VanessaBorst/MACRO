import numpy as np
import os

import random
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler

import global_config
from utils import get_project_root, ensure_dir


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _consistency_check_data_split_cv(file_name, desired_ids):
    if os.path.isfile(file_name):
        # Existing -> Check for consistency
        with open(file_name, "r") as file:
            saved_ids_for_fold = np.loadtxt(file, dtype=int)
            assert sorted(saved_ids_for_fold) == sorted(desired_ids), \
                "Data Split Error during cross-fold-validation! Check this again!"
    else:
        # Not existing -> Create
        with open(file_name, "w+") as file:
            np.savetxt(file, sorted(desired_ids.astype(int)), fmt='%i', delimiter=",")


class BaseDataLoader(DataLoader):
    """
    Base class for all data loaders:
    Handles batch generation, data shuffling, and validation data splitting.
    """

    def __init__(self, dataset, batch_size, shuffle, validation_split=None, num_workers=1, pin_memory=False,
                 cross_valid=False, train_idx=None, valid_idx=None, test_idx=None, cv_train_mode=True,
                 fold_id=None, total_num_folds=None, stratified_k_fold=False,
                 collate_fn=default_collate,
                 single_batch=False, worker_init_fn=seed_worker, generator=torch.Generator()):
        """
        single_batch: If set to True, this reduces the training set to a single batch and turns off the validation set.
        Training on a single batch should quickly overfit and reach accuracy 1.0.
        This is a recommended step for debugging networks, see https://twitter.com/karpathy/status/1013244313327681536
        """

        generator.manual_seed(global_config.SEED)

        if not cross_valid:
            self.validation_split = validation_split
            self.shuffle = shuffle

            self.batch_idx = 0
            self.batch_size = batch_size
            self.n_samples = len(dataset)

            # Update 10.11.23; before: np.random.seed(0)
            # np.random.seed(global_config.SEED)

            if not single_batch:
                self.sampler, self.valid_sampler = self._split_sampler(self.validation_split)
            else:
                idx_full = np.arange(self.n_samples)
                np.random.shuffle(idx_full)
                # Use only the number of samples contained in one batch and try to overfit them
                num_samples = batch_size if not batch_size % 2 == 1 else batch_size - 1
                self.sampler = SubsetRandomSampler(idx_full[:num_samples])
                # Set the new batch_size to 2 to update the gradient after each two samples
                self.batch_size = 2
                self.n_samples = num_samples
                self.valid_sampler = None
                self.shuffle = None

        else:
            if cv_train_mode:
                self.batch_size = batch_size

                # Update 10.11.23; before: np.random.seed(0)
                # np.random.seed(global_config.SEED)

                train_sampler = SubsetRandomSampler(train_idx)
                valid_sampler = SubsetRandomSampler(valid_idx)

                # Write it to file for reproducibility, if not yet existing
                # If existing, check that the split is always the same (for fixed SEED)
                if stratified_k_fold:
                    path = os.path.join(get_project_root(), "cross_fold_log", f"{total_num_folds}_fold",
                                        "stratified", f"seed_{global_config.SEED}")
                else:
                    path = os.path.join(get_project_root(), "cross_fold_log", f"{total_num_folds}_fold",
                                    f"seed_{global_config.SEED}")

                if "drop_invalid" in dataset._input_dir:
                    path = os.path.join(path, "drop_invalid")
                file_name = os.path.join(path, "cv_valid_" + str(fold_id + 1) + ".txt")
                ensure_dir(path)
                _consistency_check_data_split_cv(file_name, desired_ids=valid_idx)

                # turn off shuffle option which is mutually exclusive with sampler
                self.shuffle = False
                self.n_samples = len(train_idx)

                self.sampler = train_sampler
                self.valid_sampler = valid_sampler
            else:
                self.batch_size = batch_size

                test_sampler = SubsetRandomSampler(test_idx)

                if total_num_folds is not None:
                    # For threshold optimization and raw inference on an arbitrary data amount (train, valid OR test),
                    # the data split is checked before and the idx can vary
                    # (For the optimization, depending on whether only the valid idx or the train +  valid idx are used)
                    # In this case, they are passed as test_idx parameter, since no training is needed
                    # Write it to file for reproducibility, if not yet existing
                    # If existing, check that the split is always the same (for fixed SEED)
                    if stratified_k_fold:
                        path = os.path.join(get_project_root(), "cross_fold_log", f"{total_num_folds}_fold",
                                            "stratified", f"seed_{global_config.SEED}")
                    else:
                        path = os.path.join(get_project_root(), "cross_fold_log", f"{total_num_folds}_fold",
                                            f"seed_{global_config.SEED}")
                    if "drop_invalid" in dataset._input_dir:
                        path = os.path.join(path, "drop_invalid")
                    file_name = os.path.join(path, "cv_test_" + str(fold_id + 1) + ".txt")
                    ensure_dir(path)
                    _consistency_check_data_split_cv(file_name, desired_ids=test_idx)

                # turn off shuffle option which is mutually exclusive with sampler
                self.shuffle = False
                self.n_samples = len(test_idx)

                self.sampler = test_sampler

        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': self.batch_size,
            'shuffle': self.shuffle,
            'collate_fn': collate_fn,
            'num_workers': num_workers,
            'worker_init_fn': worker_init_fn,
            'generator': generator
        }
        super().__init__(sampler=self.sampler, pin_memory=pin_memory, **self.init_kwargs)

    def _split_sampler(self, split):
        if split == 0.0:
            return None, None

        idx_full = np.arange(self.n_samples)

        np.random.shuffle(idx_full)

        if isinstance(split, int):
            assert split > 0
            assert split < self.n_samples, "validation set size is configured to be larger than entire dataset."
            len_valid = split
        else:
            len_valid = int(self.n_samples * split)

        valid_idx = idx_full[0:len_valid]
        train_idx = np.delete(idx_full, np.arange(0, len_valid))

        # For certain unlucky combinations of batch size (bs) and the length of the training set it can happen
        # that training fails with an exception like the following:
        # "ValueError: Expected more than 1 value per channel when training, got input size [1, 4096]"
        # The cause of the error is a batch of size 1 being fed into the net which doesn't work with batchnorm layers.
        # Avoid this by removing the last sample in that case

        if len(valid_idx) % self.batch_size == 1:
            valid_idx = valid_idx[:-1]
        if len(train_idx) % self.batch_size == 1:
            train_idx = train_idx[:-1]

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        # turn off shuffle option which is mutually exclusive with sampler
        self.shuffle = False
        self.n_samples = len(train_idx)

        return train_sampler, valid_sampler

    def split_validation(self):
        if self.valid_sampler is None:
            return None
        else:
            return DataLoader(sampler=self.valid_sampler, **self.init_kwargs)
