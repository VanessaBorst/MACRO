import os

import random
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset

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
                 fold_id=None, total_num_folds=None,
                 collate_fn=default_collate, worker_init_fn=seed_worker, generator=torch.Generator()):

        generator.manual_seed(global_config.SEED)

        # If a dedicated validation split is given (PTB-XL) as path, treat it separately
        # Attention: This has nothing to do with cross-validation, where the validation set is passed with valid_idx
        if isinstance(validation_split, Dataset):
            self.single_run_valid_set_provided = True
            self.valid_n_samples = len(validation_split)
            assert not cross_valid, "This should never happen. If it does, something went wrong!"
        else:
            self.single_run_valid_set_provided = False

        if not cross_valid:
            self.validation_split = validation_split
            self.shuffle = shuffle

            self.batch_idx = 0
            self.batch_size = batch_size
            self.n_samples = len(dataset)

            self.sampler, self.valid_sampler = self._split_sampler(self.validation_split)

        else:
            if cv_train_mode:
                self.batch_size = batch_size

                # Update 10.11.23; before: np.random.seed(0)
                # np.random.seed(global_config.SEED)

                train_sampler = SubsetRandomSampler(train_idx)
                valid_sampler = SubsetRandomSampler(valid_idx)

                # Write it to file for reproducibility, if not yet existing
                # If existing, check that the split is always the same (for fixed SEED)
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
                    # For raw inference on an arbitrary data amount (train, valid OR test),
                    # the data split is checked before and the idx can vary
                    # (For the optimization, depending on whether only the valid idx or the train +  valid idx are used)
                    # In this case, they are passed as test_idx parameter, since no training is needed
                    # Write it to file for reproducibility, if not yet existing
                    # If existing, check that the split is always the same (for fixed SEED)
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
        if self.single_run_valid_set_provided:
            # Use the defined split
            idx_full_train = np.arange(self.n_samples)
            idx_full_valid = np.arange(self.valid_n_samples)

            if self.shuffle:
                np.random.shuffle(idx_full_train)
                np.random.shuffle(idx_full_valid)

            train_idx = idx_full_train
            valid_idx = idx_full_valid

        else:
            if split == 0.0:
                return None, None

            idx_full = np.arange(self.n_samples)

            if self.shuffle:
                np.random.shuffle(idx_full)


            if isinstance(split, int):
                assert split > 0
                assert split < self.n_samples, "validation set size is configured to be larger than entire dataset."
                len_valid = split
            else:
                len_valid = int(self.n_samples * split)

            valid_idx = idx_full[0:len_valid]
            train_idx = np.delete(idx_full, np.arange(0, len_valid))

            self.n_samples = len(train_idx)

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

        return train_sampler, valid_sampler

    def split_validation(self):
        if self.valid_sampler is None:
            return None
        else:
            if self.single_run_valid_set_provided:
                return DataLoader(dataset=self.validation_split,
                                  sampler=self.valid_sampler,
                                  **{k: v for k, v in self.init_kwargs.items() if k != 'dataset'})
            else:
                return DataLoader(sampler=self.valid_sampler, **self.init_kwargs)
