import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler


class BaseDataLoader(DataLoader):
    """
    Base class for all data loaders:
    Handles batch generation, data shuffling, and validation data splitting.
    """

    def __init__(self, dataset, batch_size, shuffle, validation_split, num_workers, collate_fn=default_collate,
                 single_batch=False):
        """
        single_batch: If set to True, this reduces the training set to a single batch and turns off the validation set.
        Training on a single batch should quickly overfit and reach accuracy 1.0.
        This is a recommended step for debugging networks, see https://twitter.com/karpathy/status/1013244313327681536
        """
        self.validation_split = validation_split
        self.shuffle = shuffle

        self.batch_idx = 0
        self.batch_size = batch_size
        self.n_samples = len(dataset)

        np.random.seed(0)

        if not single_batch:
            self.sampler, self.valid_sampler = self._split_sampler(self.validation_split)
        else:
            idx_full = np.arange(self.n_samples)
            np.random.shuffle(idx_full)
            # Use only the number of samples contained in one batch and try to overfit them
            num_samples = batch_size if not batch_size % 2 == 1 else batch_size-1
            self.sampler = SubsetRandomSampler(idx_full[:num_samples])
            # Set the new batch_size to 2 to update the gradient after each two samples
            self.batch_size = 2
            self.n_samples = num_samples
            self.valid_sampler = None
            self.shuffle = None

        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': self.batch_size,
            'shuffle': self.shuffle,
            'collate_fn': collate_fn,
            'num_workers': num_workers
        }
        super().__init__(sampler=self.sampler, **self.init_kwargs)

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
