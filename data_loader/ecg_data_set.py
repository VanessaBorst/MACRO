import functools
import operator
from typing import Tuple, List
import numpy as np
from scipy.io import loadmat
from sklearn.utils import compute_class_weight

from torch.utils.data import Dataset
import os
import torch

import pickle as pk
import pandas as pd
from utils import get_project_root


class ECGDataset(Dataset):
    """
    ECG dataset
    Read the record names in __init__ but leaves the reading of actual data to __getitem__.
    This is memory efficient because all the records are not stored in the memory at once but read as required.
    """

    def __init__(self, input_dir, transform=None):
        """

        :param input_dir: Path  -> Path to the directory containing the preprocessed pickle files for each record
        :param multi_label_training: bool -> If set to true, multi-label training is applie
        :param transform: callable, optional -> Optional transform(s) to be applied on a sample.
        """

        records = []
        for file in os.listdir(input_dir):
            if ".pk" not in file:
                continue
            records.append(file)

        self._input_dir = input_dir
        self.records = records
        # Save list of classes occurring in the dataset
        _, meta = pk.load(open(os.path.join(self._input_dir, records[0]), "rb"))
        self.class_labels = meta["classes_one_hot"].index.values
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx) -> Tuple[pd.DataFrame, List[int], int, str]:
        if torch.is_tensor(idx):
            idx = idx.tolist()

        record_name = self.records[idx]
        # record is a df, meta a dict
        record, meta = pk.load(open(os.path.join(self._input_dir, record_name), "rb"))
        # Ensure that the record is not containing any unknown class label
        assert all(label in self.class_labels for label in meta["classes_encoded"])

        if self.transform:
            record = self.transform(record)

        return record, meta["classes_encoded"], meta["classes_one_hot"], len(record.index), record_name

    def get_target_distribution(self, idx_list, multi_labels):
        """
        Can be used to determine the target classes distribution
        :param idx_list: list of ids, should contain all ids contained in the train, valid or test set
        :return: Class distribution
        :param multi_labels: If set to False, only the first label is considered for determining the target dist.

        Can be used e.g. as torch.Tensor(train.get_target_distribution()).to(device)
        """
        classes = []
        for idx in idx_list:
            _, classes_encoded, classes_one_hot, _, record_name = self.__getitem__(idx)
            if multi_labels:
                classes.append(classes_one_hot)
            else:
                # Only consider the first label
                classes_one_hot[:] = 0
                classes_one_hot[classes_encoded[0]] = 1

        target_sum = pd.DataFrame(classes).sum()
        # Get the weights as ndarray
        class_weights = target_sum.div(target_sum.sum()).values
        return class_weights

    def get_inverse_class_frequency(self, idx_list, multi_labels=True):
        """
        Can be used to determine the target classes distribution
        :param idx_list: list of ids, should contain all ids contained in the train, valid or test set
        :return: Class distribution
        :param multi_labels: If set to False, only the first label is considered for determining the target dist.
        """

        classes = []
        for idx in idx_list:
            _, classes_encoded, classes_one_hot, _, record_name = self.__getitem__(idx)
            if multi_labels:
                classes.append(classes_encoded)
            else:
                # Only consider the first label
                classes.append(classes_encoded[0])

        # Flatten the classes to a one-dimensional array
        classes = functools.reduce(operator.iconcat, classes, [])
        # TODO the above is not completely correct, since the following method interprets the array
        #  length as number of samples
        class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(classes), y=classes)

        return class_weights

if __name__ == '__main__':
    dataset = ECGDataset("data/CinC_CPSC/train/preprocessed/no_sampling/")
    target_distribution = dataset.get_target_distribution([0,1,4,5,6,71,10,101])
    class_weights = dataset.get_inverse_class_frequency([0,1,4,5,6,71,10,101])
