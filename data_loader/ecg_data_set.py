from typing import Tuple, List
import numpy as np
from scipy.io import loadmat

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
