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
        Args:
            input_dir (Path): Path to the directory containing the preprocessed pickle files for each record
            transform (callable, optional): Optional transform(s) to be applied on a sample.
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
        self.class_labels = meta["classes_encoded"].index.values
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
        classes_one_hot = meta["classes_one_hot"]

        if self.transform:
            record = self.transform(record)

        return record, meta["classes_encoded"], len(record.index), record_name


class ECGDataset_Old(Dataset):
    """
    ECG dataset
    Read the record names in __init__ but leaves the reading of actual data to __getitem__.
    This is memory efficient because all the records are not stored in the memory at once but read as required.
    """

    def __init__(self, input_dir, transform=None):
        """
        Args:
            input_dir (Path): Path to the directory containing the wfdb .mat and .hea files for each record
            transform (callable, optional): Optional transform(s) to be applied on a sample.
        """
        header_files = []
        input_path = os.path.join(get_project_root(), input_dir)
        for f in os.listdir(input_path):
            g = os.path.join(input_path, f)
            if not f.lower().startswith('.') and f.lower().endswith('hea') and os.path.isfile(g):
                header_files.append(g)
        self.header_files = header_files
        self.transform = transform
        self.encoding = {
            "426783006": 1,
            "164889003": 2,
            "270492004": 3,
            "164909002": 4,
            "59118001":  5,
            "284470004": 6,
            "164884008": 7,
            "429622005": 8,
            "164931005": 9
        }

    def __len__(self):
        return len(self.header_files)

    def __getitem__(self, idx) -> Tuple[np.ndarray, str, int]:
        if torch.is_tensor(idx):
            idx = idx.tolist()

        header_file = self.header_files[idx]
        classes = []
        with open(header_file, 'r') as f:
            for line in f:
                if line.startswith('#Dx'):
                    tmp = line.split(': ')[1].split(',')
                    for c in tmp:
                        classes.append(self.encoding[c.strip()])
        mat_file = header_file.replace('.hea', '.mat')
        x = loadmat(mat_file)
        record = np.asarray(x['val'], dtype=np.float64)
        # record = record[:, :3000]       #  Remove this later and use padding!

        if self.transform:
            record = self.transform(record)

        record_name = header_file[header_file.rfind('/') + 1:].replace('.hea', '')
        return record, classes, len(record[0]), record_name
