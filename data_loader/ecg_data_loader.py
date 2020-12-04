import collections
from typing import Tuple

import torch
from scipy.io import loadmat
from torch.utils.data.dataset import T_co
from torchvision import datasets, transforms
from base import BaseDataLoader
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import os
import numpy as np

from data_loader.loading_utils import _custom_collate


class ECGDataLoader(BaseDataLoader):
    """
    ECG data loading using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1):
        trsfm = None
        # transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.RandomCrop([12, 5000]),
        #     transforms.Normalize((0,), (1,))      # TODO Check if this works for non-image data -> probably not
        # ])
        self.data_dir = data_dir
        self.dataset = ECGDataset(self.data_dir, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers, _custom_collate)


class ECGDataset(Dataset):
    """
    ECG dataset
    Read the record names in __init__ but leaves the reading of actual data images to __getitem__.
    This is memory efficient because all the records are not stored in the memory at once but read as required.
    """

    def __init__(self, input_dir, transform=None):
        """
        Args:
            input_dir (Path): Path to the directory containing the wfdb .mat and .hea files for each record
            transform (callable, optional): Optional transform(s) to be applied on a sample.
        """
        header_files = []
        for f in os.listdir(input_dir):
            g = os.path.join(input_dir, f)
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

    def __getitem__(self, idx) -> Tuple[np.ndarray, str]:
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
        # record = record[:, :3000]       # Remove this later and use padding!

        if self.transform:
            record = self.transform(record)

        record_name = header_file[header_file.rfind('/') + 1:].replace('.hea', '')
        return record, classes, len(record[0]), record_name


dirname = os.path.dirname(__file__)
input_directory = os.path.join(dirname, '../data/CPSC/raw/Training_WFDB')
ecg_data_loader = ECGDataLoader(input_directory, 4, True)

for batch_num, batch_data in enumerate(ecg_data_loader):
    # padded_records, labels, lengths, record_names = zip(*batch_data)
    padded_records = batch_data[0]
    labels = batch_data[1]
    lengths = batch_data[2]
    record_names = batch_data[3]
    print(padded_records.shape)
    print(labels)

    for i in range(len(padded_records)):
        plt.plot(padded_records[i][0])
        plt.title("Lead I of record " + str(record_names[i])+ " of batch " + str(batch_num) + ": Label = " + str(labels[i]))
        plt.show()
