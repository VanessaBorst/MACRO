from torchvision import datasets, transforms
from base import BaseDataLoader
from data_loader.ecg_data_set import ECGDataset
from data_loader.loading_utils import _collate_pad_or_truncate
from functools import partial


class ECGDataLoader(BaseDataLoader):
    """
    ECG data loading using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, seq_len=72000,
                 single_batch=False):
        self.data_dir = data_dir
        self.dataset = ECGDataset(self.data_dir)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers,
                         collate_fn=partial(_collate_pad_or_truncate, seq_len=seq_len), single_batch=single_batch)

