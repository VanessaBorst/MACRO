from functools import partial

from base import BaseDataLoader
from data_loader.ecg_data_set import ECGDataset
from data_loader.loading_utils import _collate_pad_or_truncate


class ECGDataLoader(BaseDataLoader):
    """
    ECG data loading using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, seq_len=72000,
                 single_batch=False):
        self.data_dir = data_dir
        self.dataset = ECGDataset(self.data_dir)
        # 30.06.: Removed collate_fn param: collate_fn=partial(_collate_pad_or_truncate, seq_len=seq_len)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers, single_batch=single_batch)


