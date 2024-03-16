from functools import partial

from base import BaseDataLoader
from data_loader.ecg_data_set import ECGDataset
from data_loader.loading_utils import _collate_pad_or_truncate


class ECGDataLoader(BaseDataLoader):
    """
    ECG data loading using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, pin_memory=False,
                 cross_valid=False, train_idx=None, valid_idx=None, test_idx=None, cv_train_mode=True, fold_id=None,
                 total_num_folds=None, single_batch=False):
        self.data_dir = data_dir
        self.dataset = ECGDataset(self.data_dir)
        # 30.06.: Removed collate_fn param: collate_fn=partial(_collate_pad_or_truncate, seq_len=seq_len)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers, pin_memory,
                         cross_valid=cross_valid, train_idx=train_idx, valid_idx=valid_idx,
                         test_idx=test_idx, cv_train_mode=cv_train_mode, fold_id=fold_id,
                         total_num_folds=total_num_folds, single_batch=single_batch)


