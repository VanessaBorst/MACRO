
from base import BaseDataLoader
from data_loader.ecg_data_set import ECGDataset


class ECGDataLoader(BaseDataLoader):
    """
    ECG data loading using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, pin_memory=False,
                 cross_valid=False, train_idx=None, valid_idx=None, test_idx=None, cv_train_mode=True, fold_id=None,
                 total_num_folds=None):
        self.data_dir = data_dir
        self.dataset = ECGDataset(self.data_dir)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers, pin_memory,
                         cross_valid=cross_valid, train_idx=train_idx, valid_idx=valid_idx,
                         test_idx=test_idx, cv_train_mode=cv_train_mode, fold_id=fold_id,
                         total_num_folds=total_num_folds)


