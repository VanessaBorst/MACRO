from base import BaseDataLoader
from data_loader.ecg_data_set import ECGDataset
import os


class ECGDataLoader(BaseDataLoader):
    """
    ECG data loading using BaseDataLoader
    """

    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, pin_memory=False,
                 cross_valid=False, train_idx=None, valid_idx=None, test_idx=None, cv_train_mode=True, fold_id=None,
                 total_num_folds=None):
        # self.data_dir = data_dir
        if isinstance(validation_split, str):
            # If a dedicated validation split is given (PTB-XL), treat it separately
            assert os.path.exists(os.path.dirname(validation_split)), "The provided validation folder does not exist!"
            self.valid_dataset = ECGDataset(validation_split)

        # Set the training (or test) dataset
        self.dataset = ECGDataset(data_dir)
        super().__init__(dataset=self.dataset, batch_size=batch_size, shuffle=shuffle,
                         validation_split=self.valid_dataset if hasattr(self, 'valid_dataset') else validation_split,
                         num_workers=num_workers, pin_memory=pin_memory,
                         cross_valid=cross_valid, train_idx=train_idx, valid_idx=valid_idx,
                         test_idx=test_idx, cv_train_mode=cv_train_mode, fold_id=fold_id,
                         total_num_folds=total_num_folds)
