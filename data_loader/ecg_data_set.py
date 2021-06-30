import functools
import operator
import os
import pickle as pk
from typing import Tuple, List

import numpy as np
import pandas as pd
import torch
from sklearn.utils import compute_class_weight
from torch.utils.data import Dataset


class ECGDataset(Dataset):
    """
    ECG dataset
    Read the record names in __init__ but leaves the reading of actual data to __getitem__.
    This is memory efficient because all the records are not stored in the memory at once but read as required.
    """

    def __init__(self, input_dir, transform=None):
        """

        :param input_dir: Path  -> Path to the directory containing the preprocessed pickle files for each record
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

    def get_class_freqs_and_target_distribution(self, idx_list, multi_label_training):
        """
        Can be used to determine  the class frequencies and the target classes distribution
        :param idx_list: list of ids, should contain all ids contained in the train, valid or test set
        :param multi_label_training: If true, all labels are considered, otherwise only the first label is counted
        :return:  Class frequencies and class distribution

        Deprecated: Can be used e.g. as torch.Tensor(train.get_target_distribution()).to(device)
        """
        classes = []
        for idx in idx_list:
            _, classes_encoded, classes_one_hot, _, record_name = self.__getitem__(idx)
            if multi_label_training:
                classes.append(classes_one_hot)
            else:
                # Only consider the first label
                classes_one_hot[:] = 0
                classes_one_hot[classes_encoded[0]] = 1
                classes.append(classes_one_hot)

        # Get the class freqs as Pandas series
        class_freqs = pd.DataFrame(classes).sum()
        # Get the class distribution as Pandas series
        class_dist = class_freqs.div(class_freqs.sum())

        # Return both as as ndarray
        return class_freqs.values, class_dist.values

    def get_ml_pos_weights(self, idx_list):
        """
        Can be used to determine  the class frequencies and the target classes distribution
        :param idx_list: list of ids, should contain all ids contained in the train, valid or test set
        :return:  Pos weights, one weight per class

        """
        classes = []
        for idx in idx_list:
            _, _, classes_one_hot, _, _ = self.__getitem__(idx)
            classes.append(classes_one_hot)

        # Get the class freqs as Pandas series
        class_freqs = pd.DataFrame(classes).sum()

        # Calculate the number of pos and negative samples per class
        df = pd.DataFrame({'num_pos_samples': class_freqs})

        # Each class should occur at least ones
        assert not df['num_pos_samples'].isin([0]).any(), "Each class should occur at least ones"

        df['num_neg_samples'] = df.apply(lambda row: len(idx_list) - row.values).values
        df["ratio_neg_to_pos"] = df.num_neg_samples / (df.num_pos_samples)
        # If num_pos_samples can be 0, a dummy term needs to be added to it to avoid dividing by 0
        # df["ratio_neg_to_pos"] = df.num_neg_samples / (df.num_pos_samples + 1e-5)

        # Return the ratio as as ndarray
        return df["ratio_neg_to_pos"].values

    def get_inverse_class_frequency(self, idx_list, multi_label_training):
        """
        Can be used to determine the inverse class frequencies
        :param idx_list: list of ids, should contain all ids contained in the train, valid or test set
        :param multi_label_training: If true, all labels are considered, otherwise only the first label is counted

        :return:  Inverse class frequencies
        """

        classes = []
        for idx in idx_list:
            _, classes_encoded, classes_one_hot, _, record_name = self.__getitem__(idx)
            if multi_label_training:
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
    # class_freqs, target_distribution = dataset.get_class_freqs_and_target_distribution([0, 1, 4, 5, 6, 71, 10, 99, 76],
    #                                                                                    multi_label_training=True)
    pos_weights = dataset.get_ml_pos_weights([0, 1, 4, 5, 6, 71, 10, 99, 76])
    class_weights = dataset.get_inverse_class_frequency([0, 1, 4, 5, 6, 71, 10, 99, 76], multi_label_training=True)
    print("Done")
