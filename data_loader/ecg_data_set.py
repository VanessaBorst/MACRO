import functools
import json
import operator
import os
import pickle
import pickle as pk
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd
import torch
from sklearn.utils import compute_class_weight
from torch.utils.data import Dataset

from utils import get_project_root


class ECGDataset(Dataset):
    """
    ECG dataset
    Read the record names in __init__ but leaves the reading of actual data to __getitem__.
    This is memory efficient because all the records are not stored in the memory at once but read as required.
    """

    def __init__(self, input_dir):
        """

        :param input_dir: Path  -> Path to the directory containing the preprocessed pickle files for each record
        :param transform: callable, optional -> Optional transform(s) to be applied on a sample.
        """

        records = []
        for file in sorted(os.listdir(input_dir)):
            if ".pk" not in file:
                continue
            records.append(file)

        self._input_dir = input_dir
        self.records = records
        # Save list of classes occurring in the dataset
        _, meta = pk.load(open(os.path.join(self._input_dir, records[0]), "rb"))
        self.class_labels = meta["classes_one_hot"].index.values

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        record_name = self.records[idx]
        # record is a df, meta a dict
        record, meta = pk.load(open(os.path.join(self._input_dir, record_name), "rb"))
        # Ensure that the record is not containing any unknown class label
        assert all(label in self.class_labels for label in meta["classes_encoded"])

        # Removed len(record.index) as length, now they are all 72000
        # torch.tensor(record.values).float()
        return record.values.astype("float32"), \
               str(meta["classes_encoded"]), meta["classes_encoded"][0], \
               meta["classes_one_hot"].values, record_name

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
            _, _, first_class_encoded, classes_one_hot, record_name = self.__getitem__(idx)
            if multi_label_training:
                classes.append(classes_one_hot)
            else:
                # Only consider the first label
                classes_one_hot[:] = 0
                classes_one_hot[first_class_encoded] = 1
                classes.append(classes_one_hot)

        # Get the class freqs as Pandas series
        class_freqs = pd.DataFrame(classes).sum()
        # Get the class distribution as Pandas series
        class_dist = class_freqs.div(class_freqs.sum())

        # Return both as as ndarray
        return class_freqs.values, class_dist.values

    def get_ml_pos_weights(self, idx_list, mode=None):
        """
        Can be used to determine  the class frequencies and the target classes distribution
        :param idx_list: list of ids, should contain all ids contained in the train, valid or test set
        :param mode: should be 'train' or 'valid'
        :param log_dir: dir to save the record names to
        :return:  Pos weights, one weight per class

        """
        if mode == "test" and "test" not in self._input_dir:
            # Catch the case where the model is tested on the validation set
            file_name = "data_loader/pos_weights_ml_valid.p"
        else:
            file_name = "data_loader/pos_weights_ml_" + mode + ".p"

        file_name = os.path.join(get_project_root(), file_name)
        if os.path.isfile(file_name):
            with open(file_name, "rb") as file:
                df = pickle.load(file)
            # inverse_class_freqs = df['Inverse_class_freq']
        else:
            classes = []
            record_names = []
            for idx in idx_list:
                _, _, _, classes_one_hot, record_name = self.__getitem__(idx)
                classes.append(classes_one_hot)
                record_names.append(record_name)

            # Dump the record names to pickle to ensure they are the same between VMs
            project_root = get_project_root()
            if mode is not None:
                with open(os.path.join(project_root, 'data_loader', 'Record_names_' + str(mode)) + ".p", 'wb+') as file:
                    pickle.dump(record_names, file)

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

            with open(Path(file_name), "wb") as file:
                pickle.dump(df, file)

        # Return the ratio as as ndarray
        return df["ratio_neg_to_pos"].values

    def get_inverse_class_frequency(self, idx_list, multi_label_training, mode):
        """
        Can be used to determine the inverse class frequencies
        :param idx_list: list of ids, should contain all ids contained in the train, valid or test set
        :param multi_label_training: If true, all labels are considered, otherwise only the first label is counted

        :return:  Inverse class frequencies
        """
        if mode == "test" and "test" not in self._input_dir:
            # Catch the case where the model is tested on the validation set
            file_name = "data_loader/class_freq_ml_valid.p" if multi_label_training \
                else "data_loader/class_freq_sl_valid.p"
        else:
            file_name = "data_loader/class_freq_ml_" + mode + ".p" if multi_label_training \
                else "data_loader/class_freq_sl_" + mode + ".p"
        file_name = os.path.join(get_project_root(), file_name)

        if os.path.isfile(file_name):
            with open(file_name, "rb") as file:
                df = pickle.load(file)
            inverse_class_freqs = df['Inverse_class_freq']
        else:
            classes = []
            for idx in idx_list:
                _, _, first_class_encoded, classes_one_hot, record_name = self.__getitem__(idx)
                if multi_label_training:
                    classes.append(classes_one_hot)
                else:
                    # Only consider the first label
                    classes_one_hot[:] = 0
                    classes_one_hot[first_class_encoded] = 1
                    classes.append(classes_one_hot)

            # Get the class freqs as Pandas series
            class_freqs = pd.DataFrame(classes).sum()

            # Each class should occur at least ones
            assert not class_freqs.isin([0]).any(), "Each class should occur at least ones"

            # Calculate the inverse class freqs
            inverse_class_freqs = class_freqs.apply(lambda x: class_freqs.sum() / x)

            df = pd.concat([class_freqs, inverse_class_freqs], axis=1)
            df.columns = ['Class_freq', 'Inverse_class_freq']

            with open(Path(file_name), "wb") as file:
                pickle.dump(df, file)

        # Return them as as ndarray
        return inverse_class_freqs.values


if __name__ == '__main__':
    dataset = ECGDataset("data/CinC_CPSC/train/preprocessed/no_sampling/eq_len_72000/")

    # for idx in range(0, len(dataset)):
    #     _, classes_encoded, _, _, _ = dataset.__getitem__(idx)
    #     if json.loads(classes_encoded)==[9]:
    #         break

    # class_freqs, target_distribution = dataset.get_class_freqs_and_target_distribution([0, 1, 4, 5, 6, 71, 10, 99, 31],
    #                                                                                     multi_label_training=True)
    # pos_weights = dataset.get_ml_pos_weights([0, 1, 4, 5, 6, 71, 10, 99, 31])

    class_weights1 = dataset.get_inverse_class_frequency_old([0, 1, 2, 4, 5, 6, 7, 71, 10, 99, 31, 28, 9, 33, 15 ],
                                                             multi_label_training=False)
    class_weights2 = dataset.get_inverse_class_frequency([0, 1, 2, 4, 5, 6, 7, 71, 10, 99, 31, 28, 9, 33, 15 ], multi_label_training=False)
    print("Done")
