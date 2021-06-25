from math import sqrt

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from torch.utils.tensorboard import SummaryWriter

from model.multi_label_metrics import class_wise_confusion_matrices_multi_label
from model.single_label_metrics import overall_confusion_matrix, class_wise_confusion_matrices_single_label
import numpy as np

smooth = 1e-6


class MetricTracker:
    """
    Internally keeps track of the loss and the metrics by means of a dataframe
    If specified, each update of the metrics is also passed to the TensorboardWriter
    """

    def __init__(self, keys_iter: list, keys_epoch: list, keys_epoch_class_wise: list, labels: list, writer=None):
        self.writer = writer
        # Create a dataframe containing one row per key (e.g. one per metric and another one for the loss)
        self._data_iter = pd.DataFrame(index=keys_iter,
                                       columns=['current', 'sum', 'square_sum', 'counts', 'mean', 'square_avg', 'std'],
                                       dtype=np.float64)
        all_keys_epoch_class_wise = [keys_epoch_class_wise[idx_ftn] + '_class_' + str(labels[idx_class])
                                 for idx_class in range(0, len(labels))
                                 for idx_ftn in range(0, len(keys_epoch_class_wise))]
        self._labels = labels
        self._data_epoch = pd.DataFrame(index=keys_epoch + all_keys_epoch_class_wise, columns=['mean'], dtype=np.float64)
        self.reset()

    def reset(self):
        for col in self._data_iter.columns:
            self._data_iter[col].values[:] = 0

    def iter_update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data_iter.at[key, 'current'] = value
        self._data_iter.at[key, 'sum'] += value * n
        self._data_iter.at[key, 'square_sum'] += value * value * n
        self._data_iter.at[key, 'counts'] += n

    def epoch_update(self, key, value, epoch):
        if self.writer is not None:
            self.writer.add_scalar(key, value, global_step=epoch)
        self._data_epoch.at[key, 'mean'] = value

    # class_wise_epoch_update(met.__name__, met(target=targets, output=outputs, **additional_kwargs), epoch)
    def class_wise_epoch_update(self, key, values, epoch):
        for idx_class in range(0, len(values)):
            full_key_name = key + '_class_' + str(self._labels[idx_class])
            if self.writer is not None:
                self.writer.add_scalar(full_key_name, values[idx_class], global_step=epoch)
            self._data_epoch.at[full_key_name, 'mean'] = values[idx_class]

    def current(self):
        return dict(self._data_iter['current'])

    def avg(self):
        for key, row in self._data_iter.iterrows():
            self._data_iter.at[key, 'mean'] = row['sum'] / row['counts']
            self._data_iter.at[key, 'square_avg'] = row['square_sum'] / row['counts']

    def std(self):
        for key, row in self._data_iter.iterrows():
            self._data_iter.at[key, 'std'] = sqrt(row['square_avg'] - row['mean']**2 + smooth)

    def result(self):
        self.avg()
        self.std()
        iter_result = self._data_iter[['mean', 'std']]
        epoch_result = self._data_epoch
        return pd.concat([iter_result, epoch_result])


class ConfusionMatrixTracker:
    """
        Internally keeps track of the confusion matrix and the class-wise confusion matrices by means of dataframes
        If specified, each update of the metrics is also passed to the TensorboardWriter
    """

    def __init__(self, *keys, writer=None, multi_label_training=True):
        """
        Initializes the internal dataframes
        :param keys: labels of the classes that exist in the data
        :param writer: a SummaryWriter or TensorboardWriter
        """
        self.writer = writer
        self.multi_label_training = multi_label_training
        # Create a list of dataframes, one per class
        self._class_wise_cms = [pd.DataFrame([[0, 0], [0, 0]]).astype('int64').rename_axis(key, axis=1) for key in keys]

        if not self.multi_label_training:
            # Create a dataframe containing the classification results (rows: Ground Truth, cols: Predictions)
            self._cm = pd.DataFrame(0, index=keys, columns=keys).astype('int64')
        else:
            # Normal confusion matrix not defined for multi-label-classification
            self._cm = None

        self.reset()

    @property
    def cm(self):
        return self._cm

    @property
    def class_wise_cms(self):
        return self._class_wise_cms

    def reset(self):
        if self._cm is not None:
            for col in self._cm.columns:
                self._cm[col].values[:] = 0
        for df in self._class_wise_cms:
            for col in df.columns:
                df[col].values[:] = 0

    def update_cm(self, upd_cm):
        """
        Updates the internal data frames with the given values
        Should never be called in the ML case
        Parameters
        ----------
        :param upd_cm: Dataframe whose columns and indices should match those of self._cm
        """
        # Assert the method is only called for the single-label case
        assert not self.multi_label_training, \
            "In the multilabel case, an overall confusion matrix can not be determined"

        # Assert the df indices and columns match
        assert all(upd_cm.columns == self._cm.columns), \
            "The two given confusion matrices have different columns"
        pd.testing.assert_index_equal(upd_cm.index, self._cm.index)
        # Update the overall confusion matrix by adding the entries
        self._cm = self._cm.add(upd_cm)

        # # Plot the global confusion matrix and send it to the writer
        # if self.writer is not None:
        #     plt.figure(figsize=(10, 7))
        #     plt.title("Overall confusion matrix")
        #     fig_ = sns.heatmap(self._cm, annot=True, cmap='Spectral').get_figure()
        #     # plt.show()
        #     plt.close(fig_)  # close the curren figure
        #
        #     # Params of SummaryWriter.add_image:
        #     #   tag (string): Data identifier
        #     #   img_tensor (torch.Tensor, numpy.array, or string/blobname): Image data
        #     #   global_step (int): Global step value to record
        #     self.writer.add_figure("Overall confusion matrix", fig_)

    def update_class_wise_cms(self, upd_class_wise_cms):
        """
        Updates the internal data frames with the given values

        Parameters
        ----------
        :param upd_class_wise_cms: List of dataframes, one per class,
                the class which is represented in each dataframe should be contained as name of axis 1 (columns)
        """

        # Assert that the number of class_wise cms is correct and the labels are the same
        assert len(upd_class_wise_cms) == len(self._class_wise_cms), \
            "The number of class_wise confusion matrices doesn't match"
        for idx in range(0, len(upd_class_wise_cms)):
            assert upd_class_wise_cms[idx].columns.name == self._class_wise_cms[idx].columns.name
            self._class_wise_cms[idx] = self._class_wise_cms[idx].add(upd_class_wise_cms[idx])

        # for idx in range(len(self._classwise_cms)):
        #     class_cm = self._classwise_cms[idx]
        #     plt.figure(figsize=(10, 7))
        #     plt.title("Confusion matrix for class " + str(class_cm.columns.name))
        #     fig_ = sns.heatmap(class_cm, annot=True, cmap='Spectral').get_figure()
        #     # plt.show()
        #     plt.close(fig_)  # close the curren figure
        #     self.writer.add_figure("Confusion matrix for class " + str(class_cm.columns.name), fig_)

    def send_cms_to_writer(self, epoch):
        """
            Sends the internal confusion matrices to the SummaryWriter/TensorboardWriter
            and should be called at the end of each training/validation epoch
            (not for each batch separately as the results look confusing then)
        """

        if self.writer is not None:
            if not self.multi_label_training:
                # Plot the global confusion matrix and send it to the writer
                plt.figure(figsize=(10, 7))
                plt.title("Overall confusion matrix")
                fig_ = sns.heatmap(self._cm, annot=True, cmap='Spectral').get_figure()
                # plt.show()
                plt.close(fig_)  # close the curren figure

                # Params of SummaryWriter.add_image:
                #   tag (string): Data identifier
                #   img_tensor (torch.Tensor, numpy.array, or string/blobname): Image data
                #   global_step (int): Global step value to record
                self.writer.add_figure("Overall confusion matrix", fig_, global_step=epoch)

            # Plot the class-wise confusion matrices and send them to the writer
            for idx in range(len(self._class_wise_cms)):
                class_cm = self._class_wise_cms[idx]
                plt.figure(figsize=(10, 7))
                plt.title("Confusion matrix for class " + str(class_cm.columns.name))
                fig_ = sns.heatmap(class_cm, annot=True, cmap='Spectral').get_figure()
                # plt.show()
                plt.close(fig_)  # close the curren figure
                self.writer.add_figure("Confusion matrix for class " + str(class_cm.columns.name),
                                       fig_, global_step=epoch)


if __name__ == '__main__':
    # Single Label
    output = [0, 2, 5, 6]
    target = [1, 8, 7, 6]
    cm = overall_confusion_matrix(output, target, False, [0, 1, 2, 3, 4, 5, 6, 7, 8])
    class_wise_cms = class_wise_confusion_matrices_single_label(output, target, False, [0, 1, 2, 3, 4, 5, 6, 7, 8])
    tracker = ConfusionMatrixTracker(*[0, 1, 2, 3, 4, 5, 6, 7, 8], writer=SummaryWriter(log_dir="saved/tmp"),
                                     multi_label_training=False)
    tracker.update_cm(cm)
    tracker.update_class_wise_cms(class_wise_cms)

    # Multi Label
    output = [[0,0,0,1,0,0,1,0,0], [1,0,0,0,0,0,0,0,0], [0,0,0,0,1,0,0,0,0]]
    target = [[1,0,0,1,0,0,1,0,0], [1,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,1,0]]
    class_wise_cms = class_wise_confusion_matrices_multi_label(output, target, False, [0, 1, 2, 3, 4, 5, 6, 7, 8])
    tracker = ConfusionMatrixTracker(*[0, 1, 2, 3, 4, 5, 6, 7, 8], writer=SummaryWriter(log_dir="saved/tmp"),
                                     multi_label_training=True)
    tracker.update_class_wise_cms(class_wise_cms)
