import os.path
from math import sqrt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

smooth = 0  # 1e-6


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
        self._epoch = 0
        self._data_epoch = pd.DataFrame(index=keys_epoch + all_keys_epoch_class_wise, columns=['mean'],
                                        dtype=np.float64)
        # Set the print option of pandas to at least twice the num of metrics, since they may later be printed together
        # (one time for train, one time for valid)
        pd.set_option('display.max_rows', 100)
        self.reset()

    @property
    def epoch(self):
        return self._epoch

    @epoch.setter
    def epoch(self, value):
        assert value > self._epoch  # The epoch should increase with time
        self._epoch = value

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

    def epoch_update(self, key, value):
        if self.writer is not None:
            self.writer.add_scalar(key, value, global_step=self._epoch)
        self._data_epoch.at[key, 'mean'] = value.item()

    # class_wise_epoch_update(met.__name__, met(target=targets, output=outputs, **additional_kwargs), epoch)
    def class_wise_epoch_update(self, key, values):
        for idx_class in range(0, len(values)):
            full_key_name = key + '_class_' + str(self._labels[idx_class])
            if self.writer is not None:
                self.writer.add_scalar(full_key_name, values[idx_class], global_step=self._epoch)
            self._data_epoch.at[full_key_name, 'mean'] = values[idx_class].item()

    def current(self):
        return dict(self._data_iter['current'])

    def avg(self):
        for key, row in self._data_iter.iterrows():
            self._data_iter.at[key, 'mean'] = row['sum'] / row['counts']
            self._data_iter.at[key, 'square_avg'] = row['square_sum'] / row['counts']

    def std(self):
        for key, row in self._data_iter.iterrows():
            self._data_iter.at[key, 'std'] = sqrt(row['square_avg'] - row['mean'] ** 2 + smooth)

    def result(self, include_epoch_metrics):
        self.avg()
        self.std()
        iter_result = self._data_iter[['mean', 'std']]
        # Send the mean epoch values for the iteration metrics to the tensorboard writer as well
        if self.writer is not None:
            for key, row in iter_result.iterrows():
                self.writer.add_scalar('epoch_' + key, iter_result.at[key, 'mean'], global_step=self._epoch)
        return pd.concat([iter_result, self._data_epoch]) if include_epoch_metrics else iter_result


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
                fig_overall_cm, ax = plt.subplots(figsize=(10, 7))
                ax.set_title("Overall confusion matrix")
                sns.heatmap(self._cm, annot=True, cmap='Spectral', ax=ax)
                # Params of SummaryWriter.add_image:
                #   tag (string): Data identifier
                #   img_tensor (torch.Tensor, numpy.array, or string/blobname): Image data
                #   global_step (int): Global step value to record
                self.writer.add_figure("Overall confusion matrix", fig_overall_cm, global_step=epoch)
                fig_overall_cm.clear()
                plt.close(fig_overall_cm)  # close the current figure

            # Also create a one-containing-all-single-cms-figure
            fig_all_cms, axs = plt.subplots(3, 3, figsize=(15, 8))
            for idx, ax in enumerate(axs.flatten()):
                # Plot the class-wise confusion matrices and send them to the writer one-by-one
                self._plot_cm_for_single_class(idx, ax, epoch)
            fig_all_cms.tight_layout()
            self.writer.add_figure("Overview single confusion matrices", fig_all_cms.get_figure(), global_step=epoch)
            fig_all_cms.clear()
            plt.close(fig_all_cms)

    def save_result_cms_to_file(self, path):
        """
            Writes the confusion matrices to the provided file
            and should be called at the end of testing
        """
        if not self.multi_label_training:
            # Plot the global confusion matrix and send it to the file
            fig_overall_cm, ax = plt.subplots(figsize=(10, 7))
            ax.set_title("Overall confusion matrix")
            sns.heatmap(self._cm, annot=True, cmap='Spectral', ax=ax)
            fig_overall_cm.savefig(os.path.join(path, 'overall_cm.pdf'))
            fig_overall_cm.clear()
            plt.close(fig_overall_cm)  # close the current figure

        # Also create a one-containing-all-single-cms-figure
        fig_all_cms, axs = plt.subplots(3, 3, figsize=(15, 8))
        for idx, ax in enumerate(axs.flatten()):
            # Plot the class-wise confusion matrices one-by-one
            self._plot_cm_for_single_class(idx, ax)
        fig_all_cms.tight_layout()
        fig_all_cms.savefig(os.path.join(path, 'class_wise_cms.pdf'))
        fig_all_cms.clear()
        plt.close(fig_all_cms)

    def _plot_cm_for_single_class(self, idx, ax, epoch=None):
        class_cm = self._class_wise_cms[idx]

        heatmap = sns.heatmap(class_cm, annot=True, cmap='Spectral', ax=ax)
        heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=12)
        heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=12)
        ax.set_ylabel('True label')
        ax.set_xlabel('Predicted label')
        ax.set_title("Confusion Matrix for class " + str(class_cm.columns.name))

        # # Recreate the cm-related part to another figure to add it directly to the TensorBoardWriter
        # plt.figure(figsize=(10, 7))
        # plt.title("Confusion matrix for class " + str(class_cm.columns.name))
        # single_cm_fig = sns.heatmap(class_cm, annot=True, cmap='Spectral').get_figure()
        # # plt.show()
        # plt.close(single_cm_fig)  # close the current figure
        # self.writer.add_figure("Confusion matrix for class " + str(class_cm.columns.name),
        #                        single_cm_fig, global_step=epoch)


