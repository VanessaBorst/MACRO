import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from torch.utils.tensorboard import SummaryWriter

from model.metric import get_confusion_matrix, get_class_wise_confusion_matrix


class MetricTracker:
    """
    Internally keeps track of the loss and the metrics by means of a dataframe
    If specified, each update of the metrics is also passed to the TensorboardWriter
    """
    def __init__(self, *keys, writer=None):
        self.writer = writer
        # Create a dataframe containing one row per key (e.g. one per metric and another one for the loss)
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)


class ClassificationTracker:
    """
        Internally keeps track of the confusion matrix and the class-wise confusion matrices by means dataframes
        If specified, each update of the metrics is also passed to the TensorboardWriter
    """

    def __init__(self, *keys, writer=None):
        self.writer = writer
        # Create a dataframe containing the classification results (rows: Ground Truth, cols: Predictions)
        self._cm = pd.DataFrame(0, index=keys, columns=keys).astype('int64')
        # Create a list of dataframes, one per class
        self._classwise_cms = [pd.DataFrame([[0, 0], [0, 0]]).astype('int64').rename_axis(key, axis=1) for key in keys]
        self.reset()

    def reset(self):
        for col in self._cm.columns:
            self._cm[col].values[:] = 0
        for df in self._classwise_cms:
            for col in df.columns:
                df[col].values[:] = 0

    def update_cms(self, upd_cm, upd_class_wise_cms):
        """
            Updates the internal data frames with the given values

             Parameters
        ----------
        :param upd_cm: Dataframe whose columns and indices should match those of self._cm
        :param upd_class_wise_cms: List of dataframes, one per class,
                the class which is represented in each dataframe should be contained as name of axis 1 (columns)
        """
        # Assert the df indices and columns match
        assert all(upd_cm.columns == self._cm.columns), \
            "The two given confusion matrices have different columns"
        pd.testing.assert_index_equal(upd_cm.index, self._cm.index)
        # Update the overall confusion matrix by adding the entries
        self._cm = self._cm.add(upd_cm)

        # Assert that the number of class_wise cms is correct and the labels are the same
        assert len(upd_class_wise_cms) == len(self._classwise_cms), \
            "The number of class_wise confusion matrices doesn't match"
        for idx in range(0, len(upd_class_wise_cms)):
            assert upd_class_wise_cms[idx].columns.name == self._classwise_cms[idx].columns.name
            self._classwise_cms[idx] = self._classwise_cms[idx].add(upd_class_wise_cms[idx])

        # Plot the global confusion matrix and send it to the writer
        if self.writer is not None:
            plt.figure(figsize=(10, 7))
            plt.title("Overall confusion matrix")
            fig_ = sns.heatmap(self._cm, annot=True, cmap='Spectral').get_figure()
            # plt.show()
            plt.close(fig_)  # close the curren figure

            # Params of SummaryWriter.add_image:
            #   tag (string): Data identifier
            #   img_tensor (torch.Tensor, numpy.array, or string/blobname): Image data
            #   global_step (int): Global step value to record
            self.writer.add_figure("Overall confusion matrix", fig_)

            for idx in range(len(self._classwise_cms)):
                class_cm = self._classwise_cms[idx]
                plt.figure(figsize=(10, 7))
                plt.title("Confusion matrix for class " + str(class_cm.columns.name))
                fig_ = sns.heatmap(class_cm, annot=True, cmap='Spectral').get_figure()
                # plt.show()
                plt.close(fig_)  # close the curren figure
                self.writer.add_figure("Confusion matrix for class " + str(class_cm.columns.name), fig_)


if __name__ == '__main__':
    output = [0,2,5,6]
    target = [1,8,7,6]
    cm = get_confusion_matrix(output,target,[0,1,2,3,4,5,6,7,8])
    class_wise_cms = get_class_wise_confusion_matrix(output,target,[0,1,2,3,4,5,6,7,8])
    tracker = ClassificationTracker(*[0,1,2,3,4,5,6,7,8], writer=SummaryWriter(log_dir="saved/tmp"))
    tracker.update_cms(cm, class_wise_cms)
