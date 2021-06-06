import torch
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix, accuracy_score, top_k_accuracy_score
import pandas as pd


def _convert_logprob_to_prediction(logprob_output):
    return torch.argmax(logprob_output, dim=1)


def accuracy(output, target, log_probs=True):
    """
     Parameters
     ----------
     :param output: dimension=(minibatch,C);
        Per entry, the log-probabilities of each class should be contained when log_probs is set to true, otherwise
        a list of class indices in the range [0, C-1] should be passed
        (obtaining log-probabilities is achieved by adding a LogSoftmax layer in the last layer of the network)
     :param target: dimension= (N)
        Per entry, a class index in the range [0, C-1] as integer
    """
    with torch.no_grad():
        pred = _convert_logprob_to_prediction(output) if log_probs else output
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()  # Counts number of equal entries + converts the tensor to an number
        # Alternative:  accuracy_score(y_true=target, y_pred=_convert_logprob_to_prediction(output))
    return correct / len(target)


def top_k_acc(output, target, labels, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
        test = top_k_accuracy_score(y_true=target, y_score=output, k=k, labels=labels)
    return correct / len(target)


def get_confusion_matrix(output, target, labels):
    """
        Creates a num_labels x num_labels sized confusion matrix whose i-th row and j-th column entry indicates
        the number of samples with true label being i-th class and predicted label being j-th class

        :param output: List of integers predicted by the network
        :param target: List of integers of the real labels (ground truth)
        :param labels: List of labels to index the matrix
        :return: Dataframes of size (num_labels x num_labels) representing the confusion matrix
    """
    cm = confusion_matrix(y_true=target, y_pred=output, labels=labels)
    df_cm = pd.DataFrame(cm, index=labels, columns=labels)
    return df_cm


def get_class_wise_confusion_matrix(output, target, labels):
    """
    Creates a 2x2 confusion matrix per class contained in labels
    CM(0,0) -> TN, CM(1,0) -> FN, CM(0,1) -> FP, CM(1,1) -> TP
    The name of axis 1 is set to the respective label
    :param output: List of integers predicted by the network
    :param target: List of integers of the real labels (ground truth)
    :param labels: List of integers representing all classes that can occur
    :return: List of dataframes

    Could be extended to the multilabel-indicator-case
    (see https://scikit-learn.org/stable/modules/generated/sklearn.metrics.multilabel_confusion_matrix.html)
    """
    class_wise_cms = multilabel_confusion_matrix(y_true=target, y_pred=output, labels=labels)

    df_class_wise_cms = [pd.DataFrame(class_wise_cms[idx]).astype('int64').rename_axis(labels[idx], axis=1)
                         for idx in range(0, len(class_wise_cms))]
    return df_class_wise_cms
