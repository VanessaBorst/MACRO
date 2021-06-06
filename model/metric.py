import numpy as np
import torch
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix, \
    accuracy_score, top_k_accuracy_score, roc_auc_score, f1_score, balanced_accuracy_score
import pandas as pd


# This file contains multiclass classification metrics (currently not adapted for multi-label classification!)
# Macro metrics: Macro-level metrics gives equal weight to each class
#       => Each class has the same weight in the average
#       => There is no distinction between highly and poorly populated classes
# Micro metrics: Micro-level metrics weight all items equally.
#       => Considers all the units together, without taking into consideration possible differences between classes
#       => Classes with more observations will have more influence in the metric
#       => Biases the classes towards the most populated class
#       => Micro-Average Precision and Recall are the same values, therefore the MicroAverage F1-Score is also the same
#           (and corresponds to the accuracy when all classes are considered)


def _convert_logprob_to_prediction(logprob_output):
    return torch.argmax(logprob_output, dim=1)


def _f1(output, log_probs, target, labels, average):
    """
    Compute the F1 score, also known as balanced F-score or F-measure.
    In the multi-class and multi-label case, this is the average of the F1 score of each class with
    weighting depending on the average parameter.

    The following parameter description applies for the multiclass case
    :param output: output: dimension=(N,C) or (N);
        Per entry, the log-probabilities of each class should be contained when log_probs is set to true
        Otherwise, a list of class indices in the range [0, C-1] should be passed for each of the N samples
    :param log_probs: If the outputs are log probs, set param to True
    :param target: dimension= (N)
        Per entry, a class index in the range [0, C-1] as integer (ground truth)
    :param labels: The set of labels to include when average != 'binary', and their order if average is None.
    :param average: Determines the type of averaging performed on the data (if not None).
        Parameter values useful for this application:
        None: The scores for each class are returned
        'micro': Calculate metrics globally by counting the total true positives, false negatives and false positives.
        'macro': Calculate metrics for each label, and find their unweighted mean.
                    This does not take label imbalance into account.
        'weighted': Calculate metrics for each label, and find their average weighted by support
        (the number of true instances for each label).
        This alters ‘macro’ to account for label imbalance; can result in F-score that is not between precision & recall

    :return: The F1 score, also known as balanced F-score or F-measure.

    Note: Could be extended to multilabel classification
    (see https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html)
    """
    with torch.no_grad():
        pred = _convert_logprob_to_prediction(output) if log_probs else output
        return f1_score(y_true=target, y_pred=pred, labels=labels, average=average)


def _roc_auc(output, log_probs, target, labels, average, multi_class_cfg, ):
    """
    The following parameter description applies for the multiclass case
    :param output: dimension=(N,C)
        Per entry, the (log) probability estimates of each class should be contained and
        Later they MUST sum to 1 -> if log probs are provided, set log_prob param to True
    :param log_probs: If the outputs are log probs and do NOT necessarily sum to 1, set param to True
    :param target: dimension= (N)
        Per entry, a class index in the range [0, C-1] as integer (ground truth)
    :param labels: Needed for multiclass targets. List of labels that index the classes in output
    :param average: In the multiclass-case, either ‘macro’ or ‘weighted’ or None
        None:       The scores for each class are returned
        ‘macro’:    Calculate metrics for each label, and find their unweighted mean.
                    This does not take label imbalance into account.
        ‘weighted’: Calculate metrics for each label, and find their average, weighted by support
                    (the number of true instances for each label).
    :param multi_class_cfg: Should be specified
        'ovr': One-vs-rest. Computes the AUC of each class against the rest, sensitive to class imbalance even
            when average == 'macro', because class imbalance affects the composition of each of the ‘rest’ groupings.
        'ovo': One-vs-one. Computes the average AUC of all possible pairwise combinations of classes [5].
            Insensitive to class imbalance when average == 'macro'.
    :return: Area Under the Receiver Operating Characteristic Curve (ROC AUC) for the multiclass case

    Note: Could be extended to binary and multilabel classification, but some restrictions apply
    (see https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html)
    """
    with torch.no_grad():
        # If log_probs are provided, convert them to real probabilities
        pred = output if not log_probs else np.exp(output)
        return roc_auc_score(y_true=target, y_score=pred, labels=labels, average=average, multi_class=multi_class_cfg)


def accuracy(output, log_probs, target):
    """
    Calculates the (TOP-1) accuracy for the multiclass case

    Parameters
    ----------
    :param output: dimension=(N,C) or (N);
        Per entry, the log-probabilities of each class should be contained when log_probs is set to true, otherwise
        a list of class indices in the range [0, C-1] should be passed for each of the N samples
        (obtaining log-probabilities is achieved by adding a LogSoftmax layer in the last layer of the network)
    :param log_probs: If set to True, the output should have dimension (N,C), otherwise dimension (N)
    :param target: dimension= (N)
        Per entry, a class index in the range [0, C-1] as integer (ground truth)

    Note: Could be extended to the multilabel-indicator-case
    (see https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html)
    """
    with torch.no_grad():
        pred = _convert_logprob_to_prediction(output) if log_probs else output
        assert pred.shape[0] == len(target)
        return accuracy_score(y_true=target, y_pred=pred)


def balanced_accuracy(output, log_probs, target):
    """
    Compute the balanced accuracy.

    The balanced accuracy in binary and multiclass classification problems to deal with imbalanced datasets.
    It is defined as the average of recall obtained on each class.
    This implementation is equivalent to accuracy_score with class-balanced sample weights

    Parameters
    ----------
    :param output: dimension=(N,C) or (N);
        Per entry, the log-probabilities of each class should be contained when log_probs is set to true, otherwise
        a list of class indices in the range [0, C-1] should be passed for each of the N samples
        (obtaining log-probabilities is achieved by adding a LogSoftmax layer in the last layer of the network)
    :param log_probs: If set to True, the output should have dimension (N,C), otherwise dimension (N)
    :param target: dimension= (N)
        Per entry, a class index in the range [0, C-1] as integer (ground truth)
    """
    with torch.no_grad():
        pred = _convert_logprob_to_prediction(output) if log_probs else output
        assert pred.shape[0] == len(target)
        return balanced_accuracy_score(y_true=target, y_pred=pred)


def top_k_acc(output, target, labels, k=3):
    """
    Calculates the TOP-k accuracy for the multiclass case
    A prediction is considered correct when the true label is associated with one of the k highest predicted scores

    Parameters
     ----------
    :param output: dimension=(minibatch,C)=(n_samples, n_classes)
        Predicted scores. These can be either probability estimates or non-thresholded decision values
        Example: Per entry, the log-probabilities of each class is contained
    :param target: dimension= (N)
        Per entry, a class index in the range [0, C-1] as integer
    :param labels: List of labels that index the classes in output -> should always be provided
    :param k: Number of most likely outcomes considered to find the correct label
    """
    with torch.no_grad():
        return top_k_accuracy_score(y_true=target, y_score=output, k=k, labels=labels)


def mirco_f1(output, log_probs, target, labels):
    """See documentation for _f1 """
    return _f1(output, log_probs, target, labels, "micro")


def macro_f1(output, log_probs, target, labels):
    """See documentation for _f1 """
    return _f1(output, log_probs, target, labels, "macro")


def weighted_f1(output, log_probs, target, labels):
    """See documentation for _f1 """
    return _f1(output, log_probs, target, labels, "weighted")


def macro_roc_auc_ovo(output, log_probs, target, labels):
    """See documentation for _roc_auc """
    return _roc_auc(output, log_probs, target, labels, "macro", "ovo")


def weighted_roc_auc_ovo(output, log_probs, target, labels):
    """See documentation for _roc_auc """
    return _roc_auc(output, log_probs, target, labels, "weighted", "ovo")


def macro_roc_auc_ovr(output, log_probs, target, labels):
    """See documentation for _roc_auc """
    return _roc_auc(output, log_probs, target, labels, "macro", "ovr")


def weighted_roc_auc_ovr(output, log_probs, target, labels):
    """See documentation for _roc_auc """
    return _roc_auc(output, log_probs, target, labels, "weighted", "ovr")


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

    Note: Could be extended to the multilabel-indicator-case
    (see https://scikit-learn.org/stable/modules/generated/sklearn.metrics.multilabel_confusion_matrix.html)
    """
    class_wise_cms = multilabel_confusion_matrix(y_true=target, y_pred=output, labels=labels)

    df_class_wise_cms = [pd.DataFrame(class_wise_cms[idx]).astype('int64').rename_axis(labels[idx], axis=1)
                         for idx in range(0, len(class_wise_cms))]
    return df_class_wise_cms
