import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix, \
    accuracy_score, top_k_accuracy_score, roc_auc_score, f1_score, balanced_accuracy_score, classification_report


from torchmetrics import Precision, Accuracy, Recall, ROC, F1Score
from torchmetrics.classification.auroc import AUROC

# ----------------------------------- SKlearn Metric -----------------------------------------------
# For details, see https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
# Included metrics: F1, ROC AUC, Accuracy, Balanced Accuracy, Top-K Accuracy, Confusion Matrix, Classification Report
# Short summary for the multi-class case:
# @param output: dimension=(N,C) => the log-probabilities of each class or logits (depending on logits param)
# @param target: dimension= (N) => A class index in the range [0, C-1] as integer (ground truth) per sample
# @param logits: If set to True, the vectors are expected to contain logits/raw scores,
#                otherwise the vectors are expected to contain log-probabilities of each class
#                This is achieved by adding a LogSoftmax layer in the last layer of the network
# @param labels: The set of labels to include when average != 'binary', and their order if average is None.
# @param average: Should be None, 'micro', 'macro', 'weighted', or 'samples'


def _convert_log_probs_to_prediction(log_prob_output):
    return torch.argmax(log_prob_output, dim=1)


def _convert_logits_to_prediction(logits):
    # Convert the logits to probabilities and take the one with the highest one as final prediction
    # We are in the single-label case, so apply Softmax first and then return the maximum value
    softmax_probs = torch.nn.functional.softmax(logits, dim=1)
    # Should be the same as directly taking the maximum of raw logits (if x1<x2, then softmax(x1)<softmax(x2))
    assert (torch.argmax(softmax_probs, dim=1) == torch.argmax(logits, dim=1)).all()
    return torch.argmax(softmax_probs, dim=1)


def _sk_f1(output, target, logits, labels, average):
    """
    Compute the F1 score
    """
    with torch.no_grad():
        if not logits:
            pred = _convert_log_probs_to_prediction(output)
        else:
            pred = _convert_logits_to_prediction(output)
        assert pred.shape[0] == len(target)
        return f1_score(y_true=target, y_pred=pred, labels=labels, average=average)


def _sk_roc_auc(output, target, logits, labels, average, multi_class_cfg):
    """
    Compute the ROC AUC score
    @param multi_class_cfg: 'ovr' (one-vs-rest) or 'ovo' (one-vs-one)
    """
    with torch.no_grad():
        # From the docs:
        # In the multiclass case, y_score corresponds to an array of shape (n_samples, n_classes) of probability
        # estimates. The probability estimates must sum to 1 across the possible classes

        # Here, in both cases, the scores do not sum to one
        # Either they are logSoftmax outputs or logits, so transform them
        if logits:
            # Apply softmax on the logits
            probs = torch.nn.functional.softmax(output, dim=1)
        else:
            # Transform log(softmax(x)) to become softmax(x)
            probs = torch.exp(output)

        assert probs.shape[0] == len(target)
        return roc_auc_score(y_true=target, y_score=probs, labels=labels, average=average,
                             multi_class=multi_class_cfg)


def sk_accuracy(output, target, logits):
    """
    Calculates the (TOP-1) accuracy for the multiclass case
    """
    with torch.no_grad():
        if not logits:
            pred = _convert_log_probs_to_prediction(output)
        else:
            pred = _convert_logits_to_prediction(output)
        assert pred.shape[0] == len(target)
        return accuracy_score(y_true=target, y_pred=pred)


def sk_balanced_accuracy(output, target, logits):
    """
    Compute the balanced accuracy.

    The balanced accuracy in binary and multiclass classification problems to deal with imbalanced datasets.
    It is defined as the average of recall obtained on each class.
    This implementation is equivalent to accuracy_score with class-balanced sample weights
    """
    with torch.no_grad():
        if not logits:
            pred = _convert_log_probs_to_prediction(output)
        else:
            pred = _convert_logits_to_prediction(output)
        assert pred.shape[0] == len(target)
        return balanced_accuracy_score(y_true=target, y_pred=pred)


def sk_top_k_acc(output, target, labels, k=3):
    """
    Calculates the TOP-k accuracy for the multiclass case
    A prediction is considered correct when the true label is associated with one of the k highest predicted scores
    :@param k: Number of most likely outcomes considered to find the correct label
    """
    with torch.no_grad():
        return top_k_accuracy_score(y_true=target, y_score=output, k=k, labels=labels)


def micro_sk_f1(output, target, logits, labels):
    """See documentation for _f1
    """
    return _sk_f1(output, target, logits, labels, "micro")


def macro_sk_f1(output, target, logits, labels):
    """See documentation for _f1
    """
    return _sk_f1(output, target, logits, labels, "macro")


def weighted_sk_f1(output, target, logits, labels):
    """See documentation for _f1
    """
    return _sk_f1(output, target, logits, labels, "weighted")


def class_wise_sk_f1(output, target, logits, labels):
    """See documentation for _f1 """
    return _sk_f1(output, target, logits, labels, None)


def macro_sk_roc_auc_ovo(output, target, logits, labels):
    """See documentation for _roc_auc """
    return _sk_roc_auc(output, target, logits, labels, "macro", "ovo")


def weighted_sk_roc_auc_ovo(output, target, logits, labels):
    """See documentation for _roc_auc """
    return _sk_roc_auc(output, target, logits, labels, "weighted", "ovo")


def macro_sk_roc_auc_ovr(output, target, logits, labels):
    """See documentation for _roc_auc """
    return _sk_roc_auc(output, target, logits, labels, "macro", "ovr")


def weighted_sk_roc_auc_ovr(output, target, logits, labels):
    """See documentation for _roc_auc """
    return _sk_roc_auc(output, target, logits, labels, "weighted", "ovr")


def overall_confusion_matrix_sk(output, target, logits, labels):
    """
        Creates a num_labels x num_labels sized confusion matrix whose i-th row and j-th column entry indicates
        the number of samples with true label being i-th class and predicted label being j-th class
        :@return: Dataframe of size (num_labels x num_labels) representing the confusion matrix
    """
    with torch.no_grad():
        if not logits:
            pred = _convert_log_probs_to_prediction(output)
        else:
            pred = _convert_logits_to_prediction(output)
        assert pred.shape[0] == len(target)
        cm = confusion_matrix(y_true=target, y_pred=pred, labels=labels)
        df_cm = pd.DataFrame(cm, index=labels, columns=labels)
        return df_cm


def class_wise_confusion_matrices_single_label_sk(output, target, logits, labels):
    """
    Creates a 2x2 confusion matrix per class contained in labels
    CM(0,0) -> TN, CM(1,0) -> FN, CM(0,1) -> FP, CM(1,1) -> TP
    The name of axis 1 is set to the respective label
    :@return: List of dataframes

    Note: Multiclass data will be treated as if binarized under a one-vs-rest transformation
    """
    with torch.no_grad():
        if not logits:
            pred = _convert_log_probs_to_prediction(output)
        else:
            pred = _convert_logits_to_prediction(output)
        assert pred.shape[0] == len(target)
        class_wise_cms = multilabel_confusion_matrix(y_true=target, y_pred=pred, labels=labels)

        df_class_wise_cms = [pd.DataFrame(class_wise_cms[idx]).astype('int64').rename_axis(labels[idx], axis=1)
                             for idx in range(0, len(class_wise_cms))]
        return df_class_wise_cms


def sk_classification_summary(output, target, logits, labels, output_dict,
                              target_names=["IAVB", "AF", "LBBB", "PAC", "RBBB", "SNR", "STD", "STE", "VEB"]):
    """
        Compute the classification report
        @param output_dict: If True, return output as dict.
        @param target_names: Optional display names matching the labels (same order).
    """
    with torch.no_grad():
        if not logits:
            pred = _convert_log_probs_to_prediction(output)
        else:
            pred = _convert_logits_to_prediction(output)
        assert pred.shape[0] == len(target)
        return classification_report(y_true=target, y_pred=pred, labels=labels, digits=3, target_names=target_names,
                                     output_dict=output_dict)


# ----------------------------------- TORCHMETRICS -----------------------------------------------
# For details, see https://lightning.ai/docs/torchmetrics/stable/
# Included metrics: Precision, Recall, ROC AUC, ROC, F1, Accuracy
# Short summary for the multi-class case:
# @param output: dimension=(N,C) => A float tensor containing the log-probabilities of each class or logits
# @param target: dimension= (N) => A int tensor with the class index in the range [0, C-1] (ground truth) per sample
# @param logits: If set to True, the vectors are expected to contain logits/raw scores,
#                otherwise the vectors are expected to contain log-probabilities of each class
#                This is achieved by adding a LogSoftmax layer in the last layer of the network
# @param labels:
# @param average: Should be None, 'micro', 'macro', or 'weighted' (NOT 'samples')
#
# NOTE REGARDING TORCHMETRICS:
# "If preds is a floating point we apply torch.argmax along the C dimension to automatically convert probabilities/
#  logits into an int tensor." => Here, we do the conversion before and pass int tensors as prediction to TorchMetrics

def _torch_precision(output, target, logits, labels, average):
    """
        Compute the precision
    """
    with torch.no_grad():
        assert average in ['macro', 'micro', 'weighted', None], \
            "Average must be one of 'macro', 'micro', 'weighted', None"

        if not logits:
            pred = _convert_log_probs_to_prediction(output)
        else:
            pred = _convert_logits_to_prediction(output)

        assert pred.shape[0] == len(target)
        precision = Precision(task='multiclass', num_classes=len(labels), average=average)
        return precision(pred, target)


def _torch_recall(output, target, logits, labels, average):
    """
        Compute the recall
    """
    with torch.no_grad():
        assert average in ['macro', 'micro', 'weighted', None], \
            "Average must be one of 'macro', 'micro', 'weighted', None"

        if not logits:
            pred = _convert_log_probs_to_prediction(output)
        else:
            pred = _convert_logits_to_prediction(output)

        assert pred.shape[0] == len(target)
        recall = Recall(task='multiclass', num_classes=len(labels), average=average)
        return recall(pred, target)


def _torch_roc_auc(output, target, logits, labels, average):
    """
    Compute the Area Under the Receiver Operating Characteristic Curve (ROC AUC) for the multiclass case
    For multiclass the metric is calculated by iteratively treating each class as the positive class and all other
    classes as the negative, which is referred to as the one-vs-rest approach. One-vs-one is currently not supported.
    """
    with torch.no_grad():
        assert average in ['macro', 'micro', 'weighted', None], \
            "Average must be one of 'macro', 'micro', 'weighted', None"

        # Preds should be a float tensor of shape (N, C, ...) containing probabilities or logits for each observation.
        # Here, we have either log softmax outputs or logits (and we transform both to normal probabilities)
        if logits:
            probs = torch.nn.functional.softmax(output, dim=1)
        else:
            probs = torch.exp(output)

        assert probs.shape[0] == len(target)
        auroc = AUROC(task='multiclass', num_classes=len(labels), average=average)
        return auroc(probs, target)


def torch_roc(output, target, logits, labels):
    """
    Compute the Receiver Operating Characteristic Curve (ROC) for the multiclass case
    """
    with torch.no_grad():

        # Preds should be a float tensor of shape (N, C, ...) containing probabilities or logits for each observation.
        # Here, we have either log softmax outputs or logits (and we transform both to normal probabilities)
        if logits:
            probs = torch.nn.functional.softmax(output, dim=1)
        else:
            probs = torch.exp(output)

        assert probs.shape[0] == len(target)
        # returns a tuple (fpr, tpr, thresholds)
        auroc = ROC(task='multiclass', num_classes=len(labels))
        return auroc(probs, target)



def _torch_f1(output, target, logits, labels, average):
    """
        Compute F-1 score.
    """
    with torch.no_grad():
        assert average in ['macro', 'micro', 'weighted', None], \
            "Average must be one of 'macro', 'micro', 'weighted', None"

        if not logits:
            pred = _convert_log_probs_to_prediction(output)
        else:
            pred = _convert_logits_to_prediction(output)

        assert pred.shape[0] == len(target)

        f1 = F1Score(task='multiclass', num_classes=len(labels), average=average)
        return f1(pred, target)


def _torch_accuracy(output, target, logits, labels, average):
    """
        Compute the accuracy
    """
    with torch.no_grad():
        assert average in ['macro', 'micro', 'weighted', None], \
            "Average must be one of 'macro', 'micro', 'weighted', None"

        if not logits:
            pred = _convert_log_probs_to_prediction(output)
        else:
            pred = _convert_logits_to_prediction(output)

        assert pred.shape[0] == len(target)

        accuracy = Accuracy(task='multiclass', num_classes=len(labels), average=average)
        return accuracy(pred, target)


def class_wise_torch_accuracy(output, target, logits, labels):
    """See documentation for _torch_accuracy """
    return _torch_accuracy(output, target, logits, labels, average=None)


def weighted_torch_accuracy(output, target, logits, labels):
    """See documentation for _torch_accuracy """
    return _torch_accuracy(output, target, logits, labels, average="weighted")


def macro_torch_accuracy(output, target, logits, labels):
    """See documentation for _torch_accuracy """
    return _torch_accuracy(output, target, logits, labels, average="macro")


def micro_torch_accuracy(output, target, logits, labels):
    """See documentation for _torch_accuracy """
    return _torch_accuracy(output, target, logits, labels, average="micro")


def class_wise_torch_f1(output, target, logits, labels):
    """See documentation for _torch_f1 """
    return _torch_f1(output, target, logits, labels, average=None)


def weighted_torch_f1(output, target, logits, labels):
    """See documentation for _torch_f1 """
    return _torch_f1(output, target, logits, labels, average="weighted")


def macro_torch_f1(output, target, logits, labels):
    """See documentation for _torch_f1 """
    return _torch_f1(output, target, logits, labels, average="macro")


def class_wise_torch_roc_auc(output, target, logits, labels):
    """See documentation for _torch_auc """
    return _torch_roc_auc(output, target, logits, labels, average=None)


def weighted_torch_roc_auc(output, target, logits, labels):
    """See documentation for _torch_auc """
    # Seems to be consistent to sklearn's 'weighted_roc_auc_ovr', may differ from 'weighted_roc_auc_ovo',
    return _torch_roc_auc(output, target, logits, labels, average="weighted")


def macro_torch_roc_auc(output, target, logits, labels):
    """See documentation for _torch_auc """
    # Seems to be consistent to sklearn's 'weighted_roc_auc_ovr', may differ from 'weighted_roc_auc_ovo',
    return _torch_roc_auc(output, target, logits, labels, average="macro")


def class_wise_torch_precision(output, target, logits, labels):
    """See documentation for _torch_auc """
    return _torch_precision(output, target, logits, labels, average=None)


def weighted_torch_precision(output, target, logits, labels):
    """See documentation for _torch_precision """
    return _torch_precision(output, target, logits, labels, average='weighted')


def macro_torch_precision(output, target, logits, labels):
    """See documentation for _torch_precision """
    return _torch_precision(output, target, logits, labels, average='macro')


def class_wise_torch_recall(output, target, logits, labels):
    """See documentation for _torch_recall """
    return _torch_recall(output, target, logits, labels, average=None)


def weighted_torch_recall(output, target, logits, labels):
    """See documentation for _torch_recall """
    return _torch_recall(output, target, logits, labels, average='weighted')


def macro_torch_recall(output, target, logits, labels):
    """See documentation for _torch_recall """
    return _torch_recall(output, target, logits, labels, average='macro')



