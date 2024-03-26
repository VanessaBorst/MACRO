import numpy as np
import pandas as pd
import torch
from sklearn.metrics import multilabel_confusion_matrix, \
    accuracy_score, roc_auc_score, f1_score, precision_score, \
    recall_score, classification_report

from torchmetrics import AUROC, Precision, Accuracy, Recall, ROC, F1Score

THRESHOLD = 0.5


# ----------------------------------- Sklearn Metric -----------------------------------------------
# For details, see https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
# Included metrics: F1, Precision, Recall, ROC AUC, Subset Accuracy, Confusion Matrix, Classification Report
# Short summary for the multi-label case:
# @param output: dimension=(N,C) => Label indicator array / sparse matrix with estimated targets of a classifier.
# @param target:  dimension= (N,C) => Label indicator array / sparse matrix containing the ground truth data
# @param logits: If set to True, the vectors are expected to contain logits/raw scores,
#                otherwise the vectors are expected to contain Sigmoid output probabilities
# @param labels: The set of labels to include when average != 'binary', and their order if average is None.
# @param average: Should be None, 'micro', 'macro', 'weighted', or 'samples'

def _convert_sigmoid_probs_to_prediction(sigmoid_probs, threshold=THRESHOLD):
    return torch.where(sigmoid_probs > threshold, 1, 0)


def _convert_logits_to_prediction(logits, threshold=THRESHOLD):
    # We are in the multi-label case, so apply Sigmoid first and then the threshold
    # Good post: https://web.stanford.edu/~nanbhas/blog/sigmoid-softmax/
    sigmoid_probs = torch.sigmoid(logits)
    return torch.where(sigmoid_probs > threshold, 1, 0)


def _sk_f1(output, target, logits, labels, average):
    """
    Compute the F1 score
    """
    with torch.no_grad():
        if not logits:
            pred = _convert_sigmoid_probs_to_prediction(output)
        else:
            pred = _convert_logits_to_prediction(output)
        assert pred.shape[0] == len(target)
        return f1_score(y_true=target, y_pred=pred, labels=labels, average=average)


def _sk_precision(output, target, logits, labels, average):
    """
    Compute the precision
    """
    with torch.no_grad():
        if not logits:
            pred = _convert_sigmoid_probs_to_prediction(output)
        else:
            pred = _convert_logits_to_prediction(output)
        assert pred.shape[0] == len(target)
        return precision_score(y_true=target, y_pred=pred, labels=labels, average=average)


def _sk_recall(output, target, logits, labels, average):
    """
    Compute the recall
    """
    with torch.no_grad():
        if not logits:
            pred = _convert_sigmoid_probs_to_prediction(output)
        else:
            pred = _convert_logits_to_prediction(output)
        assert pred.shape[0] == len(target)
        return recall_score(y_true=target, y_pred=pred, labels=labels, average=average)


def _sk_roc_auc(output, target, logits, labels, average):
    """
    Compute the ROC AUC score
    """
    with torch.no_grad():
        # Predictions should be passed as probabilities, not as one-hot-vector!
        pred = output if not logits else torch.sigmoid(output)

        assert pred.shape[0] == len(target)
        return roc_auc_score(y_true=target, y_score=pred, labels=labels, average=average)


def sk_subset_accuracy(output, target, logits):
    """
    Calculates the (TOP-1) accuracy for the multi-label  case
    For the multi-label case, this function computes subset accuracy:
    the set of labels predicted for a sample must exactly match the corresponding set of labels in y_true
    """
    with torch.no_grad():
        if not logits:
            pred = _convert_sigmoid_probs_to_prediction(output)
        else:
            pred = _convert_logits_to_prediction(output)
        assert pred.shape[0] == len(target)
        return accuracy_score(y_true=target, y_pred=pred)


def micro_sk_f1(output, target, logits, labels):
    """See documentation for _f1 """
    return _sk_f1(output, target, logits, labels, "micro")


def macro_sk_f1(output, target, logits, labels):
    """See documentation for _f1 """
    return _sk_f1(output, target, logits, labels, "macro")


def weighted_sk_f1(output, target, logits, labels):
    """See documentation for _f1 """
    return _sk_f1(output, target, logits, labels, "weighted")


def samples_sk_f1(output, target, logits, labels):
    """See documentation for _f1 """
    return _sk_f1(output, target, logits, labels, "samples")


def class_wise_sk_f1(output, target, logits, labels):
    """See documentation for _f1 """
    return _sk_f1(output, target, logits, labels, None)


def micro_sk_precision(output, target, logits, labels):
    """See documentation for _f1 """
    return _sk_precision(output, target, logits, labels, "micro")


def macro_sk_precision(output, target, logits, labels):
    """See documentation for _f1 """
    return _sk_precision(output, target, logits, labels, "macro")


def weighted_sk_precision(output, target, logits, labels):
    """See documentation for _f1 """
    return _sk_precision(output, target, logits, labels, "weighted")


def samples_sk_precision(output, target, logits, labels):
    """See documentation for _f1 """
    return _sk_precision(output, target, logits, labels, "samples")


def class_wise_sk_precision(output, target, logits, labels):
    """See documentation for _f1 """
    return _sk_precision(output, target, logits, labels, None)


def micro_sk_recall(output, target, logits, labels):
    """See documentation for _f1 """
    return _sk_recall(output, target, logits, labels, "micro")


def macro_sk_recall(output, target, logits, labels):
    """See documentation for _f1 """
    return _sk_recall(output, target, logits, labels, "macro")


def weighted_sk_recall(output, target, logits, labels):
    """See documentation for _f1 """
    return _sk_recall(output, target, logits, labels, "weighted")


def samples_sk_recall(output, target, logits, labels):
    """See documentation for _f1 """
    return _sk_recall(output, target, logits, labels, "samples")


def class_wise_sk_recall(output, target, logits, labels):
    """See documentation for _f1 """
    return _sk_recall(output, target, logits, labels, None)


def micro_sk_roc_auc(output, target, logits, labels):
    """See documentation for _roc_auc """
    return _sk_roc_auc(output, target, logits, labels, "micro")


def macro_sk_roc_auc(output, target, logits, labels):
    """See documentation for _roc_auc """
    return _sk_roc_auc(output, target, logits, labels, "macro")


def weighted_sk_roc_auc(output, target, logits, labels):
    """See documentation for _roc_auc """
    return _sk_roc_auc(output, target, logits, labels, "weighted")


def samples_sk_roc_auc(output, target, logits, labels):
    """See documentation for _roc_auc """
    return _sk_roc_auc(output, target, logits, labels, "samples")


def class_wise_sk_roc_auc(output, target, logits, labels):
    """See documentation for _roc_auc """
    return _sk_roc_auc(output, target, logits, labels, average=None)

def class_wise_confusion_matrices_multi_label_sk(output, target, logits, labels):
    """
    Compute class-wise (default) multilabel confusion matrix to evaluate the accuracy of a classification,
    and output confusion matrices for each class or sample.

    Creates a 2x2 confusion matrix per class contained in labels
    CM(0,0) -> TN, CM(1,0) -> FN, CM(0,1) -> FP, CM(1,1) -> TP
    The name of axis 1 is set to the respective label
    :@return: List of dataframes
    """
    with torch.no_grad():
        if not logits:
            pred = _convert_sigmoid_probs_to_prediction(output)
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
            pred = _convert_sigmoid_probs_to_prediction(output)
        else:
            pred = _convert_logits_to_prediction(output)
        assert pred.shape[0] == len(target)
        return classification_report(y_true=target, y_pred=pred, labels=labels, digits=3, target_names=target_names,
                                     output_dict=output_dict)


# ----------------------------------- TORCHMETRICS -----------------------------------------------
# For details, see https://lightning.ai/docs/torchmetrics/stable/
# Included metrics: Precision, Recall, ROC AUC, ROC, F1, Accuracy
# Short summary for the multi-label case:
# @param output: dimension=(N,C) => : An int tensor or float tensor,
# @param target:  dimension= (N,C) => An int tensor containing the ground truth data
# @param logits: If set to True, the vectors are expected to contain logits/raw scores,
#                otherwise the vectors are expected to contain Sigmoid output probabilities
# @param labels: The set of labels, used for passing the num_classes parameter to TorchMetrics
# @param average: Should be None, 'micro', 'macro', or 'weighted' (NOT 'samples');
#                   'samples' is not supported in newer lib versions, new param multidim_average was introduced for this
# NOTE:
# The default Threshold for transforming probability or logit predictions to binary (0,1) predictions is 0.5,
# and corresponds to input being probabilities.


 
def _torch_precision(output, target, logits, labels, average):
    """
    Compute the precision
    """
    with torch.no_grad():
        assert average in ['macro', 'micro', 'weighted', None], \
            "Average must be one of 'macro', 'micro', 'weighted', None"

        pred = output if not logits else torch.sigmoid(output)
        precision = Precision(task='multilabel', num_classes=len(labels), average=average, threshold=THRESHOLD)
        return precision(pred, target)


def _torch_recall(output, target, logits, labels, average):
    """
    Compute the recall
    """
    with torch.no_grad():
        assert average in ['macro', 'micro', 'weighted', None], \
            "Average must be one of 'macro', 'micro', 'weighted', None"

        pred = output if not logits else torch.sigmoid(output)
        recall = Recall(task='multilabel', num_classes=len(labels), average=average, threshold=THRESHOLD)
        return recall(pred, target)


def _torch_roc_auc(output, target, logits, labels, average):
    """
    Compute the Area Under the Receiver Operating Characteristic Curve (ROC AUC) for the multilabel case
    """
    with torch.no_grad():

        assert average in ['macro', 'micro', 'weighted', None], \
            "Average must be one of 'macro', 'micro', 'weighted', None"

        # In newer lib versions, pred should be a tensor of shape (N, C, ...) with probabilities, where C is the number
        # of classes, or logits  (logits are expected if preds has values outside [0,1] range)
        # Here, we stick to calculating the probs before
        pred = output if not logits else torch.sigmoid(output)

        # In multilabel settings, num_labels is used instead of num_classes for the AUROC metric
        auroc = AUROC(task='multilabel', num_labels=len(labels), average=average)
        return auroc(pred, target)


def torch_roc(output, target, logits, labels):
    """
    Compute the Receiver Operating Characteristic (ROC).
    """
    with torch.no_grad():

        # In newer lib versions, pred should be a tensor of shape (N, C, ...) with probabilities, where C is the number
        # of classes, or logits  (logits are expected if preds has values outside [0,1] range)
        # Here, we stick to calculating the probs before
        pred = output if not logits else torch.sigmoid(output)

        # In multilabel settings, num_labels is used instead of num_classes for the ROC metric
        roc = ROC(task='multilabel',  num_labels=len(labels))
        # returns a tuple (fpr, tpr, thresholds)
        return roc(pred, target)


def _torch_f1(output, target, logits, labels, average):
    """
    Compute F-1 score.
    """
    with torch.no_grad():
        assert average in ['macro', 'micro', 'weighted', None], \
            "Average must be one of 'macro', 'micro', 'weighted', None"

        pred = output if not logits else torch.sigmoid(output)
        f1 = F1Score(task='multilabel', num_classes=len(labels), average=average, threshold=THRESHOLD)
        return f1(pred, target)


def _torch_accuracy(output, target, logits, labels, average):
    """
    Compute the accuracy
    """
    with torch.no_grad():

        assert average in ['macro', 'micro', 'weighted', None], \
            "Average must be one of 'macro', 'micro', 'weighted', None"

        pred = output if not logits else torch.sigmoid(output)
        # In multilabel settings, num_labels is used instead of num_classes for the Accuracy metric
        accuracy = Accuracy(task='multilabel', num_labels=len(labels), average=average, threshold=THRESHOLD)
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
    return _torch_roc_auc(output, target, logits, labels, average="weighted")


def macro_torch_roc_auc(output, target, logits, labels):
    """See documentation for _torch_auc """
    return _torch_roc_auc(output, target, logits, labels, average="macro")


def class_wise_torch_precision(output, target, logits, labels):
    """See documentation for _torch_precision """
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
