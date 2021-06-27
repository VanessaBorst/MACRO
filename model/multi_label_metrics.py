import csv

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix, \
    accuracy_score, top_k_accuracy_score, roc_auc_score, f1_score, balanced_accuracy_score, precision_score, \
    recall_score
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
from torch import nn

THRESHOLD = 0.5


def _convert_sigmoid_probs_to_prediction(sigmoid_probs, threshold=THRESHOLD):
    return torch.where(sigmoid_probs > threshold, 1, 0)


def _convert_logits_to_prediction(logits, threshold=THRESHOLD):
    # We are in the multi-label case, so apply Sigmoid first and then the threshold
    # Good post: https://web.stanford.edu/~nanbhas/blog/sigmoid-softmax/
    sigmoid_probs = nn.functional.sigmoid(logits)
    return torch.where(sigmoid_probs > threshold, 1, 0)


def _f1(output, target, sigmoid_probs, logits, labels, average):
    """
    Compute the F1 score, also known as balanced F-score or F-measure.
    In the multi-class and multi-label case, this is the average of the F1 score of each class with
    weighting depending on the average parameter.

    The following parameter description applies for the multi-label case
    :param output: output: dimension=(N,C)
         Label indicator array / sparse matrix containing the estimated targets as returned by a classifier.
    :param target: dimension= (N,C)
        Lbel indicator array / sparse matrix containing the ground truth data
    :param sigmoid_probs: If set to True, the vectors are expected to contain Sigmoid output probabilities
    :param logits:  If set to True, the vectors are expected to contain logits/raw scores
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
        'samples': Calculate metrics for each instance, and find their average
                    (only meaningful for multilabel classification where this differs from accuracy_score)

    :return: float or array of float, shape = [n_unique_labels]
        -> The F1 score, also known as balanced F-score or F-measure.
    """
    with torch.no_grad():
        assert sigmoid_probs ^ logits, "In the multi-label case, exactly one of the two must be true"
        if sigmoid_probs:
            pred = _convert_sigmoid_probs_to_prediction(output)
        else:
            pred = _convert_logits_to_prediction(output)
        assert pred.shape[0] == len(target)
        return f1_score(y_true=target, y_pred=pred, labels=labels, average=average)


def _precision(output, target, sigmoid_probs, logits, labels, average):
    """
    Compute the precision

    The following parameter description applies for the multi-label case
    :param output: output: dimension=(N,C)
         Label indicator array / sparse matrix containing the estimated targets as returned by a classifier.
    :param target: dimension= (N,C)
        Lbel indicator array / sparse matrix containing the ground truth data
    :param sigmoid_probs: If set to True, the vectors are expected to contain Sigmoid output probabilities
    :param logits:  If set to True, the vectors are expected to contain logits/raw scores
    :param labels: The set of labels to include when average != 'binary', and their order if average is None.
    :param average: Determines the type of averaging performed on the data (if not None).
        Parameter values useful for this application:
        None: The scores for each class are returned
        'micro': Calculate metrics globally by counting the total true positives, false negatives and false positives.
        'macro': Calculate metrics for each label, and find their unweighted mean.
                    This does not take label imbalance into account.
        'weighted': Calculate metrics for each label, and find their average weighted by support
                    (the number of true instances for each label).
                    This alters ‘macro’ to account for label imbalance; can result in F-score not between precision/recall
        'samples': Calculate metrics for each instance, and find their average
                    (only meaningful for multilabel classification where this differs from accuracy_score)
    :return: float (if average is not None) or array of float of shape (n_unique_labels,)
                -> Precision of the positive class in binary classification or weighted average of the precision of each
                    class for the multiclass task.
    """
    with torch.no_grad():
        assert sigmoid_probs ^ logits, "In the multi-label case, exactly one of the two must be true"
        if sigmoid_probs:
            pred = _convert_sigmoid_probs_to_prediction(output)
        else:
            pred = _convert_logits_to_prediction(output)
        assert pred.shape[0] == len(target)
        return precision_score(y_true=target, y_pred=pred, labels=labels, average=average)


def _recall(output, target, sigmoid_probs, logits, labels, average):
    """
    Compute the recall

    The following parameter description applies for the multi-label case
    :param output: output: dimension=(N,C)
         Label indicator array / sparse matrix containing the estimated targets as returned by a classifier.
    :param target: dimension= (N,C)
        Lbel indicator array / sparse matrix containing the ground truth data
    :param sigmoid_probs: If set to True, the vectors are expected to contain Sigmoid output probabilities
    :param logits:  If set to True, the vectors are expected to contain logits/raw scores
    :param labels: The set of labels to include when average != 'binary', and their order if average is None.
    :param average: Determines the type of averaging performed on the data (if not None).
        Parameter values useful for this application:
        None: The scores for each class are returned
        'micro': Calculate metrics globally by counting the total true positives, false negatives and false positives.
        'macro': Calculate metrics for each label, and find their unweighted mean.
                    This does not take label imbalance into account.
        'weighted': Calculate metrics for each label, and find their average weighted by support
                    (the number of true instances for each label).
                    This alters ‘macro’ to account for label imbalance; can result in F-score not between precision/recall
        'samples': Calculate metrics for each instance, and find their average
                    (only meaningful for multilabel classification where this differs from accuracy_score)
    :return: float (if average is not None) or array of float of shape (n_unique_labels,)
                -> Recall of the positive class in binary classification or weighted average of the recall of each class
                        for the multiclass task.
        """
    with torch.no_grad():
        assert sigmoid_probs ^ logits, "In the multi-label case, exactly one of the two must be true"
        if sigmoid_probs:
            pred = _convert_sigmoid_probs_to_prediction(output)
        else:
            pred = _convert_logits_to_prediction(output)
        assert pred.shape[0] == len(target)
        return recall_score(y_true=target, y_pred=pred, labels=labels, average=average)


def _roc_auc(output, target, sigmoid_probs, logits, labels, average, multi_class_cfg):
    """
    The following parameter description applies for the multiclass case
    :param output: dimension=(N,C)
        Per entry, the (log) probability estimates of each class should be contained and
        Later they MUST sum to 1 -> if log probs are provided instead of real probs, set log_prob param to True
    :param sigmoid_probs: If set to True, the vectors are expected to contain Sigmoid output probabilities
    :param logits:  If set to True, the vectors are expected to contain logits/raw scores
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
        assert sigmoid_probs ^ logits, "In the multi-label case, exactly one of the two must be true"
        if sigmoid_probs:
            pred = _convert_sigmoid_probs_to_prediction(output)
        else:
            pred = _convert_logits_to_prediction(output)
        assert pred.shape[0] == len(target)
        return roc_auc_score(y_true=target, y_score=pred, labels=labels, average=average, multi_class=multi_class_cfg)


def subset_accuracy(output, target, sigmoid_probs, logits):
    """
    Calculates the (TOP-1) accuracy for the multi-label  case
    For the multi-label case, this function computes subset accuracy:
    the set of labels predicted for a sample must exactly match the corresponding set of labels in y_true

    Parameters
    ----------
    :param output: dimension=(N,C)
        Label indicator array / sparse matrix as returned by the classifier
    :param target: dimension= (N,C)
        Label indicator array / sparse matrix containing the Ground Truth
    :param sigmoid_probs: If set to True, the vectors are expected to contain Sigmoid output probabilites
    :param logits:  If set to True, the vectors are expected to contain logits/raw scores

    """
    with torch.no_grad():
        assert sigmoid_probs ^ logits, "In the multi-label case, exactly one of the two must be true"
        if sigmoid_probs:
            pred = _convert_sigmoid_probs_to_prediction(output)
        else:
            pred = _convert_logits_to_prediction(output)
        assert pred.shape[0] == len(target)
        return accuracy_score(y_true=target, y_pred=pred)


def accuracy(output, target, sigmoid_probs, logits):
    """
    Calculates the (TOP-1) accuracy for the multi-label  case
    For the multi-label case, this function computes subset accuracy:
    the set of labels predicted for a sample must exactly match the corresponding set of labels in y_true

    Parameters
    ----------
    :param output: dimension=(N,C)
        Label indicator array / sparse matrix as returned by the classifier
    :param target: dimension= (N,C)
        Label indicator array / sparse matrix containing the Ground Truth
    :param sigmoid_probs: If set to True, the vectors are expected to contain Sigmoid output probabilites
    :param logits:  If set to True, the vectors are expected to contain logits/raw scores

    """
    with torch.no_grad():
        assert sigmoid_probs ^ logits, "In the multi-label case, exactly one of the two must be true"
        if sigmoid_probs:
            pred = _convert_sigmoid_probs_to_prediction(output)
        else:
            pred = _convert_logits_to_prediction(output)
        assert pred.shape[0] == len(target)
        return accuracy_score(y_true=target, y_pred=pred)


def mirco_f1(output, target, sigmoid_probs, logits, labels):
    """See documentation for _f1 """
    return _f1(output, target, sigmoid_probs, logits, labels, "micro")


def macro_f1(output, target, sigmoid_probs, logits, labels):
    """See documentation for _f1 """
    return _f1(output, target, sigmoid_probs, logits, labels, "macro")


def weighted_f1(output, target, sigmoid_probs, logits,  labels):
    """See documentation for _f1 """
    return _f1(output, target, sigmoid_probs, logits, labels, "weighted")


def samples_f1(output, target, sigmoid_probs, logits, labels):
    """See documentation for _f1 """
    return _f1(output, target, sigmoid_probs, logits, labels, "samples")


def class_wise_f1(output, target, sigmoid_probs, logits, labels):
    """See documentation for _f1 """
    return _f1(output, target, sigmoid_probs, logits, labels, None)


def micro_precision(output, target, sigmoid_probs, logits, labels):
    """See documentation for _f1 """
    return _precision(output, target, sigmoid_probs, logits, labels, "micro")


def macro_precision(output, target, sigmoid_probs, logits, labels):
    """See documentation for _f1 """
    return _precision(output, target, sigmoid_probs, logits, labels, "macro")


def weighted_precision(output, target, sigmoid_probs, logits, labels):
    """See documentation for _f1 """
    return _precision(output, target, sigmoid_probs, logits, labels, "weighted")


def sample_precision(output, target, sigmoid_probs, logits, labels):
    """See documentation for _f1 """
    return _precision(output, target, sigmoid_probs, logits, labels, "samples")


def class_wise_precision(output, target, sigmoid_probs, logits, labels):
    """See documentation for _f1 """
    return _precision(output, target, sigmoid_probs, logits, labels, None)


def micro_recall(output, target, sigmoid_probs, logits, labels):
    """See documentation for _f1 """
    return _recall(output, target, sigmoid_probs, logits, labels, "micro")


def macro_recall(output, target, sigmoid_probs, logits, labels):
    """See documentation for _f1 """
    return _recall(output, target, sigmoid_probs, logits, labels, "macro")


def weighted_recall(output, target, sigmoid_probs, logits, labels):
    """See documentation for _f1 """
    return _recall(output, target, sigmoid_probs, logits, labels, "weighted")


def sample_recall(output, target, sigmoid_probs, logits, labels):
    """See documentation for _f1 """
    return _recall(output, target, sigmoid_probs, logits, labels, "samples")


def class_wise_recall(output, target, sigmoid_probs, logits, labels):
    """See documentation for _f1 """
    return _recall(output, target, sigmoid_probs, logits, labels, None)


def macro_roc_auc_ovo(output, target, sigmoid_probs, logits, labels):
    """See documentation for _roc_auc """
    return _roc_auc(output, target, sigmoid_probs, logits, labels, "macro", "ovo")


def weighted_roc_auc_ovo(output, target, sigmoid_probs, logits, labels):
    """See documentation for _roc_auc """
    return _roc_auc(output, target, sigmoid_probs, logits, labels, "weighted", "ovo")


def macro_roc_auc_ovr(output, target, sigmoid_probs, logits, labels):
    """See documentation for _roc_auc """
    return _roc_auc(output, target, sigmoid_probs, logits, labels, "macro", "ovr")


def weighted_roc_auc_ovr(output, target, sigmoid_probs, logits, labels):
    """See documentation for _roc_auc """
    return _roc_auc(output, target, sigmoid_probs, logits, labels, "weighted", "ovr")


def class_wise_roc_auc_ovr(output, target, sigmoid_probs, logits, labels):
    """See documentation for _roc_auc """
    return _roc_auc(output, target, sigmoid_probs, logits, labels, None, "ovr")


def class_wise_roc_auc_ovo(output, target, sigmoid_probs, logits, labels):
    """See documentation for _roc_auc """
    return _roc_auc(output, target, sigmoid_probs, logits, labels, None, "ovo")


# A Normal confusion matrix does not make sense in the multi-label context, but a label-wise one can be computed
def class_wise_confusion_matrices_multi_label(output, target, sigmoid_probs, logits, labels):
    """
    Creates a 2x2 confusion matrix per class contained in labels
    CM(0,0) -> TN, CM(1,0) -> FN, CM(0,1) -> FP, CM(1,1) -> TP
    The name of axis 1 is set to the respective label

    :param output: dimension=(N,C)
        Label indicator array / sparse matrix } of shape (n_samples, n_classes)  as returned by the classifier
    :param target: dimension= (N,C)
        Label indicator array / sparse matrix } of shape (n_samples, n_classes) containing the Ground Truth
    :param sigmoid_probs: If set to True, the vectors are expected to contain Sigmoid output probabilities
    :param logits:  If set to True, the vectors are expected to contain logits/raw scores
    :param labels: List of integers representing all classes that can occur
    :return: List of dataframes

    """
    with torch.no_grad():
        assert sigmoid_probs ^ logits, "In the multi-label case, exactly one of the two must be true"
        if sigmoid_probs:
            pred = _convert_sigmoid_probs_to_prediction(output)
        else:
            pred = _convert_logits_to_prediction(output)
        assert pred.shape[0] == len(target)
        class_wise_cms = multilabel_confusion_matrix(y_true=target, y_pred=pred, labels=labels)

        df_class_wise_cms = [pd.DataFrame(class_wise_cms[idx]).astype('int64').rename_axis(labels[idx], axis=1)
                             for idx in range(0, len(class_wise_cms))]
        return df_class_wise_cms


def _convert_multi_label_probs_to_single_prediction(sigmoid_prob_output):
    return torch.argmax(sigmoid_prob_output, dim=1)


def _convert_multi_label_logits_to_single_prediction(logits_output):
    # Convert the logits to probabilities and take the one with the highest one as final prediction
    softmax_probs = torch.nn.Softmax(logits_output, dim=1)
    # Should be the same as directly taking the maximum of raw logits
    assert torch.argmax(softmax_probs, dim=1) == torch.argmax(logits_output, dim=1)
    return torch.argmax(softmax_probs, dim=1)


def cpsc_score_adapted(output, target, sigmoid_probs, logits):
    '''
    cspc2018_challenge score
    Written by:  Xingyao Wang, Feifei Liu, Chengyu Liu
                 School of Instrument Science and Engineering
                 Southeast University, China
                 chengyu@seu.edu.cns
    Adapted by: Vanessa Borst
    Output and Target are no longer csv file paths but arrays of size ()
    '''

    '''
    Score the prediction answers by comparing answers.csv and REFERENCE.csv in validation_set folder,
    The scoring uses a F1 measure, which is an average of the nine F1 values from each classification
    type. The specific score rules will be found on http://www.icbeb.org/Challenge.html.
    Matrix A follows the format as:
                                         Predicted
                          Normal  AF  I-AVB  LBBB  RBBB  PAC  PVC  STD  STE
                   Normal  N11   N12   N13   N14   N15   N16  N17  N18  N19
                   AF      N21   N22   N23   N24   N25   N26  N27  N28  N29
                   I-AVB   N31   N32   N33   N34   N35   N36  N37  N38  N39
                   LBBB    N41   N42   N43   N44   N45   N46  N47  N48  N49
    Reference      RBBB    N51   N52   N53   N54   N55   N56  N57  N58  N59
                   PAC     N61   N62   N63   N64   N65   N66  N67  N68  N69
                   PVC     N71   N72   N73   N74   N75   N76  N77  N78  N79
                   STD     N81   N82   N83   N84   N85   N86  N87  N88  N89
                   STE     N91   N92   N93   N94   N95   N96  N97  N98  N99

    For each of the nine types, F1 is defined as:
    Normal: F11=2*N11/(N1x+Nx1) AF: F12=2*N22/(N2x+Nx2) I-AVB: F13=2*N33/(N3x+Nx3) LBBB: F14=2*N44/(N4x+Nx4) RBBB: F15=2*N55/(N5x+Nx5)
    PAC: F16=2*N66/(N6x+Nx6)    PVC: F17=2*N77/(N7x+Nx7)    STD: F18=2*N88/(N8x+Nx8)    STE: F19=2*N99/(N9x+Nx9)

    The final challenge score is defined as:
    F1 = (F11+F12+F13+F14+F15+F16+F17+F18+F19)/9

    In addition, we also calculate the F1 measures for each of the four sub-abnormal types:
                AF: Faf=2*N22/(N2x+Nx2)                         Block: Fblock=2*(N33+N44+N55)/(N3x+Nx3+N4x+Nx4+N5x+Nx5)
    Premature contraction: Fpc=2*(N66+N77)/(N6x+Nx6+N7x+Nx7)    ST-segment change: Fst=2*(N88+N99)/(N8x+Nx8+N9x+Nx9)

    The static of predicted answers and the final score are saved to score.txt in local path.
    '''
    with torch.no_grad():
        assert sigmoid_probs ^ logits, "In the multi-label case, exactly one of the two must be true"
        if sigmoid_probs:
            # ndarray of size (sample_num, )
            answers = _convert_multi_label_probs_to_single_prediction(output).numpy()
        else:
            # ndarray of size (sample_num, )
            answers = _convert_multi_label_logits_to_single_prediction(output).numpy()

        # list of sample_num ndarrays of size (1, ) or (2, ) or (3,)
        reference = [np.nonzero(sample_vec == 1)[0] for sample_vec in target.numpy()]

        assert len(answers) == len(reference), "Answers and References should have equal length"

        A = np.zeros((9, 9), dtype=np.float)

        for sample_idx in range(0, len(answers)):
            pred_class = answers[sample_idx]
            reference_classes = reference[sample_idx]
            if pred_class in reference_classes:
                A[pred_class][pred_class] += 1
            else:
                A[reference_classes[0]][pred_class] += 1

        F11 = 2 * A[0][0] / (np.sum(A[0, :]) + np.sum(A[:, 0]))
        F12 = 2 * A[1][1] / (np.sum(A[1, :]) + np.sum(A[:, 1]))
        F13 = 2 * A[2][2] / (np.sum(A[2, :]) + np.sum(A[:, 2]))
        F14 = 2 * A[3][3] / (np.sum(A[3, :]) + np.sum(A[:, 3]))
        F15 = 2 * A[4][4] / (np.sum(A[4, :]) + np.sum(A[:, 4]))
        F16 = 2 * A[5][5] / (np.sum(A[5, :]) + np.sum(A[:, 5]))
        F17 = 2 * A[6][6] / (np.sum(A[6, :]) + np.sum(A[:, 6]))
        F18 = 2 * A[7][7] / (np.sum(A[7, :]) + np.sum(A[:, 7]))
        F19 = 2 * A[8][8] / (np.sum(A[8, :]) + np.sum(A[:, 8]))

        F1 = (F11 + F12 + F13 + F14 + F15 + F16 + F17 + F18 + F19) / 9

        ## Following is calculating scores for 4 types: AF, Block, Premature contraction, ST-segment change.
        # TODO adapt the indices of the formulas

        # # Class AF -> 164889003 -> Index 1
        # Faf = 2 * A[1][1] / (np.sum(A[1, :]) + np.sum(A[:, 1]))
        # # Classes I-AVB, LBBB, RBBB -> 270492004, 164909002, 59118001 -> Indices 0, 2, 4
        # Fblock = 2 * (A[2][2] + A[3][3] + A[4][4]) / (np.sum(A[2:5, :]) + np.sum(A[:, 2:5]))
        # # Classes PAC, PVC -> 284470004, 164884008 -> Indices 3, 8
        # Fpc = 2 * (A[5][5] + A[6][6]) / (np.sum(A[5:7, :]) + np.sum(A[:, 5:7]))
        # # Classes STD, STE -> 429622005, 164931005 -> Indices 6, 7
        # Fst = 2 * (A[7][7] + A[8][8]) / (np.sum(A[7:9, :]) + np.sum(A[:, 7:9]))

        # print(A)
        print('Total Record Number: ', np.sum(A))
        return F1


def cpsc_score(answers_csv_path, reference_csv_path):
    '''
    cspc2018_challenge score
    Written by:  Xingyao Wang, Feifei Liu, Chengyu Liu
                 School of Instrument Science and Engineering
                 Southeast University, China
                 chengyu@seu.edu.cns
    Adapted by: Vanessa Borst
    '''

    '''
    Score the prediction answers by comparing answers.csv and REFERENCE.csv in validation_set folder,
    The scoring uses a F1 measure, which is an average of the nice F1 values from each classification
    type. The specific score rules will be found on http://www.icbeb.org/Challenge.html.
    Matrix A follows the format as:
                                         Predicted
                          Normal  AF  I-AVB  LBBB  RBBB  PAC  PVC  STD  STE
                   Normal  N11   N12   N13   N14   N15   N16  N17  N18  N19
                   AF      N21   N22   N23   N24   N25   N26  N27  N28  N29
                   I-AVB   N31   N32   N33   N34   N35   N36  N37  N38  N39
                   LBBB    N41   N42   N43   N44   N45   N46  N47  N48  N49
    Reference      RBBB    N51   N52   N53   N54   N55   N56  N57  N58  N59
                   PAC     N61   N62   N63   N64   N65   N66  N67  N68  N69
                   PVC     N71   N72   N73   N74   N75   N76  N77  N78  N79
                   STD     N81   N82   N83   N84   N85   N86  N87  N88  N89
                   STE     N91   N92   N93   N94   N95   N96  N97  N98  N99

    For each of the nine types, F1 is defined as:
    Normal: F11=2*N11/(N1x+Nx1) AF: F12=2*N22/(N2x+Nx2) I-AVB: F13=2*N33/(N3x+Nx3) LBBB: F14=2*N44/(N4x+Nx4) RBBB: F15=2*N55/(N5x+Nx5)
    PAC: F16=2*N66/(N6x+Nx6)    PVC: F17=2*N77/(N7x+Nx7)    STD: F18=2*N88/(N8x+Nx8)    STE: F19=2*N99/(N9x+Nx9)

    The final challenge score is defined as:
    F1 = (F11+F12+F13+F14+F15+F16+F17+F18+F19)/9

    In addition, we also calculate the F1 measures for each of the four sub-abnormal types:
                AF: Faf=2*N22/(N2x+Nx2)                         Block: Fblock=2*(N33+N44+N55)/(N3x+Nx3+N4x+Nx4+N5x+Nx5)
    Premature contraction: Fpc=2*(N66+N77)/(N6x+Nx6+N7x+Nx7)    ST-segment change: Fst=2*(N88+N99)/(N8x+Nx8+N9x+Nx9)

    The static of predicted answers and the final score are saved to score.txt in local path.
    '''
    answers = dict()
    reference = dict()
    A = np.zeros((9, 9), dtype=np.float)
    with open(answers_csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            answers.setdefault(row['Recording'], []).append(row['Result'])
        f.close()
    with open(reference_csv_path) as ref:
        reader = csv.DictReader(ref)
        for row in reader:
            reference.setdefault(row['Recording'], []).append(
                [row['First_label'], row['Second_label'], row['Third_label']])
        ref.close()

    for key in answers.keys():
        value = []
        for item in answers[key]:
            predict = np.int(item)
        for item in reference[key][0]:
            if item == '':
                item = 0
            value.append(np.int(item))

        if predict in value:
            A[predict - 1][predict - 1] += 1
        else:
            A[value[0] - 1][predict - 1] += 1

    F11 = 2 * A[0][0] / (np.sum(A[0, :]) + np.sum(A[:, 0]))
    F12 = 2 * A[1][1] / (np.sum(A[1, :]) + np.sum(A[:, 1]))
    F13 = 2 * A[2][2] / (np.sum(A[2, :]) + np.sum(A[:, 2]))
    F14 = 2 * A[3][3] / (np.sum(A[3, :]) + np.sum(A[:, 3]))
    F15 = 2 * A[4][4] / (np.sum(A[4, :]) + np.sum(A[:, 4]))
    F16 = 2 * A[5][5] / (np.sum(A[5, :]) + np.sum(A[:, 5]))
    F17 = 2 * A[6][6] / (np.sum(A[6, :]) + np.sum(A[:, 6]))
    F18 = 2 * A[7][7] / (np.sum(A[7, :]) + np.sum(A[:, 7]))
    F19 = 2 * A[8][8] / (np.sum(A[8, :]) + np.sum(A[:, 8]))

    F1 = (F11 + F12 + F13 + F14 + F15 + F16 + F17 + F18 + F19) / 9

    ## following is calculating scores for 4 types: AF, Block, Premature contraction, ST-segment change.

    Faf = 2 * A[1][1] / (np.sum(A[1, :]) + np.sum(A[:, 1]))
    Fblock = 2 * (A[2][2] + A[3][3] + A[4][4]) / (np.sum(A[2:5, :]) + np.sum(A[:, 2:5]))
    Fpc = 2 * (A[5][5] + A[6][6]) / (np.sum(A[5:7, :]) + np.sum(A[:, 5:7]))
    Fst = 2 * (A[7][7] + A[8][8]) / (np.sum(A[7:9, :]) + np.sum(A[:, 7:9]))

    # print(A)
    print('Total File Number: ', np.sum(A))

    print("F11: ", F11)
    print("F12: ", F12)
    print("F13: ", F13)
    print("F14: ", F14)
    print("F15: ", F15)
    print("F16: ", F16)
    print("F17: ", F17)
    print("F18: ", F18)
    print("F19: ", F19)
    print("F1: ", F1)

    print("Faf: ", Faf)
    print("Fblock: ", Fblock)
    print("Fpc: ", Fpc)
    print("Fst: ", Fst)

    with open('score.txt', 'w') as score_file:
        # print (A, file=score_file)
        print('Total File Number: %d\n' % (np.sum(A)), file=score_file)
        print('F11: %0.3f' % F11, file=score_file)
        print('F12: %0.3f' % F12, file=score_file)
        print('F13: %0.3f' % F13, file=score_file)
        print('F14: %0.3f' % F14, file=score_file)
        print('F15: %0.3f' % F15, file=score_file)
        print('F16: %0.3f' % F16, file=score_file)
        print('F17: %0.3f' % F17, file=score_file)
        print('F18: %0.3f' % F18, file=score_file)
        print('F19: %0.3f\n' % F19, file=score_file)
        print('F1: %0.3f\n' % F1, file=score_file)
        print('Faf: %0.3f' % Faf, file=score_file)
        print('Fblock: %0.3f' % Fblock, file=score_file)
        print('Fpc: %0.3f' % Fpc, file=score_file)
        print('Fst: %0.3f' % Fst, file=score_file)

        score_file.close()


if __name__ == '__main__':
    test = torch.nn.functional.one_hot(torch.arange(0, 5) % 3)
    reference_path = 'info/csv/REFERENCE_cpsc.csv'
    answers_path = 'info/csv/fake_answers.csv'
    cpsc_score(answers_path, reference_path)
    # x = torch.randn(3, 2)
    # print(x)
    # test = _convert_sigmoid_probs_to_prediction(x)
    # print(test)