import csv

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import multilabel_confusion_matrix, \
    accuracy_score, roc_auc_score, f1_score, precision_score, \
    recall_score, classification_report

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

# Details:
# https://datascience.stackexchange.com/questions/15989/micro-average-vs-macro-average-performance-in-a-multiclass-classification-settin/16001
from torchmetrics import AUROC, F1, Precision, Accuracy, Recall, ROC

THRESHOLD = 0.5


# ----------------------------------- Sklearn Metric -----------------------------------------------

def _convert_sigmoid_probs_to_prediction(sigmoid_probs, thresholds):
    if isinstance(thresholds, dict):
        thresholds = list(thresholds.values())
    ts = torch.tensor(thresholds).unsqueeze(0)
    return torch.where(sigmoid_probs > ts, 1, 0)


def _convert_logits_to_prediction(logits, thresholds):
    # We are in the multi-label case, so apply Sigmoid first and then the threshold
    # Good post: https://web.stanford.edu/~nanbhas/blog/sigmoid-softmax/
    sigmoid_probs = torch.sigmoid(logits)
    return _convert_sigmoid_probs_to_prediction(sigmoid_probs, thresholds)

def _sk_f1(output, target, sigmoid_probs, logits, labels, thresholds, average):
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
            pred = _convert_sigmoid_probs_to_prediction(output, thresholds)
        else:
            pred = _convert_logits_to_prediction(output, thresholds)
        assert pred.shape[0] == len(target)
        return f1_score(y_true=target, y_pred=pred, labels=labels, average=average)


def _sk_precision(output, target, sigmoid_probs, logits, labels, thresholds, average):
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
            pred = _convert_sigmoid_probs_to_prediction(output, thresholds)
        else:
            pred = _convert_logits_to_prediction(output, thresholds)
        assert pred.shape[0] == len(target)
        return precision_score(y_true=target, y_pred=pred, labels=labels, average=average)


def _sk_recall(output, target, sigmoid_probs, logits, labels, thresholds, average):
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
            pred = _convert_sigmoid_probs_to_prediction(output, thresholds)
        else:
            pred = _convert_logits_to_prediction(output, thresholds)
        assert pred.shape[0] == len(target)
        return recall_score(y_true=target, y_pred=pred, labels=labels, average=average)


def _sk_roc_auc(output, target, sigmoid_probs, logits, labels, average):
    """
    The following parameter description applies for the multilabel case
    :param output: dimension=(N,C)
        Per entry, the (log) probability estimates of each class should be contained
    :param sigmoid_probs: If set to True, the vectors are expected to contain Sigmoid output probabilities
    :param logits:  If set to True, the vectors are expected to contain logits/raw scores
    :param target: dimension= (N,C)
        Label indicator array / sparse matrix containing the ground truth data
    :param labels: Needed for multiclass targets. List of labels that index the classes in output
    :param average: In the multilabel-case, either ‘macro’, 'micro', ‘weighted’, 'samples' or None
    :return: Area Under the Receiver Operating Characteristic Curve (ROC AUC) for the multilabel case
    """
    with torch.no_grad():
        assert sigmoid_probs ^ logits, "In the multi-label case, exactly one of the two must be true"

        # Predictions should be passes as probabilities, not as one-hot-vector!
        pred = output if sigmoid_probs else torch.sigmoid(output)

        assert pred.shape[0] == len(target)
        return roc_auc_score(y_true=target, y_score=pred, labels=labels, average=average)


def sk_subset_accuracy(output, target, sigmoid_probs, logits, thresholds):
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
            pred = _convert_sigmoid_probs_to_prediction(output, thresholds)
        else:
            pred = _convert_logits_to_prediction(output, thresholds)
        assert pred.shape[0] == len(target)
        return accuracy_score(y_true=target, y_pred=pred)


def micro_sk_f1(output, target, sigmoid_probs, logits, labels, thresholds):
    """See documentation for _f1 """
    return _sk_f1(output, target, sigmoid_probs, logits, labels, thresholds, "micro")


def macro_sk_f1(output, target, sigmoid_probs, logits, labels, thresholds):
    """See documentation for _f1 """
    return _sk_f1(output, target, sigmoid_probs, logits, labels, thresholds, "macro")


def weighted_sk_f1(output, target, sigmoid_probs, logits, labels, thresholds):
    """See documentation for _f1 """
    return _sk_f1(output, target, sigmoid_probs, logits, labels, thresholds, "weighted")


def samples_sk_f1(output, target, sigmoid_probs, logits, labels, thresholds):
    """See documentation for _f1 """
    return _sk_f1(output, target, sigmoid_probs, logits, labels, thresholds, "samples")


def class_wise_sk_f1(output, target, sigmoid_probs, logits, labels, thresholds):
    """See documentation for _f1 """
    return _sk_f1(output, target, sigmoid_probs, logits, labels, thresholds,  None)


def micro_sk_precision(output, target, sigmoid_probs, logits, labels, thresholds):
    """See documentation for _f1 """
    return _sk_precision(output, target, sigmoid_probs, logits, labels, thresholds, "micro")


def macro_sk_precision(output, target, sigmoid_probs, logits, labels, thresholds):
    """See documentation for _f1 """
    return _sk_precision(output, target, sigmoid_probs, logits, labels, thresholds, "macro")


def weighted_sk_precision(output, target, sigmoid_probs, logits, labels, thresholds):
    """See documentation for _f1 """
    return _sk_precision(output, target, sigmoid_probs, logits, labels, thresholds, "weighted")


def samples_sk_precision(output, target, sigmoid_probs, logits, labels, thresholds):
    """See documentation for _f1 """
    return _sk_precision(output, target, sigmoid_probs, logits, labels, thresholds, "samples")


def class_wise_sk_precision(output, target, sigmoid_probs, logits, labels, thresholds):
    """See documentation for _f1 """
    return _sk_precision(output, target, sigmoid_probs, logits, labels, thresholds, None)


def micro_sk_recall(output, target, sigmoid_probs, logits, labels, thresholds):
    """See documentation for _f1 """
    return _sk_recall(output, target, sigmoid_probs, logits, labels, thresholds, "micro")


def macro_sk_recall(output, target, sigmoid_probs, logits, labels, thresholds):
    """See documentation for _f1 """
    return _sk_recall(output, target, sigmoid_probs, logits, labels, thresholds, "macro")


def weighted_sk_recall(output, target, sigmoid_probs, logits, labels, thresholds):
    """See documentation for _f1 """
    return _sk_recall(output, target, sigmoid_probs, logits, labels, thresholds, "weighted")


def samples_sk_recall(output, target, sigmoid_probs, logits, labels, thresholds):
    """See documentation for _f1 """
    return _sk_recall(output, target, sigmoid_probs, logits, labels, thresholds, "samples")


def class_wise_sk_recall(output, target, sigmoid_probs, logits, labels, thresholds):
    """See documentation for _f1 """
    return _sk_recall(output, target, sigmoid_probs, logits, labels, thresholds, None)


def micro_sk_roc_auc(output, target, sigmoid_probs, logits, labels, thresholds):
    """See documentation for _roc_auc """
    return _sk_roc_auc(output, target, sigmoid_probs, logits, labels, thresholds,  "micro")


def macro_sk_roc_auc(output, target, sigmoid_probs, logits, labels, thresholds):
    """See documentation for _roc_auc """
    return _sk_roc_auc(output, target, sigmoid_probs, logits, labels, thresholds, "macro")


def weighted_sk_roc_auc(output, target, sigmoid_probs, logits, labels, thresholds):
    """See documentation for _roc_auc """
    return _sk_roc_auc(output, target, sigmoid_probs, logits, labels, thresholds, "weighted")


def samples_sk_roc_auc(output, target, sigmoid_probs, logits, labels, thresholds):
    """See documentation for _roc_auc """
    return _sk_roc_auc(output, target, sigmoid_probs, logits, labels, thresholds, "samples")


def class_wise_sk_roc_auc(output, target, sigmoid_probs, logits, labels, thresholds):
    """See documentation for _roc_auc """
    return _sk_roc_auc(output, target, sigmoid_probs, logits, labels, thresholds, average=None)


# A Normal confusion matrix does not make sense in the multi-label context, but a label-wise one can be computed
def class_wise_confusion_matrices_multi_label_sk(output, target, sigmoid_probs, logits, labels, thresholds):
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
            pred = _convert_sigmoid_probs_to_prediction(output, thresholds)
        else:
            pred = _convert_logits_to_prediction(output, thresholds)
        assert pred.shape[0] == len(target)
        class_wise_cms = multilabel_confusion_matrix(y_true=target, y_pred=pred, labels=labels)

        df_class_wise_cms = [pd.DataFrame(class_wise_cms[idx]).astype('int64').rename_axis(labels[idx], axis=1)
                             for idx in range(0, len(class_wise_cms))]
        return df_class_wise_cms


def sk_classification_summary(output, target, sigmoid_probs, logits, labels, output_dict, thresholds,
                              target_names=["IAVB", "AF", "LBBB", "PAC", "RBBB", "SNR", "STD", "STE", "VEB"]):
    with torch.no_grad():
        assert sigmoid_probs ^ logits, "In the multi-label case, exactly one of the two must be true"
        if sigmoid_probs:
            pred = _convert_sigmoid_probs_to_prediction(output, thresholds)
        else:
            pred = _convert_logits_to_prediction(output, thresholds)
        assert pred.shape[0] == len(target)
        return classification_report(y_true=target, y_pred=pred, labels=labels, digits=3, target_names=target_names,
                                     output_dict=output_dict)


# ----------------------------------- Further Metrics -----------------------------------------------


def _convert_multi_label_probs_to_single_prediction(sigmoid_prob_output):
    return torch.argmax(sigmoid_prob_output, dim=1)


def _convert_multi_label_logits_to_single_prediction(logits_output):
    # Convert the logits to probabilities and take the one with the highest one as final predicti
    softmax_probs = torch.nn.functional.softmax(logits_output, dim=1)
    # Should be the same as directly taking the maximum of raw logits
    assert (torch.argmax(softmax_probs, dim=1) == torch.argmax(logits_output, dim=1)).all()
    # In very seldom cases the following is not true, probably because of numerical instability:
    # torch.sigmoid(torch.Tensor([7.4171776772])) == torch.sigmoid(torch.Tensor([7.4171915054,]))
    # even though the first logit is smaller!
    # assert (torch.argmax(softmax_probs, dim=1) == torch.argmax(torch.sigmoid(logits_output), dim=1)).all()
    return torch.argmax(softmax_probs, dim=1)


def cpsc_score(output, target, sigmoid_probs, logits):
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
                   
    UDPATE:
                                             Predicted
                            IAVB  AF    LBBB  PAC   RBBB  SNR  STD  STE  VEB
                   IAVB     N11   N12   N13   N14   N15   N16  N17  N18  N19
                   AF       N21   N22   N23   N24   N25   N26  N27  N28  N29
                   LBBB     N31   N32   N33   N34   N35   N36  N37  N38  N39
                   PAC      N41   N42   N43   N44   N45   N46  N47  N48  N49
    Reference      RBBB     N51   N52   N53   N54   N55   N56  N57  N58  N59
                   SNR      N61   N62   N63   N64   N65   N66  N67  N68  N69
                   STD      N71   N72   N73   N74   N75   N76  N77  N78  N79
                   STE      N81   N82   N83   N84   N85   N86  N87  N88  N89
                   VEB      N91   N92   N93   N94   N95   N96  N97  N98  N99

   

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

        # Following is calculating scores for 4 types: AF, Block, Premature contraction, ST-segment change.

        # Class AF -> 164889003 -> Index 1
        Faf = 2 * A[1][1] / (np.sum(A[1, :]) + np.sum(A[:, 1]))
        """                    
        Block: Fblock=2*(N33+N44+N55)/(N3x+Nx3+N4x+Nx4+N5x+Nx5)
        Premature contraction: Fpc=2*(N66+N77)/(N6x+Nx6+N7x+Nx7)    
        ST-segment change: Fst=2*(N88+N99)/(N8x+Nx8+N9x+Nx9)
        """
        # Block: Classes I-AVB, LBBB, RBBB -> 270492004, 164909002, 59118001 -> Indices 0, 2, 4
        Fblock = 2 * (A[0][0] + A[2][2] + A[4][4]) / \
                 (np.sum(A[0, :]) + np.sum(A[:, 0]) + np.sum(A[2, :]) + np.sum(A[:, 2]) + np.sum(A[4, :]) + np.sum(
                     A[:, 4]))

        # # Classes PAC, PVC -> 284470004, 164884008 -> Indices 3, 8
        Fpc = 2 * (A[3][3] + A[8][8]) / (np.sum(A[3, :]) + np.sum(A[:, 3]) + np.sum(A[8, :]) + np.sum(A[:, 8]))

        # # Classes STD, STE -> 429622005, 164931005 -> Indices 6, 7
        Fst = 2 * (A[6][6] + A[7][7]) / (np.sum(A[6:8, :]) + np.sum(A[:, 6:8]))
        test = 2 * (A[6][6] + A[7][7]) / (np.sum(A[6, :]) + np.sum(A[:, 6]) + np.sum(A[7, :]) + np.sum(A[:, 7]))
        assert Fst == test

        # print(A)
        # print('Total Record Number: ', np.sum(A))
        return F1, Faf, Fblock, Fpc, Fst


# ----------------------------------- TORCHMETRICS -----------------------------------------------

def _torch_roc_auc(output, target, sigmoid_probs, logits, labels, average):
    """
    The following parameter description applies for the multilabel case
    For non-binary input, if the preds and target tensor have the same size the input will be interpretated as
    multilabel and if preds have one dimension more than the target tensor the input will be interpretated as multiclass.
    :param output: (N, C, ...) (multiclass) tensor with probabilities, where C is the number of classes.
    :param sigmoid_probs: If the outputs are log sigmoid probs and do NOT necessarily sum to 1, set param to True
    :param logits:  If set to True, the vectors are expected to contain logits/raw scores
    :param target: (N, ...) or (N, C, ...) with integer labels
    :param labels: List of labels that index the classes in output
    :param average: Either ‘macro’, 'micro', ‘weighted’, or None
    :return: Area Under the Receiver Operating Characteristic Curve (ROC AUC) for the multilabel case
    """
    with torch.no_grad():
        sigmoid_probs ^ logits, "In the multi-label case, exactly one of the two must be true"

        # Pred should be a tensor with probabilities of shape (N, C, ...), where C is the number of classes.
        pred = output if sigmoid_probs else torch.sigmoid(output)

        auroc = AUROC(num_classes=len(labels), average=average, pos_label=1)
        return auroc(pred, target)


def torch_roc(output, target, sigmoid_probs, logits, labels):
    """
    The following parameter description applies for the multilabel case
    :param output: (N, C, ...) (multiclass/multilabel) tensor with probabilities, where C is the number of classes.
    :param sigmoid_probs: If the outputs are log sigmoid probs and do NOT necessarily sum to 1, set param to True
    :param logits:  If set to True, the vectors are expected to contain logits/raw scores
    :param target: (N, ...) or (N, C, ...) with integer labels
    :param labels: List of labels that index the classes in output
    :return: Receiver Operating Characteristic Curve (ROC) for the multilabel case
    """
    with torch.no_grad():
        sigmoid_probs ^ logits, "In the multi-label case, exactly one of the two must be true"

        # Pred should be a tensor with probabilities of shape (N, C, ...), where C is the number of classes.
        pred = output if sigmoid_probs else torch.sigmoid(output)

        roc = ROC(num_classes=len(labels), pos_label=1)
        # returns a tuple (fpr, tpr, thresholds)
        return roc(pred, target)

def class_wise_torch_roc_auc(output, target, sigmoid_probs, logits, labels):
    """See documentation for _torch_auc """
    return _torch_roc_auc(output, target, sigmoid_probs, logits, labels, average=None)


def weighted_torch_roc_auc(output, target, sigmoid_probs, logits, labels):
    """See documentation for _torch_auc """
    return _torch_roc_auc(output, target, sigmoid_probs, logits, labels, average="weighted")


def macro_torch_roc_auc(output, target, sigmoid_probs, logits, labels):
    """See documentation for _torch_auc """
    return _torch_roc_auc(output, target, sigmoid_probs, logits, labels, average="macro")


def class_wise_torch_acc(output, target, sigmoid_probs, logits, labels, thresholds): # average=none
    with torch.no_grad():
        assert sigmoid_probs ^ logits, "In the single-label case, exactly one of the two must be true"

        if isinstance(thresholds, dict):
            thresholds = list(thresholds.values())

        # Function accepts logits or probabilities from a model output or integer class values in prediction.
        pred = output if sigmoid_probs else torch.sigmoid(output)
        # Calculate the accuracy with different thresholds
        class_wise_accs = []
        for label_idx in len(thresholds):
            threshold = thresholds[label_idx]
            # Function accepts logits or probabilities from a model output or integer class values in prediction.
            # The default Threshold for transforming probability or logit predictions to binary (0,1) predictions,
            # in the case of binary or multi-label inputs is 0.5 and corresponds to input being probabilities.
            accuracy = Accuracy(num_classes=len(labels), average=None, threshold=threshold)
            all_accs = accuracy(pred, target)
            class_wise_accs.append(all_accs[label_idx])




