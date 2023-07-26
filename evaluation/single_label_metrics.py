import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix, \
    accuracy_score, top_k_accuracy_score, roc_auc_score, f1_score, balanced_accuracy_score, classification_report


# Macro metrics: Macro-level metrics gives equal weight to each class
#       => Each class has the same weight in the average
#       => There is no distinction between highly and poorly populated classes
# Micro metrics: Micro-level metrics weight all items equally.
#       => Considers all the units together, without taking into consideration possible differences between classes
#       => Classes with more observations will have more influence in the metric
#       => Biases the classes towards the most populated class
#       => Micro-Average Precision and Recall are the same values, therefore the MicroAverage F1-Score is also the same
#           (and corresponds to the accuracy when all classes are considered)
from torchmetrics import F1, Precision, Accuracy, Recall, ROC
from torchmetrics.classification.auroc import AUROC


# ----------------------------------- SKlearn Metric -----------------------------------------------

def _convert_log_probs_to_prediction(log_prob_output):
    return torch.argmax(log_prob_output, dim=1)


def _convert_logits_to_prediction(logits):
    # Convert the logits to probabilities and take the one with the highest one as final prediction
    # We are in the single-label case, so apply Softmax first and then return the maximum value
    softmax_probs = torch.nn.functional.softmax(logits, dim=1)
    # Should be the same as directly taking the maximum of raw logits (if x1<x2, then softmax(x1)<softmax(x2))
    assert (torch.argmax(softmax_probs, dim=1) == torch.argmax(logits, dim=1)).all()
    return torch.argmax(softmax_probs, dim=1)


def _sk_f1(output, target, log_probs, logits, labels, average):
    """
    Compute the F1 score, also known as balanced F-score or F-measure.
    In the multi-class and multi-label case, this is the average of the F1 score of each class with
    weighting depending on the average parameter.

    The following parameter description applies for the multiclass case
    :param output: output: dimension=(N,C) or (N);
        Per entry, the log-probabilities of each class should be contained when log_probs is set to true
        Otherwise a list of the predicted class indices in the range [0, C-1] should be passed for each of the N samples
    :param log_probs: If the outputs are log probs, set param to True
    :param logits:  If set to True, the vectors are expected to contain logits/raw scores
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
        assert log_probs ^ logits, "In the single-label case, exactly one of the two must be true"
        if log_probs:
            pred = _convert_log_probs_to_prediction(output)
        else:
            pred = _convert_logits_to_prediction(output)
        assert pred.shape[0] == len(target)
        return f1_score(y_true=target, y_pred=pred, labels=labels, average=average)


def _sk_roc_auc(output, target, log_probs, logits, labels, average, multi_class_cfg):
    """
    The following parameter description applies for the multiclass case
    :param output: dimension=(N,C)
        Per entry, the (log) probability estimates of each class should be contained and
        Later they MUST sum to 1 -> if log probs are provided instead of real probs, set log_prob param to True
    :param log_probs: If the outputs are log softmax probs and do NOT necessarily sum to 1, set param to True
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
    :param multi_class_cfg: Must be specified
        'ovr': One-vs-rest. Computes the AUC of each class against the rest, sensitive to class imbalance even
            when average == 'macro', because class imbalance affects the composition of each of the ‘rest’ groupings.
        'ovo': One-vs-one. Computes the average AUC of all possible pairwise combinations of classes [5].
            Insensitive to class imbalance when average == 'macro'.
    :return: Area Under the Receiver Operating Characteristic Curve (ROC AUC) for the multiclass case

    Note: Could be extended to binary and multilabel classification, but some restrictions apply
    (see https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html)
    """
    with torch.no_grad():
        # In the multiclass case, y_score corresponds to an array of shape (n_samples, n_classes) of probability
        # estimates. The probability estimates must sum to 1 across the possible classes
        assert log_probs ^ logits, "In the single-label case, exactly one of the two must be true"
        # In both cases, the scores do not sum to one
        # Either they are logmax outputs or logits, so transform them
        if logits:
            # Apply softmax on the logits
            probs = torch.nn.functional.softmax(output, dim=1)
        else:
            # Transform log(softmax(x)) to become softmax(x)
            probs = torch.exp(output)

        assert probs.shape[0] == len(target)
        return roc_auc_score(y_true=target, y_score=probs, labels=labels, average=average,
                             multi_class=multi_class_cfg)


def sk_accuracy(output, target, log_probs, logits):
    """
    Calculates the (TOP-1) accuracy for the multiclass case

    Parameters
    ----------
    :param output: dimension=(N,C) or (N);
        Per entry, the log-probabilities of each class should be contained when log_probs is set to true,
        Otherwise a list of the predicted class indices in the range [0, C-1] should be passed for each of the N samples
        (obtaining log-probabilities is achieved by adding a LogSoftmax layer in the last layer of the network)
    :param log_probs: If set to True, the output should have dimension (N,C), otherwise dimension (N)
    :param logits:  If set to True, the vectors are expected to contain logits/raw scores
    :param target: dimension= (N)
        Per entry, a class index in the range [0, C-1] as integer (ground truth)

    Note: Could be extended to the multilabel-indicator-case
    (see https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html)
    """
    with torch.no_grad():
        assert log_probs ^ logits, "In the single-label case, exactly one of the two must be true"
        if log_probs:
            pred = _convert_log_probs_to_prediction(output)
        else:
            pred = _convert_logits_to_prediction(output)
        assert pred.shape[0] == len(target)
        return accuracy_score(y_true=target, y_pred=pred)


def sk_balanced_accuracy(output, target, log_probs, logits):
    """
    Compute the balanced accuracy.

    The balanced accuracy in binary and multiclass classification problems to deal with imbalanced datasets.
    It is defined as the average of recall obtained on each class.
    This implementation is equivalent to accuracy_score with class-balanced sample weights

    Parameters
    ----------
    :param output: dimension=(N,C) or (N);
        Per entry, the log-probabilities of each class should be contained when log_probs is set to true
        Otherwise a list of the predicted class indices in the range [0, C-1] should be passed for each of the N samples
        (obtaining log-probabilities is achieved by adding a LogSoftmax layer in the last layer of the network)
    :param log_probs: If set to True, the output should have dimension (N,C), otherwise dimension (N)
    :param logits:  If set to True, the vectors are expected to contain logits/raw scores
    :param target: dimension= (N)
        Per entry, a class index in the range [0, C-1] as integer (ground truth)
    """
    with torch.no_grad():
        assert log_probs ^ logits, "In the single-label case, exactly one of the two must be true"
        if log_probs:
            pred = _convert_log_probs_to_prediction(output)
        else:
            pred = _convert_logits_to_prediction(output)
        assert pred.shape[0] == len(target)
        return balanced_accuracy_score(y_true=target, y_pred=pred)


def sk_top_k_acc(output, target, labels, k=3):
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


def micro_sk_f1(output, target, log_probs, logits, labels):
    """See documentation for _f1
    Compute the f1-score using the global count of true positives / false negatives, etc.
    (Sums the number of true positives / false negatives for each class).
    """
    return _sk_f1(output, target, log_probs, logits, labels, "micro")


def macro_sk_f1(output, target, log_probs, logits, labels):
    """See documentation for _f1
    Takes the average of the f1-score for each class: that's the avg / total result
    """
    return _sk_f1(output, target, log_probs, logits, labels, "macro")


def weighted_sk_f1(output, target, log_probs, logits, labels):
    """See documentation for _f1
    Compute a weighted average of the f1-score.
    Weights the f1-score by the support of the class:
    the more elements a class has, the more important the f1-score for this class in the computation.
    """
    return _sk_f1(output, target, log_probs, logits, labels, "weighted")


def class_wise_sk_f1(output, target, log_probs, logits, labels):
    """See documentation for _f1 """
    return _sk_f1(output, target, log_probs, logits, labels, None)


def macro_sk_roc_auc_ovo(output, target, log_probs, logits, labels):
    """See documentation for _roc_auc """
    return _sk_roc_auc(output, target, log_probs, logits, labels, "macro", "ovo")


def weighted_sk_roc_auc_ovo(output, target, log_probs, logits, labels):
    """See documentation for _roc_auc """
    return _sk_roc_auc(output, target, log_probs, logits, labels, "weighted", "ovo")


def macro_sk_roc_auc_ovr(output, target, log_probs, logits, labels):
    """See documentation for _roc_auc """
    return _sk_roc_auc(output, target, log_probs, logits, labels, "macro", "ovr")


def weighted_sk_roc_auc_ovr(output, target, log_probs, logits, labels):
    """See documentation for _roc_auc """
    return _sk_roc_auc(output, target, log_probs, logits, labels, "weighted", "ovr")


def overall_confusion_matrix_sk(output, target, log_probs, logits, labels):
    """
        Creates a num_labels x num_labels sized confusion matrix whose i-th row and j-th column entry indicates
        the number of samples with true label being i-th class and predicted label being j-th class

        :param output: dimension=(N,C) or (N);
            Per entry, the log-probabilities of each class should be contained when log_probs is set to true,
            otherwise a list of the predicted class indices in the range [0, C-1] should be passed for each of the N samples
        :param log_probs: If set to True, the output should have dimension (N,C), otherwise dimension (N)
        :param logits:  If set to True, the vectors are expected to contain logits/raw scores
        :param target: List of integers of the real labels (ground truth)
        :param labels: List of labels to index the matrix
        :return: Dataframes of size (num_labels x num_labels) representing the confusion matrix
    """
    with torch.no_grad():
        assert log_probs ^ logits, "In the single-label case, exactly one of the two must be true"
        if log_probs:
            pred = _convert_log_probs_to_prediction(output)
        else:
            pred = _convert_logits_to_prediction(output)
        assert pred.shape[0] == len(target)
        cm = confusion_matrix(y_true=target, y_pred=pred, labels=labels)
        df_cm = pd.DataFrame(cm, index=labels, columns=labels)
        return df_cm


def class_wise_confusion_matrices_single_label_sk(output, target, log_probs, logits, labels):
    """
    Creates a 2x2 confusion matrix per class contained in labels
    CM(0,0) -> TN, CM(1,0) -> FN, CM(0,1) -> FP, CM(1,1) -> TP
    The name of axis 1 is set to the respective label
    :param output: dimension=(N,C) or (N);
        Per entry, the log-probabilities of each class should be contained when log_probs is set to true,
        otherwise a list of the predicted class indices in the range [0, C-1] should be passed for each of the N samples
    :param log_probs: If set to True, the output should have dimension (N,C), otherwise dimension (N)
    :param logits:  If set to True, the vectors are expected to contain logits/raw scores
    :param target: List of integers of the real labels (ground truth)
    :param labels: List of integers representing all classes that can occur
    :return: List of dataframes

    Note: Could be extended to the multilabel-indicator-case
    (see https://scikit-learn.org/stable/modules/generated/sklearn.metrics.multilabel_confusion_matrix.html)
    """
    with torch.no_grad():
        assert log_probs ^ logits, "In the single-label case, exactly one of the two must be true"
        if log_probs:
            pred = _convert_log_probs_to_prediction(output)
        else:
            pred = _convert_logits_to_prediction(output)
        assert pred.shape[0] == len(target)
        class_wise_cms = multilabel_confusion_matrix(y_true=target, y_pred=pred, labels=labels)

        df_class_wise_cms = [pd.DataFrame(class_wise_cms[idx]).astype('int64').rename_axis(labels[idx], axis=1)
                             for idx in range(0, len(class_wise_cms))]
        return df_class_wise_cms


def sk_classification_summary(output, target, log_probs, logits, labels, output_dict,
                              target_names=["IAVB", "AF", "LBBB", "PAC", "RBBB", "SNR", "STD", "STE", "VEB"]):
    with torch.no_grad():
        assert log_probs ^ logits, "In the single-label case, exactly one of the two must be true"
        if log_probs:
            pred = _convert_log_probs_to_prediction(output)
        else:
            pred = _convert_logits_to_prediction(output)
        assert pred.shape[0] == len(target)
        return classification_report(y_true=target, y_pred=pred, labels=labels, digits=3, target_names=target_names,
                                     output_dict=output_dict)

# ----------------------------------- Further Metrics -----------------------------------------------

def cpsc_score(output, target, log_probs, logits):
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
        assert log_probs ^ logits, "In the single-label case, exactly one of the two must be true"
        if log_probs:
            # ndarray of size (sample_num, )
            answers = _convert_log_probs_to_prediction(output).numpy()
        else:
            # ndarray of size (sample_num, )
            answers = _convert_logits_to_prediction(output).numpy()

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

        # print(A)
        # print('Total Record Number: ', np.sum(A))
        return F1, Faf, Fblock, Fpc, Fst


# ----------------------------------- TORCHMETRICS -----------------------------------------------

def _torch_precision(output, target, log_probs, logits, labels, average):
    with torch.no_grad():
        assert log_probs ^ logits, "In the single-label case, exactly one of the two must be true"

        if log_probs:
            pred = _convert_log_probs_to_prediction(output)
        else:
            pred = _convert_logits_to_prediction(output)

        assert pred.shape[0] == len(target)
        precision = Precision(num_classes=len(labels), average=average)
        return precision(pred, target)


def _torch_recall(output, target, log_probs, logits, labels, average):
    with torch.no_grad():
        assert log_probs ^ logits, "In the single-label case, exactly one of the two must be true"

        if log_probs:
            pred = _convert_log_probs_to_prediction(output)
        else:
            pred = _convert_logits_to_prediction(output)

        assert pred.shape[0] == len(target)
        recall = Recall(num_classes=len(labels), average=average)
        return recall(pred, target)


def _torch_roc_auc(output, target, log_probs, logits, labels, average):
    """
    The following parameter description applies for the multiclass case
    For non-binary input, if the preds and target tensor have the same size the input will be interpretated as
    multilabel and if preds have one dimension more than the target tensor the input will be interpretated as multiclass.
    :param output: (N, C, ...) (multiclass) tensor with probabilities, where C is the number of classes.
    :param log_probs: If the outputs are log softmax probs and do NOT necessarily sum to 1, set param to True
    :param logits:  If set to True, the vectors are expected to contain logits/raw scores
    :param target: (N, ...) or (N, C, ...) with integer labels
    :param labels: List of labels that index the classes in output
    :param average: Either ‘macro’, ‘weighted’ or None (´micro' only works for multilabel)
        None:       The scores for each class are returned
        ‘macro’:    Computes metric for each class and uniformly averages them
        ‘weighted’: Computes metric for each class and does a weighted-average, where each class is weighted by
                    their support (accounts for class imbalance)
    :return: Area Under the Receiver Operating Characteristic Curve (ROC AUC) for the multiclass case
    """
    with torch.no_grad():
        # In the multiclass case, y_score corresponds to an array of shape (n_samples, n_classes) of class
        # estimates. The class estimates must sum to 1 across the possible classes
        assert log_probs ^ logits, "In the single-label case, exactly one of the two must be true"
        # In both cases, the scores do not sum to one
        # Either they are logmax outputs or logits, so transform them
        if logits:
            probs = torch.nn.functional.softmax(output, dim=1)
        else:
            probs = torch.exp(output)

        assert probs.shape[0] == len(target)
        auroc = AUROC(num_classes=len(labels), average=average)
        return auroc(probs, target)


def torch_roc(output, target, log_probs, logits, labels):
    """
    The following parameter description applies for the multiclass case
    :param output: (N, C, ...) (multiclass/multilabel) tensor with probabilities, where C is the number of classes.
    :param log_probs: If the outputs are log softmax probs and do NOT necessarily sum to 1, set param to True
    :param logits:  If set to True, the vectors are expected to contain logits/raw scores
    :param target: (N, ...) or (N, C, ...) with integer labels
    :param labels: List of labels that index the classes in output
    :return: Receiver Operating Characteristic Curve (ROC) for the multiclass case
    """
    with torch.no_grad():
        # In the multiclass case, y_score corresponds to an array of shape (n_samples, n_classes) of class
        # estimates. The class estimates must sum to 1 across the possible classes
        assert log_probs ^ logits, "In the single-label case, exactly one of the two must be true"
        # In both cases, the scores do not sum to one
        # Either they are logmax outputs or logits, so transform them
        if logits:
            probs = torch.nn.functional.softmax(output, dim=1)
        else:
            probs = torch.exp(output)

        assert probs.shape[0] == len(target)
        # returns a tuple (fpr, tpr, thresholds)
        auroc = ROC(num_classes=len(labels))
        return auroc(probs, target)



def _torch_f1(output, target, log_probs, logits, labels, average):
    """
    The following parameter description applies for the multiclass case
    :param output: (N, C, ...), accepts logits or probabilities from a model output or integer class values
    :param log_probs: If the outputs are log softmax probs and do NOT necessarily sum to 1, set param to True
    :param logits:  If set to True, the vectors are expected to contain logits/raw scores
    :param target: dimension= (N) (long tensor)
    :param labels: Needed for multiclass targets. List of labels that index the classes in output
    :param average: Either ‘macro’,´micro', ‘weighted’, 'samples' or None
    :return: F1 score for the multilabel case
    """
    with torch.no_grad():
        assert log_probs ^ logits, "In the single-label case, exactly one of the two must be true"
        # Instead of probs, the converted integer values can be passed as well, i.e., _convert_logits_to_prediction(output)
        if log_probs:
            pred = _convert_log_probs_to_prediction(output)
        else:
            pred = _convert_logits_to_prediction(output)

        assert pred.shape[0] == len(target)

        f1 = F1(num_classes=len(labels), average=average)
        return f1(pred, target)


def _torch_accuracy(output, target, log_probs, logits, labels, average):
    """
    The following parameter description applies for the multiclass case
    :param output: (N, C, ...), accepts logits or probabilities from a model output or integer class values
    :param log_probs: If the outputs are log softmax probs and do NOT necessarily sum to 1, set param to True
    :param logits:  If set to True, the vectors are expected to contain logits/raw scores
    :param target: dimension= (N) (long tensor)
    :param labels: Needed for multiclass targets. List of labels that index the classes in output
    :param average: Either ‘macro’,´micro', ‘weighted’, 'samples' or None
    :return: F1 score for the multilabel case
    """
    with torch.no_grad():
        assert log_probs ^ logits, "In the single-label case, exactly one of the two must be true"
        if log_probs:
            pred = _convert_log_probs_to_prediction(output)
        else:
            pred = _convert_logits_to_prediction(output)

        assert pred.shape[0] == len(target)

        accuracy = Accuracy(num_classes=len(labels), average=average)
        return accuracy(pred, target)


def class_wise_torch_accuracy(output, target, log_probs, logits, labels):
    """See documentation for _torch_accuracy """
    return _torch_accuracy(output, target, log_probs, logits, labels, average=None)


def weighted_torch_accuracy(output, target, log_probs, logits, labels):
    """See documentation for _torch_accuracy """
    return _torch_accuracy(output, target, log_probs, logits, labels, average="weighted")


def macro_torch_accuracy(output, target, log_probs, logits, labels):
    """See documentation for _torch_accuracy """
    return _torch_accuracy(output, target, log_probs, logits, labels, average="macro")


def micro_torch_accuracy(output, target, log_probs, logits, labels):
    """See documentation for _torch_accuracy """
    return _torch_accuracy(output, target, log_probs, logits, labels, average="micro")


def class_wise_torch_f1(output, target, log_probs, logits, labels):
    """See documentation for _torch_f1 """
    return _torch_f1(output, target, log_probs, logits, labels, average=None)


def weighted_torch_f1(output, target, log_probs, logits, labels):
    """See documentation for _torch_f1 """
    return _torch_f1(output, target, log_probs, logits, labels, average="weighted")


def macro_torch_f1(output, target, log_probs, logits, labels):
    """See documentation for _torch_f1 """
    return _torch_f1(output, target, log_probs, logits, labels, average="macro")


def class_wise_torch_roc_auc(output, target, log_probs, logits, labels):
    """See documentation for _torch_auc """
    return _torch_roc_auc(output, target, log_probs, logits, labels, average=None)


def weighted_torch_roc_auc(output, target, log_probs, logits, labels):
    """See documentation for _torch_auc """
    # Seems to be consistent to sklearn's 'weighted_roc_auc_ovr', may differ from 'weighted_roc_auc_ovo',
    return _torch_roc_auc(output, target, log_probs, logits, labels, average="weighted")


def macro_torch_roc_auc(output, target, log_probs, logits, labels):
    """See documentation for _torch_auc """
    # Seems to be consistent to sklearn's 'weighted_roc_auc_ovr', may differ from 'weighted_roc_auc_ovo',
    return _torch_roc_auc(output, target, log_probs, logits, labels, average="macro")


def class_wise_torch_precision(output, target, log_probs, logits, labels):
    """See documentation for _torch_auc """
    return _torch_precision(output, target, log_probs, logits, labels, average=None)


def weighted_torch_precision(output, target, sigmoid_probs, logits, labels):
    """See documentation for _torch_precision """
    return _torch_precision(output, target, sigmoid_probs, logits, labels, average='weighted')


def macro_torch_precision(output, target, sigmoid_probs, logits, labels):
    """See documentation for _torch_precision """
    return _torch_precision(output, target, sigmoid_probs, logits, labels, average='macro')


def class_wise_torch_recall(output, target, sigmoid_probs, logits, labels):
    """See documentation for _torch_recall """
    return _torch_recall(output, target, sigmoid_probs, logits, labels, average=None)


def weighted_torch_recall(output, target, sigmoid_probs, logits, labels):
    """See documentation for _torch_recall """
    return _torch_recall(output, target, sigmoid_probs, logits, labels, average='weighted')


def macro_torch_recall(output, target, sigmoid_probs, logits, labels):
    """See documentation for _torch_recall """
    return _torch_recall(output, target, sigmoid_probs, logits, labels, average='macro')


if __name__ == '__main__':
    # Single Label
    # output = [0, 2, 5, 6]
    target = [1, 8, 7, 6]
    output = [[0.1, 0.1, 0.1, 0.1, 0.2, 0.1, 0.1, 0.1, 0.1],
              [0.1, 0.1, 0.1, 0.1, 0.2, 0.1, 0.1, 0.1, 0.1],
              [0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
              [0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]]

    test = roc_auc_score(y_true=target, y_score=output, labels=[0,1,2,3,4,5,6,7,8], average=None, multi_class="ovo")

    print("Finished")

