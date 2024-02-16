import itertools

from bayes_opt import BayesianOptimization
import evaluation.multi_label_metrics as module_metric
import numpy as np
import torch
from sklearn.metrics import f1_score, precision_recall_curve

T_OPTIMIZE_INIT = 1000
T_OPTIMIZER_GP = 50


def apply_threshold(sigmoid_probs, thresholds):
    ts = torch.tensor(thresholds).unsqueeze(0)
    return torch.where(sigmoid_probs > ts, 1, 0)


def apply_single_threshold(sigmoid_probs, threshold):
    torch.where(sigmoid_probs > threshold, 1, 0)


def optimize_ts(logits, target, labels, fast=False):
    sigmoid_probs = torch.sigmoid(logits)

    def evaluate_ts(**thresholds):
        # Convert dict to list
        thresholds = list(thresholds.values())
        thresholds = [round(threshold, 2) for threshold in thresholds]
        res_binar = apply_threshold(sigmoid_probs, thresholds)

        macro_f1 = f1_score(y_true=target, y_pred=res_binar, labels=labels, average="macro")

        return macro_f1

    func = evaluate_ts

    param_names = ['t' + str(k) for k in range(target.shape[1])]
    bounds_lw = 0 * np.ones(target.shape[1])
    bounds_up = 1 * np.ones(target.shape[1])

    # Dict of shape {'t0': (0.0, 1.0), ... , 't8': (0.0, 1.0)}
    pbounds = dict(zip(param_names, zip(bounds_lw, bounds_up)))

    optimizer = BayesianOptimization(f=func, pbounds=pbounds, random_state=1)

    optimizer.probe(
        params=0.5 * np.ones(target.shape[1]),
        lazy=True,
    )

    if fast:
        optimizer.maximize(init_points=200, n_iter=0)
    else:
        optimizer.maximize(init_points=T_OPTIMIZE_INIT, n_iter=T_OPTIMIZER_GP)

    thresholds = optimizer.max['params']

    if isinstance(thresholds, dict):
        thresholds = list(thresholds.values())

    return [round(threshold, 2) for threshold in thresholds]


def optimize_ts_based_on_roc_auc(logits, target, labels):
    sigmoid_probs = torch.sigmoid(logits)

    fpr_all, tpr_all, thresholds_all = module_metric.torch_roc(output=logits, target=target,
                                                               sigmoid_probs=False,
                                                               logits=True, labels=labels)

    # Find the best threshold per class
    best_thresholds = []
    for class_idx in range(0, target.shape[1]):
        fpr = fpr_all[class_idx]
        tpr = tpr_all[class_idx]
        thresholds = thresholds_all[class_idx]

        # Calc Youden's J statistic
        J = tpr - fpr
        best_idx = np.argmax(J)
        best_thresholds.append(thresholds[best_idx])

    res_binar = apply_threshold(sigmoid_probs, best_thresholds)
    macro_f1 = f1_score(y_true=target, y_pred=res_binar, labels=labels, average="macro")

    print('Best Macro F1: ' + str(macro_f1) + ' @ thresholds =' + str(best_thresholds))
    return best_thresholds


def optimize_ts_based_on_f1(logits, target, labels):
    sigmoid_probs = torch.sigmoid(logits)

    # Find the best threshold per class
    best_thresholds = []
    for class_idx in range(0, target.shape[1]):
        # Get only label and score for the current class
        y_true = target[:, class_idx]
        y_score = sigmoid_probs[:, class_idx]
        precision, recall, thresholds = precision_recall_curve(y_true, y_score)

        # Convert to F1-Score
        fscore = (2 * precision * recall) / (precision + recall)
        best_idx = np.argmax(fscore)
        best_thresholds.append(thresholds[best_idx])

    res_binar = apply_threshold(sigmoid_probs, best_thresholds)
    macro_f1 = f1_score(y_true=target, y_pred=res_binar, labels=labels, average="macro")

    print('Best Macro F1: ' + str(macro_f1) + ' @ thresholds =' + str(best_thresholds))
    return [round(threshold, 1) for threshold in best_thresholds]



def optimize_ts_manual(logits, target, labels):
    # TODO WIP!!!!
    sigmoid_probs = torch.sigmoid(logits)

    def evaluate_ts(thresholds):
        res_binar = apply_threshold(sigmoid_probs, thresholds)

        macro_f1 = f1_score(y_true=target, y_pred=res_binar, labels=labels, average="macro")

        return macro_f1

    best = []
    best_score = -1
    # Way too much! Must be done for each class separately -> TODO (compare https://github.com/onlyzdd/ecg-diagnosis)
    base_options = [np.arange(0, 1, 0.1) for _ in range(0, target.shape[1])]
    all_combinations = list(itertools.product(*base_options))
    for t in all_combinations:
        score = evaluate_ts(t)
        if score > best_score:
            best_score = score
            best = t
    print('Best scores: ' + str(round(best_score, 5)) + ' @ thresholds =' + str(best))

    return best