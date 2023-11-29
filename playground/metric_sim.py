import os
import pickle as pk
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

from evaluation.multi_label_metrics import _sk_f1
from utils import get_project_root

INPUT_DIR = "data/CinC_CPSC/cross_valid/500Hz/no_crop"

records = []
labels = []

for file_name in sorted(os.listdir(os.path.join(get_project_root(), INPUT_DIR))):
    if ".pk" not in file_name:
        continue
    record, meta = pk.load(open(os.path.join(get_project_root(),INPUT_DIR, file_name), "rb"))

    records.append(record.values.astype("float32"))
    labels.append(meta["classes_one_hot"].values)

# Simulate prediction for each record as the majority class, i.e. the class at index 0 in the one-hot vectors
preds = [[0, 0, 0, 0, 1, 0, 0, 0, 0] for record in records]
label_list = [0, 1, 2, 3, 4, 5, 6, 7, 8]
sum_of_ones = np.sum(labels, axis=0)        # liefert [ 722 1221  236  616 1857  918  869  220  700]
f1_scores = f1_score(y_true=labels, y_pred=preds, labels=label_list, average=None)
macro_avg_f1 = f1_score(y_true=labels, y_pred=preds, labels=label_list, average='macro')
weighted_avg_f1 = f1_score(y_true=labels, y_pred=preds, labels=label_list, average='weighted')

precision_scores = precision_score(y_true=labels, y_pred=preds, labels=label_list, average=None)
recall_scores = recall_score(y_true=labels, y_pred=preds, labels=label_list, average=None)

print("Summary (for prediction of majority class at index 4):")
print("-------")
print("Macro avg. F1: ", macro_avg_f1)
print("Weighted avg. F1: ", weighted_avg_f1)
print("F1 scores per class: ", f1_scores)
print("-------")
print("Precision per class: ", precision_scores)
print("Recall per class: ", recall_scores)
