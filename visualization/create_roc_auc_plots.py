import argparse

import pickle

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import RocCurveDisplay, auc
from torchmetrics import AUROC

from utils import ensure_dir, read_json

# Needed for working with SSH Interpreter...
import os



def _extract_predictions_and_target_dicts_for_gb_models(main_path, num_classes, total_num_folds, target_names):
    # Initialize a dictionary to store predictions grouped by class
    predictions_by_class = {class_idx: [] for class_idx in range(num_classes)}
    targets_by_class = {class_idx: [] for class_idx in range(num_classes)}

    for i in range(total_num_folds):
        for class_idx in range(num_classes):
            class_name = target_names[class_idx]
            # Load the classifier for this class and fold
            with open(os.path.join(main_path, "Fold_" + str(i + 1), f"best_model_{class_name}.p"), 'rb') as file:
                classifier = pickle.load(file)
            # Load the test set for this fold
            with open(os.path.join(main_path, "Fold_" + str(i + 1), f"X_test_{class_name}.p"), 'rb') as file:
                X_test = pickle.load(file)
            with open(os.path.join(main_path, "Fold_" + str(i + 1), f"y_test_{class_name}.p"), 'rb') as file:
                y_test = pickle.load(file)
            # Get the predictions for the current class from the current fold
            class_predictions = classifier.predict_proba(X_test)[:, 1]
            predictions_by_class[class_idx].append(class_predictions)
            # Append the targets for the current class from the current fold
            targets_by_class[class_idx].append(y_test)

    return predictions_by_class, targets_by_class


def _extract_predictions_and_target_dicts_for_raw_models(main_path, num_classes, total_num_folds):
    predictions_fold_wise = []
    targets_fold_wise = []
    for i in range(total_num_folds):
        # Load the outputs and targets for this fold
        with open(os.path.join(main_path, "output_logging", "Fold_" + str(i + 1), "test_det_outputs.p"), 'rb') as file:
            det_outputs = pickle.load(file)
        with open(os.path.join(main_path, "output_logging", "Fold_" + str(i + 1), "test_det_targets.p"), 'rb') as file:
            det_targets = pickle.load(file)
        predictions_fold_wise.append(det_outputs)
        targets_fold_wise.append(det_targets)
    # Initialize a dictionary to store predictions grouped by class
    predictions_by_class = {class_idx: [] for class_idx in range(num_classes)}
    targets_by_class = {class_idx: [] for class_idx in range(num_classes)}
    # Iterate over each class
    for class_idx in range(num_classes):
        # Collect predictions for the current class from all folds
        for fold_idx in range(total_num_folds):
            # Get the predictions for the current class from the current fold
            class_predictions = predictions_fold_wise[fold_idx][:, class_idx]
            predictions_by_class[class_idx].append(class_predictions)
            # Get the targets for the current class from the current fold
            class_targets = targets_fold_wise[fold_idx][:, class_idx]
            targets_by_class[class_idx].append(class_targets)
    return predictions_by_class, targets_by_class


def create_roc_curve_report(main_path, save_path=None):
    total_num_folds = 10
    target_names = ["IAVB", "AF", "LBBB", "PAC", "RBBB", "SNR", "STD", "STE", "VEB"]
    num_classes = len(target_names)

    params = {'font.size': 26,
              'axes.labelsize': 30,
              'axes.titlesize': 30}
    plt.rcParams.update(params)

    if "gradient_boosting" not in main_path:
        config = read_json(os.path.join(main_path, "config.json"))
        assert config["arch"]["args"]["multi_label_training"], \
            "Multi-Label Training should be enabled when running this script"

        preds_are_logits = True
        predictions_by_class, targets_by_class = _extract_predictions_and_target_dicts_for_raw_models(main_path,
                                                                                                      num_classes,
                                                                                                      total_num_folds)
    else:
        preds_are_logits = False
        predictions_by_class, targets_by_class = _extract_predictions_and_target_dicts_for_gb_models(main_path,
                                                                                                     num_classes,
                                                                                                     total_num_folds,
                                                                                                     target_names)

    # Required:
    # predictions_by_class: dict with keys as class indices and values as lists of predictions for each fold
    # targets_by_class: dict with keys as class indices and values as lists of targets for each fold
    fig, axs = plt.subplots(3, 3, figsize=(36, 36))
    axis_0 = 0
    axis_1 = 0
    line_width = 2


    # Create one plot per class
    desired_order = ['SNR', 'AF', 'IAVB', 'LBBB', 'RBBB', 'PAC', 'VEB', 'STD', 'STE']
    for class_name in desired_order:
        class_idx = target_names.index(class_name)
        try:
            tprs = []
            aucs = []
            mean_fpr = np.linspace(0, 1, 100)

            ax = axs[axis_0, axis_1]

            # Load the outputs and targets for each fold and plot the ROC curve
            for fold_idx in range(total_num_folds):

                y_true = targets_by_class[class_idx][fold_idx]

                if preds_are_logits:
                    y_pred_probs = torch.sigmoid(predictions_by_class[class_idx][fold_idx])
                else:
                    y_pred_probs = predictions_by_class[class_idx][fold_idx]

                viz = RocCurveDisplay.from_predictions(y_true=y_true,
                                                       y_pred=y_pred_probs,
                                                       name=f'Fold {fold_idx + 1}',
                                                       alpha=0.3,
                                                       lw=line_width,
                                                       ax=ax)

                interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
                interp_tpr[0] = 0.0
                tprs.append(interp_tpr)
                aucs.append(viz.roc_auc)

            ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k', label='Chance', alpha=.8)

            mean_tpr = np.mean(tprs, axis=0)
            mean_tpr[-1] = 1.0
            mean_auc = np.mean(aucs)
            std_auc = np.std(aucs)

            ax.plot(
                mean_fpr,
                mean_tpr,
                color="b",
                label=r"Mean ROC (AUC = %0.3f $\pm$ %0.3f)" % (mean_auc, std_auc),
                lw=2,
                alpha=0.8,
            )

            std_tpr = np.std(tprs, axis=0)
            tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
            tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
            ax.fill_between(
                mean_fpr,
                tprs_lower,
                tprs_upper,
                color="grey",
                alpha=0.2,
                label=r"$\pm$ 1 std. dev.",
            )

            ax.set(
                xlabel="False Positive Rate",
                ylabel="True Positive Rate"
            )
            ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
                   title=class_name.replace('VEB', 'PVC'))
            ax.legend(loc="lower right")

            extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            # Pad the saved area by 25% in the x-direction and 25% in the y-direction
            fig.savefig(f"{save_path}/roc_curves_class_{class_name.replace('VEB', 'PVC')}_test_set.pdf",
                        dpi=100, facecolor='w', edgecolor='b', orientation='portrait', transparent=False,
                        bbox_inches=extent.expanded(1.25, 1.25),
                        pad_inches=0.1)
            print(f"Created ROC figure for {class_name.replace('VEB', 'PVC')}")

            axis_1 = (axis_1 + 1) % 3
            if axis_1 == 0:
                axis_0 += 1

        except (AttributeError, OverflowError) as detail:
            print(target_names[class_idx] + " Failed due to ", detail)

    # Finished plotting all classes in one figure
    plt.tight_layout(pad=2, h_pad=3, w_pad=1.5)
    plt.savefig(os.path.join(save_path, "roc_curves_all_classes.pdf"),
                dpi=400)
    plt.show()
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MACRO Paper: Cross-Validation Evaluation')
    parser.add_argument('-p', '--path', default=None, type=str,
                        help='main path to CV runs(default: None)')
    args = parser.parse_args()

    model_name = ""
    if args.path is not None:
        model_name = args.path.split('/')[7]
    save_dir = os.path.join('../figures', 'ROC', f'{model_name}')
    if "gradient_boosting" in args.path:
        gradient_boosting_suffix = args.path.split('/')[-1]
        save_dir = os.path.join(save_dir, gradient_boosting_suffix)
    ensure_dir(save_dir)
    create_roc_curve_report(main_path=args.path, save_path=save_dir)
