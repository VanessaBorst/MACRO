import argparse

import pickle

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import RocCurveDisplay, auc, roc_auc_score
from torchmetrics.classification import BinaryROC, BinaryAUROC

import global_config

from utils import ensure_dir, read_json
from torchmetrics import AUROC, Precision, Accuracy, Recall, ROC, F1Score

# Needed for working with SSH Interpreter...
import os

os.environ["CUDA_VISIBLE_DEVICES"] = global_config.CUDA_VISIBLE_DEVICES


# Create ROC curve reports similar
# to "A pipeline and comparative study of 12 machine learning models for text classification"

# def create_roc_curve_report_new(main_path):
#     config = read_json(os.path.join(main_path, "config.json"))
#     total_num_folds = config["data_loader"]["cross_valid"]["k_fold"]
#     assert config["arch"]["args"]["multi_label_training"], \
#         "Multi-Label Training should be enabled when running this script"
#
#     # Create one plot per class
#     target_names = ["IAVB", "AF", "LBBB", "PAC", "RBBB", "SNR", "STD", "STE", "VEB"]
#     desired_order = ['SNR', 'AF', 'IAVB', 'LBBB', 'RBBB', 'PAC', 'VEB', 'STD', 'STE']
#
#     for i in range(len(target_names)):
#         # Load the outputs and targets for each fold
#         for fold in range(total_num_folds):
#             pass
#
#
#     for a in range(len(models)):
#         model = models[a]
#         model_name = determine_model_name(model)
#         try:
#             tprs = []
#             aucs = []
#             mean_fpr = np.linspace(0, 1, 100)
#             i = 0
#             fig, ax = plt.subplots()
#             # Load the outputs and targets for each fold and plot the ROC curve
#             for i in range(total_num_folds):
#
#                 fpr, tpr, thresholds = module_metric.torch_roc(output=det_outputs, target=det_targets,
#                                                                sigmoid_probs=_param_dict["sigmoid_probs"],
#                                                                logits=_param_dict["logits"],
#                                                                labels=_param_dict["labels"])
#                 roc_auc_scores = module_metric.class_wise_torch_roc_auc(output=det_outputs, target=det_targets,
#                                                                         sigmoid_probs=_param_dict["sigmoid_probs"],
#                                                                         logits=_param_dict["logits"],
#                                                                         labels=_param_dict["labels"])
#                 fpr_class_i = fpr[i].numpy()
#                 tpr_class_i = tpr[i].numpy(
#             for i, (train, test) in enumerate(cv.split(X, y)):
#                 model.fit(X[train], y[train])
#                 viz = plot_roc_curve(model, X[test], y[test],
#                                      name='ROC fold {}'.format(i),
#                                      alpha=0.3, lw=1, ax=ax)
#                 interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
#                 interp_tpr[0] = 0.0
#                 tprs.append(interp_tpr)
#                 aucs.append(viz.roc_auc)
#
#             ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
#                     label='Chance', alpha=.8)
#
#             mean_tpr = np.mean(tprs, axis=0)
#             mean_tpr[-1] = 1.0
#             mean_auc = auc(mean_fpr, mean_tpr)
#             std_auc = np.std(aucs)
#             ax.plot(mean_fpr, mean_tpr, color='b',
#                     label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
#                     lw=2, alpha=.8)
#
#             std_tpr = np.std(tprs, axis=0)
#             tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
#             tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
#             ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
#                             label=r'$\pm$ 1 std. dev.')
#
#             ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
#                    title=model_name)
#             ax.legend(loc="lower right")
#             plt.savefig(
#                 "../results/plots/test/roc_%s_%0i_features_%0i_test.pdf" % (
#                     model_name, feature_size, len(X)),
#                 dpi=100, facecolor='w', edgecolor='b', orientation='portrait', transparent=False, bbox_inches=None,
#                 pad_inches=0.1)
#             print("Created %s ROC figure" % model_name)
#             plt.close()
#         except (AttributeError, OverflowError) as detail:
#             print(model_name + " Failed due to ", detail)

def create_roc_curve_report(main_path, save_path=None):
    config = read_json(os.path.join(main_path, "config.json"))
    total_num_folds = config["data_loader"]["cross_valid"]["k_fold"]
    assert config["arch"]["args"]["multi_label_training"], \
        "Multi-Label Training should be enabled when running this script"

    target_names = ["IAVB", "AF", "LBBB", "PAC", "RBBB", "SNR", "STD", "STE", "VEB"]
    desired_order = ['SNR', 'AF', 'IAVB', 'LBBB', 'RBBB', 'PAC', 'VEB', 'STD', 'STE']
    num_classes = 9

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

    fig, axs = plt.subplots(3, 3, figsize=(18, 18))
    axis_0 = 0
    axis_1 = 0
    line_width = 2

    # Create one plot per class
    for class_idx in range(num_classes):
        # model = models[a]
        class_name = target_names[class_idx]
        try:
            tprs = []
            fprs = []
            aucs = []
            mean_fpr = np.linspace(0, 1, 100)

            ax = axs[axis_0, axis_1]

            # Load the outputs and targets for each fold and plot the ROC curve
            for fold_idx in range(total_num_folds):
                # fpr, tpr, thresholds = roc(preds=predictions_by_class[class_idx][fold_idx],
                #                            target=targets_by_class[class_idx][fold_idx])
                #
                # au_roc_score = au_roc(preds=predictions_by_class[class_idx][fold_idx],
                #                       target=targets_by_class[class_idx][fold_idx])
                #
                # ax.plot(fpr, tpr, lw=line_width,
                #         label=f'Fold {fold_idx + 1}: AUC = %0.3f)' % au_roc_score)

                y_true = targets_by_class[class_idx][fold_idx]
                y_pred_probs = torch.sigmoid(predictions_by_class[class_idx][fold_idx])
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
            mean_auc = auc(mean_fpr, mean_tpr)
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
                   title=class_name)
            ax.legend(loc="lower right")

            extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            # Pad the saved area by 25% in the x-direction and 25% in the y-direction
            fig.savefig(f"{save_path}/roc_curves_class_{class_name}_test_set.pdf",
                        dpi=100, facecolor='w', edgecolor='b', orientation='portrait', transparent=False,
                        bbox_inches=extent.expanded(1.25, 1.25),
                        pad_inches=0.1)
            print(f"Created ROC figure for {class_name}")

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
    save_dir = os.path.join('figures', 'ROC', f'{model_name}')
    ensure_dir(save_dir)
    create_roc_curve_report(main_path=args.path, save_path=save_dir)
