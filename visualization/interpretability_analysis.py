import argparse
import pickle

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import torch
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, average_precision_score
from torchmetrics import AUROC, Accuracy

import global_config
from utils import ensure_dir

# Needed for working with SSH Interpreter...
import os
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = global_config.CUDA_VISIBLE_DEVICES


def plot_activations_per_classifier_multilabel(branchNets_activations,
                                               MBM_activations,
                                               GB_all_probs,
                                               GB_13_probs,
                                               labels, lead_names, class_names=None):
    """
    Plot lead-specific activations for different CVD classes in a multilabel setting.
    @param branchNets_activations: List of activations from each lead-specific BranchNet
    @param MBM_activations:  Activations from multibranch MACRO
    @param GB_all_probs: Predicted Probs from GB-all classifiers
    @param GB_13_probs: Predicted Probs from GB-13 classifiers
    @param labels: Corresponding multilabel CVD class labels for each input (binary format)
    @param lead_names: Names of the leads (e.g., ["Lead I", "Lead II", ...])
    @param class_names: Name of the classes
    """
    branchNets_activations = [act.detach().cpu().numpy() for act in branchNets_activations]

    # Use sigmoid probs; the GB all predictions are already probs
    branchNets_activations = [np.array(torch.sigmoid(torch.tensor(activation))) for activation in
                              branchNets_activations]
    MBM_activations = np.array(torch.sigmoid(MBM_activations))

    num_classes = labels.shape[1]  # Number of classes (9 in this case)
    num_leads = len(lead_names)

    # Calculate global min and max across all activations for consistent y-axis scaling
    global_min = 0  # round(min([act.min() for act in branchNets_activations]))
    global_max = 1  # round(max([act.max() for act in branchNets_activations]))

    # for cls in range(num_classes):
    #     # Iterate over each class
    #     plt.figure(figsize=(20, 15))
    #     plt.suptitle(f"Activations for Class {cls if class_names is None else f'{class_names[cls]} (class {cls})'}"
    #                  , fontsize=16)
    #     for lead_idx in range(num_leads):
    #         plt.subplot(3, 4, lead_idx + 1)
    #         lead_activations = activations[lead_idx]
    #
    #         # Filter activations where the current class is active (label = 1)
    #         cls_activations = lead_activations[labels[:, cls] == 1]
    #         if cls_activations.size > 0:
    #             mean_activation = cls_activations.mean(axis=0)
    #             plt.plot(mean_activation, label=f"Class {cls if class_names is None else class_names[cls]}")
    #
    #         plt.title(f"Lead: {lead_names[lead_idx]}")
    #         plt.xlabel("Activation Units")
    #         plt.ylabel("Magnitude")
    #         plt.ylim(global_min, global_max)
    #         plt.legend()
    #
    #     plt.tight_layout()
    #     plt.show()

    # Define colors for each class using a colormap
    class_colors = plt.colormaps.get_cmap('tab10')
    class_colors = matplotlib.colors.ListedColormap(name="custom_vanessa",
                                                    colors=[*[class_colors(idx) for idx in range(7)],
                                                            *[class_colors(8), class_colors(9)]])

    # Create a summary plot for all classes and leads, creating one subplot per lead
    plt.figure(figsize=(20, 30))

    for lead_idx in range(num_leads):
        plt.subplot(5, 3, lead_idx + 1)
        for cls in range(num_classes):  # Iterate over each class
            single_branchNet_activation = branchNets_activations[lead_idx]

            # Filter activations where the current class is active (label = 1)
            cls_activations = single_branchNet_activation[labels[:, cls] == 1]
            if cls_activations.size > 0:
                mean_activation = cls_activations.mean(axis=0)
                plt.plot(mean_activation, color=class_colors(cls),
                         label=f"Class {cls if class_names is None else f'{class_names[cls]} ({cls})'}")

        plt.title(f"BN-Lead_{lead_names[lead_idx]}")
        plt.xlabel("Activation Units")
        plt.ylabel("Magnitude")
        plt.ylim(global_min, global_max)  # Set the y-axis range to the global min and max

    # Also plot the predicted probs for MB-M, GB-all and GB-13

    # MB-M
    plt.subplot(5, 3, num_leads + 1)
    for cls in range(num_classes):  # Iterate over each class
        # Filter activations where the current class is active (label = 1)
        cls_activations_MBM = MBM_activations[labels[:, cls] == 1]
        if cls_activations_MBM.size > 0:
            mean_activation_MBM = cls_activations_MBM.mean(axis=0)
            plt.plot(mean_activation_MBM, color=class_colors(cls),
                     label=f"Class {cls if class_names is None else f'{class_names[cls]} ({cls})'}")
    plt.title(f"Multibranch-MACRO")
    plt.xlabel("Activation Units")
    plt.ylabel("Magnitude")
    plt.ylim(global_min, global_max)

    # GB-all
    plt.subplot(5, 3, num_leads + 2)
    for cls in range(num_classes):  # Iterate over each class
        # Filter activations where the current class is active (label = 1)
        cls_probs_GB_all = GB_all_probs[labels[:, cls] == 1]
        if cls_probs_GB_all.size > 0:
            mean_activation_GB_all = cls_probs_GB_all.mean(axis=0)
            plt.plot(mean_activation_GB_all, color=class_colors(cls),
                     label=f"Class {cls if class_names is None else f'{class_names[cls]} ({cls})'}")
    plt.title(f"GB-all")
    plt.xlabel("Activation Units")
    plt.ylabel("Magnitude")
    plt.ylim(global_min, global_max)

    # GB-13
    plt.subplot(5, 3, num_leads + 3)
    for cls in range(num_classes):  # Iterate over each class
        # Filter activations where the current class is active (label = 1)
        cls_probs_GB_13 = GB_13_probs[labels[:, cls] == 1]
        if cls_probs_GB_13.size > 0:
            mean_activation_GB_13 = cls_probs_GB_13.mean(axis=0)
            plt.plot(mean_activation_GB_13, color=class_colors(cls),
                     label=f"Class {cls if class_names is None else f'{class_names[cls]} ({cls})'}")
    plt.title(f"GB-13")
    plt.xlabel("Activation Units")
    plt.ylabel("Magnitude")
    plt.ylim(global_min, global_max)

    # Create a single legend above the plots
    legend_elements = [Line2D([0], [0], color=class_colors(cls), lw=2,
                              label=f'Class {cls}' if class_names is None else f'{class_names[cls]} (ID={cls})')
                       for cls in range(num_classes)]
    plt.figlegend(handles=legend_elements, loc='upper center', ncol=9, fontsize=20, bbox_to_anchor=(0.5, 1.025))
    plt.tight_layout()
    plt.savefig("/home/vab30xh/projects/2024-macro-final/figures/interpretability/activations_per_classifier.pdf",
                bbox_inches='tight')
    plt.show()


def plot_activations_per_class_multilabel(branchNets_activations,
                                          MBM_activations,
                                          GB_all_probs,
                                          GB_13_probs,
                                          labels, lead_names, class_names=None):
    """
    Plot lead-specific activations for different CVD classes in a multilabel setting.
    @param branchNets_activations: List of activations from each lead-specific BranchNet
    @param MBM_activations:  Activations from multibranch MACRO
    @param GB_all_probs: Predicted Probs from GB-all classifiers
    @param GB_13_probs: Predicted Probs from GB-13 classifiers
    @param labels: Corresponding multilabel CVD class labels for each input (binary format)
    @param lead_names: Names of the leads (e.g., ["Lead I", "Lead II", ...])
    @param class_names: Name of the classes
    """
    branchNets_activations = [act.detach().cpu().numpy() for act in branchNets_activations]

    # Use sigmoid probs; the GB all predictions are already probs
    branchNets_activations = [np.array(torch.sigmoid(torch.tensor(activation)))
                              for activation in branchNets_activations]
    MBM_activations = np.array(torch.sigmoid(MBM_activations))

    num_classes = labels.shape[1]  # Number of classes (9 in this case)
    num_leads = len(lead_names)

    # Calculate global min and max across all activations for consistent y-axis scaling
    global_min = 0
    global_max = 1

    # Generate 15 distinct colors
    # hsv_cmap = plt.cm.get_cmap('hsv', 15)
    tab20_cmap = plt.colormaps.get_cmap('tab20')
    lead_colors = [tab20_cmap(i) for i in range(15)]


    # Create a summary plot for each class showing activations per lead
    plt.figure(figsize=(20, 20))
    for cls in range(num_classes):
        plt.subplot(3, 3, cls + 1)
        # Plot activations for all leads for the current class
        for lead_idx in range(num_leads):
            single_branchNet_activation = branchNets_activations[lead_idx]  # Activations for the current lead
            # Filter activations where the current class is active (label = 1)
            cls_activations = single_branchNet_activation[labels[:, cls] == 1]
            if cls_activations.size > 0:
                mean_activation = cls_activations.mean(axis=0)  # Mean activation for the current class
                plt.plot(mean_activation, color=lead_colors[lead_idx],
                         label=f"BranchNet for Lead {lead_names[lead_idx]}")
        # Also plot the predicted probs for MB-M, GB-all and GB-13
        cls_activations_MBM = MBM_activations[labels[:, cls] == 1]
        if cls_activations_MBM.size > 0:
            mean_activation_MBM = cls_activations_MBM.mean(axis=0)  # Mean activation for the current class
            plt.plot(mean_activation_MBM, color=lead_colors[num_leads], marker='o', label=f"MB-M")
        cls_probs_GB_all = GB_all_probs[labels[:, cls] == 1]
        if cls_probs_GB_all.size > 0:
            mean_activation_GB_all = cls_probs_GB_all.mean(axis=0)  # Mean activation for the current class
            plt.plot(mean_activation_GB_all, color=lead_colors[num_leads + 1], marker='*', linestyle='dashed',
                     label=f"GB-all")
        cls_probs_GB_13 = GB_13_probs[labels[:, cls] == 1]
        if cls_probs_GB_13.size > 0:
            mean_activation_GB_13 = cls_probs_GB_13.mean(axis=0)  # Mean activation for the current class
            plt.plot(mean_activation_GB_13, color=lead_colors[num_leads + 2], marker='v', linestyle='dotted',
                     label=f"GB-13")
        plt.title(f"Class {cls if class_names is None else f'{class_names[cls]} (ID={cls})'}")
        plt.xlabel("Activation Units")
        plt.ylabel("Magnitude")
        plt.ylim(global_min, global_max)  # Set the y-axis range to the global min and max
    # Create a single legend below the plots
    legend_elements_branchNets = [Line2D([0], [0], color=lead_colors[lead_idx], lw=2,
                                         label=f'BN-{lead_names[lead_idx]}') for lead_idx in range(num_leads)]
    further_legend_elements = [Line2D([0], [0], color=lead_colors[num_leads], marker='o', lw=2, label=f'MB-M'),
                               Line2D([0], [0], color=lead_colors[num_leads + 1], marker='*', linestyle='dashed', lw=2,
                                      label=f'GB-all'),
                               Line2D([0], [0], color=lead_colors[num_leads + 2], marker='v', linestyle='dotted',
                                      lw=2, label=f'GB-13')]
    legend_elements = [*legend_elements_branchNets, *further_legend_elements]
    plt.figlegend(handles=legend_elements, loc='upper center', ncol=8, fontsize=20, bbox_to_anchor=(0.5, 1.07))
    # plt.subplots_adjust(hspace=0.5)  # Increase the hspace value to increase vertical space
    plt.tight_layout()
    plt.savefig("/home/vab30xh/projects/2024-macro-final/figures/interpretability/activations_per_class.pdf",
                bbox_inches='tight')
    plt.show()


def perform_single_vs_multi_label_analysis(labels, logits_MBM, GB_all_probs, GB_13_probs, class_names):
    def process_predictions(pred_logits):
        """Helper function to process logits into probabilities and binary predictions."""
        probs = torch.sigmoid(pred_logits)
        predictions = (probs > 0.5).int()  # Binarize logits to predictions
        return probs, predictions

    # Process each set of predictions
    probs_MBM, preds_MBM = process_predictions(logits_MBM)
    probs_GB_all, preds_GB_all = GB_all_probs, (GB_all_probs > 0.5).astype(int)
    probs_GB_13, preds_GB_13 = GB_13_probs, (GB_13_probs > 0.5).astype(int)

    # Convert tensors to numpy arrays for easier manipulation
    labels_np = labels.numpy()
    probs_MBM_np = probs_MBM.numpy()
    preds_MBM_np = preds_MBM.numpy()

    # Identify single-labeled and multi-labeled indices
    single_labeled_indices = np.where(labels_np.sum(axis=1) == 1)[0]
    multi_labeled_indices = np.where(labels_np.sum(axis=1) > 1)[0]

    # Performance metrics calculation function
    def calculate_metrics(labels, predictions, probs, average='macro'):
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average=average)
        auroc = AUROC(task='multilabel', num_labels=len(labels[0]), average=average)
        auc = auroc(torch.tensor(probs), torch.tensor(labels))
        acc_metric = Accuracy(task='multilabel', num_labels=len(labels[0]), average=average)
        acc = acc_metric(torch.tensor(probs), torch.tensor(labels))
        return precision, recall, f1, auc, acc

    # Function to evaluate each set of predictions
    def evaluate_predictions(probs_np, preds_np, name):
        # Separate labels and predictions for single-labeled and multi-labeled recordings
        single_probs = probs_np[single_labeled_indices]
        single_preds = preds_np[single_labeled_indices]
        multi_probs = probs_np[multi_labeled_indices]
        multi_preds = preds_np[multi_labeled_indices]

        # Performance on single-labeled recordings
        single_precision, single_recall, single_f1, single_auc, single_acc = calculate_metrics(
            labels_np[single_labeled_indices],
            single_preds,
            single_probs
        )

        # Performance on multi-labeled recordings
        multi_precision, multi_recall, multi_f1, multi_auc, multi_acc = calculate_metrics(
            labels_np[multi_labeled_indices],
            multi_preds,
            multi_probs
        )

        # Print the results
        print(f"Performance on Single-Labeled Recordings ({name}):\n"
              f"Precision: {single_precision:.4f}, "
              f"Recall: {single_recall:.4f}, "
              f"F1-Score: {single_f1:.4f}, "
              f"AUC: {single_auc:.4f}",
              f"Acc: {single_acc:.4f}")

        print(f"Performance on Multi-Labeled Recordings ({name}):\n"
              f"Precision: {multi_precision:.4f}, "
              f"Recall: {multi_recall:.4f}, "
              f"F1-Score: {multi_f1:.4f}, ",
              f"AUC: {multi_auc:.4f}",
              f"Acc: {multi_acc:.4f}")

    # Evaluate each prediction set
    evaluate_predictions(probs_MBM_np, preds_MBM_np, "MBM")
    evaluate_predictions(probs_GB_all, preds_GB_all, "GB_all")
    evaluate_predictions(probs_GB_13, preds_GB_13, "GB_13")

    # Analysis of class distribution in multi-labeled recordings
    multi_label_class_counts = labels_np[multi_labeled_indices].sum(axis=0)
    total_class_counts = labels_np.sum(axis=0)

    print("\nClass Distribution in Multi-Labeled Recordings (vs. Total):")
    for i in range(labels.shape[1]):
        print(f"Class {class_names[i]}: {multi_label_class_counts[i]} (in multi-labeled) / {total_class_counts[i]} (total)",
              f"({(multi_label_class_counts[i] / total_class_counts[i]) * 100:.2f}%)")

    # Check if minority classes like STE are predominantly in multi-labeled recordings
    if 'STE' in class_names:
        minority_class_index = class_names.index('STE')
        minority_class_in_multilabels = multi_label_class_counts[minority_class_index]
        minority_class_total = total_class_counts[minority_class_index]
        print(f"\nMinority Class 'STE' in Multi-Labeled Recordings: "
              f"{minority_class_in_multilabels}/{minority_class_total} "
              f"({(minority_class_in_multilabels / minority_class_total) * 100:.2f}%)")


def run_interpretability_analysis_on_cross_fold_data(main_path, total_num_folds):
    print(f"Starting with some interpretability analysis for the 10 test folds of the {total_num_folds}-fold CV")

    base_dir = main_path
    save_dir = os.path.join("../figures", "interpretability")
    ensure_dir(save_dir)

    lead_names = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    class_names = ["IAVB", "AF", "LBBB", "PAC", "RBBB", "SNR", "STD", "STE", "PVC"]  # VEB = PVC
    # desired_order = ['SNR', 'AF', 'IAVB', 'LBBB', 'RBBB', 'PAC', 'PVC', 'STD', 'STE']

    single_lead_activations = []
    MBM_activations = []
    GB_all_probs = []
    GB_13_probs = []
    targets = []

    for k in range(total_num_folds):
        print("Starting fold " + str(k + 1))

        # Adapt the log and save paths for the current fold
        data_dir = os.path.join(base_dir, "output_logging", "Fold_" + str(k + 1))

        ###################### Final FC Activations ######################

        # Load the detached data for the current TEST fold
        path = os.path.join(data_dir, "test_det_single_lead_outputs.p")
        with open(path, 'rb') as file:
            test_det_single_lead_outputs = pickle.load(file)
        path = os.path.join(data_dir, "test_det_outputs.p")
        with open(path, 'rb') as file:
            test_det_outputs = pickle.load(file)

        # Load the detached targets for the current TEST fold
        path = os.path.join(data_dir, "test_det_targets.p")
        with open(path, 'rb') as file:
            test_det_targets = pickle.load(file)

        # Load the predicted GB probs
        path = os.path.join(base_dir, "ML models", "gradient_boosting", "Fold_" + str(k + 1),
                            "y_pred_probs_all_classes.p")
        with open(path, 'rb') as file:
            test_GB_all_outputs = pickle.load(file)

        path = os.path.join(base_dir, "ML models", "gradient_boosting_individual_features", "Fold_" + str(k + 1),
                            "y_pred_probs_all_classes.p")
        with open(path, 'rb') as file:
            test_GB_13_outputs = pickle.load(file)

        single_lead_activations.append(test_det_single_lead_outputs)
        MBM_activations.append(test_det_outputs)
        GB_all_probs.append(test_GB_all_outputs)
        GB_13_probs.append(test_GB_13_outputs)
        targets.append(test_det_targets)

    # All activation and target information across the folds is loaded and can be analysed now

    # Concatenate all tensors along the sample dimension (dim=0)
    all_activations_MBM = torch.cat(MBM_activations, dim=0)
    all_probs_GB_all = np.concatenate(GB_all_probs, axis=0)
    all_probs_GB_13 = np.concatenate(GB_13_probs, axis=0)
    all_targets = torch.cat(targets, dim=0)
    merged_tensor_single_leads = torch.cat(single_lead_activations, dim=0)

    # Split the merged tensor by leads
    # This will give a list of 12 tensors, each of shape (total_records, num_classes)
    activations_by_BrachNets = [merged_tensor_single_leads[:, lead_idx, :] for lead_idx in range(12)]

    # ###################### Visualization of Network Activations ######################

    # # Set global font sizes using rcParams
    # plt.rcParams.update({
    #     'font.size': 18,  # Base font size for everything
    #     'axes.titlesize': 20,  # Font size for subplot titles
    #     'axes.labelsize': 18,  # Font size for x and y labels
    #     'xtick.labelsize': 16,  # Font size for x-axis tick labels
    #     'ytick.labelsize': 16,  # Font size for y-axis tick labels
    #     'legend.fontsize': 16,  # Font size for legend text
    #     'figure.titlesize': 23  # Font size for the overall figure title
    # })
    #
    # plot_activations_per_classifier_multilabel(branchNets_activations=activations_by_BrachNets,
    #                                            MBM_activations=all_activations_MBM,
    #                                            GB_all_probs=all_probs_GB_all,
    #                                            GB_13_probs=all_probs_GB_13,
    #                                            labels=all_targets,
    #                                            lead_names=lead_names,
    #                                            class_names=class_names)
    # plot_activations_per_class_multilabel(branchNets_activations=activations_by_BrachNets,
    #                                       MBM_activations=all_activations_MBM,
    #                                       GB_all_probs=all_probs_GB_all,
    #                                       GB_13_probs=all_probs_GB_13,
    #                                       labels=all_targets,
    #                                       lead_names=lead_names,
    #                                       class_names=class_names)

    ###################### Analysis Single vs Multi Label ######################
    class_names = ["IAVB", "AF", "LBBB", "PAC", "RBBB", "SNR", "STD", "STE", "PVC"]
    perform_single_vs_multi_label_analysis(labels=all_targets,
                                           logits_MBM=all_activations_MBM,
                                           GB_all_probs=all_probs_GB_all,
                                           GB_13_probs=all_probs_GB_13,
                                           class_names=class_names)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MACRO Paper: Feature Importance Visualization on Cross-Fold Data')
    parser.add_argument('-p', '--path', default=None, type=str,
                        help='main path to CV runs(default: None)')
    # Example path: "../savedVM/models/Multibranch_MACRO_CV/0201_104057_ml_bs64convRedBlock_333_0.2_6_false_0.2_24/"
    parser.add_argument('--num_folds', default=10, type=int,
                        help='number of folds within the CV (default: 10)')
    args = parser.parse_args()
    run_interpretability_analysis_on_cross_fold_data(args.path, args.num_folds)
