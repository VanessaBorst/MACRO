import argparse
import pickle

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import torch
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, average_precision_score
from torchmetrics import AUROC

import global_config
from utils import ensure_dir

# Needed for working with SSH Interpreter...
import os
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = global_config.CUDA_VISIBLE_DEVICES


def plot_activations_per_lead_multilabel(activations, labels, lead_names, class_names=None, use_sigmoid=False):
    """
    Plot lead-specific activations for different CVD classes in a multilabel setting.
    @param activations: List of activations from each lead
    @paramaram labels: Corresponding multilabel CVD class labels for each input (binary format)
    @param lead_names: Names of the leads (e.g., ["Lead I", "Lead II", ...])
    @param class_names: Name of the classes
    @param use_sigmoid: Whether to use raw logits or sigmoid probs for plotting
    """
    activations = [act.detach().cpu().numpy() for act in activations]
    if use_sigmoid:
        activations = [np.array(torch.sigmoid(torch.tensor(activation))) for activation in activations]
    num_classes = labels.shape[1]  # Number of classes (9 in this case)
    num_leads = len(lead_names)

    # Calculate global min and max across all activations for consistent y-axis scaling
    global_min = round(min([act.min() for act in activations]))
    global_max = round(max([act.max() for act in activations]))

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

    # Create a summary plot for all classes and leads, creating one subplot per lead
    plt.figure(figsize=(20, 15))
    for lead_idx in range(num_leads):
        plt.subplot(3, 4, lead_idx + 1)
        for cls in range(num_classes):  # Iterate over each class
            lead_activations = activations[lead_idx]

            # Filter activations where the current class is active (label = 1)
            cls_activations = lead_activations[labels[:, cls] == 1]
            if cls_activations.size > 0:
                mean_activation = cls_activations.mean(axis=0)
                plt.plot(mean_activation, color=class_colors(cls),
                         label=f"Class {cls if class_names is None else f'{class_names[cls]} ({cls})'}")

        plt.title(f"Lead: {lead_names[lead_idx]}")
        plt.xlabel("Activation Units")
        plt.ylabel("Magnitude")
        plt.ylim(global_min, global_max)  # Set the y-axis range to the global min and max

    # Create a single legend below the plots
    legend_elements = [Line2D([0], [0], color=class_colors(cls), lw=2,
                              label=f"Class {cls if class_names is None else f'{class_names[cls]} ({cls})'}")
                       for cls in range(num_classes)]
    plt.figlegend(handles=legend_elements, loc='upper center', ncol=num_classes, fontsize=12, title="Classes",
                  bbox_to_anchor=(0.5, 0.95))
    # plt.tight_layout()  # Adjust layout to accommodate the legend
    plt.show()


def plot_activations_per_class_multilabel(activations, labels, lead_names, class_names=None, use_sigmoid=False):
    """
    Plot activations per class, showing activations for all 12 leads in each plot with consistent y-axis scaling
    and a single legend for lead colors.

    :param activations: List of activations from each lead, each of shape (num_samples, num_classes)
    :param labels: Corresponding multilabel CVD class labels for each input (binary format, shape: num_samples x num_classes)
    :param lead_names: Names of the leads (e.g., ["Lead I", "Lead II", ...])
    """
    activations = [act.detach().cpu().numpy() for act in activations]
    if use_sigmoid:
        if use_sigmoid:
            activations = [np.array(torch.sigmoid(torch.tensor(activation))) for activation in activations]
    num_classes = labels.shape[1]  # Number of classes (9 in this case)
    num_leads = len(lead_names)

    # Calculate global min and max across all activations for consistent y-axis scaling
    global_min = round(min([act.min() for act in activations]))
    global_max = round(max([act.max() for act in activations]))

    # Define colors for each lead using a colormap
    lead_colors = plt.colormaps.get_cmap('Paired')

    # Create a summary plot for each class showing activations per lead
    plt.figure(figsize=(20, 15))
    for cls in range(num_classes):
        plt.subplot(3, 3, cls + 1)

        # Plot activations for all leads for the current class
        for lead_idx in range(num_leads):
            lead_activations = activations[lead_idx]  # Activations for the current lead
            # Filter activations where the current class is active (label = 1)
            cls_activations = lead_activations[labels[:, cls] == 1]
            if cls_activations.size > 0:
                mean_activation = cls_activations.mean(axis=0)  # Mean activation for the current class
                plt.plot(mean_activation, color=lead_colors(lead_idx), label=f"{lead_names[lead_idx]}")

        plt.title(f"Class {cls if class_names is None else f'{class_names[cls]} ({cls})'}")
        plt.xlabel("Activation Units")
        plt.ylabel("Magnitude")
        plt.ylim(global_min, global_max)  # Set the y-axis range to the global min and max

    # Create a single legend below the plots
    legend_elements = [Line2D([0], [0], color=lead_colors(lead_idx), lw=2,
                              label=f'{lead_names[lead_idx]}') for lead_idx in range(num_leads)]
    # Adjust bbox_to_anchor and loc for better placement of the legend
    plt.figlegend(handles=legend_elements, loc='upper center', ncol=num_leads, fontsize=12, title="Leads")

    # plt.tight_layout(rect=[0, 0.1, 1, 0.95])
    plt.show()


def perform_single_vs_multi_label_analysis(labels, pred_logits, class_names):
     # Replace with your actual logits tensor of shape (num_records, num_classes)
    predictions = (torch.sigmoid(pred_logits) > 0.5).int()  # Binarize logits to predictions

    # Convert tensors to numpy arrays for easier manipulation
    labels_np = labels.numpy()
    predictions_np = predictions.numpy()

    # Identify single-labeled and multi-labeled indices
    single_labeled_indices = np.where(labels_np.sum(axis=1) == 1)[0]
    multi_labeled_indices = np.where(labels_np.sum(axis=1) > 1)[0]

    # Separate labels and predictions for single-labeled and multi-labeled recordings
    single_labels = labels_np[single_labeled_indices]
    single_preds = predictions_np[single_labeled_indices]
    multi_labels = labels_np[multi_labeled_indices]
    multi_preds = predictions_np[multi_labeled_indices]

    # Calculate performance metrics
    def calculate_metrics(labels, predictions, average='macro'):
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average=average)
        auroc = AUROC(task='multilabel', num_labels=len(labels), average=average)
        auc = 0
        # auc = auroc(predictions, labels)
        # auc = roc_auc_score(labels, predictions, average=average, multi_class='ovr')
        ap = average_precision_score(labels, predictions, average=average)
        return precision, recall, f1, auc, ap

    # Performance on single-labeled recordings
    single_precision, single_recall, single_f1, single_auc, single_ap = calculate_metrics(single_labels, single_preds)

    # Performance on multi-labeled recordings
    multi_precision, multi_recall, multi_f1, multi_auc, multi_ap = calculate_metrics(multi_labels, multi_preds)

    # Print the results
    print(f"Performance on Single-Labeled Recordings:\n"
          f"Precision: {single_precision:.4f}, "
          f"Recall: {single_recall:.4f}, "
          f"F1-Score: {single_f1:.4f}, "
          f"AUC: {single_auc:.4f}, "
          f"AP: {single_ap:.4f}")

    print(f"Performance on Multi-Labeled Recordings:\n"
          f"Precision: {multi_precision:.4f}, "
          f"Recall: {multi_recall:.4f}, "
          f"F1-Score: {multi_f1:.4f}, "
          f"AUC: {multi_auc:.4f}, "
          f"AP: {multi_ap:.4f}")

    # Analysis of class distribution in multi-labeled recordings
    multi_label_class_counts = multi_labels.sum(axis=0)
    total_class_counts = labels_np.sum(axis=0)

    print("\nClass Distribution in Multi-Labeled Recordings (vs. Total):")
    for i in range(labels.shape[1]):
        print(f"Class {i}: {multi_label_class_counts[i]} (in multi-labeled) / {total_class_counts[i]} (total)")

    # Check if minority classes like STE are predominantly in multi-labeled recordings
    minority_class_index = class_names.index('STE')
    minority_class_in_multilabels = multi_label_class_counts[minority_class_index]
    minority_class_total = total_class_counts[minority_class_index]
    print(f"\nMinority Class {minority_class_index} in Multi-Labeled Recordings: "
          f"{minority_class_in_multilabels}/{minority_class_total} "
          f"({(minority_class_in_multilabels / minority_class_total) * 100:.2f}%)")


def run_interpretability_analysis_on_cross_fold_data(main_path, total_num_folds):
    print(f"Starting with some interpretability analysis for the 10 test folds of the {total_num_folds}-fold CV")

    base_dir = main_path
    save_dir = os.path.join("../figures", "interpretability")
    ensure_dir(save_dir)

    lead_names = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    class_names = ["IAVB", "AF", "LBBB", "PAC", "RBBB", "SNR", "STD", "STE", "PVC"]     # VEB = PVC
    # desired_order = ['SNR', 'AF', 'IAVB', 'LBBB', 'RBBB', 'PAC', 'PVC', 'STD', 'STE']

    single_lead_activations = []
    MBM_activations = []
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

        single_lead_activations.append(test_det_single_lead_outputs)
        MBM_activations.append(test_det_outputs)
        targets.append(test_det_targets)

    # All activation and target information across the folds is loaded and can be analysed now

    # Concatenate all tensors along the sample dimension (dim=0)
    all_activations_MBM = torch.cat(MBM_activations)
    all_targets = torch.cat(targets, dim=0)
    merged_tensor_single_leads = torch.cat(single_lead_activations, dim=0)

    # Split the merged tensor by leads
    # This will give a list of 12 tensors, each of shape (total_records, num_classes)
    activations_by_leads = [merged_tensor_single_leads[:, lead_idx, :] for lead_idx in range(12)]


    # ###################### Visualization of Lead Activations ######################
    #
    plot_activations_per_lead_multilabel(activations_by_leads, all_targets, lead_names, class_names, use_sigmoid=True)
    plot_activations_per_class_multilabel(activations_by_leads, all_targets, lead_names,  class_names, use_sigmoid=True)

    ###################### Analysis Single vs Multi Label ######################
    # Example tensors (use your actual data)
    temp_target = torch.randint(0, 2, (1000, 9))  # Replace with your actual labels tensor of shape (num_records, num_classes)
    temp_pred = torch.randn(1000, 9)
    #all_targets = temp_target
    #MBM_activations = temp_pred
    class_names = ["IAVB", "AF", "LBBB", "PAC", "RBBB", "SNR", "STD", "STE", "PVC"]
    perform_single_vs_multi_label_analysis(all_targets, all_activations_MBM, class_names)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MACRO Paper: Feature Importance Visualization on Cross-Fold Data')
    parser.add_argument('-p', '--path', default=None, type=str,
                        help='main path to CV runs(default: None)')
    # Example path: "../savedVM/models/Multibranch_MACRO_CV/0201_104057_ml_bs64convRedBlock_333_0.2_6_false_0.2_24/"
    parser.add_argument('--num_folds', default=10, type=int,
                        help='number of folds within the CV (default: 10)')
    args = parser.parse_args()
    run_interpretability_analysis_on_cross_fold_data(args.path, args.num_folds)
