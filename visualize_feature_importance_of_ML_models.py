import argparse
import pickle

import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import shap
from matplotlib.gridspec import GridSpec

import global_config
from utils import ensure_dir

# Needed for working with SSH Interpreter...
import os

os.environ["CUDA_VISIBLE_DEVICES"] = global_config.CUDA_VISIBLE_DEVICES


def visualize_feature_importance_on_cross_fold_data(main_path, total_num_folds):
    print(f"Starting with {total_num_folds}-fold cross validation for the visualization of the feature importance")

    base_dir = main_path
    folder_name = "GB_individual_features" if "individual_features" in base_dir else "GB"
    folder_name = f"{folder_name}_reduced" if "reduced" in base_dir else folder_name
    save_dir = os.path.join("figures", "feature_importance", folder_name)
    ensure_dir(save_dir)

    # Store the shap values across folds for later visualization
    class_names = ['SNR', 'AF', 'IAVB', 'LBBB', 'RBBB', 'PAC', 'VEB', 'STD', 'STE']
    shap_values_dict = dict.fromkeys(class_names, [])
    test_sets = []
    feature_importance_list = []

    for k in range(total_num_folds):
        print("Starting fold " + str(k + 1))

        # Adapt the log and save paths for the current fold
        data_dir = os.path.join(base_dir, "Fold_" + str(k + 1))

        ###################### Feature Importance ######################

        # Load the feature importance data for the current fold
        # The file already contains the feature importances for all classes
        path = os.path.join(data_dir, "feature_importance.p")
        with open(path, 'rb') as file:
            feature_importance = pickle.load(file)
        feature_importance.rename(
            index={
                'class_IAVB': 'I-AVB',
                'class_AF': 'AF',
                'class_LBBB': 'LBBB',
                'class_PAC': 'PAC',
                'class_RBBB': 'RBBB',
                'class_SNR': 'SNR',
                'class_STD': 'STD',
                'class_STE': 'STE',
                'class_VEB': 'VEB'
            },
            columns=lambda x: x.replace("branchNet_", "BN_").replace("multibranch", "MB")  # (red.) individual features
            .replace("BrN","BN_").replace("c", ""),  # all features
            inplace=True)

        fig_width = 40 if "individual_features" not in base_dir else 10
        fig, ax = plt.subplots(figsize=(fig_width, 10), dpi=300)
        ax.set_title(f"Feature Importance for Fold {k + 1}")
        # Re-sort according to desired class order
        feature_importance = feature_importance.reindex(
            [class_name.replace('IAVB', 'I-AVB') for class_name in class_names])
        sns.heatmap(feature_importance, xticklabels=True, yticklabels=True, ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        plt.savefig(os.path.join(save_dir, f"feature_importance_fold_{k + 1}.pdf"))
        plt.show()

        feature_importance_list.append(feature_importance)

        # ###################### SHAP ######################
        #
        # # Analyze the SHAP values for the current fold
        # # It has to be done for each class seperately
        # for idx in range(len(class_names)):
        #     class_name = class_names[idx]
        #
        #     # Load the model for the current fold and class, then visualize the SHAP values per class
        #     path = os.path.join(data_dir, f"best_model_{class_name}.p")
        #     with open(path, 'rb') as file:
        #         clf = pickle.load(file)
        #     with open(os.path.join(data_dir, f"X_test_{class_name}.p"), 'rb') as file:
        #         X_test = pickle.load(file)
        #
        #     # Explaining model
        #     explainer = shap.TreeExplainer(clf)
        #     shap_values = explainer.shap_values(X_test.numpy())             # ndarray
        #     shap_values_explanation = explainer(X_test.numpy())             # Explanation object
        #
        #     if shap_values_explanation.feature_names is None:
        #         shap_values_explanation.feature_names = list(feature_importance)  # get column names of df
        #
        #
        #     # Store the SHAP values for the current class and fold
        #     shap_values_dict[class_name].append(shap_values)
        #     test_sets.append(X_test.numpy())
        #
        #     # Apply different SHAP visualization methods based on
        #     # https://medium.com/dataman-in-ai/the-shap-with-more-elegant-charts-bc3e73fa1c0c
        #
        #     # fig = plt.figure(figsize=(20, 20))
        #     # gs = gridspec.GridSpec(3,4, height_ratios=[0.05,1,0.2], width_ratios=[1,0.2,0.2,1])
        #     # gs.update(left=0.05, right=0.95, bottom=0.08, top=0.93, wspace=0.02, hspace=0.03)
        #
        #     fig = plt.figure(constrained_layout=False, figsize=(20, 30))
        #     spec = GridSpec(ncols=2, nrows=2, height_ratios=[1,2], width_ratios=[1,2],
        #                     figure=fig)
        #     ax1 = fig.add_subplot(spec[0, 0])
        #     ax2 = fig.add_subplot(spec[0, 1])
        #     ax3 = fig.add_subplot(spec[1, :])
        #
        #
        #     # Bar plot
        #     plt.sca(ax1)
        #     shap.plots.bar(shap_values_explanation, max_display=10, show=False)
        #     # shap.plots.bar(shap_values_explanation.cohorts(2).abs.mean(0))
        #     new_y_labels = [item.get_text() if "Sum of" not in item.get_text() else "Others"
        #                     for item in ax1.get_ymajorticklabels()]
        #     ax1.set_yticklabels(new_y_labels)
        #     ax1.title.set_text(f'Bar Plot for Class {class_name} in Fold {k + 1}')
        #
        #     # Heatmap
        #     plt.sca(ax2)
        #     shap.plots.heatmap(shap_values_explanation, show=False)
        #     new_y_labels = [item.get_text() if "Sum of" not in item.get_text() else "Others"
        #                     for item in ax2.get_ymajorticklabels()]
        #     ax2.set_yticklabels(new_y_labels)
        #     ax2.title.set_text(f'Heatmap for Class {class_name} in Fold {k + 1}')
        #
        #     # Summary plot
        #     plt.sca(ax3)
        #     shap.summary_plot(shap_values, X_test.numpy(),
        #                       feature_names=shap_values_explanation.feature_names, show=False)
        #     ax3.title.set_text(f'Summary Plot for Class {class_name} in Fold {k + 1}')
        #
        #     # plt.tight_layout()
        #     plt.savefig(os.path.join(save_dir, f"shap_class_{class_name}_fold_{k + 1}.pdf"),bbox_inches='tight')
        #     plt.show()



    # All feature importance and shap data across the folds is loaded and can be visualized now

    # Begin with the visualization of the feature importance
    # Merge feature importances into a single DataFrame
    mean_feature_importance_df = sum(feature_importance_list) / len(feature_importance_list)
    fig, ax = plt.subplots(figsize=(fig_width, 10), dpi=300)
    ax.set_title(f"Average Feature Importance Across the 10 Folds")
    sns.heatmap(mean_feature_importance_df, xticklabels=True, yticklabels=True, ax=ax)
    plt.savefig(os.path.join(save_dir, f"average_feature_importance.pdf"))
    plt.show()

    # # Alternative:
    # # joined = pd.concat(feature_importance_list).reset_index()
    # # means_df = joined.groupby('index').mean()
    #
    # fig, axs = plt.subplots(3, 3, figsize=(18, 18))
    # axis_0 = 0
    # axis_1 = 0
    #
    # for class_name in class_names:
    #     shap_values_for_class = shap_values_dict[class_name]
    #     mean_shap_values = np.mean(shap_values_for_class, axis=0)
    #     shap.summary_plot(mean_shap_values, test_sets)

    print(f"Finished additional run of cross-fold-validation for the visualization of the feature importance")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MACRO Paper: Feature Importance Visualization on Cross-Fold Data')
    parser.add_argument('-p', '--path', default=None, type=str,
                        help='main path to CV runs(default: None)')
    parser.add_argument('--num_folds', default=10, type=int,
                        help='number of folds within the CV (default: 10)')
    args = parser.parse_args()
    visualize_feature_importance_on_cross_fold_data(args.path, args.num_folds)
