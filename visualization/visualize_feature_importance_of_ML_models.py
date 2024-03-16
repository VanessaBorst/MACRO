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
    save_dir = os.path.join("../figures", "feature_importance", folder_name)
    ensure_dir(save_dir)

    # Store the shap values across folds for later visualization
    class_names = ['SNR', 'AF', 'IAVB', 'LBBB', 'RBBB', 'PAC', 'VEB', 'STD', 'STE']
    shap_values_dict = dict.fromkeys(class_names, [])
    test_sets = []
    feature_importance_list = []

    params = {'font.size': 40,
              'axes.labelsize': 40,
              'axes.titlesize': 40}
    plt.rcParams.update(params)

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
            columns=lambda x: x.replace("branchNet_", "BN ").replace("multibranch", "MB")  # (red.) individual features
            .replace("BrN","BN ").replace("c", ""),  # all features
            inplace=True)

        fig_width = 60 if "individual_features" not in base_dir else 20
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


    # All feature importance and shap data across the folds is loaded and can be visualized now

    # Begin with the visualization of the feature importance
    # Merge feature importances into a single DataFrame
    mean_feature_importance_df = sum(feature_importance_list) / len(feature_importance_list)
    fig, ax = plt.subplots(figsize=(fig_width, 10), dpi=300)
    # ax.set_title(f"Average Feature Importance Across the 10 Folds")
    sns.heatmap(mean_feature_importance_df, xticklabels=True, yticklabels=True, ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    plt.savefig(os.path.join(save_dir, f"average_feature_importance.pdf"),bbox_inches='tight')
    plt.show()

    print(f"Finished additional run of cross-fold-validation for the visualization of the feature importance")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MACRO Paper: Feature Importance Visualization on Cross-Fold Data')
    parser.add_argument('-p', '--path', default=None, type=str,
                        help='main path to CV runs(default: None)')
    # Example path: "../savedVM/models/Multibranch_MACRO_CV/0201_104057_ml_bs64convRedBlock_333_0.2_6_false_0.2_24/ML models/gradient_boosting_individual_features"
    parser.add_argument('--num_folds', default=10, type=int,
                        help='number of folds within the CV (default: 10)')
    args = parser.parse_args()
    visualize_feature_importance_on_cross_fold_data(args.path, args.num_folds)
