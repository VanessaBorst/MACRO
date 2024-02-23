import argparse
import pickle

import seaborn as sns
import matplotlib.pyplot as plt

import global_config
from utils import ensure_dir

# Needed for working with SSH Interpreter...
import os

os.environ["CUDA_VISIBLE_DEVICES"] = global_config.CUDA_VISIBLE_DEVICES


def visualize_feature_importance_on_cross_fold_data(main_path, total_num_folds):

    print(f"Starting with {total_num_folds}-fold cross validation for the visualization of the feature importance")

    base_dir = main_path

    for k in range(total_num_folds):
        print("Starting fold " + str(k + 1))

        # Adapt the log and save paths for the current fold
        save_dir = os.path.join(base_dir, "Fold_" + str(k + 1))
        ensure_dir(save_dir)

        # Load the feature importance data for the current fold
        path = os.path.join(save_dir, "feature_importance.p")
        with open(path , 'rb') as file:
            feature_importance = pickle.load(file)
            fig, ax = plt.subplots(figsize=(40, 5))
            sns.heatmap(feature_importance, xticklabels=True, yticklabels=True, ax=ax)
            plt.show()
            plt.savefig(os.path.join(save_dir, "feature_importance.png"))


    # All feature importance data across the folds is loaded and can be visualized now
    # It can be stored in the base_dir

    print(f"Finished additional run of cross-fold-validation for the visualization of the feature importance")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MACRO Paper: Feature Importance Visualization on Cross-Fold Data')
    parser.add_argument('-p', '--path', default=None, type=str,
                        help='main path to CV runs(default: None)')
    parser.add_argument('--num_folds', default=10, type=int,
                        help='number of folds within the CV (default: 10)')
    args = parser.parse_args()
    visualize_feature_importance_on_cross_fold_data(args.path, args.num_folds)
