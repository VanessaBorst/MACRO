import argparse

import global_config
from ML_ensemble.retrieve_detached_single_run_tensors import load_config_and_setup_paths
from ML_ensemble.train_ML_models_with_detached_tensors import retrieve_detached_data, get_all_predictions_as_probs, \
    get_all_predictions_as_logits
from ML_models_for_multibranch_ensemble import train_ML_model, train_stacking_classifier

from utils import extract_target_names_for_PTB_XL

# Needed for working with SSH Interpreter...
import os

os.environ["CUDA_VISIBLE_DEVICES"] = global_config.CUDA_VISIBLE_DEVICES
global_config.suppress_warnings()


def train_ML_models_on_predfined_splits(main_path, strategy=None, use_logits=False, individual_features=False,
                                        reduced_individual_features=False, scoring_grid_search="f1"):
    assert strategy in ["decision_tree", "ridge", "lasso", "gradient_boosting", "elastic_net", "ada_boost", "xg_boost"], \
        "The given strategy is not supported for training ML models!"
    assert not reduced_individual_features or individual_features, \
        "Reduced individual features can only be used if individual features are used!"

    strategy_name = strategy if not use_logits else strategy + " with logits"
    strategy_name = strategy_name + "_individual_features" if individual_features else strategy_name
    strategy_name = strategy_name + "_reduced" if reduced_individual_features else strategy_name

    config = load_config_and_setup_paths(main_path, sub_dir=os.path.join("ML models",
                                                                         f"{strategy_name}_{scoring_grid_search}"))
    assert config['arch']['type'] == 'FinalModelMultiBranch', \
        "The ML models can only be trained for multi-branch models!"

    data_dir = config["data_loader"]["args"]["data_dir"]
    if "PTB_XL" in data_dir:
        target_names = extract_target_names_for_PTB_XL(data_dir)
    else:
        target_names = ["IAVB", "AF", "LBBB", "PAC", "RBBB", "SNR", "STD", "STE", "VEB"]

    print(f"Starting with ML training for {strategy_name} classifier")
    if use_logits:
        print("Using the raw logits for the training of the ML models")

    det_targets_test, det_targets_train, det_targets_valid, predicted_test, predicted_train, predicted_valid = \
        _retrieve_train_valid_test_data(config, use_logits)

    _, _, = train_ML_model(X_train=predicted_train,
                           X_valid=predicted_valid,
                           X_test=predicted_test,
                           # Ground Truth
                           y_train=det_targets_train,
                           y_valid=det_targets_valid,
                           y_test=det_targets_test,
                           # Save Path
                           save_path=config.save_dir,
                           # Stratgey
                           strategy=strategy,
                           individual_features=individual_features,
                           reduced_individual_features=reduced_individual_features,
                           # Target Names
                           target_names=target_names,
                           # Further params
                           scoring_grid_search=scoring_grid_search,
                           use_SMOTE=False,
                           use_class_weights=False)  # True if "PTB_XL" in data_dir else False)

    print(f"Finished training and evaluation of {strategy_name} models")

def _retrieve_train_valid_test_data(config, use_logits):
    # Load the detached outputs to train the ML model
    detached_storage_path = os.path.join(config.save_dir.parent.parent, "output_logging")
    det_outputs_test, det_outputs_train, det_outputs_valid, \
        det_single_lead_outputs_test, det_single_lead_outputs_train, det_single_lead_outputs_valid, \
        det_targets_test, det_targets_train, det_targets_valid = retrieve_detached_data(detached_storage_path)
    # Train the ML models
    predicted_train = get_all_predictions_as_probs(det_outputs_train, det_single_lead_outputs_train) \
        if not use_logits else get_all_predictions_as_logits(det_outputs_train, det_single_lead_outputs_train)
    predicted_valid = get_all_predictions_as_probs(det_outputs_valid, det_single_lead_outputs_valid) \
        if not use_logits else get_all_predictions_as_logits(det_outputs_valid, det_single_lead_outputs_valid)
    predicted_test = get_all_predictions_as_probs(det_outputs_test, det_single_lead_outputs_test) \
        if not use_logits else get_all_predictions_as_logits(det_outputs_test, det_single_lead_outputs_test)
    return det_targets_test, det_targets_train, det_targets_valid, predicted_test, predicted_train, predicted_valid


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MACRO Paper: Strategy-based Evaluation')
    parser.add_argument('-p', '--path', default=None, type=str,
                        help='main path to CV runs(default: None)')
    parser.add_argument('--strategy', default=None, type=str,
                        help='strategy to use for final model prediction (default: None)')
    parser.add_argument('--use_logits', action='store_true',
                        help='Use the raw logits for the training of the ML models')
    parser.add_argument('--individual_features', action='store_true',
                        help='Use the individual features per class for the training of the ML models')
    parser.add_argument('--reduced_individual_features', action='store_true',
                        help='Use the reduced individual features per class for the training of the ML models '
                             'but omit the multibranch features')
    parser.add_argument('--scoring_grid_search', type=str, default="f1",
                        help='Metric used to apply the grid search for param tuning. Should be f1 or roc_auc')
    args = parser.parse_args()
    train_ML_models_on_predfined_splits(main_path=args.path,
                                        strategy=args.strategy,
                                        use_logits=args.use_logits,
                                        individual_features=args.individual_features,
                                        reduced_individual_features=args.reduced_individual_features,
                                        scoring_grid_search=args.scoring_grid_search)