import argparse
import os

import global_config
from ML_ensemble.ML_models_for_multibranch_ensemble import train_stacking_classifier
from ML_ensemble.retrieve_detached_single_run_tensors import load_config_and_setup_paths
from ML_ensemble.train_ML_models_with_detached_single_run_tensors import _retrieve_train_valid_test_data
from utils import ensure_dir, extract_target_names_for_PTB_XL

global_config.suppress_warnings()

def train_stacking_classifiers_on_predfined_splits(main_path, use_logits=False, meta_model="xg_boost"):
    config = load_config_and_setup_paths(main_path, sub_dir=os.path.join("ML models",
                                                                         f"stacking_classifiers_{meta_model}"))
    assert config['arch']['type'] == 'FinalModelMultiBranch', \
        "The ML models can only be trained for multi-branch models!"

    data_dir = config["data_loader"]["args"]["data_dir"]
    if "PTB_XL" in data_dir:
        target_names = extract_target_names_for_PTB_XL(data_dir)
    else:
        target_names = ["IAVB", "AF", "LBBB", "PAC", "RBBB", "SNR", "STD", "STE", "VEB"]

    det_targets_test, det_targets_train, det_targets_valid, predicted_test, predicted_train, predicted_valid = \
        _retrieve_train_valid_test_data(config, use_logits)

    train_stacking_classifier(X_train=predicted_train,
                              X_valid=predicted_valid,
                              X_test=predicted_test,
                              # Ground Truth
                              y_train=det_targets_train,
                              y_valid=det_targets_valid,
                              y_test=det_targets_test,
                              save_path=config.save_dir,
                              target_names=target_names,
                              meta_model=meta_model)

    print(f"Finished training and evaluation of stacking classifiers")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MACRO Paper: Strategy-based Evaluation')
    parser.add_argument('-p', '--path', default=None, type=str,
                        help='main path to CV runs(default: None)')
    parser.add_argument('--use_logits', action='store_true',
                        help='Use the raw logits for the training of the ML models')
    parser.add_argument('--meta_model', type=str, default="xg_boost",
                        help='Define meta model for the stacking classifier (logistic_regression, xg_boost)')
    args = parser.parse_args()
    train_stacking_classifiers_on_predfined_splits(args.path, args.use_logits)
