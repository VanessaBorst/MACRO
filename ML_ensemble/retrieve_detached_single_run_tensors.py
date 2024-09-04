import argparse

import pickle
from pathlib import Path

import global_config
from ML_ensemble.retrieve_detached_cross_fold_tensors import prepare_model_for_inference

from parse_config import ConfigParser

from train import _set_seed
from utils import ensure_dir, read_json

import data_loader.data_loaders as module_data

# Needed for working with SSH Interpreter...
import os
import torch
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = global_config.CUDA_VISIBLE_DEVICES
global_config.suppress_warnings()

def run_inference_on_predefined_split(config,
                                      data_type=None):

    logger = config.get_logger(f'run_inference_on_split_{data_type}')

    match data_type:
        case "train":
            data_dir = config['data_loader']['args']['data_dir']
        case "valid":
            data_dir = config['data_loader']['args']['validation_split']
        case "test":
            data_dir = config['data_loader']['test_dir']
        case _:
            raise ValueError("The predefined split does not exist")

    data_loader = getattr(module_data, config['data_loader']['type'])(
        data_dir,
        batch_size=64,
        shuffle=False,
        validation_split=0.0,
        num_workers=4,
        cross_valid=False,
        test_idx=None,
        cv_train_mode=False,
        fold_id=None
    )

    device, model = prepare_model_for_inference(config, logger)

    with torch.no_grad():
        # Store the intermediate targets. Always store the output scores
        outputs_list = []
        targets_list = []
        single_lead_outputs_list = []

        for batch_idx, (padded_records, _, first_labels, labels_one_hot, record_names) in \
                enumerate(tqdm(data_loader)):
            data, target = padded_records.to(device), labels_one_hot.to(device)
            data = data.permute(0, 2, 1)  # switch seq_len and feature_size (12 = #leads)
            output = model(data)
            multi_lead_branch_active = False
            if type(output) is tuple:
                if isinstance(output[1], list):
                    # multi-branch network
                    # first element is the overall network output, the second one a list of the single lead branches
                    multi_lead_branch_active = True
                    output, single_lead_outputs = output
                    detached_single_lead_outputs = torch.stack(single_lead_outputs).detach().cpu()  # Shape: [12, 64, 9]
                    single_lead_outputs_list.append(detached_single_lead_outputs)
                else:
                    # single-branch network
                    output, attention_weights = output

            # Detach tensors needed for further tracing and metrics calculation to remove them from the graph
            detached_output = output.detach().cpu()
            detached_target = target.detach().cpu()
            outputs_list.append(detached_output)
            targets_list.append(detached_target)

            # INNER_EPOCH METRICS COULD BE CALCULATED WITH DETACHED TENSORS HERE

    # Get detached tensors from the list for further evaluation
    # For this, create a tensor from the dynamically filled list
    det_outputs = torch.cat(outputs_list).detach().cpu()
    det_targets = torch.cat(targets_list).detach().cpu()
    det_single_lead_outputs = torch.cat(([branch_outputs for branch_outputs in single_lead_outputs_list]), dim=1)\
        .permute(1,0,2).detach().cpu() if multi_lead_branch_active else None

    # Store the tensors for further evaluation
    with open(os.path.join(config.save_dir, f"{data_type}_det_outputs.p"), 'wb') as file:
        pickle.dump(det_outputs, file)
    with open(os.path.join(config.save_dir, f"{data_type}_det_targets.p"), 'wb') as file:
        pickle.dump(det_targets, file)
    if det_single_lead_outputs is not None:
        with open(os.path.join(config.save_dir, f"{data_type}_det_single_lead_outputs.p"), 'wb') as file:
            pickle.dump(det_single_lead_outputs, file)

    # ------------ Metrics could be calculated here ------------------------------------

    # ------------ ROC Plots could be created here------------------------------------

    # ------------ Confusion matrices could be determined here------------------------

    # ------------------- Summary Report -------------------
    # summary_dict = module_metric.sk_classification_summary(output=det_outputs, target=det_targets,
    #                                                        logits=_param_dict["logits"],
    #                                                        labels=_param_dict["labels"],
    #                                                        output_dict=True)
    #
    # df_sklearn_summary = pd.DataFrame.from_dict(summary_dict)

    print(f"Finished {data_type} inference")




def retrieve_detached_single_run_tensors_for_further_evaluation(main_path):
    config = load_config_and_setup_paths(main_path, sub_dir="output_logging")


    # # Adapt the log and save paths for the current fold
    # config.save_dir = Path(os.path.join(base_save_dir, "Fold_" + str(k + 1)))
    # config.log_dir = Path(os.path.join(base_log_dir, "Fold_" + str(k + 1)))
    # ensure_dir(config.save_dir)
    # ensure_dir(config.log_dir)
    # update_logging_setup_for_tune_or_cross_valid(config.log_dir)

    # Skip the training and load the trained model
    config.resume = os.path.join(main_path, "model_best.pth")


    run_inference_on_predefined_split(config=config, data_type="train")
    run_inference_on_predefined_split(config=config, data_type="valid")
    run_inference_on_predefined_split(config=config, data_type="test")

    print("Finished additional run for the predefined splits to retrieve detached tensors for further evaluation")




# def prepare_result_data_structures(total_num_folds, include_valid_results=False):
#     class_wise_metrics = ["precision", "recall", "f1-score", "torch_roc_auc", "torch_accuracy", "support"]
#     folds = ['fold_' + str(i) for i in range(1, total_num_folds + 1)]
#     iterables = [folds, class_wise_metrics]
#     multi_index = pd.MultiIndex.from_product(iterables, names=["fold", "metric"])
#     if include_valid_results:
#         valid_results = pd.DataFrame(columns=folds)
#     test_results_class_wise = pd.DataFrame(columns=['SNR', 'AF', 'IAVB', 'LBBB', 'RBBB', 'PAC', 'VEB', 'STD', 'STE',
#                                                     'macro avg', 'weighted avg'], index=multi_index)
#     test_results_single_metrics = pd.DataFrame(columns=['loss', 'sk_subset_accuracy'])
#     if not include_valid_results:
#         return class_wise_metrics, folds, test_results_class_wise, test_results_single_metrics
#     else:
#         return class_wise_metrics, folds, test_results_class_wise, test_results_single_metrics, valid_results


def load_config_and_setup_paths(main_path, sub_dir=None):
    # Load the main config
    config = read_json(os.path.join(main_path, "config.json"))
    config = ConfigParser(config=config, create_save_log_dir=False)

    # Update paths to old paths (otherwise set to current date)
    config.save_dir = Path(main_path) if sub_dir is None else Path(os.path.join(main_path, sub_dir))
    ensure_dir(config.save_dir)
    log_dir = Path(main_path.replace("models", "log"))
    config.log_dir = log_dir if sub_dir is None else Path(os.path.join(log_dir, sub_dir))
    ensure_dir(config.log_dir)

    assert not config["data_loader"]["cross_valid"]["enabled"], \
        "Cross-valid should NOT be enabled when running this script"
    assert "PTB_XL" in config["data_loader"]["args"]["data_dir"], \
        "This script does only work with PTB-XL and its fixed splits!"

    # fix random seeds for reproducibility
    global_config.SEED = config.config.get("SEED", global_config.SEED)
    _set_seed(global_config.SEED)

    return config




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MACRO Paper: Cross-Validation Evaluation')
    parser.add_argument('-p', '--path', default=None, type=str,
                        help='main path to CV runs(default: None)')
    args = parser.parse_args()
    retrieve_detached_single_run_tensors_for_further_evaluation(args.path)
