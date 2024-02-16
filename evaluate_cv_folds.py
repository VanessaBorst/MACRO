import argparse
import copy

import time
import pickle
import pandas as pd
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt


import global_config
from data_loader.ecg_data_set import ECGDataset
from logger import update_logging_setup_for_tune_or_cross_valid

from parse_config import ConfigParser
from test import test_model
from train import _set_seed
from utils import ensure_dir, read_json
import evaluation.multi_label_metrics as module_metric

import data_loader.data_loaders as module_data

# Needed for working with SSH Interpreter...
import os

os.environ["CUDA_VISIBLE_DEVICES"] = global_config.CUDA_VISIBLE_DEVICES


def test_fold(config, data_dir, test_idx, k_fold, total_num_folds):
    df_class_wise_results, df_single_metric_results = test_model(config, tune_config=None, cv_active=True,
                                                                 cv_data_dir=data_dir, test_idx=test_idx,
                                                                 k_fold=k_fold, total_num_folds=total_num_folds)
    return df_class_wise_results, df_single_metric_results


def create_roc_curve_report(main_path):
    config = read_json(os.path.join(main_path, "config.json"))
    total_num_folds = config["data_loader"]["cross_valid"]["k_fold"]
    assert config["arch"]["args"]["multi_label_training"], \
        "Multi-Label Training should be enabled when running this script"

    # Dummy data loader for getting the class labels
    data_loader = getattr(module_data, config['data_loader']['type'])(
        config["data_loader"]["cross_valid"]["data_dir"],
        batch_size=64,
        shuffle=False,
        validation_split=0.0,
        num_workers=4,
        cross_valid=True,
        test_idx=np.array([14, 16, 17, 32, 34, 39, 60, 89, 93, 96, 111, 113, 126, 133, 180, 191, 194, 231, 255, 262, 268, 271, 281, 287, 293, 295, 321, 323, 329, 330, 342, 361, 371, 389, 390, 407, 413, 416, 424, 434, 438, 441, 443, 454, 455, 460, 464, 481, 508, 538, 544, 545, 555, 557, 581, 588, 604, 609, 612, 639, 672, 686, 687, 695, 703, 713, 719, 731, 744, 757, 762, 763, 788, 794, 819, 821, 826, 836, 837, 845, 855, 862, 867, 874, 881, 892, 915, 922, 924, 931, 936, 942, 950, 979, 998, 1007, 1038, 1061, 1092, 1100, 1104, 1105, 1111, 1118, 1128, 1132, 1133, 1146, 1150, 1153, 1159, 1221, 1222, 1230, 1235, 1246, 1248, 1257, 1258, 1282, 1305, 1312, 1326, 1329, 1336, 1343, 1346, 1363, 1378, 1381, 1382, 1389, 1401, 1406, 1418, 1425, 1427, 1438, 1439, 1442, 1443, 1452, 1474, 1507, 1510, 1513, 1514, 1521, 1524, 1537, 1543, 1551, 1557, 1558, 1593, 1594, 1602, 1639, 1647, 1649, 1656, 1662, 1669, 1675, 1684, 1704, 1731, 1734, 1755, 1759, 1760, 1764, 1771, 1789, 1792, 1794, 1795, 1847, 1871, 1873, 1888, 1891, 1897, 1902, 1922, 1968, 1994, 1996, 2004, 2009, 2010, 2011, 2023, 2040, 2046, 2050, 2056, 2072, 2080, 2108, 2140, 2159, 2161, 2174, 2185, 2191, 2201, 2213, 2216, 2227, 2240, 2241, 2255, 2270, 2289, 2301, 2338, 2341, 2359, 2360, 2378, 2388, 2395, 2397, 2412, 2414, 2425, 2428, 2446, 2467, 2478, 2480, 2483, 2490, 2495, 2507, 2512, 2521, 2523, 2528, 2533, 2541, 2543, 2551, 2557, 2563, 2566, 2567, 2574, 2578, 2587, 2599, 2610, 2620, 2621, 2622, 2624, 2625, 2637, 2640, 2654, 2657, 2666, 2684, 2709, 2721, 2741, 2747, 2751, 2766, 2768, 2778, 2795, 2800, 2810, 2819, 2838, 2843, 2854, 2856, 2863, 2871, 2878, 2880, 2884, 2894, 2917, 2936, 2937, 2944, 2945, 2947, 2969, 2989, 2999, 3003, 3026, 3034, 3047, 3054, 3064, 3072, 3078, 3082, 3084, 3086, 3092, 3095, 3101, 3116, 3121, 3132, 3143, 3155, 3178, 3200, 3224, 3226, 3230, 3252, 3258, 3261, 3273, 3287, 3295, 3306, 3310, 3316, 3323, 3325, 3329, 3331, 3348, 3356, 3358, 3365, 3367, 3373, 3377, 3387, 3393, 3406, 3427, 3429, 3454, 3458, 3463, 3465, 3476, 3481, 3486, 3505, 3512, 3518, 3527, 3537, 3539, 3556, 3561, 3573, 3574, 3579, 3582, 3622, 3624, 3651, 3655, 3656, 3665, 3668, 3687, 3700, 3702, 3708, 3719, 3727, 3737, 3741, 3751, 3753, 3796, 3815, 3831, 3839, 3860, 3862, 3866, 3875, 3881, 3891, 3892, 3895, 3899, 3914, 3918, 3933, 3937, 3940, 3943, 3956, 3959, 3963, 3971, 3978, 3997, 4004, 4007, 4011, 4022, 4028, 4029, 4030, 4043, 4050, 4060, 4063, 4097, 4099, 4102, 4104, 4112, 4118, 4142, 4143, 4149, 4166, 4169, 4173, 4184, 4195, 4201, 4202, 4231, 4235, 4242, 4245, 4254, 4277, 4293, 4295, 4309, 4310, 4313, 4321, 4343, 4344, 4355, 4360, 4384, 4386, 4391, 4418, 4421, 4436, 4451, 4454, 4455, 4462, 4486, 4496, 4501, 4507, 4508, 4538, 4541, 4547, 4550, 4576, 4584, 4590, 4591, 4602, 4610, 4626, 4628, 4636, 4645, 4648, 4671, 4679, 4692, 4703, 4718, 4721, 4725, 4731, 4736, 4756, 4761, 4765, 4782, 4783, 4841, 4846, 4851, 4861, 4862, 4867, 4904, 4933, 4939, 4940, 4947, 4951, 4955, 4958, 4968, 4976, 4981, 4982, 4985, 5000, 5009, 5015, 5016, 5036, 5046, 5060, 5088, 5093, 5109, 5126, 5150, 5182, 5189, 5206, 5208, 5218, 5236, 5243, 5246, 5250, 5255, 5256, 5257, 5264, 5306, 5307, 5314, 5318, 5325, 5327, 5338, 5339, 5342, 5345, 5353, 5367, 5373, 5397, 5410, 5417, 5421, 5424, 5451, 5461, 5474, 5484, 5494, 5497, 5530, 5531, 5533, 5543, 5550, 5571, 5586, 5594, 5613, 5616, 5626, 5630, 5642, 5646, 5650, 5660, 5664, 5679, 5703, 5722, 5731, 5746, 5747, 5762, 5795, 5815, 5837, 5845, 5857, 5862, 5879, 5880, 5888, 5915, 5921, 5934, 5957, 5971, 5973, 5975, 5986, 5992, 6010, 6027, 6049, 6052, 6065, 6077, 6084, 6085, 6086, 6087, 6113, 6114, 6115, 6118, 6139, 6169, 6173, 6182, 6195, 6202, 6226, 6233, 6236, 6245, 6250, 6257, 6262, 6269, 6284, 6302, 6306, 6309, 6354, 6357, 6359, 6364, 6368, 6382, 6409, 6427, 6433, 6440, 6445, 6448, 6468, 6481, 6483, 6484, 6494, 6503, 6504, 6505, 6515, 6517, 6527, 6533, 6544, 6545, 6553, 6564, 6566, 6570, 6578, 6601, 6611, 6615, 6624, 6625, 6628, 6640, 6648, 6653, 6659, 6681, 6694, 6717, 6740, 6752, 6771, 6775, 6782, 6806, 6812, 6816, 6833, 6834, 6836, 6838, 6857, 6862, 6870, 6874]),   #Dummy value
        cv_train_mode=False,
        fold_id=0,
        total_num_folds=total_num_folds
    )

    _param_dict = {
        "labels": data_loader.dataset.class_labels, # array([0, 1, 2, 3, 4, 5, 6, 7, 8])
        "sigmoid_probs": config["metrics"]["additional_metrics_args"].get("sigmoid_probs", False),
        "logits": config["metrics"]["additional_metrics_args"].get("logits", False),
    }

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    fig, ax = plt.subplots()

    for i in range(total_num_folds):
        # Load the outputs and targets for this fold
        with open(os.path.join(main_path, "Fold_" + str(i + 1), "additional_eval", "det_outputs.p"), 'rb') as file:
            det_outputs = pickle.load(file)
        with open(os.path.join(main_path, "Fold_" + str(i + 1), "additional_eval", "det_targets.p"), 'rb') as file:
            det_targets = pickle.load(file)

        # Plot the ROC curve for this fold
        fpr, tpr, thresholds = module_metric.torch_roc(output=det_outputs, target=det_targets,
                                                       sigmoid_probs=_param_dict["sigmoid_probs"],
                                                       logits=_param_dict["logits"],
                                                       labels=_param_dict["labels"])
        roc_auc_scores = module_metric.class_wise_torch_roc_auc(output=det_outputs, target=det_targets,
                                                                sigmoid_probs=_param_dict["sigmoid_probs"],
                                                                logits=_param_dict["logits"],
                                                                labels=_param_dict["labels"])

        fig, axs = plt.subplots(3, 3, figsize=(18, 10))
        axis_0 = 0
        axis_1 = 0
        line_width = 2
        target_names = ["IAVB", "AF", "LBBB", "PAC", "RBBB", "SNR", "STD", "STE", "VEB"]
        desired_order = ['SNR', 'AF', 'IAVB', 'LBBB', 'RBBB', 'PAC', 'VEB', 'STD', 'STE']
        class_shares = {'SNR': ' (12.5%)',
                        'AF': ' (16.6%)',
                        'I-AVB': ' (9.8%)',
                        'LBBB': ' (3.2%)',
                        'RBBB': ' (25.2%)',
                        'PAC': ' (8.4%)',
                        'PVC': ' (9.5%)',
                        'STD': ' (11.8%)',
                        'STE': ' (3.0%)'}
        for i in range(0, 9):
            desired_class = desired_order[i]
            idx = target_names.index(desired_class)
            fpr_class_i = fpr[idx].numpy()
            tpr_class_i = tpr[idx].numpy()
            # Scale values by a factor of 1000 to better match the cpsc raw values
            axs[axis_0, axis_1].plot(fpr_class_i, tpr_class_i, color='darkorange', lw=line_width,
                                     label='ROC curve (AUC = %0.3f)' % roc_auc_scores[idx])
            axs[axis_0, axis_1].plot([0, 1], [0, 1], color='navy', lw=line_width, linestyle='--')
            axs[axis_0, axis_1].tick_params(axis='both', which='major', labelsize=20)
            axs[axis_0, axis_1].set_yticks([0.25, 0.5, 0.75, 1])

            if axis_0 == 2:
                axs[axis_0, axis_1].set_xlabel('False Positive Rate', fontsize=20)
            if axis_1 == 0:
                axs[axis_0, axis_1].set_ylabel('True Positive Rate', fontsize=20)

            axs[axis_0, axis_1].set_xlim([0.0, 1.0])
            axs[axis_0, axis_1].set_ylim([0.0, 1.05])
            axs[axis_0, axis_1].legend(loc="lower right", fontsize=20)

            class_name = str(target_names[idx]).replace('IAVB', 'I-AVB').replace('VEB', 'PVC')
            axs[axis_0, axis_1].set_title('Class ' + class_name + class_shares[class_name], fontsize=20)
            # Also save the single plots per class
            file_name = 'roc_curve_' + target_names[idx] + '.pdf'
            # 'ROC curve for class ' + str(target_names[idx]) + str())
            extent = axs[axis_0, axis_1].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            # Pad the saved area by 30% in the x-direction and 35% in the y-direction
            # TODO: AttributeError: 'collections.OrderedDict' object has no attribute 'test_output_dir'
            fig.savefig(config.test_output_dir / file_name, bbox_inches=extent.expanded(1.3, 1.35))
            axis_1 = (axis_1 + 1) % 3
            if axis_1 == 0:
                axis_0 += 1
        plt.tight_layout(pad=2, h_pad=3, w_pad=1.5)
        plt.savefig(config.test_output_dir / "roc_curves_with_shares.pdf")


        # viz = plot_roc_curve(model, X[test], y[test],
        #                      name='ROC fold {}'.format(i),
        #                      alpha=0.3, lw=1, ax=ax)
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)

    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
            label='Chance', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(mean_fpr, mean_tpr, color='b',
            label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
            lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                    label=r'$\pm$ 1 std. dev.')

    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
           title=model_name)
    ax.legend(loc="lower right")
    plt.savefig(
        "../results/plots/test/roc_%s_%0i_features_%0i_test.pdf" % (
            model_name, feature_size, len(X)),
        dpi=100, facecolor='w', edgecolor='b', orientation='portrait', transparent=False, bbox_inches=None,
        pad_inches=0.1)
    print("Created %s ROC figure" % model_name)
    plt.close()


def evaluate_cross_validation(main_path):
    config = load_config_and_setup_paths(main_path, "additional_eval")

    total_num_folds = config["data_loader"]["cross_valid"]["k_fold"]
    data_dir = config["data_loader"]["cross_valid"]["data_dir"]

    # Update data dir in Dataloader ARGs!
    config["data_loader"]["args"]["data_dir"] = data_dir
    dataset = ECGDataset(data_dir)
    n_samples = len(dataset)

    # Get the main config and the dirs for logging and saving checkpoints
    base_config = copy.deepcopy(config)
    base_save_dir = config.save_dir
    base_log_dir = config.log_dir

    # Divide the samples into k distinct sets
    fold_data = split_dataset_into_folds(n_samples, total_num_folds)

    # Save the results of each run
    class_wise_metrics, folds, test_results_class_wise, test_results_single_metrics = \
        prepare_result_data_structures(total_num_folds)

    print("Starting with " + str(total_num_folds) + "-fold cross validation")
    valid_fold_index = total_num_folds - 2
    test_fold_index = total_num_folds - 1

    for k in range(total_num_folds):
        print("Starting fold " + str(k))
        # Get the idx for valid and test samples, train idx not needed
        # TODO ADAPT DIR PATHS
        train_idx, valid_idx, test_idx = get_train_valid_test_indices(base_save_dir,
                                                                      dataset,
                                                                      fold_data,
                                                                      k,
                                                                      test_fold_index,
                                                                      valid_fold_index)

        # Adapt the log and save paths for the current fold
        config.save_dir = Path(os.path.join(base_save_dir, "Fold_" + str(k + 1), "additional_eval"))
        config.log_dir = Path(os.path.join(base_log_dir, "Fold_" + str(k + 1), "additional_eval"))
        ensure_dir(config.save_dir)
        ensure_dir(config.log_dir)
        update_logging_setup_for_tune_or_cross_valid(config.log_dir)

        # Skip the training and load the trained model
        config.resume = os.path.join(base_save_dir, "Fold_" + str(k + 1), "model_best.pth")

        #  Do the testing and add the fold results to the dfs
        config.test_output_dir = Path(os.path.join(base_save_dir, "Fold_" + str(k + 1), "additional_eval"))
        ensure_dir(config.test_output_dir)
        fold_eval_class_wise, fold_eval_single_metrics = test_fold(config, data_dir=data_dir,
                                                                   test_idx=test_idx,
                                                                   k_fold=k, total_num_folds=total_num_folds)

        # Class-Wise Metrics
        test_results_class_wise.loc[(folds[k], fold_eval_class_wise.index), fold_eval_class_wise.columns] = \
            fold_eval_class_wise.values
        # Single Metrics
        pd_series = fold_eval_single_metrics.loc['value']
        pd_series.name = folds[k]
        test_results_single_metrics = test_results_single_metrics.append(pd_series)

        # Update the indices and reset the config (including resume!)
        valid_fold_index = (valid_fold_index + 1) % (total_num_folds)
        test_fold_index = (test_fold_index + 1) % (total_num_folds)
        config = copy.deepcopy(base_config)

    # Summarize the results of the cross validation and write everything to files
    # --------------------------- Test Class-Wise Metrics ---------------------------
    iterables_summary = [["mean", "median"], class_wise_metrics]
    multi_index = pd.MultiIndex.from_product(iterables_summary, names=["merging", "metric"])
    test_results_class_wise_summary = pd.DataFrame(
        columns=['SNR', 'AF', 'IAVB', 'LBBB', 'RBBB', 'PAC', 'VEB', 'STD', 'STE',
                 'macro avg', 'weighted avg'], index=multi_index)
    for metric in class_wise_metrics:
        test_results_class_wise_summary.loc[('mean', metric)] = test_results_class_wise.xs(metric, level=1).mean()
        test_results_class_wise_summary.loc[('median', metric)] = test_results_class_wise.xs(metric, level=1).median()

    ensure_dir(os.path.join(base_save_dir, "additional_eval"))
    path = os.path.join(base_save_dir, "additional_eval", "test_results_class_wise.p")
    with open(path, 'wb') as file:
        pickle.dump(test_results_class_wise, file)
    path = os.path.join(base_save_dir, "additional_eval", "test_results_class_wise_summary.p")
    with open(path, 'wb') as file:
        pickle.dump(test_results_class_wise_summary, file)

    # --------------------------- Test Single Metrics ---------------------------
    test_results_single_metrics.loc['mean'] = test_results_single_metrics.mean()
    test_results_single_metrics.loc['median'] = test_results_single_metrics[:][:-1].median()

    path = os.path.join(base_save_dir, "additional_eval", "test_results_single_metrics.p")
    with open(path, 'wb') as file:
        pickle.dump(test_results_single_metrics, file)

    #  --------------------------- Omit Train Result ---------------------------

    # --------------------------- Test Metrics To Latex---------------------------
    with open(os.path.join(base_save_dir, "additional_eval", 'cross_validation_results.tex'), 'w') as file:
        file.write("Class-Wise Summary:\n\n")
        test_results_class_wise_summary.to_latex(buf=file, index=False, float_format="{:0.3f}".format,
                                                 escape=False)
        file.write("\n\n\nSingle Metrics:\n\n")
        test_results_single_metrics.to_latex(buf=file, index=False, float_format="{:0.3f}".format,
                                             escape=False)

    print("Finished additional eval of cross-fold-validation")


def prepare_result_data_structures(total_num_folds, include_valid_results=False):
    class_wise_metrics = ["precision", "recall", "f1-score", "torch_roc_auc", "torch_accuracy", "support"]
    folds = ['fold_' + str(i) for i in range(1, total_num_folds + 1)]
    iterables = [folds, class_wise_metrics]
    multi_index = pd.MultiIndex.from_product(iterables, names=["fold", "metric"])
    if include_valid_results:
        valid_results = pd.DataFrame(columns=folds)
    test_results_class_wise = pd.DataFrame(columns=['SNR', 'AF', 'IAVB', 'LBBB', 'RBBB', 'PAC', 'VEB', 'STD', 'STE',
                                                    'macro avg', 'weighted avg'], index=multi_index)
    test_results_single_metrics = pd.DataFrame(columns=['loss', 'sk_subset_accuracy'])
    if not include_valid_results:
        return class_wise_metrics, folds, test_results_class_wise, test_results_single_metrics
    else:
        return class_wise_metrics, folds, test_results_class_wise, test_results_single_metrics, valid_results


def load_config_and_setup_paths(main_path, sub_dir=None):
    # Load the main config
    config = read_json(os.path.join(main_path, "config.json"))
    config = ConfigParser(config=config, create_save_log_dir=False)

    # Update paths to old paths (otherwise set to current date)
    config.save_dir = Path(main_path) if sub_dir is None else os.path.join(Path(main_path), sub_dir)
    log_dir = Path(main_path.replace("models", "log"))
    config.log_dir = log_dir if sub_dir is None else os.path.join(log_dir, sub_dir)

    assert config["data_loader"]["cross_valid"]["enabled"], "Cross-valid should be enabled when running this script"

    # fix random seeds for reproducibility
    global_config.SEED = config.config.get("SEED", global_config.SEED)
    _set_seed(global_config.SEED)

    return config


def get_train_valid_test_indices(main_path, dataset, fold_data, k, test_fold_index, valid_fold_index,
                                 merge_train_into_valid_idx=False):
    train_sets = [fold for id, fold in enumerate(fold_data)
                  if id != valid_fold_index and id != test_fold_index]
    train_idx = np.concatenate(train_sets, axis=0)
    valid_idx = fold_data[valid_fold_index]
    test_idx = fold_data[test_fold_index]
    # Load the old data split for sanity check
    with open(os.path.join(main_path, "Fold_" + str(k + 1), "data_split.csv"), "r") as file:
        data = pd.read_csv(file, index_col=0, header=None)
        dict = {
            "train_records": data.loc["train_records"],
            "valid_records": data.loc["valid_records"].dropna(),
            "test_records": data.loc["test_records"].dropna()
        }
    assert np.array(dataset.records)[train_idx].tolist() == dict["train_records"].tolist(), \
        "Train indices do not match the old data split"
    assert np.array(dataset.records)[valid_idx].tolist() == dict["valid_records"].tolist(), \
        "Valid indices do not match the old data split"
    assert np.array(dataset.records)[test_idx].tolist() == dict["test_records"].tolist(), \
        "Test indices do not match the old data split"

    if merge_train_into_valid_idx:
        valid_idx = np.concatenate([fold_data[valid_fold_index], train_idx], axis=0)

    return train_idx, valid_idx, test_idx


def split_dataset_into_folds(n_samples, total_num_folds):
    idx_full = np.arange(n_samples)
    np.random.shuffle(idx_full)

    fold_size = n_samples // total_num_folds
    fold_data = []
    lower_limit = 0
    for i in range(0, total_num_folds):
        if i != total_num_folds - 1:
            fold_data.append(idx_full[lower_limit:lower_limit + fold_size])
            lower_limit = lower_limit + fold_size
        else:
            # Last fold may be a bit larger
            fold_data.append(idx_full[lower_limit:n_samples])
    return fold_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MACRO Paper: Cross-Validation Evaluation')
    parser.add_argument('-p', '--path', default=None, type=str,
                        help='main path to CV runs(default: None)')
    args = parser.parse_args()
    evaluate_cross_validation(args.path)
    create_roc_curve_report(args.path)
